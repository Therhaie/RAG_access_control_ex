from __future__ import annotations
import argparse
import json
import os
import shutil
import time
import warnings
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tenacity import retry, stop_after_attempt, wait_exponential

import numpy as np
import chromadb
from chromadb.config import Settings
from scipy.stats import ortho_group

# --- Imports from your project ---
from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX
from plot_PCA import get_all_chunk_ids, get_list_id_targeted_chunk

# --- Constants (configurable via CLI) ---
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results_experiment")
GT_FILE = Path("RAGBench_whole/merged_id_triplets_with_metadata2.json")
PATH_TIME_DATABASE_CREATION = Path("time_database_creation.jsonl")

# --- Logging ---
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CLI Arguments ---
def parse_args():
    parser = argparse.ArgumentParser(description="Access-control embedding experiment (parallel)")
    parser.add_argument("--gt", default=str(GT_FILE), help="Path to ground truth file.")
    parser.add_argument("--top-k", type=int, default=20, help="Number of results to retrieve per query.")
    parser.add_argument("--large-value-chunk", type=float, default=10.0, help="High value for chunk encoding.")
    parser.add_argument("--large-value-query", type=float, default=10.0, help="High value for query encoding.")
    parser.add_argument("--number-clusters", type=int, default=20, help="Number of clusters for extra dimensions.")
    parser.add_argument("--batch-size-db", type=int, default=500, help="Batch size for ChromaDB upserts.")
    parser.add_argument("--batch-size-results", type=int, default=300, help="Batch size for writing results to file.")
    parser.add_argument("--workers", type=int, default=64, help="Number of worker threads for parallel tasks.")
    parser.add_argument("--normalize-after", action="store_true", default=True, help="Normalize vectors after augmentation.")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose output.")
    return parser.parse_args()

args = parse_args()

# Override constants with CLI values
DEFAULT_TOP_K = args.top_k
DEFAULT_LARGE_VAL = args.large_value_chunk
DEFAULT_LARGE_VAL_QUERY = args.large_value_query
NUMBER_CLUSTER = args.number_clusters
BATCH_SIZE = args.batch_size_db
NUMBER_THREADS = args.workers
DISTANCE_METRIC = "cosine"

# --- Paths ---
ORIGINAL_CHROMA = os.path.join(os.getcwd(), "./experiment_chroma_db")
ORIGINAL_COLLECTION = "experimental_baseline_db"
AUG_CHROMA_BASE = os.path.join(os.getcwd(), "./chroma_extra_dim_experiment")
AUGMENTED_NAME = f"augmented_db{DISTANCE_METRIC}val_query_half{DEFAULT_LARGE_VAL}{NUMBER_CLUSTER}"
META_CHROMA_BASE = os.path.join(os.getcwd(), "./chroma_meta_db_experiment")
META_NAME = "meta_access_control_experiment"
ROTATED_CHROMA = os.path.join(os.getcwd(), "./chroma_rotated_db_log")
ROTATED_COLLECTION = "rotated_experiment"

# --- Data Structures ---
@dataclass
class ExtraDimConfig:
    large_value: float = DEFAULT_LARGE_VAL
    large_value_query: float = DEFAULT_LARGE_VAL_QUERY
    normalize_after: bool = args.normalize_after
    n_queries: int = NUMBER_CLUSTER

    @property
    def config_id(self) -> str:
        lv = f"{self.large_value:.0e}".replace("+", "")
        lv_query = f"{self.large_value_query:.0e}".replace("+", "")
        na = "na1" if self.normalize_after else "na0"
        return f"q{self.n_queries}_lv{lv}_lvq{lv_query}_{na}"

@dataclass
class RawQueryResults:
    user_index: int
    query_index: str
    high_value_encoding_chunk: float
    high_value_encoding_query: float
    top_k: int
    number_user: int
    distance_used: str
    list_ground_truth: List[str]
    list_retrieved_meta_auth: List[str]
    list_retrieved_meta_unauth: List[str]
    list_retrieved_rot_auth: List[str]
    list_retrieved_rot_unauth: List[str]
    list_retrieved_aug_auth: List[str]
    list_retrieved_aug_unauth: List[str]
    embedding_retrieved_ground_truth: Dict[str, List[float]]
    embedding_retrieved_meta_auth: Dict[str, List[float]]
    embedding_retrieved_meta_unauth: Dict[str, List[float]]
    embedding_retrieved_rot_auth: Dict[str, List[float]]
    embedding_retrieved_rot_unauth: Dict[str, List[float]]
    embedding_retrieved_aug_auth: Dict[str, List[float]]
    embedding_retrieved_aug_unauth: Dict[str, List[float]]
    embedding_query_meta_auth: List[float]
    embedding_query_meta_unauth: List[float]
    embedding_query_rot_auth: List[float]
    embedding_query_rot_unauth: List[float]
    embedding_query_aug_auth: List[float]
    embedding_query_aug_unauth: List[float]
    t_embed_query_s: float
    t_augment_auth_s: float
    t_augment_unauth_s: float
    t_query_aug_auth_s: float
    t_query_aug_unauth_s: float
    t_query_meta_auth_s: float
    t_query_meta_unauth_s: float
    t_apply_rotation_s: float
    t_query_orig_s: float
    t_query_rotated_s: float
    timestamp: str = datetime.now(timezone.utc).isoformat()

    def to_json_dict(self) -> Dict:
        """Convert all fields to JSON-serializable types."""
        result = asdict(self)
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, dict):
                result[key] = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()}
            elif isinstance(value, list):
                result[key] = [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
        return result

# --- Timing and Logging ---
TIMING_LOG: List[Dict] = []

def timed(label: str):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                result = fn(*args, **kwargs)
            finally:
                TIMING_LOG.append({
                    "label": label,
                    "start_iso": start_iso,
                    "duration_s": round(time.perf_counter() - start, 6),
                    "args_repr": str(args[:2])[:120],
                })
            return result
        return wrapper
    return decorator

def save_timing_log(eval_results: Optional[List] = None) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    output = {"entries": TIMING_LOG, "summary": _compute_timing_summary()}
    if eval_results:
        output["retrieval_stats"] = _compute_retrieval_stats(eval_results)
    with open(LOGS_DIR / f"dim_timing_largeval_{DEFAULT_LARGE_VAL}_topk_{DEFAULT_TOP_K}_extra_dim_untargeted.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Timing log saved to {LOGS_DIR}")

def _compute_timing_summary() -> Dict:
    buckets: Dict[str, List[float]] = {}
    for entry in TIMING_LOG:
        buckets.setdefault(entry["label"], []).append(entry["duration_s"])
    summary = {}
    for label, durations in sorted(buckets.items()):
        arr = np.array(durations)
        summary[label] = {
            "n": int(len(arr)),
            "mean_s": round(float(arr.mean()), 6),
            "std_s": round(float(arr.std()), 6),
            "min_s": round(float(arr.min()), 6),
            "max_s": round(float(arr.max()), 6),
        }
    return summary

def _compute_retrieval_stats(eval_results: List) -> Dict:
    fields = [
        "targeted_in_auth_query_aug_db",
        "targeted_in_unauth_query_aug_db",
        "targeted_in_auth_meta",
        "targeted_in_unauth_meta",
        "targeted_in_auth_query_rot_db",
        "targeted_in_unauth_query_rot_db",
    ]
    totals = np.array([r.n_targeted_chunks for r in eval_results], dtype=float)
    stats = {}
    for f in fields:
        counts = np.array([getattr(r, f, 0) for r in eval_results], dtype=float)
        fracs = np.where(totals > 0, counts / totals, 0.0)
        stats[f] = {
            "mean_count": round(float(counts.mean()), 4),
            "std_count": round(float(counts.std()), 4),
            "min_count": round(float(counts.min()), 4),
            "max_count": round(float(counts.max()), 4),
            "mean_frac": round(float(fracs.mean()), 4),
            "std_frac": round(float(fracs.std()), 4),
            "n_queries": int(len(counts)),
        }
    return stats

# --- Rotation Helpers ---
def _seed_from_key(key: str) -> int:
    return int(hashlib.sha256(str(key).encode()).hexdigest()[:8], 16) % (2 ** 31)

def make_rotation_matrix(dim: int, seed: int) -> np.ndarray:
    return ortho_group.rvs(dim=dim, random_state=seed).astype(np.float32)

def apply_rotation(vec: np.ndarray, R: np.ndarray) -> np.ndarray:
    if vec.ndim == 1:
        return R @ vec
    return (R @ vec.T).T

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

class RotationRegistry:
    def __init__(self, dim: int):
        self.dim = dim
        self._store: Dict[str, Dict] = {}

    def get_or_create(self, key: str) -> np.ndarray:
        if key not in self._store:
            seed = _seed_from_key(key)
            self._store[key] = {"seed": seed, "matrix": make_rotation_matrix(self.dim, seed)}
        return self._store[key]["matrix"]

    def get(self, key: str) -> Optional[np.ndarray]:
        entry = self._store.get(key)
        return entry["matrix"] if entry else None

    def seed_of(self, key: str) -> Optional[int]:
        entry = self._store.get(key)
        return entry["seed"] if entry else None

    def to_serialisable(self) -> Dict:
        return {k: v["seed"] for k, v in self._store.items()}

    @classmethod
    def from_serialisable(cls, data: Dict, dim: int) -> "RotationRegistry":
        reg = cls(dim=dim)
        for key, seed in data.items():
            reg._store[key] = {"seed": seed, "matrix": make_rotation_matrix(dim, seed)}
        return reg

# --- Augmentation Helpers ---
def augment_chunk(base_vec: np.ndarray, cfg: ExtraDimConfig, query_index: Optional[int], untargeted: bool = False) -> np.ndarray:
    extra = np.zeros(NUMBER_CLUSTER, dtype=np.float32)
    if untargeted:
        return np.concatenate([base_vec.astype(np.float32), extra])
    if query_index is not None:
        extra[query_index] = cfg.large_value
    return np.concatenate([base_vec.astype(np.float32), extra])

def augment_query(base_vec: np.ndarray, cfg: ExtraDimConfig, query_index: int, authorised: bool) -> np.ndarray:
    extra = np.zeros(NUMBER_CLUSTER, dtype=np.float32)
    if authorised:
        extra[query_index] = cfg.large_value_query
    return np.concatenate([base_vec.astype(np.float32), extra])

# --- ChromaDB Helpers ---
def _get_client(path: str) -> chromadb.PersistentClient:
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _upsert_batch(collection, batch_ids: List[str], batch_embeddings: List, batch_documents: List, batch_metadatas: List) -> None:
    collection.upsert(
        ids=batch_ids,
        embeddings=batch_embeddings,
        documents=batch_documents,
        metadatas=batch_metadatas
    )

def _get_original_collection(path: str, name: str):
    return _get_client(path).get_collection(name=name)

def _get_aug_collection(cfg_id: str):
    return _get_client(AUG_CHROMA_BASE).get_or_create_collection(
        name=f"{AUGMENTED_NAME}_{cfg_id}", metadata={"hnsw:space": DISTANCE_METRIC}
    )

def _get_meta_collection():
    return _get_client(META_CHROMA_BASE).get_or_create_collection(
        name=META_NAME, metadata={"hnsw:space": DISTANCE_METRIC}
    )

def _get_or_create_rotated_collection(path: str, name: str):
    return _get_client(path).get_or_create_collection(
        name=name, metadata={"hnsw:space": DISTANCE_METRIC}
    )

# --- Fetch Chunk ---
# @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
# def fetch_chunk(collection, triplet_index: str, document_id: str, phrase_seq: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
#     try:
#         result = collection.get(
#             where={
#                 "$and": [
#                     {"triplet_index": {"$eq": triplet_index}},
#                     {"document_id": {"$eq": document_id}},
#                     {"phrase_seq": {"$eq": phrase_seq}},
#                 ]
#             },
#             include=["embeddings", "documents"],
#         )
#         embs = result.get("embeddings", [[]])
#         docs = result.get("documents", [[]])
#         if len(embs[0]) > 0:
#             return np.array(embs[0], dtype=np.float32), (docs[0] if docs else None)
#     except Exception as e:
#         warnings.warn(f"fetch_chunk failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
#     return None, None


@timed("fetch_chunk_from_original_db")
def fetch_chunk(collection, triplet_index: str, document_id: str, phrase_seq: str) -> Tuple[Optional[np.ndarray], Optional[str]]:
    try:
        result = collection.get(
            where={
                "$and": [
                    {"triplet_index": {"$eq": triplet_index}},
                    {"document_id": {"$eq": document_id}},
                    {"phrase_seq": {"$eq": phrase_seq}},
                ]
            },
            include=["embeddings", "documents"],
        )
        embs = result.get("embeddings", [[]])
        docs = result.get("documents", [[]])
        if len(embs[0]) > 0:
            return np.array(embs[0], dtype=np.float32), (docs[0] if docs else None)
    except Exception as e:
        warnings.warn(f"fetch_chunk failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
    return None, None


# --- Build Augmented DB ---
def _build_aug_record(
    record: Dict,
    query_index: int,
    cfg: ExtraDimConfig,
    orig_collection,
    aug_collection,
) -> int:
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in stable_chunks:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{cfg.config_id}_{triplet_index}_{did}_{pseq}"
            base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if base_vec is None:
                continue
            aug_vec = augment_chunk(base_vec, cfg, query_index=query_index)
            batch_ids.append(cid)
            batch_embeddings.append(aug_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": triplet_index,
                "document_id": did,
                "phrase_seq": pseq,
                "query_index": query_index,
                "config_id": cfg.config_id,
                "restricted": True,
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        aug_collection.upsert(
    ids=batch_ids,
    embeddings=batch_embeddings,
    documents=batch_documents,
    metadatas=batch_metadatas
)
        # _upsert_batch(aug_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
    return n_ok

def build_aug_db_parallel(
    gt_records: List[Dict],
    cfg: ExtraDimConfig,
    orig_collection,
    aug_collection,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> None:
    logger.info("Building augmented DB (parallel with threads)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, record in enumerate(gt_records):
            triplet_index = record.get("id_triplets")
            if not triplet_index or not record.get("targeted_chunk"):
                continue
            futures.append(
                executor.submit(
                    _build_aug_record,
                    record,
                    i % NUMBER_CLUSTER,
                    cfg,
                    orig_collection,
                    aug_collection,
                )
            )
        for future in as_completed(futures):
            future.result()
    logger.info("✅ Augmented DB build complete")

def _process_aug_untargeted_batch(
    batch: List[str],
    cfg: ExtraDimConfig,
    orig_collection,
    aug_collection,
) -> int:
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in batch:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{cfg.config_id}_{chunk_id}"
            base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if base_vec is None:
                continue
            aug_vec = augment_chunk(base_vec, cfg, query_index=None, untargeted=True)
            batch_ids.append(cid)
            batch_embeddings.append(aug_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": tid,
                "document_id": did,
                "phrase_seq": pseq,
                "query_index": None,
                "config_id": cfg.config_id,
                "restricted": False,
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        aug_collection.upsert(
    ids=batch_ids,
    embeddings=batch_embeddings,
    documents=batch_documents,
    metadatas=batch_metadatas
)
        # _upsert_batch(aug_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
    return n_ok

def _build_aug_db_all_untargeted_chunks_parallel(
    gt_records: List[Dict],
    cfg: ExtraDimConfig,
    orig_collection,
    aug_collection,
    batch_size: int = BATCH_SIZE,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> None:
    list_of_chunk_ids = set(get_all_chunk_ids(gt_records))
    list_of_targeted_chunk_ids = set(get_list_id_targeted_chunk(gt_records))
    untargeted_chunks = list(list_of_chunk_ids - list_of_targeted_chunk_ids)
    logger.info(f"Found {len(untargeted_chunks)} untargeted chunks to add to augmented DB.")
    batches = [untargeted_chunks[i:i + batch_size] for i in range(0, len(untargeted_chunks), batch_size)]
    total_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_aug_untargeted_batch, batch, cfg, orig_collection, aug_collection)
            for batch in batches
        ]
        for future in as_completed(futures):
            total_ok += future.result()
    logger.info(f"✅ Added {total_ok} untargeted chunks to augmented DB")

# --- Build Metadata DB ---
def _build_meta_record(
    record: Dict,
    orig_collection,
    meta_collection,
) -> int:
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in stable_chunks:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"meta_{triplet_index}_{did}_{pseq}"
            base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if base_vec is None:
                continue
            batch_ids.append(cid)
            batch_embeddings.append(base_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": triplet_index,
                "document_id": did,
                "phrase_seq": pseq,
                "restricted": f"{triplet_index}_True",
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        meta_collection.upsert(
    ids=batch_ids,
    embeddings=batch_embeddings,
    documents=batch_documents,
    metadatas=batch_metadatas
)
        # _upsert_batch(meta_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
    return n_ok

def build_meta_db_parallel(
    gt_records: List[Dict],
    orig_collection,
    meta_collection,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> None:
    logger.info("Building metadata DB (parallel with threads)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for record in gt_records:
            if not record.get("id_triplets") or not record.get("targeted_chunk"):
                continue
            futures.append(
                executor.submit(
                    _build_meta_record,
                    record,
                    orig_collection,
                    meta_collection,
                )
            )
        for future in as_completed(futures):
            future.result()
    logger.info("✅ Metadata DB build complete")

def _process_meta_untargeted_batch(
    batch: List[str],
    orig_collection,
    meta_collection,
) -> int:
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in batch:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{tid}_{did}_{pseq}"
            base_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if base_vec is None:
                continue
            batch_ids.append(cid)
            batch_embeddings.append(base_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": tid,
                "document_id": did,
                "phrase_seq": pseq,
                "restricted": False,
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        # _upsert_batch(meta_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
        meta_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    return n_ok

def build_meta_db_add_untargeted_chunks_parallel(
    gt_records: List[Dict],
    orig_collection,
    meta_collection,
    batch_size: int = BATCH_SIZE,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> None:
    list_of_chunk_ids = set(get_all_chunk_ids(gt_records))
    list_of_targeted_chunk_ids = set(get_list_id_targeted_chunk(gt_records))
    untargeted_chunks = list(list_of_chunk_ids - list_of_targeted_chunk_ids)
    batches = [untargeted_chunks[i:i + batch_size] for i in range(0, len(untargeted_chunks), batch_size)]
    total_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_meta_untargeted_batch, batch, orig_collection, meta_collection)
            for batch in batches
        ]
        for future in as_completed(futures):
            total_ok += future.result()
    logger.info(f"✅ Added {total_ok} untargeted chunks to metadata DB")

# --- Build Rotated DB ---
@timed("build_rotated_db_single_record")
def _build_record_parallel(
    record: Dict,
    registry: RotationRegistry,
    orig_collection,
    rot_collection,
    user_index: int,
) -> int:
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    R = registry.get_or_create(str(user_index))
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in stable_chunks:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{triplet_index}_{did}_{pseq}"
            orig_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if orig_vec is None:
                warnings.warn(f"Build: chunk not found {chunk_id} — skipping.")
                continue
            rot_vec = apply_rotation(orig_vec, R)
            batch_ids.append(cid)
            batch_embeddings.append(rot_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": triplet_index,
                "document_id": did,
                "phrase_seq": pseq,
                "rotation_seed": int(registry.seed_of(str(user_index))),
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        # _upsert_batch(rot_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
        rot_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    return n_ok

@timed("build_rotated_db")
def build_rotated_db_parallel(
    gt_records: List[Dict],
    registry: RotationRegistry,
    orig_collection,
    rot_collection,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> RotationRegistry:
    logger.info(f"Building rotated database (parallel) ({len(gt_records)} records)")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, record in enumerate(gt_records):
            triplet_index = record.get("id_triplets")
            if not triplet_index or not record.get("targeted_chunk"):
                continue
            futures.append(
                executor.submit(
                    _build_record_parallel,
                    record,
                    registry,
                    orig_collection,
                    rot_collection,
                    i % NUMBER_CLUSTER,
                )
            )
        for future in as_completed(futures):
            future.result()
    logger.info(f"✅ Build complete — rotated DB")
    return registry

@timed("build_rotated_db_untargeted_chunks")
def _process_rot_untargeted_batch(
    batch: List[str],
    orig_collection,
    rot_collection,
) -> int:
    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []
    n_ok = 0
    for chunk_id in batch:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{tid}_{did}_{pseq}"
            orig_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if orig_vec is None:
                continue
            batch_ids.append(cid)
            batch_embeddings.append(orig_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": tid,
                "document_id": did,
                "phrase_seq": pseq,
                "rotation_seed": None,
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue
    if batch_ids:
        # _upsert_batch(rot_collection, batch_ids, batch_embeddings, batch_documents, batch_metadatas)
        rot_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    return n_ok

@timed("build_rotated_db_untargeted_chunks")
def _build_rotated_db_untargeted_chunks_parallel(
    gt_records: List[Dict],
    registry: RotationRegistry,
    orig_collection,
    rot_collection,
    batch_size: int = BATCH_SIZE,
    max_workers: int = NUMBER_THREADS,
    verbose: bool = True,
) -> int:
    list_of_chunk_ids = set(get_all_chunk_ids(gt_records))
    list_of_targeted_chunk_ids = set(get_list_id_targeted_chunk(gt_records))
    untargeted_chunks = list(list_of_chunk_ids - list_of_targeted_chunk_ids)
    batches = [untargeted_chunks[i:i + batch_size] for i in range(0, len(untargeted_chunks), batch_size)]
    total_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                _process_rot_untargeted_batch,
                batch,
                orig_collection,
                rot_collection,
            )
            for batch in batches
        ]
        for future in as_completed(futures):
            total_ok += future.result()
    logger.info(f"✅ Added {total_ok} untargeted chunks to rotated DB")
    return total_ok




# --- Query Phase ---
def _query_record_parallel(args):
    record, query_index, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k = args
    question = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id = f"triplet_{triplet_index}"
    R = registry.get_or_create(str(query_index))

    # Step 1: Embed raw query
    t0 = time.perf_counter()
    raw_q = np.array(embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32)
    t_embed_query_s = time.perf_counter() - t0

    # Step 2: Augment and rotate
    t0 = time.perf_counter()
    auth_q = augment_query(raw_q, cfg, query_index, authorised=True)
    t_augment_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    unauth_q = augment_query(raw_q, cfg, query_index, authorised=False)
    t_augment_unauth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rot_query_vec = apply_rotation(raw_q, R)
    t_apply_rotation_s = time.perf_counter() - t0

    # Step 3: Retrieve
    n_results = min(top_k, max(aug_collection.count(), meta_collection.count(), rot_collection.count()))

    t0 = time.perf_counter()
    orig_res = orig_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=n_results,
        include=["metadatas", "embeddings"],
    )
    t_query_orig_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    aug_res_auth = aug_collection.query(
        query_embeddings=[auth_q.tolist()],
        n_results=n_results,
        include=["metadatas", "embeddings"],
    )
    t_query_aug_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    aug_res_unauth = aug_collection.query(
        query_embeddings=[unauth_q.tolist()],
        n_results=n_results,
        include=["metadatas", "embeddings"],
    )
    t_query_aug_unauth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta_res_auth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=n_results,
        where={"restricted": {"$eq": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta_res_unauth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=n_results,
        where={"restricted": {"$ne": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_unauth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rot_res = rot_collection.query(
        query_embeddings=[rot_query_vec.tolist()],
        n_results=n_results,
        include=["metadatas", "embeddings"],
    )
    t_query_rotated_s = time.perf_counter() - t0

    rot_res_unrot = rot_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=n_results,
        include=["metadatas", "embeddings"],
    )

    # Format results
    def _ids(metas):
        return [f"{m.get('triplet_index', '?')}|{m.get('document_id', '?')}|{m.get('phrase_seq', '?')}" for m in metas]

    orig_topk = _ids(orig_res["metadatas"][0])
    aug_topk_auth = _ids(aug_res_auth["metadatas"][0])
    aug_topk_unauth = _ids(aug_res_unauth["metadatas"][0])
    meta_topk_auth = _ids(meta_res_auth["metadatas"][0])
    meta_topk_unauth = _ids(meta_res_unauth["metadatas"][0])
    rot_topk_auth = _ids(rot_res["metadatas"][0])
    rot_topk_unrot = _ids(rot_res_unrot["metadatas"][0])

    # Collect embeddings
    embedding_ground_truth = {}
    for chunk_id in stable_chunks:
        tid, did, pseq = chunk_id.split("|")
        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            embedding_ground_truth[chunk_id] = base_vec.tolist()

    embedding_retrieved_meta_auth = {meta_topk_auth[i]: meta_res_auth["embeddings"][0][i] for i in range(len(meta_topk_auth))}
    embedding_retrieved_meta_unauth = {meta_topk_unauth[i]: meta_res_unauth["embeddings"][0][i] for i in range(len(meta_topk_unauth))}
    embedding_retrieved_aug_auth = {aug_topk_auth[i]: aug_res_auth["embeddings"][0][i] for i in range(len(aug_topk_auth))}
    embedding_retrieved_aug_unauth = {aug_topk_unauth[i]: aug_res_unauth["embeddings"][0][i] for i in range(len(aug_topk_unauth))}
    embedding_retrieved_rot_auth = {rot_topk_auth[i]: rot_res["embeddings"][0][i] for i in range(len(rot_topk_auth))}
    embedding_retrieved_rot_unauth = {rot_topk_unrot[i]: rot_res_unrot["embeddings"][0][i] for i in range(len(rot_topk_unrot))}

    yield RawQueryResults(
        user_index=query_index,
        query_index=query_id,
        high_value_encoding_chunk=cfg.large_value,
        high_value_encoding_query=cfg.large_value_query,
        top_k=top_k,
        number_user=NUMBER_CLUSTER,
        distance_used=DISTANCE_METRIC,
        list_ground_truth=stable_chunks,
        list_retrieved_meta_auth=meta_topk_auth,
        list_retrieved_meta_unauth=meta_topk_unauth,
        list_retrieved_aug_auth=aug_topk_auth,
        list_retrieved_aug_unauth=aug_topk_unauth,
        list_retrieved_rot_auth=rot_topk_auth,
        list_retrieved_rot_unauth=rot_topk_unrot,
        embedding_retrieved_ground_truth=embedding_ground_truth,
        embedding_retrieved_meta_auth=embedding_retrieved_meta_auth,
        embedding_retrieved_meta_unauth=embedding_retrieved_meta_unauth,
        embedding_retrieved_aug_auth=embedding_retrieved_aug_auth,
        embedding_retrieved_aug_unauth=embedding_retrieved_aug_unauth,
        embedding_retrieved_rot_auth=embedding_retrieved_rot_auth,
        embedding_retrieved_rot_unauth=embedding_retrieved_rot_unauth,
        embedding_query_meta_auth=raw_q.tolist(),
        embedding_query_meta_unauth=raw_q.tolist(),
        embedding_query_rot_auth=rot_query_vec.tolist(),
        embedding_query_rot_unauth=raw_q.tolist(),
        embedding_query_aug_auth=auth_q.tolist(),
        embedding_query_aug_unauth=unauth_q.tolist(),
        t_embed_query_s=float(t_embed_query_s),
        t_augment_auth_s=float(t_augment_auth_s),
        t_augment_unauth_s=float(t_augment_unauth_s),
        t_apply_rotation_s=float(t_apply_rotation_s),
        t_query_orig_s=float(t_query_orig_s),
        t_query_aug_auth_s=float(t_query_aug_auth_s),
        t_query_aug_unauth_s=float(t_query_aug_unauth_s),
        t_query_meta_auth_s=float(t_query_meta_auth_s),
        t_query_meta_unauth_s=float(t_query_meta_unauth_s),
        t_query_rotated_s=float(t_query_rotated_s),
    )

def run_query_phase_parallel_batched(
    gt_records: List[Dict],
    cfg: ExtraDimConfig,
    registry: RotationRegistry,
    embedder,
    orig_collection,
    aug_collection,
    meta_collection,
    rot_collection,
    top_k: int = DEFAULT_TOP_K,
    batch_size: int = args.batch_size_results,
    output_file: Path = RESULTS_DIR / "raw_results.jsonl",
    verbose: bool = True,
) -> Path:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'a', encoding='utf-8') as f:
        with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
            args_list = [
                (record, idx % NUMBER_CLUSTER, cfg, registry, embedder,
                 orig_collection, aug_collection, meta_collection, rot_collection, top_k)
                for idx, record in enumerate(gt_records)
                if record.get("id_triplets") and record.get("targeted_chunk")
            ]
            futures = [executor.submit(_query_record_parallel, args) for args in args_list]
            for future in as_completed(futures):
                for result in future.result():
                    f.write(json.dumps(result.to_json_dict()) + '\n')
                    if verbose:
                        logger.info(f"✅ Result saved to {output_file}")
    return output_file

# --- Main Experiment ---
@timed("run_experiment")
def run_experiment_parallel(
    gt_path: str = str(GT_FILE),
    cfg: Optional[ExtraDimConfig] = None,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = True,
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: List[Dict] = json.load(fh)

    if cfg is None:
        cfg = ExtraDimConfig(
            large_value=args.large_value_chunk,
            large_value_query=args.large_value_query,
            normalize_after=args.normalize_after,
        )
    cfg.n_queries = NUMBER_CLUSTER

    logger.info(f"Starting experiment: GT file={gt_path}, Config={cfg.config_id}, Top-K={top_k}")

    # Clear existing databases
    db_paths = [AUG_CHROMA_BASE, META_CHROMA_BASE, ROTATED_CHROMA]
    for db_path in db_paths:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    # Initialize model and collections
    embedder = get_embedding_model()
    dim = len(embedder.embed_query("probe"))
    orig_coll = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    aug_coll = _get_aug_collection(cfg.config_id)
    meta_coll = _get_meta_collection()
    rot_coll = _get_or_create_rotated_collection(ROTATED_CHROMA, ROTATED_COLLECTION)
    registry = RotationRegistry(dim=dim)

    # Build databases
    t0 = time.time()
    build_aug_db_parallel(gt_records, cfg, orig_coll, aug_coll, verbose=verbose)
    time_aug_db_creation = time.time() - t0

    t0 = time.time()
    build_meta_db_parallel(gt_records, orig_coll, meta_coll, verbose=verbose)
    time_meta_db_creation = time.time() - t0

    t0 = time.time()
    build_rotated_db_parallel(gt_records, registry, orig_coll, rot_coll, verbose=verbose)
    time_rot_db_creation = time.time() - t0

    t0 = time.time()
    _build_aug_db_all_untargeted_chunks_parallel(gt_records, cfg, orig_coll, aug_coll, verbose=verbose)
    time_aug_db_creation_untargeted = time.time() - t0

    t0 = time.time()
    build_meta_db_add_untargeted_chunks_parallel(gt_records, orig_coll, meta_coll, verbose=verbose)
    time_meta_db_creation_untargeted = time.time() - t0

    t0 = time.time()
    _build_rotated_db_untargeted_chunks_parallel(gt_records, registry, orig_coll, rot_coll, verbose=verbose)
    time_rot_db_creation_untargeted = time.time() - t0

    # Log database creation times
    result_row = {
        'top_k': top_k,
        'time_aug_db_creation': time_aug_db_creation,
        'time_meta_db_creation': time_meta_db_creation,
        'time_rot_db_creation': time_rot_db_creation,
        'time_aug_db_creation_untargeted': time_aug_db_creation_untargeted,
        'time_meta_db_creation_untargeted': time_meta_db_creation_untargeted,
        'time_rot_db_creation_untargeted': time_rot_db_creation_untargeted,
    }
    with open(PATH_TIME_DATABASE_CREATION, 'a') as jsonl_file:
        jsonl_file.write(json.dumps(result_row) + '\n')

    # Query phase
    output_file = RESULTS_DIR / f"raw_results_topk_{top_k}.jsonl"
    run_query_phase_parallel_batched(
        gt_records, cfg, registry, embedder, orig_coll, aug_coll, meta_coll, rot_coll,
        top_k=top_k,
        batch_size=args.batch_size_results,
        output_file=output_file,
        verbose=verbose
    )
    logger.info(f"✅ Experiment complete. Results saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment_parallel(
        gt_path=args.gt,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
from __future__ import annotations
import argparse
import functools
import json
import os
import shutil
import time
import warnings
import hashlib
import pickle
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Dict, Set, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import chromadb
import numpy as np
from chromadb.config import Settings

from config import COLLECTION
from ingestion_pipeline import get_embedding_model
from query_pipeline import BGE_QUERY_PREFIX

from plot_PCA import get_all_chunk_ids, get_list_id_targeted_chunk
from rotation_experiment import _seed_from_key

from scipy.stats import ortho_group

# Constants and paths (unchanged)

DEFAULT_TOP_K = 20
DEFAULT_LARGE_VAL = 10
DEFAULT_LARGE_VAL_QUERY = DEFAULT_LARGE_VAL 
NUMBER_CLUSTER = 20
DISTANCE_METRIC = "cosine"
LOGS_DIR = Path("logs")
RESULTS_DIR = Path("results_experiment_extra_dim")
RAW_RESULTS_FILE = RESULTS_DIR / "raw_results.pkl"

GT_FILE = Path("RAGBench_whole/merged_id_triplets_with_metadata2.json")
DIM_RESULTS_FILE = RESULTS_DIR / "dim_results.json"
DIM_REGISTRY_FILE = RESULTS_DIR / "dim_registry.json"
TIMING_FILE = LOGS_DIR / f"dim_timing_largeval_{DEFAULT_LARGE_VAL}_topk_{DEFAULT_TOP_K}_extra_dim_untargeted.json"
ORIGINAL_CHROMA = os.path.join(os.getcwd(), "./experiment_chroma_db")
ORIGINAL_COLLECTION = "experimental_baseline_db"
AUG_CHROMA_BASE = os.path.join(os.getcwd(), "./chroma_extra_dim_experiment")
AUGMENTED_NAME = f"augmented_db{DISTANCE_METRIC}val_query_half{DEFAULT_LARGE_VAL}{NUMBER_CLUSTER}"
META_CHROMA_BASE = os.path.join(os.getcwd(), "./chroma_meta_db_experiment")
META_NAME = "meta_access_control_experiment"
ROTATED_CHROMA      = os.path.join(os.getcwd(), "./chroma_rotated_db_log")
ROTATED_COLLECTION  = "rotated_experiment"

NUMBER_THREADS = 64
BATCH_SIZE = 500

# --- Timing and logging (unchanged) ---
TIMING_LOG: List[Dict] = []

def timed(label: str):
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            args_repr = str(args[:2])[:120]
            start = time.perf_counter()
            start_iso = datetime.now(timezone.utc).isoformat()
            try:
                result = fn(*args, **kwargs)
            finally:
                TIMING_LOG.append({
                    "label": label,
                    "start_iso": start_iso,
                    "duration_s": round(time.perf_counter() - start, 6),
                    "args_repr": args_repr,
                })
            return result
        return wrapper
    return decorator

def _compute_timing_summary() -> dict:
    """Group log entries by label, return per-label mean ± std / min / max / n."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for entry in TIMING_LOG:
        buckets[entry["label"]].append(entry["duration_s"])

    summary = {}
    for label, durations in sorted(buckets.items()):
        arr = np.array(durations)
        summary[label] = {
            "n":      int(len(arr)),
            "mean_s": round(float(arr.mean()), 6),
            "std_s":  round(float(arr.std()),  6),
            "min_s":  round(float(arr.min()),  6),
            "max_s":  round(float(arr.max()),  6),
        }
    return summary


def _compute_retrieval_stats(eval_results: "list[QueryEvalResult]") -> dict:
    """
    Mean / std of every targeted_in_* counter across all evaluated queries,
    both as raw counts and as fractions of n_targeted_chunks.
    """
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
        counts = np.array([getattr(r, f) for r in eval_results], dtype=float)
        fracs  = np.where(totals > 0, counts / totals, 0.0)
        stats[f] = {
            "mean_count": round(float(counts.mean()), 4),
            "std_count":  round(float(counts.std()),  4),
            "min_count":  round(float(counts.min()),  4),
            "max_count":  round(float(counts.max()),  4),
            "mean_frac":  round(float(fracs.mean()),  4),
            "std_frac":   round(float(fracs.std()),   4),
            "n_queries":  int(len(counts)),
        }
    return stats


def save_timing_log(
    eval_results: "list[QueryEvalResult] | None" = None,
) -> None:
    """
    Flush TIMING_LOG to logs/dim_timing.json.

    Structure:
        {
          "entries":         [ …per-call records… ],
          "summary":         { label: {n, mean_s, std_s, min_s, max_s} },
          "retrieval_stats": { metric: {mean_count, std_count, mean_frac, …} }
        }
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    output: dict = {
        "entries": TIMING_LOG,
        "summary": _compute_timing_summary(),
    }
    if eval_results:
        output["retrieval_stats"] = _compute_retrieval_stats(eval_results)

    with open(TIMING_FILE, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)

    print(f"\n⏱  Timing log → {TIMING_FILE}  ({len(TIMING_LOG)} entries)")
    print("\n  ── Timing summary ─────────────────────────────────────────────")
    for label, s in output["summary"].items():
        print(
            f"  {label:<50}"
            f"  n={s['n']:<4}"
            f"  mean={s['mean_s']:.4f}s"
            f"  std={s['std_s']:.4f}s"
            f"  [{s['min_s']:.4f}s … {s['max_s']:.4f}s]"
        )
    if "retrieval_stats" in output:
        print("\n  ── Retrieval stats ─────────────────────────────────────────────")
        for key, s in output["retrieval_stats"].items():
            print(
                f"  {key:<50}"
                f"  mean={s['mean_count']:.2f} ± {s['std_count']:.2f} chunks"
                f"  ({s['mean_frac']*100:.1f}% ± {s['std_frac']*100:.1f}%)"
            )

# --- Rotation helpers ---

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
    """
    Maps each triplet_index to exactly one orthogonal rotation matrix.
    Once assigned the rotation never changes.
    """

    def __init__(self, dim: int):
        self.dim    = dim
        self._store: dict[str, dict] = {}

    def get_or_create(self, triplet_index: str) -> np.ndarray:
        if triplet_index not in self._store:
            seed = _seed_from_key(triplet_index)
            self._store[triplet_index] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(self.dim, seed),
            }
        return self._store[triplet_index]["matrix"]

    def get(self, triplet_index: str) -> np.ndarray | None:
        entry = self._store.get(triplet_index)
        return entry["matrix"] if entry else None

    def seed_of(self, triplet_index: str) -> int | None:
        entry = self._store.get(triplet_index)
        return entry["seed"] if entry else None

    def all_keys(self) -> list[str]:
        return list(self._store.keys())

    def to_serialisable(self) -> dict:
        return {k: v["seed"] for k, v in self._store.items()}

    @classmethod
    def from_serialisable(cls, data: dict, dim: int) -> "RotationRegistry":
        reg = cls(dim=dim)
        for key, seed in data.items():
            reg._store[key] = {
                "seed":   seed,
                "matrix": make_rotation_matrix(dim, seed),
            }
        return reg


# --- Data structures (unchanged) ---
@dataclass
class ExtraDimConfig:
    large_value: float = DEFAULT_LARGE_VAL
    normalize_after: bool = True
    n_queries: int = 0

    @property
    def config_id(self) -> str:
        lv = f"{self.large_value:.0e}".replace("+", "")
        na = "na1" if self.normalize_after else "na0"
        return f"q{self.n_queries}_lv{lv}_{na}"




@dataclass
class ChunkSimilarities:
    """
    Four cosine-similarity measurements for one chunk, mirroring the
    rotation experiment's four pairs.

    (a) auth query (augmented)   ↔ aug chunk      → both sides carry large_value → high
    (b) unauth query (plain)     ↔ aug chunk      → zeros vs large_value → lower
    (c) auth query (augmented)   ↔ plain chunk    → large_value vs zeros → collateral
    (d) plain query (no aug)     ↔ plain chunk    → baseline, no extra dims
    """
    chunk_key:                  str
    query_index:                int
    config_id:                  str
    sim_auth_query_aug_chunk:   float   # (a)
    sim_unauth_query_aug_chunk: float   # (b)
    sim_auth_query_plain_chunk: float   # (c)
    sim_plain_query_plain_chunk: float  # (d) baseline

    # rotated
    rotation_seed            : int
    sim_orig_query_orig_chunk: float
    sim_rot_query_rot_chunk:   float
    sim_orig_query_rot_chunk:  float
    sim_rot_query_orig_chunk:  float




    @property
    def security_delta(self) -> float:
        """(b) − (a): how much similarity drops for an unauthorised query. Negative = good."""
        return self.sim_unauth_query_aug_chunk - self.sim_auth_query_aug_chunk

    @property
    def collateral_delta(self) -> float:
        """(c) − (d): change on plain chunks when auth query is used. Should be ~0."""
        return self.sim_auth_query_plain_chunk - self.sim_plain_query_plain_chunk


@dataclass
class RawQueryRetrieval:
    """Output of Phase 2 — raw retrieval data, no metrics yet."""
    query_id:           str
    question:           str
    triplet_index:      str
    query_index:        int           # 0-based position in GT list
    targeted_chunk_ids: list[str]
    rotation_seed:     int

    rot_query_vec:      list[float] 
    query_vec:          list[float]   # un-augmented base embedding

    # Per-chunk raw base vectors: chunk_key → {"base": list[float]}
    chunk_vectors: dict = field(default_factory=dict)

    # ── Method 1: augmented DB results ────────────────────────────────────────
    # auth (augmented) query → augmented DB
    aug_topk_auth_ids:   list[str] = field(default_factory=list)
    # unauth (plain/zero) query → augmented DB
    aug_topk_unauth_ids: list[str] = field(default_factory=list)

    # ── Method 2: metadata filter results ─────────────────────────────────────
    meta_topk_auth_ids:   list[str] = field(default_factory=list)   # no filter
    meta_topk_unauth_ids: list[str] = field(default_factory=list)   # filtered

    # ── Method 3: rotation results ─────────────────────────────────────
    # rotation_seed:      int
    # rot_query_vec:      list[float]   # rotated
    # Top-K IDs from original collection  (unrotated query)
    original_topk_ids:            list[str] = field(default_factory=list)
    # Top-K IDs from rotated collection   (rotated query)   — main experiment
    rotated_topk_ids:             list[str] = field(default_factory=list)
    # Top-K IDs from rotated collection   (UNrotated query) — sanity baseline
    rotated_topk_ids_unrot_query: list[str] = field(default_factory=list)

    # Precise per-step timings (Method 1) : Extra-dim augmentation
    t_embed_query_s:        float = 0.0
    t_augment_auth_s:       float = 0.0
    t_augment_unauth_s:     float = 0.0
    t_query_aug_auth_s:     float = 0.0
    t_query_aug_unauth_s:   float = 0.0
    # Precise per-step timings (Method 2) : Metadata filtering
    t_query_meta_auth_s:    float = 0.0
    t_query_meta_unauth_s:  float = 0.0
    # Precise per-step timings (Method 3) : Rotation
    t_apply_rotation_s: float = 0.0
    t_query_orig_s: float = 0.0
    t_query_rotated_s:  float = 0.0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class RawQueryResults:
    """Output of Phase 2 — raw retrieval data, no metrics yet."""
    
    # general info experiment related
    user_index:        int           # usefull for rot and aug to get the user it stem from 
    query_index:        str          # called before triplet_index
    high_value_encoding_chunk : int
    high_value_encoding_query : int
    # rotation_seed:     int
    top_k : int
    number_user : int
    distance_used : str

    # retrieval results 
    # (chunks id)
    list_ground_truth: list[str]
    list_retrieved_meta_auth: list[str] 
    list_retrieved_meta_unauth: list[str]
    list_retrieved_rot_auth: list[str]
    list_retrieved_rot_unauth: list[str]
    list_retrieved_aug_auth: list[str]
    list_retrieved_aug_unauth: list[str]
    # (embeddings)
    embedding_retrieved_ground_truth: dict[str, list[float]]
    embedding_retrieved_meta_auth: dict[str, list[float]]
    embedding_retrieved_meta_unauth: dict[str, list[float]]
    embedding_retrieved_rot_auth: dict[str, list[float]]
    embedding_retrieved_rot_unauth: dict[str, list[float]]
    embedding_retrieved_aug_auth: dict[str, list[float]]
    embedding_retrieved_aug_unauth: dict[str, list[float]]

    embedding_query_meta_auth: list[float]
    embedding_query_meta_unauth: list[float]
    embedding_query_rot_auth: list[float]
    embedding_query_rot_unauth: list[float]
    embedding_query_aug_auth: list[float]
    embedding_query_aug_unauth: list[float]

    # Precise per-step timings (Method 1) : Extra-dim augmentation
    t_embed_query_s:        float = 0.0
    t_augment_auth_s:       float = 0.0
    t_augment_unauth_s:     float = 0.0
    t_query_aug_auth_s:     float = 0.0
    t_query_aug_unauth_s:   float = 0.0
    # Precise per-step timings (Method 2) : Metadata filtering
    t_query_meta_auth_s:    float = 0.0
    t_query_meta_unauth_s:  float = 0.0
    # Precise per-step timings (Method 3) : Rotation
    t_apply_rotation_s: float = 0.0
    t_query_orig_s: float = 0.0
    t_query_rotated_s:  float = 0.0


    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )



@dataclass
class QueryEvalResult:
    """Output of Phase 3 — fully evaluated metrics for one query."""
    query_id:          str
    question:          str
    triplet_index:     str
    query_index:       int
    n_targeted_chunks: int
    config_id:         str
    rotation_seed:     int

    chunk_similarities: list[ChunkSimilarities] = field(default_factory=list)

    # Method 2 retrieval counts
    targeted_in_auth_query_aug_db:   int = 0
    targeted_in_unauth_query_aug_db: int = 0

    # Method 3 retrieval counts
    targeted_in_auth_meta:   int = 0
    targeted_in_unauth_meta: int = 0

    targeted_in_auth_query_rot_db: int = 0
    targeted_in_unauth_query_rot_db: int = 0

    # time values
    t_query_orig_s:         float = 0.0
    t_embed_query_s:        float = 0.0
    t_augment_auth_s:       float = 0.0
    t_augment_unauth_s:     float = 0.0
    t_query_aug_auth_s:     float = 0.0
    t_query_aug_unauth_s:   float = 0.0

    t_query_meta_auth_s:    float = 0.0
    t_query_meta_unauth_s:  float = 0.0

    t_apply_rotation_s:     float = 0.0
    t_query_rotated_s:      float = 0.0




    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )



# --- Helper functions ---
def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v

def augment_chunk(base_vec: np.ndarray, cfg: ExtraDimConfig, query_index: int | None, untargeted: bool = False) -> np.ndarray:
    extra = np.zeros(NUMBER_CLUSTER, dtype=np.float32)
    if untargeted:
        return np.concatenate([base_vec.astype(np.float32), extra])
    if query_index is not None:
        extra[query_index] = DEFAULT_LARGE_VAL
    return np.concatenate([base_vec.astype(np.float32), extra])

def augment_query(base_vec: np.ndarray, cfg: ExtraDimConfig, query_index: int, authorised: bool) -> np.ndarray:
    extra = np.zeros(NUMBER_CLUSTER, dtype=np.float32)
    if authorised:
        extra[query_index] = DEFAULT_LARGE_VAL_QUERY
    return np.concatenate([base_vec.astype(np.float32), extra])

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

# --- ChromaDB helpers ---
def _get_client(path: str) -> chromadb.PersistentClient:
    os.makedirs(path, exist_ok=True)
    return chromadb.PersistentClient(path=path, settings=Settings(anonymized_telemetry=False))

def _get_original_collection(path: str, name: str):
    return _get_client(path).get_collection(name=name)

def _get_aug_collection(cfg_id: str, name: str = AUGMENTED_NAME):
    return _get_client(AUG_CHROMA_BASE).get_or_create_collection(name=name, metadata={"hnsw:space": DISTANCE_METRIC})

def _get_meta_collection():
    return _get_client(META_CHROMA_BASE).get_or_create_collection(name=META_NAME, metadata={"hnsw:space": DISTANCE_METRIC})

def _get_or_create_rotated_collection(path: str, name: str):
    return _get_client(path).get_or_create_collection(
        name=name, metadata={"hnsw:space": f"{DISTANCE_METRIC}"}
    )


# --- Database builders ---

@timed("fetch_chunk_from_original_db")
def fetch_chunk(
    collection,
    triplet_index: str,
    document_id:   str,
    phrase_seq:    str,
) -> tuple[np.ndarray | None, str | None]:
    """Return (embedding, document_text) or (None, None) if not found."""
    try:
        result = collection.get(
            where={"$and": [
                {"triplet_index": {"$eq": triplet_index}},
                {"document_id":   {"$eq": document_id}},
                {"phrase_seq":    {"$eq": phrase_seq}},
            ]},
            include=["embeddings", "documents"],
        )
        embs = result.get("embeddings", [[]])
        docs = result.get("documents", [[]])
        if len(embs[0]) > 0:
            return np.array(embs[0], dtype=np.float32), (docs[0] if docs else None)
    except Exception as e:
        warnings.warn(f"fetch_chunk failed for {triplet_index}|{document_id}|{phrase_seq}: {e}")
    return None, None


#####################################################


def _build_meta_record(
    record: dict,
    orig_collection,
    meta_collection,
) -> int:
    """
    Thread-safe helper to build metadata for all targeted chunks in a record using batch upsert.
    """
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    n_ok = 0

    batch_ids = []
    batch_embeddings = []
    batch_documents = []
    batch_metadatas = []
    
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

    # Batch upsert for all targeted chunks in the record
    if batch_ids:
        meta_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

    return n_ok

def build_meta_db_parallel(
    gt_records: list,
    orig_collection,
    meta_collection,
    max_workers: int = 8,
    verbose: bool = True,
) -> None:
    """
    Parallel construction of the metadata database (per-record, batch upserts).
    """
    print(f"\n{'─' * 60}\n  Phase 1b — Building metadata DB (parallel, batched by record)\n{'─' * 60}")

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

    print("\n  ✅ Metadata DB build complete (parallel with batched upserts)")

def build_meta_db_add_untargeted_chunks_parallel(
    gt_records: list,
    orig_collection,
    meta_collection,
    batch_size: int = 100,
    max_workers: int = 8,
    verbose: bool = True,
):
    """
    Parallel addition of untargeted chunks to meta DB, with batch upserts.
    """
    # Use set for efficient chunk difference
    list_of_chunk_ids = set(get_all_chunk_ids(gt_records))
    list_of_targeted_chunk_ids = set(get_list_id_targeted_chunk(gt_records))
    untargeted_chunks = list(list_of_chunk_ids - list_of_targeted_chunk_ids)

    # Create batches
    batches = [untargeted_chunks[i:i + batch_size] for i in range(0, len(untargeted_chunks), batch_size)]

    def process_batch(batch: list) -> int:
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
            meta_collection.upsert(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas
            )
        return n_ok

    total_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_batch, batch) for batch in batches]
        for future in as_completed(futures):
            total_ok += future.result()

    if verbose:
        print(f"✅ Added {total_ok} untargeted chunks to the metadata DB (batched upsert).")




def _process_aug_untargeted_batch(
    batch: List[str],
    cfg,
    orig_collection,
    aug_collection,
) -> int:
    """Batch-process and upsert untargeted chunks into the augmented DB."""
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
    return n_ok

def _build_aug_db_all_untargeted_chunks_parallel(
    gt_records: List[Dict],
    cfg,
    orig_collection,
    aug_collection,
    batch_size: int = 100,
    max_workers: int = 8,
    verbose: bool = True,
) -> None:
    """
    Thread-based parallel addition of untargeted chunks to the augmented DB, using batch upserts.
    """
    list_of_chunk_ids = set(get_all_chunk_ids(gt_records))
    list_of_targeted_chunk_ids = set(get_list_id_targeted_chunk(gt_records))
    untargeted_chunks = list(list_of_chunk_ids - list_of_targeted_chunk_ids)
    print(f"Found {len(untargeted_chunks)} untargeted chunks to add to the augmented DB.")

    batches = [untargeted_chunks[i:i + batch_size] for i in range(0, len(untargeted_chunks), batch_size)]

    total_ok = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_aug_untargeted_batch, batch, cfg, orig_collection, aug_collection)
            for batch in batches
        ]
        for future in as_completed(futures):
            total_ok += future.result()

    if verbose:
        print(f"✅ Added {total_ok} untargeted chunks to the augmented DB (batched upsert).")

def _build_aug_record(
    record: Dict,
    query_index: int,
    cfg,
    orig_collection,
    aug_collection,
) -> int:
    """
    Thread-safe helper to build augmented chunks for a single record, using batch upsert.
    """
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    n_ok = 0

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

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
    return n_ok

def build_aug_db_parallel(
    gt_records: List[Dict],
    cfg,
    orig_collection,
    aug_collection,
    max_workers: int = 8,
    verbose: bool = True,
) -> None:
    """
    Thread-based parallel construction of the augmented DB, using batch upserts per record.
    """
    print(f"\n{'─' * 60}\n  Phase 1a — Building augmented DB (parallel with threads)\n{'─' * 60}")

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

    print(f"\n  ✅ Augmented DB build complete (parallel, batch upserts by record)")



@timed("build_rotated_db_single_record")
def _build_record_parallel(
    record,
    registry,
    orig_collection,
    rot_collection,
    user_index
) -> int:
    """
    Rotate and upsert targeted chunks for one record in a batched way (thread-safe).
    """
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    R = registry.get_or_create(user_index)
    n_ok = 0

    batch_ids, batch_embeddings, batch_documents, batch_metadatas = [], [], [], []

    for chunk_id in stable_chunks:
        try:
            tid, did, pseq = chunk_id.split("|")
            cid = f"{triplet_index}_{did}_{pseq}"

            orig_vec, content = fetch_chunk(orig_collection, tid, did, pseq)
            if orig_vec is None:
                warnings.warn(f"  ⚠  Build: chunk not found {chunk_id} — skipping.")
                continue

            rot_vec = apply_rotation(orig_vec, R)

            batch_ids.append(cid)
            batch_embeddings.append(rot_vec.tolist())
            batch_documents.append(content or "")
            batch_metadatas.append({
                "triplet_index": triplet_index,
                "document_id": did,
                "phrase_seq": pseq,
                "rotation_seed": int(registry.seed_of(user_index)),
            })
            n_ok += 1
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue

    if batch_ids:
        rot_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    return n_ok

@timed("build_rotated_db")
def build_rotated_db_parallel(
    gt_records: list,
    registry,
    orig_collection,
    rot_collection,
    max_workers: int = 8,
    verbose: bool = True
) -> "RotationRegistry":
    """Thread-based parallel construction of the rotated DB, with batched upsert per record."""
    print(f"\n{'─'*60}\n  Phase 1 — Building rotated database (parallel) ({len(gt_records)} records)\n{'─'*60}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for i, record in enumerate(gt_records):
            triplet_index = record.get("id_triplets")
            if not triplet_index or not record.get("targeted_chunk"):
                continue
            futures.append(
                executor.submit(
                    _build_record_parallel, 
                    record, registry, orig_collection, rot_collection, i % NUMBER_CLUSTER
                )
            )
        for future in as_completed(futures):
            future.result()  # Raises exceptions if any
    
    print(f"\n  ✅ Build complete — {len(futures)} chunk vectors in rotated DB")
    return registry

@timed("build_rotated_db_untargeted_chunks")
def _process_rot_untargeted_batch(
    batch: list,
    orig_collection,
    rot_collection,
    verbose: bool = True,
) -> int:
    """Batch-process and upsert untargeted chunks to the rotated DB."""
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
            if verbose:
                print(f" untargeted chunk added {chunk_id} → {cid}")
        except Exception as e:
            warnings.warn(f"Failed to process chunk {chunk_id}: {e}")
            continue

    if batch_ids:
        rot_collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )
    return n_ok

@timed("build_rotated_db_untargeted_chunks")
def _build_rotated_db_untargeted_chunks_parallel(
    gt_records: list,
    registry,
    orig_collection,
    rot_collection,
    batch_size: int = 100,
    max_workers: int = 8,
    verbose: bool = True
) -> int:
    """
    Thread-based parallel addition of untargeted chunks to the rotated DB, with batch upserts.
    """
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
                verbose
            ) for batch in batches
        ]
        for future in as_completed(futures):
            total_ok += future.result()

    return total_ok


#####################################################################################

@timed("query_phase_single_record")
def _query_record_parallel_2(args):
    """
    Thread-safe helper to process a single query record.
    Args:
        args: Tuple of (record, query_index, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rotation_collection, top_k)
    """
    record, query_index, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k = args
    question = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id = f"triplet_{triplet_index}"
    targeted_keys = set(stable_chunks)

    R             = registry.get(query_index)
    seed          = registry.seed_of(query_index)

    # Step 1: Embed raw query
    t0 = time.perf_counter()
    raw_q = np.array(embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32)
    t_embed_query_s = time.perf_counter() - t0


    # Step 2: Augment query vectors and perform the rotation
    t0 = time.perf_counter()   
    auth_q = augment_query(raw_q, cfg, query_index, authorised=True)
    t_augment_auth_s = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    unauth_q = augment_query(raw_q, cfg, query_index, authorised=False)
    t_augment_unauth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rot_query_vec = apply_rotation(raw_q, R)
    t_apply_rotation_s = time.perf_counter() - t0


    # Step 3: Retrieve from augmented DB
    n_aug = max(1, aug_collection.count())
    n_meta = max(1, meta_collection.count())
    n_rot = max(1, rot_collection.count())
    n_max = max(n_aug, n_meta, n_rot, top_k)
    n_aug = n_max
    n_meta = n_max
    n_rot = n_max

    t0 = time.perf_counter()
    orig_res = orig_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, orig_collection.count()),
        include=["metadatas", "embeddings"],
    )
    t_query_orig = time.perf_counter() - t0


    t0 = time.perf_counter()
    aug_res_auth = aug_collection.query(
        query_embeddings=[auth_q.tolist()],
        n_results=min(top_k, n_aug),
        include=["metadatas", "embeddings"],
    )
    t_query_aug_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    aug_res_unauth = aug_collection.query(
        query_embeddings=[unauth_q.tolist()],
        n_results=min(top_k, n_aug),
        include=["metadatas", "embeddings"],
    )
    t_query_aug_unauth_s = time.perf_counter() - t0

    # Step 4: Retrieve from metadata DB
    
    
    t0 = time.perf_counter()
    meta_res_auth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_meta),
        where={"restricted": {"$eq": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta_res_unauth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_meta),
        where={"restricted": {"$ne": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_unauth_s = time.perf_counter() - t0

    # Rotation retrieve

    t0 = time.perf_counter()
    rot_res = rot_collection.query(
        query_embeddings=[rot_query_vec.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas", "embeddings"],
    )
    t_query_rotated_s = time.perf_counter() - t0

    rot_res_unrot = rot_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas", "embeddings"],
    )


    # target_chunk_id = [f'{target.split("|")[0]}|{target[-2]}|{target[-1]}' for target in stable_chunks]
    target_chunk_id = list(stable_chunks)

    # get embedding from ground truth
    embedding_ground_truth = {}
    for chunk_id in target_chunk_id:
        # tid, did, pseq = chunk_id.split("|")[0], chunk_id.split("|")[1], chunk_id.split("|")[2]
        tid, did, pseq = chunk_id.split("|")

        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            embedding_ground_truth[chunk_id] = base_vec.tolist()


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

    embedding_retrieved_meta_auth = meta_res_auth["embeddings"][0]
    embedding_retrieved_meta_unauth = meta_res_unauth["embeddings"][0]
    embedding_retrieved_aug_auth = aug_res_auth["embeddings"][0]
    embedding_retrieved_aug_unauth = aug_res_unauth["embeddings"][0]
    embedding_retrieved_rot_auth = rot_res["embeddings"][0]
    embedding_retrieved_rot_unrot = rot_res_unrot["embeddings"][0]

    # create dict
    embedding_retrieved_meta_auth = {meta_topk_auth[i]: embedding_retrieved_meta_auth[i].tolist() for i in range(len(meta_res_auth["metadatas"][0]))}
    embedding_retrieved_meta_unauth = {meta_topk_unauth[i]: embedding_retrieved_meta_unauth[i].tolist() for i in range(len(meta_res_unauth["metadatas"][0]))}
    embedding_retrieved_aug_auth = {aug_topk_auth[i]: embedding_retrieved_aug_auth[i].tolist() for i in range(len(aug_res_auth["metadatas"][0]))}
    embedding_retrieved_aug_unauth = {aug_topk_unauth[i]: embedding_retrieved_aug_unauth[i].tolist() for i in range(len(aug_res_unauth["metadatas"][0]))}
    embedding_retrieved_rot_auth = {rot_topk_auth[i]: embedding_retrieved_rot_auth[i].tolist() for i in range(len(rot_res["metadatas"][0]))}
    embedding_retrieved_rot_unrot = {rot_topk_unrot[i]: embedding_retrieved_rot_unrot[i].tolist() for i in range(len(rot_res_unrot["metadatas"][0]))}   

    # Collect chunk vectors
    chunk_vectors = {}
    for chunk_id in stable_chunks:
        # tid, did, pseq = chunk_id.split("|")[0], chunk_id[-2], chunk_id[-1]
        tid, did, pseq = chunk_id.split("|")

        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            chunk_vectors[chunk_id] = {"base": base_vec.tolist(), 
                                       "rot": apply_rotation(base_vec, R).tolist(), 
                                       "aug": augment_chunk(base_vec, cfg, query_index=query_index).tolist(),
                                       "rot_query": rot_query_vec.tolist(),
                                       }

    yield RawQueryResults(
        query_index = query_id,
        user_index = query_index,
        high_value_encoding_chunk = DEFAULT_LARGE_VAL,
        high_value_encoding_query = DEFAULT_LARGE_VAL_QUERY,
        number_user = NUMBER_CLUSTER,
        distance_used = DISTANCE_METRIC,
        top_k = top_k,

        list_ground_truth = target_chunk_id,
        list_retrieved_meta_auth = meta_topk_auth,
        list_retrieved_meta_unauth = meta_topk_unauth,
        list_retrieved_aug_auth = aug_topk_auth,
        list_retrieved_aug_unauth = aug_topk_unauth,
        list_retrieved_rot_auth = rot_topk_auth,
        list_retrieved_rot_unauth = rot_topk_unrot,

        embedding_retrieved_ground_truth = embedding_ground_truth,
        embedding_retrieved_meta_auth = embedding_retrieved_meta_auth,
        embedding_retrieved_meta_unauth = embedding_retrieved_meta_unauth,
        embedding_retrieved_aug_auth = embedding_retrieved_aug_auth,
        embedding_retrieved_aug_unauth = embedding_retrieved_aug_unauth,
        embedding_retrieved_rot_auth = embedding_retrieved_rot_auth,
        embedding_retrieved_rot_unauth = embedding_retrieved_rot_unrot,

        embedding_query_meta_auth = raw_q.tolist(),
        embedding_query_meta_unauth = raw_q.tolist(),
        embedding_query_aug_auth = auth_q.tolist(),
        embedding_query_aug_unauth = unauth_q.tolist(),
        embedding_query_rot_auth = rot_query_vec.tolist(),
        embedding_query_rot_unauth = raw_q.tolist(),

        t_query_orig_s=t_query_orig,
        t_embed_query_s=t_embed_query_s,
        t_augment_auth_s=t_augment_auth_s,
        t_augment_unauth_s=t_augment_unauth_s,
        t_apply_rotation_s=t_apply_rotation_s,
        t_query_aug_auth_s=t_query_aug_auth_s,
        t_query_aug_unauth_s=t_query_aug_unauth_s,
        t_query_meta_auth_s=t_query_meta_auth_s,
        t_query_meta_unauth_s=t_query_meta_unauth_s,
        t_query_rotated_s=t_query_rotated_s,
        


    )

@timed("query_phase_single_record")
def _query_record_parallel(args):
    """
    Thread-safe helper to process a single query record.
    Args:
        args: Tuple of (record, query_index, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rotation_collection, top_k)
    """
    record, query_index, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k = args
    question = record["question"]
    triplet_index = record["id_triplets"]
    stable_chunks = record["targeted_chunk"]
    query_id = f"triplet_{triplet_index}"
    targeted_keys = set(stable_chunks)

    R             = registry.get(query_index)
    seed          = registry.seed_of(query_index)

    # Step 1: Embed raw query
    t0 = time.perf_counter()
    raw_q = np.array(embedder.embed_query(BGE_QUERY_PREFIX + question), dtype=np.float32)
    t_embed_query_s = time.perf_counter() - t0


    # Step 2: Augment query vectors and perform the rotation
    t0 = time.perf_counter()   
    auth_q = augment_query(raw_q, cfg, query_index, authorised=True)
    t_augment_auth_s = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    unauth_q = augment_query(raw_q, cfg, query_index, authorised=False)
    t_augment_unauth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    rot_query_vec = apply_rotation(raw_q, R)
    t_apply_rotation_s = time.perf_counter() - t0


    # Step 3: Retrieve from augmented DB
    n_aug = max(1, aug_collection.count())
    n_meta = max(1, meta_collection.count())
    n_rot = max(1, rot_collection.count())

    t0 = time.perf_counter()
    orig_res = orig_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, orig_collection.count()),
        include=["metadatas", "embeddings"],
    )
    t_query_orig = time.perf_counter() - t0


    t0 = time.perf_counter()
    aug_res_auth = aug_collection.query(
        query_embeddings=[auth_q.tolist()],
        n_results=min(top_k, n_aug),
        include=["metadatas", "embeddings"],
    )
    t_query_aug_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    aug_res_unauth = aug_collection.query(
        query_embeddings=[unauth_q.tolist()],
        n_results=min(top_k, n_aug),
        include=["metadatas", "embeddings"],
    )
    t_query_aug_unauth_s = time.perf_counter() - t0

    # Step 4: Retrieve from metadata DB
    
    
    t0 = time.perf_counter()
    meta_res_auth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_meta),
        where={"restricted": {"$eq": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_auth_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    meta_res_unauth = meta_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_meta),
        where={"restricted": {"$ne": f"{triplet_index}_True"}},
        include=["metadatas", "embeddings"],
    )
    t_query_meta_unauth_s = time.perf_counter() - t0

    # Rotation retrieve

    t0 = time.perf_counter()
    rot_res = rot_collection.query(
        query_embeddings=[rot_query_vec.tolist()],
        n_results=min(top_k, n_rot),
        include=["metadatas", "embeddings"],
    )
    t_query_rotated_s = time.perf_counter() - t0

    rot_res_unrot = rot_collection.query(
        query_embeddings=[raw_q.tolist()],
        n_results=min(top_k, n_rot),
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

    # Collect chunk vectors
    chunk_vectors = {}
    for chunk_id in stable_chunks:
        # tid, did, pseq = chunk_id.split("|")[0], chunk_id[-2], chunk_id[-1]
        tid, did, pseq = chunk_id.split("|")

        base_vec, _ = fetch_chunk(orig_collection, tid, did, pseq)
        if base_vec is not None:
            chunk_vectors[chunk_id] = {"base": base_vec.tolist(), 
                                       "rot": apply_rotation(base_vec, R).tolist(), 
                                       "aug": augment_chunk(base_vec, cfg, query_index=query_index).tolist(),
                                       "rot_query": rot_query_vec.tolist(),
                                       }

    # return RawQueryRetrieval(
    #     query_id=query_id,
    #     question=question,
    #     triplet_index=triplet_index,
    #     query_index=query_index,
    #     targeted_chunk_ids=list(stable_chunks),
    #     query_vec=raw_q.tolist(),
    #     chunk_vectors=chunk_vectors,
    #     aug_topk_auth_ids=aug_topk_auth,
    #     aug_topk_unauth_ids=aug_topk_unauth,
    #     meta_topk_auth_ids=meta_topk_auth,
    #     meta_topk_unauth_ids=meta_topk_unauth,
    # )

    return RawQueryRetrieval(
        query_id=query_id,
        question=question,
        triplet_index=triplet_index,
        query_index=query_index,
        targeted_chunk_ids=list(stable_chunks),
        rotation_seed=seed,
        rot_query_vec=rot_query_vec.tolist(),
        query_vec=raw_q.tolist(),
        chunk_vectors=chunk_vectors,
        original_topk_ids=orig_topk,
        aug_topk_auth_ids=aug_topk_auth,
        aug_topk_unauth_ids=aug_topk_unauth,
        meta_topk_auth_ids=meta_topk_auth,
        meta_topk_unauth_ids=meta_topk_unauth,
        rotated_topk_ids=rot_topk_auth,
        rotated_topk_ids_unrot_query=rot_topk_unrot,
        t_query_orig_s=t_query_orig,
        t_embed_query_s=t_embed_query_s,
        t_augment_auth_s=t_augment_auth_s,
        t_augment_unauth_s=t_augment_unauth_s,
        t_apply_rotation_s=t_apply_rotation_s,
        t_query_aug_auth_s=t_query_aug_auth_s,
        t_query_aug_unauth_s=t_query_aug_unauth_s,
        t_query_meta_auth_s=t_query_meta_auth_s,
        t_query_meta_unauth_s=t_query_meta_unauth_s,
        t_query_rotated_s=t_query_rotated_s,
    )



# def run_query_phase_parallel_with_pickle(
#     gt_records: List[Dict],
#     cfg: ExtraDimConfig,
#     registry: RotationRegistry,
#     embedder,
#     orig_collection,
#     aug_collection,
#     meta_collection,
#     rot_collection,
#     top_k: int = DEFAULT_TOP_K,
#     batch_size: int = 100,
#     output_file: Path = RESULTS_DIR / "raw_results.pkl",
#     verbose: bool = True,
# ):
#     output_file = RESULTS_DIR / f"raw_results_topk_{top_k}.pkl"
#     output_file.parent.mkdir(parents=True, exist_ok=True)
#     results_processed = 0

#     # Open the pickle file in append mode
#     with open(output_file, 'ab') as f:
#         for i in range(0, len(gt_records), batch_size):
#             batch = gt_records[i:i + batch_size]
#             args_list = [
#                 (record, idx % NUMBER_CLUSTER, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k)
#                 for idx, record in enumerate(batch)
#                 if record.get("id_triplets") and record.get("targeted_chunk")
#             ]

#             # Process batch in parallel
#             raw_results = []
#             with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
#                 futures = [executor.submit(_query_record_parallel_2, args) for args in args_list]
#                 for future in as_completed(futures):
#                     raw_results.append(future.result())

#             # Write batch to the same pickle file
#             pickle.dump(raw_results, f)
#             results_processed += len(raw_results)
#             if verbose:
#                 print(f"✅ Processed batch {i//batch_size}: {results_processed}/{len(gt_records)} results saved to {output_file}")

#     return output_file



# @timed("run_query_phase")
# def run_query_phase_parallel(
#     gt_records: List[Dict],
#     cfg: ExtraDimConfig,
#     registry: RotationRegistry,
#     embedder,
#     orig_collection,
#     aug_collection,
#     meta_collection,
#     rot_collection,
#     top_k: int = DEFAULT_TOP_K,
#     verbose: bool = True,
# ) -> List[RawQueryResults]:
#     """
#     Thread-based parallel query phase.
#     """
#     print(f"\n{'─' * 60}\n  Phase 2 — Query phase (parallel with threads) ({len(gt_records)} records, top-K={top_k})\n{'─' * 60}")

#     # Prepare arguments for parallel processing
#     args_list = [
#         (record, i % NUMBER_CLUSTER, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k)
#         for i, record in enumerate(gt_records)
#         if record.get("id_triplets") and record.get("targeted_chunk")
#     ]

#     # Process queries in parallel using threads
#     raw_results = []
#     with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
#         futures = [executor.submit(_query_record_parallel_2, args) for args in args_list]
#         for future in as_completed(futures):
#             raw_results.append(future.result())

#     return raw_results

###################################################


#########

# def run_query_phase_parallel_with_pickle(
#     gt_records: List[Dict],
#     cfg: ExtraDimConfig,
#     registry: RotationRegistry,
#     embedder,
#     orig_collection,
#     aug_collection,
#     meta_collection,
#     rot_collection,
#     top_k: int = DEFAULT_TOP_K,
#     batch_size: int = 100,
#     output_dir: Path = RESULTS_DIR / "batches",
#     verbose: bool = True,
# ):
#     # Create output directory if it doesn't exist
#     output_dir.mkdir(parents=True, exist_ok=True)

#     # Calculate the number of batches
#     valid_records = [
#         record for record in gt_records
#         if record.get("id_triplets") and record.get("targeted_chunk")
#     ]
#     num_batches = (len(valid_records) + batch_size - 1) // batch_size

#     # Process each batch in parallel
#     with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
#         futures = []
#         for batch_idx in range(num_batches):
#             start_idx = batch_idx * batch_size
#             end_idx = start_idx + batch_size
#             batch_records = valid_records[start_idx:end_idx]

#             args_list = [
#                 (record, idx % NUMBER_CLUSTER, cfg, registry, embedder,
#                  orig_collection, aug_collection, meta_collection, rot_collection, top_k)
#                 for idx, record in enumerate(batch_records)
#             ]

#             # Submit batch processing
#             futures.append(executor.submit(
#                 _query_record_parallel_2,
#                 args_list,
#                 output_dir,
#                 batch_idx,
#                 verbose
#             ))

#         # Wait for all futures to complete
#         for future in as_completed(futures):
#             future.result()  # This will raise exceptions if any occurred

#     return output_dir

####### version with writting in sub batches to prevent memory overload
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
    batch_size: int = 100,
    output_file: Path = RESULTS_DIR / "raw_results.pkl",
    verbose: bool = True,
):
    output_file = RESULTS_DIR / f"raw_results_hv_10_topk_{top_k}.pkl"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:  # Open in write mode
        with ThreadPoolExecutor(max_workers=NUMBER_THREADS) as executor:
            args_list = [
                (record, idx % NUMBER_CLUSTER, cfg, registry, embedder, orig_collection, aug_collection, meta_collection, rot_collection, top_k)
                for idx, record in enumerate(gt_records)
                if record.get("id_triplets") and record.get("targeted_chunk")
            ]
            futures = [executor.submit(_query_record_parallel_2, args) for args in args_list]

            for future in as_completed(futures):
                # Iterate over the generator and write each result directly
                for result in future.result():
                    pickle.dump(result, f)
                    if verbose:
                        print(f"✅ Result saved to {output_file}")

    return output_file




# --- Evaluation phase ---
@timed("evaluate_single_record")
def _evaluate_record_optimized(
    raw: RawQueryRetrieval,
    cfg: ExtraDimConfig,
) -> QueryEvalResult:
    """Optimized evaluation of a single query."""
    raw_q = np.array(raw.query_vec, dtype=np.float32)
    targeted = set(raw.targeted_chunk_ids)
    query_index = raw.query_index

    # Precompute augmented query vectors
    auth_q_aug = augment_query(raw_q, cfg, query_index, authorised=True)
    unauth_q_aug = augment_query(raw_q, cfg, query_index, authorised=False)

    auth_q_rot = np.array(raw.rot_query_vec, dtype=np.float32)
    rotation_seed = raw.rotation_seed

    # Precompute all chunk vectors and similarities in batches
    chunk_sims = []
    # target_chunk_id = [f'{target.split("|")[0]}|{target[-2]}|{target[-1]}' for target in targeted]
    target_chunk_id = list(targeted)

    # Batch process chunk similarities
    for chunk_id, vecs in raw.chunk_vectors.items():
        base_vec = np.array(vecs["base"], dtype=np.float32)

        aug_chunk = augment_chunk(base_vec, cfg, query_index=query_index)
        plain_chunk = augment_chunk(base_vec, cfg, query_index=None)

        rot_chunk = np.array(vecs["rot"], dtype=np.float32)
        rot_query = np.array(vecs["rot_query"], dtype=np.float32)

        # Compute all similarities at once
        sim_a = cosine_similarity(auth_q_aug, aug_chunk)      # (a)
        sim_b = cosine_similarity(unauth_q_aug, aug_chunk)    # (b)
        sim_c = cosine_similarity(auth_q_aug, plain_chunk)   # (c)
        sim_d = cosine_similarity(raw_q, base_vec)            # (d)

        # data for the rotation experiment
        seed = raw.rotation_seed
        # sim_e = cosine_similarity(raw_q, base_vec)            # (e) original query vs original chunk
        # sim_f = cosine_similarity(auth_q_rot, rot_chunk)    # (f) rotated query vs rotated chunk
        # sim_g = cosine_similarity(raw_q, rot_chunk)      # (g) original query vs rotated chunk
        # sim_h = cosine_similarity(auth_q_rot, base_vec)  # (h) rotated query vs original chunk

        sim_e = cosine_similarity(raw_q, base_vec)            # (e) original query vs original chunk
        sim_f = cosine_similarity(rot_query, rot_chunk)    # (f) rotated query vs rotated chunk
        sim_g = cosine_similarity(raw_q, rot_chunk)      # (g) original query vs rotated chunk
        sim_h = cosine_similarity(rot_query, base_vec)  # (h) rotated query vs original chunk

        chunk_sims.append(ChunkSimilarities(
            chunk_key=chunk_id,
            query_index=query_index,
            config_id=cfg.config_id,
            sim_auth_query_aug_chunk=sim_a,
            sim_unauth_query_aug_chunk=sim_b,
            sim_auth_query_plain_chunk=sim_c,
            sim_plain_query_plain_chunk=sim_d,
            rotation_seed=seed,
            sim_orig_query_orig_chunk=sim_e,
            sim_rot_query_rot_chunk=sim_f,
            sim_orig_query_rot_chunk=sim_g,
            sim_rot_query_orig_chunk=sim_h,
            
        ))

    # Count targeted chunks in results
    targeted_in_auth_query_aug_db = sum(1 for c in raw.aug_topk_auth_ids if c in target_chunk_id)
    targeted_in_unauth_query_aug_db = sum(1 for c in raw.aug_topk_unauth_ids if c in target_chunk_id)
    targeted_in_auth_meta = sum(1 for c in raw.meta_topk_auth_ids if c in target_chunk_id)
    targeted_in_unauth_meta = sum(1 for c in raw.meta_topk_unauth_ids if c in target_chunk_id)
    targeted_in_auth_query_rot_db = sum(1 for c in raw.rotated_topk_ids if c in target_chunk_id)
    targeted_in_unauth_query_rot_db = sum(1 for c in raw.rotated_topk_ids_unrot_query if c in target_chunk_id)


    return QueryEvalResult(
        query_id=raw.query_id,
        question=raw.question,
        triplet_index=raw.triplet_index,
        query_index=query_index,
        n_targeted_chunks=len(raw.targeted_chunk_ids),
        config_id=cfg.config_id,
        rotation_seed=rotation_seed,
        chunk_similarities=chunk_sims,
        targeted_in_auth_query_aug_db=targeted_in_auth_query_aug_db,
        targeted_in_unauth_query_aug_db=targeted_in_unauth_query_aug_db,
        targeted_in_auth_meta=targeted_in_auth_meta,
        targeted_in_unauth_meta=targeted_in_unauth_meta,
        targeted_in_auth_query_rot_db=targeted_in_auth_query_rot_db,
        targeted_in_unauth_query_rot_db=targeted_in_unauth_query_rot_db,
        t_query_orig_s=raw.t_query_orig_s,
        t_embed_query_s=raw.t_embed_query_s,
        t_augment_auth_s=raw.t_augment_auth_s,
        t_augment_unauth_s=raw.t_augment_unauth_s,
        t_query_aug_auth_s=raw.t_query_aug_auth_s,
        t_query_aug_unauth_s=raw.t_query_aug_unauth_s,
        t_query_meta_auth_s=raw.t_query_meta_auth_s,
        t_query_meta_unauth_s=raw.t_query_meta_unauth_s,
        t_apply_rotation_s=raw.t_apply_rotation_s,
        t_query_rotated_s=raw.t_query_rotated_s,
    )


@timed("evaluate_results")
def evaluate_results_parallel(
    raw_results: List[RawQueryRetrieval],
    cfg: ExtraDimConfig,
    top_k_used: int = DEFAULT_TOP_K,
    verbose: bool = True,
) -> List[QueryEvalResult]:
    """Parallelized evaluation of all queries, with JSON logging."""
    print(f"\n{'─' * 60}\n  Phase 3 — Evaluation (parallel) ({len(raw_results)} queries)\n{'─' * 60}")

    # Prepare arguments for parallel processing
    args_list = [(raw, cfg) for raw in raw_results]

    # Process queries in parallel using threads (I/O-bound)
    eval_results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(_evaluate_record_optimized, *args) for args in args_list]
        for future in as_completed(futures):
            eval_results.append(future.result())

    # Save detailed results to JSON
    json_result_log = RESULTS_DIR / "detailled_results.json"
    for ev in eval_results:
        # Configuration
        top_k = top_k_used
        nb_cluster = NUMBER_CLUSTER

        # Timing data
        t_query_orig_s = ev.t_query_orig_s
        t_embed_query_s = ev.t_embed_query_s
        t_augment_auth_s = ev.t_augment_auth_s
        t_augment_unauth_s = ev.t_augment_unauth_s
        t_query_aug_auth_s = ev.t_query_aug_auth_s
        t_query_aug_unauth_s = ev.t_query_aug_unauth_s
        t_query_meta_auth_s = ev.t_query_meta_auth_s
        t_query_meta_unauth_s = ev.t_query_meta_unauth_s
        t_apply_rotation_s = ev.t_apply_rotation_s
        t_query_rotated_s = ev.t_query_rotated_s


        # Performance metrics (normalized by number of targeted chunks)
        n_targeted = ev.n_targeted_chunks
        targeted_in_auth_query_aug_db = ev.targeted_in_auth_query_aug_db / n_targeted if n_targeted > 0 else 0
        targeted_in_unauth_query_aug_db = ev.targeted_in_unauth_query_aug_db / n_targeted if n_targeted > 0 else 0
        targeted_in_auth_meta = ev.targeted_in_auth_meta / n_targeted if n_targeted > 0 else 0
        targeted_in_unauth_meta = ev.targeted_in_unauth_meta / n_targeted if n_targeted > 0 else 0
        targeted_in_auth_query_rot_db = ev.targeted_in_auth_query_rot_db / n_targeted if n_targeted > 0 else 0
        targeted_in_unauth_query_rot_db = ev.targeted_in_unauth_query_rot_db / n_targeted if n_targeted > 0 else 0

        # Prepare data entry
        data_entry = {
            "query_id": ev.query_id,
            "config_id": ev.config_id,
            "top_k": top_k,
            "nb_cluster": nb_cluster,
            "t_query_orig_s": t_query_orig_s,
            "t_embed_query_s": t_embed_query_s,
            "t_augment_auth_s": t_augment_auth_s,
            "t_augment_unauth_s": t_augment_unauth_s,
            "t_query_aug_auth_s": t_query_aug_auth_s,
            "t_query_aug_unauth_s": t_query_aug_unauth_s,
            "t_query_meta_auth_s": t_query_meta_auth_s,
            "t_query_meta_unauth_s": t_query_meta_unauth_s,
            "t_apply_rotation_s": t_apply_rotation_s,
            "t_query_rotated_s": t_query_rotated_s,
            "targeted_in_auth_query_aug_db": targeted_in_auth_query_aug_db,
            "targeted_in_unauth_query_aug_db": targeted_in_unauth_query_aug_db,
            "targeted_in_auth_meta": targeted_in_auth_meta,
            "targeted_in_unauth_meta": targeted_in_unauth_meta,
            "targeted_in_auth_query_rot_db": targeted_in_auth_query_rot_db,
            "targeted_in_unauth_query_rot_db": targeted_in_unauth_query_rot_db,
            "n_targeted_chunks": n_targeted,
        }

        # Load existing data or initialize
        if os.path.exists(json_result_log):
            with open(json_result_log, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {"results": []}

        # Append new entry
        data["results"].append(data_entry)

        # Save back to file
        with open(json_result_log, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    # Save full evaluation results (unchanged)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(DIM_RESULTS_FILE, "w", encoding="utf-8") as fh:
        json.dump([asdict(r) for r in eval_results], fh, indent=2, ensure_ascii=False)

    if verbose:
        print(f"\n  ✅ Results → {DIM_RESULTS_FILE}")
        print(f"  ✅ Detailed results → {json_result_log}")

    return eval_results


# --- Main experiment function ---
@timed("run_experiment")
def run_experiment_parallel(
    gt_path: str = str(GT_FILE),
    cfg: ExtraDimConfig | None = None,
    top_k: int = DEFAULT_TOP_K,
    verbose: bool = True,
) -> List[QueryEvalResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(gt_path, encoding="utf-8") as fh:
        gt_records: List[dict] = json.load(fh)

    n_queries = NUMBER_CLUSTER
    if cfg is None:
        cfg = ExtraDimConfig()
    cfg.n_queries = n_queries

    print(f"\n{'═' * 60}\n  Extra-Dim + Metadata Access-Control Experiment (Parallel)\n  GT file: {gt_path} ({n_queries} records)\n  Config: {cfg.config_id}\n  Top-K: {top_k}\n{'═' * 60}")

    # clear already existing database
    db_paths = [AUG_CHROMA_BASE, META_CHROMA_BASE, ROTATED_CHROMA]

    # Plot existing DBs before deletion
    print("Databases found before deletion:")
    for db_path in db_paths:
        if os.path.exists(db_path):
            print(f" - {db_path} (EXISTS)")
        else:
            print(f" - {db_path} (NOT FOUND)")

    # Delete existing DBs to ensure a clean state
    for db_path in db_paths:
        if os.path.exists(db_path):
            shutil.rmtree(db_path)

    # Plot DBs after deletion
    print("\nDatabases after deletion attempt:")
    for db_path in db_paths:
        if os.path.exists(db_path):
            print(f" - {db_path} (EXISTS)")
        else:
            print(f" - {db_path} (DELETED or NOT FOUND)") 



    # Initialize model and collections
    embedder = get_embedding_model()
    orig_coll = _get_original_collection(ORIGINAL_CHROMA, ORIGINAL_COLLECTION)
    aug_coll = _get_aug_collection(cfg.config_id)
    meta_coll = _get_meta_collection()
    rot_coll  = _get_or_create_rotated_collection(ROTATED_CHROMA, ROTATED_COLLECTION)
    dim      = len(embedder.embed_query("probe"))
    registry = RotationRegistry(dim=dim)

    # Phase 1: Parallel DB construction
    build_aug_db_parallel(gt_records, cfg, orig_coll, aug_coll, max_workers=NUMBER_THREADS, verbose=False)
    build_meta_db_parallel(gt_records, orig_coll, meta_coll, max_workers=NUMBER_THREADS, verbose=False)
    build_rotated_db_parallel(gt_records, registry, orig_coll, rot_coll, max_workers=NUMBER_THREADS, verbose=False)

    _build_aug_db_all_untargeted_chunks_parallel(gt_records, cfg, orig_coll, aug_coll, batch_size=BATCH_SIZE, max_workers=NUMBER_THREADS, verbose=False)
    build_meta_db_add_untargeted_chunks_parallel(gt_records, orig_coll, meta_coll, batch_size=BATCH_SIZE, max_workers=NUMBER_THREADS, verbose=False)
    _build_rotated_db_untargeted_chunks_parallel(gt_records, registry, orig_coll, rot_coll, batch_size=BATCH_SIZE, max_workers=NUMBER_THREADS, verbose=False)

    # Phase 2: Parallel query phase
    # raw_results = run_query_phase_parallel_with_pickle(gt_records, cfg, registry, embedder, orig_coll, aug_coll, meta_coll, rot_coll, top_k, batch_size=BATCH_SIZE, verbose=True)
    raw_results = run_query_phase_parallel_with_pickle(gt_records, cfg, registry, embedder, orig_coll, aug_coll, meta_coll, rot_coll, top_k, verbose=False)

    # raw_results = run_query_phase_parallel(gt_records, cfg, registry, embedder, orig_coll, aug_coll, meta_coll, rot_coll, top_k, verbose=False)

    # # save_results = RAW_RESULTS_FILE + f"_topk{top_k}"
    # save_results = f"results_experiment_extra_dim/raw_results_topk{top_k}.pkl"
    # #save raw results into a pickle
    # if os.path.exists(save_results):
    #     with open(save_results, "rb") as fh:
    #         existing_results = pickle.load(fh)
    # else:
    #     existing_results = []
    
    # existing_results.extend(raw_results)
    # with open(save_results, 'wb') as f:
    #     pickle.dump(existing_results, f)

    # print(f"\n  ✅ Raw query results saved to {save_results}")
    print(f"\n  ✅ Raw query results saved for top-K={top_k} ")


    # # Phase 3: Evaluation (unchanged)
    # eval_results = evaluate_results_parallel(raw_results, cfg, top_k, verbose=verbose)
    # save_timing_log(eval_results=eval_results)
    # return eval_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Access-control embedding experiment (parallel)")
    parser.add_argument("--gt", default=str(GT_FILE))
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--large-value", type=float, default=DEFAULT_LARGE_VAL)
    parser.add_argument("--normalize-after", default=True, action="store_true")
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    user_cfg = ExtraDimConfig(
        large_value=args.large_value,
        normalize_after=args.normalize_after,
    )

    run_experiment_parallel(
        gt_path=args.gt,
        cfg=user_cfg,
        top_k=args.top_k,
        verbose=not args.quiet,
    )
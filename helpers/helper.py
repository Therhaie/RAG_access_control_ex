
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from typing import List, Optional
import random
import os
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
import umap.umap_ as umap


def get_all_chunk_ids(data) -> List[str]:
    """Extract all chunk ids from the data, in the format "triplet_index|document_id|phrase_seq"."""
    chunk_ids : List[str] = []
    for chunk in data:
        id_triplet = chunk.get("id_triplets", "")
        for sentence in chunk.get("sentences", []):
            chunk_ids.append(f"{id_triplet}|{sentence[0][-2]}|{sentence[0][-1]}")
    return chunk_ids


def get_list_id_targeted_chunk(data) -> List[str]:
    """Extract all targeted chunk ids from the data, in the format "triplet_index|document_id|phrase_seq"."""
    list_of_chunk_ids = []
    for chunk in data:
        for chunk_id in chunk.get("targeted_chunk", []):
            if chunk_id not in list_of_chunk_ids:
                list_of_chunk_ids.append(chunk_id)
    return list_of_chunk_ids


def get_id_untargeted_chunk(data, n_untargeted_chunks_displayed=10) -> List[str]:
    untargeted_chunks = []
    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    for chunk_id in list_of_chunk_ids:
        if chunk_id not in list_of_targeted_chunk_ids:
            untargeted_chunks.append(chunk_id)
    return np.random.choice(untargeted_chunks, size=min(n_untargeted_chunks_displayed, len(untargeted_chunks)), replace=False).tolist()

def get_id_clusters(data, n_user_displayed=5) :
    """Give the id of NUMBER_OF_CLUSTERS_DISPLAYED targeted chunks cluster using the SEED, """
    clusters = []
    questions : list[str] = []
    for chunk in data:
        cluster = []
        questions.append(chunk.get("question", ""))
        for targeted_chunk in chunk.get("targeted_chunk", []):
            cluster.append(f'{targeted_chunk.split("|")[0]}|{targeted_chunk[-2]}|{targeted_chunk[-1]}')
        clusters.append(cluster)
    selected_indices = random.sample(range(len(clusters)), n_user_displayed)
    return [clusters[i] for i in selected_indices], [questions[i] for i in selected_indices]
            
def fetch_embeddings(
    collection_path: str,
    collection_name: str,
    chunk_ids: List[str],
    is_rotated: bool = False,
    is_dim: bool = False,
) -> np.ndarray:
    client = chromadb.PersistentClient(
        path=collection_path,
        settings=Settings(anonymized_telemetry=False),
    )
    collection = client.get_collection(name=collection_name)

    embeddings = []
    for chunk_id in chunk_ids:
        # Parse chunk_id (format: "triplet_index|document_id|phrase_seq")
        triplet_index = str(chunk_id.split("|")[0])
        document_id = str(chunk_id.split("|")[1])
        phrase_seq = str(chunk_id.split("|")[2])
        
        chroma_id = f"{triplet_index}_{document_id}_{phrase_seq}"
        if collection_path == DIRECTORY_ROTATION_DB :
            result = collection.get(ids=[chroma_id], include=["embeddings"])
        if collection_path == DIRECTORY_AUGMENTED_DB:
            result = collection.get(
                where={
                    "$and": [
                        {"triplet_index": {"$eq": triplet_index}},
                        {"document_id":   {"$eq": document_id}},
                        {"phrase_seq":    {"$eq": phrase_seq}},
                    ]
                },
                include=["embeddings"],
            )
        elif collection_path == DIRECTORY_BASELINE_DB:
            result = collection.get(
                where={
                    "$and": [
                        {"triplet_index": {"$eq": triplet_index}},
                        {"document_id":   {"$eq": document_id}},
                        {"phrase_seq":    {"$eq": phrase_seq}},
                    ]
                },
                include=["embeddings"],
            )

        if len(result["embeddings"][0]) > 0:
            embeddings.append(result["embeddings"][0])

    return np.array(embeddings, dtype=np.float32)


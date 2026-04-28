from opti_experiment_raw_retrieve import RawQueryResults

import os
import pickle
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import astuple, fields
from typing import List, Dict, Set, Optional
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
import hashlib
from scipy.stats import ortho_group
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity



# PATH_RAW_RESULTS = "results_experiment_extra_dim/185.pkl"
# PATH_GROUND_TRUTH = "results_experiment_extra_dim/GT_results/ground_truth_"

PATH_GROUND_TRUTH = os.path.join(os.getcwd(), "results_experiment_extra_dim", "GT_results")
PATH_RAW_RESULTS = os.path.join(os.getcwd(), "results_experiment_extra_dim")
PATH_ALL_CHUNKS = os.path.join(os.getcwd(), "RAGBench_whole", "list_chunks_id.json")

# function to turn all list of ground truth into dict

def cosine_similarity_custom(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0


def rotate_vector_with_user(vec: np.ndarray, user_index: str) -> np.ndarray:
    # Seed from user_index
    seed = int(hashlib.sha256(str(user_index).encode()).hexdigest()[:8], 16) % (2 ** 31)
    # Rotation matrix
    dim = vec.shape[-1]
    R = ortho_group.rvs(dim=dim, random_state=seed).astype(np.float32)
    # Apply rotation
    if vec.ndim == 1:
        return R @ vec
    return (R @ vec.T).T


def turn_GT_list_to_dict(top_k):
    gt_path = os.path.join(PATH_GROUND_TRUTH, f'ground_truth_{top_k}.json')
    with open(gt_path, 'r') as f:
        data_list = json.load(f)
    # Convert to dict
    ground_truth_dict_ = {item['id_triplets']: item for item in data_list}

    # Save the dict as a JSON file with a dynamic filename
    save_path = os.path.join(PATH_GROUND_TRUTH, f'ground_truth_dict_{top_k}.json')
    with open(save_path, 'w') as f:
        json.dump(ground_truth_dict_, f, indent=2)


def load_ground_truths(ground_truth_file: str) -> Dict[str, Dict]:
    """Load ground truth data from JSON file."""
    with open(ground_truth_file, 'r') as f:
        return json.load(f)

def load_chunk_ids(chunk_list_file: str) -> Set[str]:
    """Load all chunk IDs from text file."""
    with open(chunk_list_file, 'r') as f:
        return set(line.strip() for line in f)



def create_untargeted_chunk_set(all_chunk_ids: set, targeted_chunks: dict) -> set:
    """
    Create a set of untargeted chunk IDs by excluding all 'targeted_chunk' elements from all_chunk_ids.
    """
    all_targeted_chunks = set()
    for value in targeted_chunks.values():
        liste = value.get('targeted_chunk', [])  # get list or empty if missing
        all_targeted_chunks.update(set(liste))
    return set(all_chunk_ids) - all_targeted_chunks


def compute_E_user(object: RawQueryResults):
    """Compute E_user for a given RawQueryResults object."""
    E_u_dic = {}
    for item in object:
        if item.user_index not in E_u_dic:
            E_u_dic[item.user_index] = set(item.list_ground_truth)
        else:
            E_u_dic[item.user_index].update(set(item.list_ground_truth))
    return E_u_dic

  
# def compute_average_distance(object, max_workers=4):
#     # Group embeddings by user
#     distance_intra_user = {}
#     nb_user = object.number_user
#     high_value_encoding = object.high_value_encoding_chunk

#     # Data Preparation: O(N)
#     for item in object:
#         for chunk in item.list_ground_truth:
#             embedding_chunk = item.embedding_retrieved_ground_truth[chunk]
#             distance_intra_user.setdefault(item.user_index, []).append(embedding_chunk)

#     # Computation per user
#     def process_user(user):
#         embeddings = np.array(distance_intra_user[user])
#         n = len(embeddings)

#         # Early exit on small sets
#         if n < 2:
#             return user, 0, 0

#         # Cosine similarity (upper triangle, excludes diag/self-pairs)
#         sim_matrix = cosine_similarity(embeddings)
#         dists = sim_matrix[np.triu_indices(n, k=1)]

#         # Augmented: same extra to both, as in your code
#         extra = np.zeros(nb_user)
#         extra[user] = high_value_encoding
#         aug_embeds = np.concatenate([embeddings, np.tile(extra, (n, 1))], axis=1)
#         sim_aug = cosine_similarity(aug_embeds)
#         dists_aug = sim_aug[np.triu_indices(n, k=1)]

#         return user, np.mean(dists), np.mean(dists_aug)

#     # Parallelized processing
#     results = []
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         futures = [executor.submit(process_user, user) for user in distance_intra_user]
#         for fut in futures:
#             results.append(fut.result())

#     # Collect results
#     average_distance_intra_user = {user: mean_orig for user, mean_orig, _ in results}
#     average_distance_intra_user_augmented = {user: mean_aug for user, _, mean_aug in results}

#     return average_distance_intra_user, average_distance_intra_user_augmented

# def distance_filter(set_retrieved, embedding_retrieved, query_embedding, threshold=0.01):
#     """Filter retrieved chunks based on cosine similarity distance to the query embedding.
#         The threshold has to be given in argument."""
#     for chunk in set_retrieved:
#         retrieved_embedding = embedding_retrieved[chunk]
#         distance = cosine_similarity_custom(retrieved_embedding, query_embedding)
#         if distance > threshold:
#             set_retrieved.remove(chunk)
#     return set_retrieved

def distance_filter(set_retrieved, embedding_retrieved, query_embedding, threshold=0.01):
    """
    Filter retrieved chunks based on cosine similarity distance to the query embedding.
    The threshold has to be given in argument.
    """
    # Use set comprehension to build a new filtered set
    filtered_set = {
        chunk for chunk in set_retrieved
        if cosine_similarity_custom(embedding_retrieved[chunk], query_embedding) > threshold
    }
    return filtered_set


def compute_average_distance(object, max_workers=4):
    # Group embeddings by user
    distance_intra_user = {}
    nb_user = object[0].number_user
    high_value_encoding = object[0].high_value_encoding_chunk

    # Data Preparation: O(N)
    for item in object:
        for chunk in item.list_ground_truth:
            embedding_chunk = item.embedding_retrieved_ground_truth[chunk]
            distance_intra_user.setdefault(item.user_index, []).append(embedding_chunk)

    # Computation per user (INTRA-user)
    def process_user(user):
        embeddings = np.array(distance_intra_user[user])
        n = len(embeddings)

        # Early exit on small sets
        if n < 2:
            return user, 0, 0

        # Cosine similarity (upper triangle, excludes diag/self-pairs)
        sim_matrix = cosine_similarity(embeddings)
        dists = sim_matrix[np.triu_indices(n, k=1)]

        # Augmented: same extra to both, as in your code
        extra = np.zeros(nb_user)
        extra[user] = high_value_encoding
        aug_embeds = np.concatenate([embeddings, np.tile(extra, (n, 1))], axis=1)
        sim_aug = cosine_similarity(aug_embeds)
        dists_aug = sim_aug[np.triu_indices(n, k=1)]

        return user, np.mean(dists), np.mean(dists_aug)

    # Parallelized processing (INTRA-user)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_user, user) for user in distance_intra_user]
        for fut in futures:
            results.append(fut.result())

    # Collect intra-user average similarities
    average_distance_intra_user = {user: mean_orig for user, mean_orig, _ in results}
    average_distance_intra_user_augmented = {user: mean_aug for user, _, mean_aug in results}

    # --- INTER-user average centroid similarity ---
    # Compute centroids per user, both normal and augmented
    user_centroids = {}
    user_centroids_aug = {}
    for user, embs in distance_intra_user.items():
        embs = np.array(embs)
        user_centroids[user] = np.mean(embs, axis=0)
        # Augmentations
        extra = np.zeros(nb_user)
        extra[user] = high_value_encoding
        aug_embs = np.concatenate([embs, np.tile(extra, (len(embs), 1))], axis=1)
        user_centroids_aug[user] = np.mean(aug_embs, axis=0)

    # Build centroid arrays
    users = sorted(user_centroids.keys())
    centroid_array = np.stack([user_centroids[u] for u in users])
    centroid_array_aug = np.stack([user_centroids_aug[u] for u in users])

    # Cosine similarity matrix between centroids
    sim_centroid = cosine_similarity(centroid_array)
    sim_centroid_aug = cosine_similarity(centroid_array_aug)

    # Get upper triangle, excluding diagonal
    n = sim_centroid.shape[0]
    inter_user_sim = sim_centroid[np.triu_indices(n, k=1)]
    inter_user_sim_aug = sim_centroid_aug[np.triu_indices(n, k=1)]

    average_centroid_inter_user = np.mean(inter_user_sim) if inter_user_sim.size else 0
    average_centroid_inter_user_aug = np.mean(inter_user_sim_aug) if inter_user_sim_aug.size else 0

    # Return both intra- and inter-user values
    return (
        average_distance_intra_user,
        average_distance_intra_user_augmented,
        average_centroid_inter_user,
        average_centroid_inter_user_aug
    )



def process_raw_query_results(top_k,
                              all_chunk_ids: Set[str]):
    # Load the entire big pickle (needs enough RAM just this first time)
    
    path = PATH_RAW_RESULTS + f"/raw_results_topk{top_k}.pkl"
    with open(path, "rb") as f:
        raw_query_results = pickle.load(f)
    
    ground_truth = PATH_GROUND_TRUTH + f'/ground_truth_dict_{top_k}.json'
    with open(ground_truth, 'r') as f:
        ground_truth_data = json.load(f)

    untargeted = create_untargeted_chunk_set(all_chunk_ids, ground_truth_data)

    dic_E_user = compute_E_user(raw_query_results) # dic
    average_distance_intra_user, average_distance_intra_user_augmented, average_centroid_inter_user, average_centroid_inter_user_aug = compute_average_distance(raw_query_results) 



    # dic_intra_user_distance, distance_intro_cluster_augmented = compute_average_distance(raw_query_results) # dic
    # dic_intra_user_distance = {user: compute_average_distance(E_u) for user, E_u in dic_E_user.items()}


    
    # list for the results
    list_AAR_meta_auth = []    
    list_AAR_rot_auth = []    
    list_AAR_aug_auth = []

    list_AAP_meta_auth = []    
    list_AAP_rot_auth = []    
    list_AAP_aug_auth = []    

    list_FAR_meta_auth = []
    list_FAR_rot_auth = []
    list_FAR_aug_auth = []

    list_FAR_user_meta_auth = []
    list_FAR_user_rot_auth = []
    list_FAR_user_aug_auth = []

    for result in raw_query_results:
        query_index = result.query_index # triplet_358
        user_index = result.user_index
        intra_user_distance = average_distance_intra_user.get(user_index, 0)
        intra_user_distance_aug = average_distance_intra_user_augmented.get(user_index, 0)

        # threshold = (average_centroid_inter_user - intra_user_distance) / 2
        # threshold_aug = (average_centroid_inter_user_aug - intra_user_distance_aug) / 2

        # threshold_intra = intra_user_distance
        # threshold_intra_aug = intra_user_distance_aug 

        query_number = query_index.split("_")[-1]  # Extract the number part after the last underscore
        targeted_chunks = ground_truth_data.get(query_index, {}).get('targeted_chunk', [])
        
        targeted_chunks = set(result.list_ground_truth)



        list_retrieved_meta_auth = set(result.list_retrieved_meta_auth)
        list_retrieved_rot_auth = set(result.list_retrieved_rot_auth)
        list_retrieved_aug_auth = set(result.list_retrieved_aug_auth)


        if False:
            # Filter the list of retrieved chunks by distance
            list_retrieved_meta_auth = distance_filter(list_retrieved_meta_auth, result.embedding_retrieved_meta_auth, result.embedding_query_meta_auth, threshold=threshold)
            list_retrieved_rot_auth = distance_filter(list_retrieved_rot_auth, result.embedding_retrieved_rot_auth, result.embedding_query_meta_auth, threshold=threshold) # not need to use the rotated query because the chunks are not rotated, and the rotation preserves distances
            list_retrieved_aug_auth = distance_filter(list_retrieved_aug_auth, result.embedding_retrieved_aug_auth, result.embedding_query_aug_auth, threshold=threshold_aug)


            list_retrieved_meta_auth = set(result.list_retrieved_meta_auth)
            list_retrieved_rot_auth = set(result.list_retrieved_rot_auth)
            list_retrieved_aug_auth = set(result.list_retrieved_aug_auth)

            # Filter the list of retrieved chunks by distance
            list_retrieved_meta_auth = distance_filter(list_retrieved_meta_auth, result.embedding_retrieved_meta_auth, result.embedding_query_meta_auth, threshold=threshold)
            list_retrieved_rot_auth = distance_filter(list_retrieved_rot_auth, result.embedding_retrieved_rot_auth, result.embedding_query_meta_auth, threshold=threshold) # not need to use the rotated query because the chunks are not rotated, and the rotation preserves distances
            list_retrieved_aug_auth = distance_filter(list_retrieved_aug_auth, result.embedding_retrieved_aug_auth, result.embedding_query_aug_auth, threshold=threshold_aug)


        VR_meta_auth = list_retrieved_meta_auth & (set(targeted_chunks) )
        VR_rot_auth = list_retrieved_rot_auth & (set(targeted_chunks) )
        VR_aug_auth = list_retrieved_aug_auth & (set(targeted_chunks))

        AAR_meta_auth = len(VR_meta_auth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_rot_auth = len(VR_rot_auth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_aug_auth = len(VR_aug_auth) / len(targeted_chunks) if targeted_chunks else 0

        list_AAR_meta_auth.append(AAR_meta_auth)
        list_AAR_rot_auth.append(AAR_rot_auth)
        list_AAR_aug_auth.append(AAR_aug_auth)

        ### compute AAP
        AAP_meta_auth = len(VR_meta_auth) / len(list_retrieved_meta_auth) if list_retrieved_meta_auth else 0
        AAP_rot_auth = len(VR_rot_auth) / len(list_retrieved_rot_auth) if list_retrieved_rot_auth else 0
        AAP_aug_auth = len(VR_aug_auth) / len(list_retrieved_aug_auth) if list_retrieved_aug_auth else 0

        list_AAP_meta_auth.append(AAP_meta_auth)
        list_AAP_rot_auth.append(AAP_rot_auth)
        list_AAP_aug_auth.append(AAP_aug_auth)

        ### FAR
        F = all_chunk_ids - (targeted_chunks | untargeted)
        FAR_meta_auth = list_retrieved_meta_auth & F
        FA_rot_auth = list_retrieved_rot_auth & F
        FA_aug_auth = list_retrieved_aug_auth & F

        list_FAR_meta_auth.append(len(FAR_meta_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_rot_auth.append(len(FA_rot_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_aug_auth.append(len(FA_aug_auth) / len(targeted_chunks) if targeted_chunks else 0)


        ### FAR_user
        F_user = all_chunk_ids - (dic_E_user[result.user_index] | untargeted)
        # FAR_user_meta_auth = list_retrieved_meta_auth & (F - dic_E_user[result.user_index])
        # FAR_user_rot_auth = list_retrieved_rot_auth & (F - dic_E_user[result.user_index])
        # FAR_user_aug_auth = list_retrieved_aug_auth & (F - dic_E_user[result.user_index])

        FAR_user_meta_auth = list_retrieved_meta_auth & (F_user)
        FAR_user_rot_auth = list_retrieved_rot_auth & (F_user)
        FAR_user_aug_auth = list_retrieved_aug_auth & (F_user)

        list_FAR_user_meta_auth.append(len(FAR_user_meta_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_user_rot_auth.append(len(FAR_user_rot_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_user_aug_auth.append(len(FAR_user_aug_auth) / len(targeted_chunks) if targeted_chunks else 0)

        ### UFAR







    
    # Average every AAR list
    avg_AAR_meta_auth = np.mean(list_AAR_meta_auth)
    avg_AAR_rot_auth = np.mean(list_AAR_rot_auth)
    avg_AAR_aug_auth = np.mean(list_AAR_aug_auth)

    avg_AAP_meta_auth = np.mean(list_AAP_meta_auth)
    avg_AAP_rot_auth = np.mean(list_AAP_rot_auth)
    avg_AAP_aug_auth = np.mean(list_AAP_aug_auth)

    avg_list_FAR_meta_auth = np.mean(list_FAR_meta_auth)
    avg_list_FAR_rot_auth = np.mean(list_FAR_rot_auth)
    avg_list_FAR_aug_auth = np.mean(list_FAR_aug_auth)

    avg_list_FAR_user_meta_auth = np.mean(list_FAR_user_meta_auth)
    avg_list_FAR_user_rot_auth = np.mean(list_FAR_user_rot_auth)
    avg_list_FAR_user_aug_auth = np.mean(list_FAR_user_aug_auth)



    print(f"Top-k: {top_k}")
    print(f"Average AAR Meta: {avg_AAR_meta_auth:.4f}")
    print(f"Average AAR Rot: {avg_AAR_rot_auth:.4f}")
    print(f"Average AAR Aug: {avg_AAR_aug_auth:.4f}")

    print("\n")

    print(f"Average AAP Meta: {avg_AAP_meta_auth:.4f}")
    print(f"Average AAP Rot: {avg_AAP_rot_auth:.4f}")
    print(f"Average AAP Aug: {avg_AAP_aug_auth:.4f}")

    print("\n")

    print(f"Average FAR Meta: {avg_list_FAR_meta_auth:.4f}")
    print(f"Average FAR Rot: {avg_list_FAR_rot_auth:.4f}")
    print(f"Average FAR Aug: {avg_list_FAR_aug_auth:.4f}")

    print("\n")

    print(f"Average FAR User Meta: {avg_list_FAR_user_meta_auth:.4f}")
    print(f"Average FAR User Rot: {avg_list_FAR_user_rot_auth:.4f}")
    print(f"Average FAR User Aug: {avg_list_FAR_user_aug_auth:.4f}")


    # Filter by top_k


if __name__ == "__main__":
    # turn_GT_list_to_dict("results_experiment_extra_dim/GT_results")
    list_k = np.unique(np.logspace(np.log10(1), np.log10(2000), num=100, dtype=int))
    for k in list_k:
        if k >= 10 and k <11:
            turn_GT_list_to_dict(k)
            with open(PATH_ALL_CHUNKS, 'r') as f:
                all_chunk_ids = set(json.load(f))
            # all_chunk_ids = load_chunk_ids("RAGBench_whole/all_chunk_ids.txt")
            process_raw_query_results(k, all_chunk_ids=all_chunk_ids)

















# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import json

# from opti_experiment_raw_retrieve import RawQueryResults



# # Group the relevant data by top-k value

# ### Helper functions

# # function to extract all the value for a given top-k value
# def extract_values_for_top_k(raw_query_results, top_k):
#     values = []
#     for result in raw_query_results:
#         if result.top_k == top_k:
#             values.append(result.value)
#     return values

# def get_GT_query_index(query_index, top_k):
#     """Returns the targeted chunks for a query, index with a given top-k value."""
#     with open(f"results_experiment_extra_dim/GT_results/ground_truth_{top_k}_copy.json", "rb") as f:
#         ground_truth = json.load(f)
#     ground_truth = {item['id_triplets']: item for item in ground_truth}
#     # data = (ground_truth['id_triplets'] == query_index).get('targeted_chunk')
#     # return data
#     # target_item = ground_truth.get(f'query_index')   # get item by key (may be None!)
#     target_item = ground_truth[f"{query_index}"]   # get item by key (may be None!)
#     if target_item is not None:
#         data = target_item.get('targeted_chunk')  # get targeted_chunk from item
#         return data
#     else:
#         return None  # or handle "not found" case as you like
    
#     print(f"keys in GT: {len(ground_truth.keys())}")
    


# ### Intermediate processing functions

# def compute_T_k_i():
#     pass


# ### Metrics functions





#     # print(get_GT_query_index(358,10))

#     with open("raw_results.pkl", "rb") as f:
#         raw_query_results = pickle.load(f)

#     # top_ks = [10, 11, 12]
#     # grouped_values = {}
#     top_k = 10
#     values_for_top_k = extract_values_for_top_k(raw_query_results, top_k)
#     print(len(values_for_top_k))

#     # for top_k in top_ks:
#     #     grouped_values[top_k] = extract_values_for_top_k(raw_query_results, top_k)

#         # Open the right GT file 


# # Load the entire big pickle (needs enough RAM just this first time)
# with open("raw_results.pkl", "rb") as f:
#     raw_query_results = pickle.load(f)

# # Group by top_k
# groups = defaultdict(list)

# for obj in raw_query_results:
#     # Assumes each obj has a .top_k attribute (may need to adapt for dict)
#     groups[obj.top_k].append(obj)

# # Save each group as a separate pickle file
# for top_k, group in groups.items():
#     filename = f"raw_results_topk_{top_k}.pkl"
#     with open(filename, "wb") as f_out:
#         pickle.dump(group, f_out)
#     print(f"Wrote {len(group)} items to {filename}")




# if __name__ == "__main__":
#     # Example of loading one of the new files
#     path = os.path.join("results_experiment_extra_dim", "raw_results_topk_10.pkl")
#     with open("results_experiment_extra_dim/raw_results_topk10.pkl", "rb") as f:
#         top_k_10_results = pickle.load(f)
#     print(f"Loaded {len(top_k_10_results)} results for top_k=10")

      
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

# PATH_RAW_RESULTS = "results_experiment_extra_dim/185.pkl"
# PATH_GROUND_TRUTH = "results_experiment_extra_dim/GT_results/ground_truth_"

PATH_GROUND_TRUTH = os.path.join(os.getcwd(), "results_experiment_extra_dim", "GT_results")
PATH_RAW_RESULTS = os.path.join(os.getcwd(), "results_experiment_extra_dim")
PATH_ALL_CHUNKS = os.path.join(os.getcwd(), "RAGBench_whole", "list_chunks_id.json")

# function to turn all list of ground truth into dict

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

# def create_untargeted_chunk_set(all_chunk_ids: Set[str], targeted_chunks: List[str]) -> Set[str]:
#     """Create a set of untargeted chunk IDs by excluding targeted chunks from all chunk IDs."""
#     all_targeted_chunks = set()
#     for element in targeted_chunks:
#         liste = targeted_chunks[element]['targeted_chunk']
#         set_liste = set(liste)
#         all_targeted_chunks.update(set_liste)

#     returna = set(all_chunk_ids) - all_targeted_chunks
#     return returna

def create_untargeted_chunk_set(all_chunk_ids: set, targeted_chunks: dict) -> set:
    """
    Create a set of untargeted chunk IDs by excluding all 'targeted_chunk' elements from all_chunk_ids.
    """
    all_targeted_chunks = set()
    for value in targeted_chunks.values():
        liste = value.get('targeted_chunk', [])  # get list or empty if missing
        all_targeted_chunks.update(set(liste))
    return set(all_chunk_ids) - all_targeted_chunks


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
    
    # list for the results
    list_AAR_meta_auth = []    
    list_AAR_meta_unauth = []
    list_AAR_rot_auth = []    
    list_AAR_rot_unauth = []
    list_AAR_aug_auth = []
    list_AAR_aug_unauth = []

    list_AAP_meta_auth = []    
    list_AAP_meta_unauth = []
    list_AAP_rot_auth = []    
    list_AAP_rot_unauth = []
    list_AAP_aug_auth = []    
    list_AAP_aug_unauth = []

    list_FAR_meta_auth = []
    list_FAR_meta_unauth = []
    list_FAR_rot_auth = []
    list_FAR_rot_unauth = []
    list_FAR_aug_auth = []
    list_FAR_aug_unauth = []

    for result in raw_query_results:
        query_index = result.query_index # triplet_358
        query_number = query_index.split("_")[-1]  # Extract the number part after the last underscore
        targeted_chunks = ground_truth_data.get(query_index, {}).get('targeted_chunk', [])
        
        targeted_chunks = set(result.list_ground_truth)

        list_retrieved_meta_auth = set(result.list_retrieved_meta_auth)
        list_retrieved_meta_unauth = set(result.list_retrieved_meta_unauth)
        list_retrieved_rot_auth = set(result.list_retrieved_rot_auth)
        list_retrieved_rot_unauth = set(result.list_retrieved_rot_unauth)
        list_retrieved_aug_auth = set(result.list_retrieved_aug_auth)
        list_retrieved_aug_unauth = set(result.list_retrieved_aug_unauth)

        # VR_meta_auth = list_retrieved_meta_auth & (set(targeted_chunks) | untargeted)
        # VR_meta_unauth = list_retrieved_meta_unauth & (set(targeted_chunks) | untargeted)
        # VR_rot_auth = list_retrieved_rot_auth & (set(targeted_chunks) | untargeted)
        # VR_rot_unauth = list_retrieved_rot_unauth & (set(targeted_chunks) | untargeted)
        # VR_aug_auth = list_retrieved_aug_auth & (set(targeted_chunks) | untargeted)
        # VR_aug_unauth = list_retrieved_aug_unauth & (set(targeted_chunks) | untargeted)

        VR_meta_auth = list_retrieved_meta_auth & (set(targeted_chunks) )
        VR_meta_unauth = list_retrieved_meta_unauth & (set(targeted_chunks) )
        VR_rot_auth = list_retrieved_rot_auth & (set(targeted_chunks) )
        VR_rot_unauth = list_retrieved_rot_unauth & (set(targeted_chunks) )
        VR_aug_auth = list_retrieved_aug_auth & (set(targeted_chunks))
        VR_aug_unauth = list_retrieved_aug_unauth & (set(targeted_chunks) )

        AAR_meta_auth = len(VR_meta_auth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_meta_unauth = len(VR_meta_unauth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_rot_auth = len(VR_rot_auth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_rot_unauth = len(VR_rot_unauth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_aug_auth = len(VR_aug_auth) / len(targeted_chunks) if targeted_chunks else 0
        AAR_aug_unauth = len(VR_aug_unauth) / len(targeted_chunks) if targeted_chunks else 0 

        list_AAR_meta_auth.append(AAR_meta_auth)
        list_AAR_meta_unauth.append(AAR_meta_unauth)
        list_AAR_rot_auth.append(AAR_rot_auth)
        list_AAR_rot_unauth.append(AAR_rot_unauth)
        list_AAR_aug_auth.append(AAR_aug_auth)
        list_AAR_aug_unauth.append(AAR_aug_unauth)

        ### compute AAP
        AAP_meta_auth = len(VR_meta_auth) / len(list_retrieved_meta_auth) if list_retrieved_meta_auth else 0
        AAP_meta_unauth = len(VR_meta_unauth) / len(list_retrieved_meta_unauth) if list_retrieved_meta_unauth else 0
        AAP_rot_auth = len(VR_rot_auth) / len(list_retrieved_rot_auth) if list_retrieved_rot_auth else 0
        AAP_rot_unauth = len(VR_rot_unauth) / len(list_retrieved_rot_unauth) if list_retrieved_rot_unauth else 0
        AAP_aug_auth = len(VR_aug_auth) / len(list_retrieved_aug_auth) if list_retrieved_aug_auth else 0
        AAP_aug_unauth = len(VR_aug_unauth) / len(list_retrieved_aug_unauth) if list_retrieved_aug_unauth else 0

        list_AAP_meta_auth.append(AAP_meta_auth)
        list_AAP_meta_unauth.append(AAP_meta_unauth)
        list_AAP_rot_auth.append(AAP_rot_auth)
        list_AAP_rot_unauth.append(AAP_rot_unauth)
        list_AAP_aug_auth.append(AAP_aug_auth)
        list_AAP_aug_unauth.append(AAP_aug_unauth)

        ### FAR
        F = all_chunk_ids - (targeted_chunks | untargeted)
        FAR_meta_auth = list_retrieved_meta_auth & F
        FA_meta_unauth = list_retrieved_meta_unauth & F
        FA_rot_auth = list_retrieved_rot_auth & F
        FA_rot_unauth = list_retrieved_rot_unauth & F
        FA_aug_auth = list_retrieved_aug_auth & F
        FA_aug_unauth = list_retrieved_aug_unauth & F

        list_FAR_meta_auth.append(len(FAR_meta_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_meta_unauth.append(len(FA_meta_unauth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_rot_auth.append(len(FA_rot_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_rot_unauth.append(len(FA_rot_unauth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_aug_auth.append(len(FA_aug_auth) / len(targeted_chunks) if targeted_chunks else 0)
        list_FAR_aug_unauth.append(len(FA_aug_unauth) / len(targeted_chunks) if targeted_chunks else 0)

        # list_FAR_meta_auth.append(len(FAR_meta_auth) / len(F) if F else 0)
        # list_FAR_meta_unauth.append(len(FA_meta_unauth) / len(F) if F else 0)
        # list_FAR_rot_auth.append(len(FA_rot_auth) / len(F) if F else 0)
        # list_FAR_rot_unauth.append(len(FA_rot_unauth) / len(F) if F else 0)
        # list_FAR_aug_auth.append(len(FA_aug_auth) / len(F) if F else 0)
        # list_FAR_aug_unauth.append(len(FA_aug_unauth) / len(F) if F else 0)




    
    # Average every AAR list
    avg_AAR_meta_auth = np.mean(list_AAR_meta_auth)
    avg_AAR_meta_unauth = np.mean(list_AAR_meta_unauth)
    avg_AAR_rot_auth = np.mean(list_AAR_rot_auth)
    avg_AAR_rot_unauth = np.mean(list_AAR_rot_unauth)
    avg_AAR_aug_auth = np.mean(list_AAR_aug_auth)
    avg_AAR_aug_unauth = np.mean(list_AAR_aug_unauth)

    avg_AAP_meta_auth = np.mean(list_AAP_meta_auth)
    avg_AAP_meta_unauth = np.mean(list_AAP_meta_unauth)
    avg_AAP_rot_auth = np.mean(list_AAP_rot_auth)
    avg_AAP_rot_unauth = np.mean(list_AAP_rot_unauth)
    avg_AAP_aug_auth = np.mean(list_AAP_aug_auth)
    avg_AAP_aug_unauth = np.mean(list_AAP_aug_unauth)

    avg_list_FAR_meta_auth = np.mean(list_FAR_meta_auth)
    avg_list_FAR_meta_unauth = np.mean(list_FAR_meta_unauth)
    avg_list_FAR_rot_auth = np.mean(list_FAR_rot_auth)
    avg_list_FAR_rot_unauth = np.mean(list_FAR_rot_unauth)
    avg_list_FAR_aug_auth = np.mean(list_FAR_aug_auth)
    avg_list_FAR_aug_unauth = np.mean(list_FAR_aug_unauth)



    print(f"Top-k: {top_k}")
    print(f"Average AAR Meta Auth: {avg_AAR_meta_auth:.4f}")
    print(f"Average AAR Meta Unauth: {avg_AAR_meta_unauth:.4f}")
    print(f"Average AAR Rot Auth: {avg_AAR_rot_auth:.4f}")
    print(f"Average AAR Rot Unauth: {avg_AAR_rot_unauth:.4f}")
    print(f"Average AAR Aug Auth: {avg_AAR_aug_auth:.4f}")
    print(f"Average AAR Aug Unauth: {avg_AAR_aug_unauth:.4f}")

    print("\n")

    print(f"Average AAP Meta Auth: {avg_AAP_meta_auth:.4f}")
    print(f"Average AAP Meta Unauth: {avg_AAP_meta_unauth:.4f}")
    print(f"Average AAP Rot Auth: {avg_AAP_rot_auth:.4f}")
    print(f"Average AAP Rot Unauth: {avg_AAP_rot_unauth:.4f}")
    print(f"Average AAP Aug Auth: {avg_AAP_aug_auth:.4f}")
    print(f"Average AAP Aug Unauth: {avg_AAP_aug_unauth:.4f}")

    print("\n")

    print(f"Average FAR Meta Auth: {avg_list_FAR_meta_auth:.4f}")
    print(f"Average FAR Meta Unauth: {avg_list_FAR_meta_unauth:.4f}")
    print(f"Average FAR Rot Auth: {avg_list_FAR_rot_auth:.4f}")
    print(f"Average FAR Rot Unauth: {avg_list_FAR_rot_unauth:.4f}")
    print(f"Average FAR Aug Auth: {avg_list_FAR_aug_auth:.4f}")
    print(f"Average FAR Aug Unauth: {avg_list_FAR_aug_unauth:.4f}")


        


    

    chunk_sim_retrieved_meta_auth = []
    chunk_sim_retrieved_meta_unauth = []
    chunk_sim_retrieved_rot_auth = []
    chunk_sim_retrieved_rot_unauth = []
    chunk_sim_retrieved_aug_auth = []
    chunk_sim_retrieved_aug_unauth = []

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

      
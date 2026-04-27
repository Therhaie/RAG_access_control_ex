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



# if __name__ == "__main__":

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
from opti_experiment_raw_retrieve import RawQueryResults

import os
import pickle
from collections import defaultdict

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


if __name__ == "__main__":
    # Example of loading one of the new files
    path = os.path.join("results_experiment_extra_dim", "raw_results_topk_10.pkl")
    with open("results_experiment_extra_dim/raw_results_topk10.pkl", "rb") as f:
        top_k_10_results = pickle.load(f)
    print(f"Loaded {len(top_k_10_results)} results for top_k=10")
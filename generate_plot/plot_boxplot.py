# # import matplotlib.pyplot as plt
# # from pathlib import Path
# # import pandas as pd
# # import json
# # import os
# # import argparse
# # import numpy as np

# # def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
# #     denom = np.linalg.norm(a) * np.linalg.norm(b)
# #     return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

# # # === 1. Load Data ===
# # path = os.path.join(
# #     os.getcwd(),
# #     'results_experiment_extra_dim',
# #     'dim_results.json'
# # )
# # path2 = os.path.join(
# #     os.getcwd(),
# #     'results_experiment_extra_dim',
# #     'detailled_results.json'
# # )

# # def parse_args():
# #     parser = argparse.ArgumentParser(description="Embedding cluster visualization experiment control.")
# #     parser.add_argument("--results-path", type=str, default=path, help="Path to the JSON file containing the results.")




# # if __name__ == "__main__":
# #     args = parse_args()
# #     # path_results = os.path.join(os.path.dirname(os.getcwd()), 'results_experiment', args.results_path)
# #     path_results = os.path.join(os.getcwd(), 'results_experiment', 'raw_results_topk_10.jsonl')

    
# #     with open(path_results, 'r') as f:
# #         results = [json.loads(line) for line in f]

# #     sim_aa_aug, sim_bb_aug, sim_ab_aug, sim_ba_aug = [], [], [], []
    
# #     for chunk in results:
# #         embedding_retrieved_ground_truth = [chunk['embedding_retrieved_ground_truth'][f'{key}'] for key in chunk['embedding_retrieved_ground_truth'].keys()]
# #         embedding_retrieved_aug_auth = [chunk['embedding_retrieved_aug_auth'][f'{key}'] for key in chunk['embedding_retrieved_aug_auth'].keys()]
# #         embedding_raw_query = chunk['embedding_query_rot_unauth']
# #         embedding_query_aug_auth = chunk['embedding_query_aug_auth']

# #         sim_aa_aug.append([cosine_similarity(embedding_raw_query, embedding_retrieved_gt) for embedding_retrieved_gt in embedding_retrieved_ground_truth])

# #         sim_bb_aug.append([cosine_similarity(embedding_query_aug_auth, embedding_retrieved_aug_auth) for embedding_retrieved_aug_auth in embedding_retrieved_aug_auth])

# #     flat_sim_aa_aug = [score for sim_list in sim_aa_aug for score in sim_list]
# #     flat_sim_bb_aug = [score for sim_list in sim_bb_aug for score in sim_list]

# #     df_aug = pd.DataFrame({
# #     'sim_aa': flat_sim_aa_aug,
# #     'sim_bb': flat_sim_bb_aug,
# #     })

# #     def plot_sim_four_way(df: pd.DataFrame, out_dir: Path, filename: str, title: str):
# #         """
# #         Box-plots for four similarity measurements side by side.
# #         Outputs PDF.
# #         """
# #         plt.rcParams.update({"font.family": "serif"})
# #         data = [
# #             df["sim_aa"].dropna().values,
# #             df["sim_bb"].dropna().values,
# #         ]
# #         labels = [
# #             "(a)\nbaseline",
# #             "(b)\nboth",
# #         ]
# #         # Colors: update if you have certain brand colors
# #         colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"]
# #         fontsize = 22

# #         fig, ax = plt.subplots(figsize=(10, 4))
# #         bp = ax.boxplot(
# #             data,
# #             patch_artist=True,
# #             widths=0.5,
# #             medianprops=dict(color="black", linewidth=2)
# #         )
# #         for patch, color in zip(bp["boxes"], colors):
# #             patch.set_facecolor(color)
# #             patch.set_alpha(0.75)
# #         ax.set_xticklabels(labels, fontsize=fontsize)
# #         ax.set_ylabel("Cosine Similarity", fontsize=fontsize)
# #         ax.tick_params(axis="y", labelsize=fontsize)
# #         ax.grid(axis="y", alpha=0.3)
# #         ax.axhline(
# #             df["sim_aa"].mean(),
# #             color=colors[0],
# #             linestyle="--",
# #             alpha=0.6,
# #             label=f"baseline mean {df['sim_aa'].mean():.3f}"
# #         )
# #         # ax.set_title(title, fontsize=fontsize+2)
# #         ax.legend(fontsize=fontsize, handlelength=4, handletextpad=1.0)
# #         out = Path(out_dir) / filename
# #         fig.tight_layout()
# #         fig.savefig(out, format='pdf', bbox_inches='tight')
# #         plt.close(fig)
# #         print(f"✔ PDF saved at: {out}")
# #         return out

# #     # === 5. Plot Both Boxplots ===

# #     output_folder = Path(os.path.join(os.getcwd(), 'plots', 'boxplots', "HV10"))  # change if needed
# #     os.makedirs(os.path.dirname(output_folder), exist_ok=True)

# #     # Plot for augmentation
# #     plot_sim_four_way(
# #         df_aug,
# #         output_folder,
# #         "box_plot_augmentation_impact.pdf",
# #         title="Augmentation Impact on Similarity"
# #     )



# #     # print(f"key in results[0]: {results[0].keys()}")

# #     # embedding_retrieved_ground_truth = []
# #     # embedding_retrieved_aug_auth = []
# #     # embedding_raw_query = embedding_query_rot_unauth
# #     # embedding_query_aug_auth = []

# #     # for chunk in results:
# #     #     embeddings_query = chunk[]

# #     # sim_aa_aug, sim_bb_aug, sim_ab_aug, sim_ba_aug = [], [], [], []
# #     # for chunk in results:
# #     #     for k in chunk['chunk_similarities']:
# #     #         sim_aa_aug.append(k['sim_plain_query_plain_chunk'])
# #     #         sim_bb_aug.append(k['sim_auth_query_aug_chunk'])
# #     #         sim_ab_aug.append(k['sim_unauth_query_aug_chunk'])
# #     #         sim_ba_aug.append(k['sim_auth_query_plain_chunk'])







# # # with open(path, 'r') as f:
# # #     results = json.load(f)

# # # # === 2. Allocate Lists ===
# # # sim_aa_rot, sim_bb_rot, sim_ab_rot, sim_ba_rot = [], [], [], []
# # # sim_aa_aug, sim_bb_aug, sim_ab_aug, sim_ba_aug = [], [], [], []

# # # for chunk in results:
# # #     for k in chunk['chunk_similarities']:
# # #         sim_aa_rot.append(k['sim_orig_query_orig_chunk'])
# # #         sim_bb_rot.append(k['sim_rot_query_rot_chunk'])
# # #         sim_ab_rot.append(k['sim_orig_query_rot_chunk'])
# # #         sim_ba_rot.append(k['sim_rot_query_orig_chunk'])

# # #         sim_aa_aug.append(k['sim_plain_query_plain_chunk'])
# # #         sim_bb_aug.append(k['sim_auth_query_aug_chunk'])
# # #         sim_ab_aug.append(k['sim_unauth_query_aug_chunk'])
# # #         sim_ba_aug.append(k['sim_auth_query_plain_chunk'])

# # # # === 3. Create DataFrames ===
# # # df_rot = pd.DataFrame({
# # #     'sim_aa': sim_aa_rot,
# # #     'sim_bb': sim_bb_rot,
# # #     'sim_ab': sim_ab_rot,
# # #     'sim_ba': sim_ba_rot,
# # # })

# # # df_aug = pd.DataFrame({
# # #     'sim_aa': sim_aa_aug,
# # #     'sim_bb': sim_bb_aug,
# # #     'sim_ab': sim_ab_aug,
# # #     'sim_ba': sim_ba_aug,
# # # })

# # # # === 4. Box Plot Function ===

# # # def plot_sim_four_way(df: pd.DataFrame, out_dir: Path, filename: str, title: str):
# # #     """
# # #     Box-plots for four similarity measurements side by side.
# # #     Outputs PDF.
# # #     """
# # #     plt.rcParams.update({"font.family": "serif"})
# # #     data = [
# # #         df["sim_aa"].dropna().values,
# # #         df["sim_bb"].dropna().values,
# # #         df["sim_ab"].dropna().values,
# # #         df["sim_ba"].dropna().values,
# # #     ]
# # #     labels = [
# # #         "(a)\nbaseline",
# # #         "(b)\nboth",
# # #         "(c)\nchunk only",
# # #         "(d)\nquery only",
# # #     ]
# # #     # Colors: update if you have certain brand colors
# # #     colors = ["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"]
# # #     fontsize = 22

# # #     fig, ax = plt.subplots(figsize=(10, 4))
# # #     bp = ax.boxplot(
# # #         data,
# # #         patch_artist=True,
# # #         widths=0.5,
# # #         medianprops=dict(color="black", linewidth=2)
# # #     )
# # #     for patch, color in zip(bp["boxes"], colors):
# # #         patch.set_facecolor(color)
# # #         patch.set_alpha(0.75)
# # #     ax.set_xticklabels(labels, fontsize=fontsize)
# # #     ax.set_ylabel("Cosine Similarity", fontsize=fontsize)
# # #     ax.tick_params(axis="y", labelsize=fontsize)
# # #     ax.grid(axis="y", alpha=0.3)
# # #     ax.axhline(
# # #         df["sim_aa"].mean(),
# # #         color=colors[0],
# # #         linestyle="--",
# # #         alpha=0.6,
# # #         label=f"baseline mean {df['sim_aa'].mean():.3f}"
# # #     )
# # #     # ax.set_title(title, fontsize=fontsize+2)
# # #     ax.legend(fontsize=fontsize, handlelength=4, handletextpad=1.0)
# # #     out = Path(out_dir) / filename
# # #     fig.tight_layout()
# # #     fig.savefig(out, format='pdf', bbox_inches='tight')
# # #     plt.close(fig)
# # #     print(f"✔ PDF saved at: {out}")
# # #     return out

# # # # === 5. Plot Both Boxplots ===

# # # output_folder = Path(os.path.join(os.getcwd(), 'results_experiment_extra_dim'))  # change if needed

# # # # Plot for rotation
# # # plot_sim_four_way(
# # #     df_rot,
# # #     output_folder,
# # #     "box_plot_rotation_impact.pdf",
# # #     title="Rotation Impact on Similarity"
# # # )

# # # # Plot for augmentation
# # # plot_sim_four_way(
# # #     df_aug,
# # #     output_folder,
# # #     "box_plot_augmentation_impact.pdf",
# # #     title="Augmentation Impact on Similarity"
# # # )


# import matplotlib.pyplot as plt
# from pathlib import Path
# import pandas as pd
# import json
# import os
# import argparse
# import numpy as np

# def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
#     denom = np.linalg.norm(a) * np.linalg.norm(b)
#     return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

# def parse_args():
#     parser = argparse.ArgumentParser(description="Embedding cluster visualization experiment control.")
#     parser.add_argument("--results-path", type=str, default="results_experiment/raw_results_topk_10.jsonl", help="Path to the JSON file containing the results.")
#     return parser.parse_args()

# def plot_sim_four_way(df: pd.DataFrame, out_dir: Path, filename: str, title: str):
#     """
#     Box-plots for two similarity measurements side by side.
#     Outputs PDF.
#     """
#     plt.rcParams.update({"font.family": "serif"})
#     data = [
#         df["sim_aa"].dropna().values,
#         df["sim_bb"].dropna().values,
#         df["sim_no_token"].dropna().values,
#         df["sim_different_query"].dropna().values,
#     ]
#     labels = [
#         "(a)\nbaseline",
#         "(b)\nboth",
#         "(c)\nno token",
#         "(d)\ndifferent token",
#     ]
#     colors = ["#4472C4", "#ED7D31"]
#     fontsize = 22

#     fig, ax = plt.subplots(figsize=(10, 4))
#     bp = ax.boxplot(
#         data,
#         patch_artist=True,
#         widths=0.5,
#         medianprops=dict(color="black", linewidth=2)
#     )
#     for patch, color in zip(bp["boxes"], colors):
#         patch.set_facecolor(color)
#         patch.set_alpha(0.75)
#     ax.set_xticklabels(labels, fontsize=fontsize)
#     ax.set_ylabel("Cosine Similarity", fontsize=fontsize)
#     ax.tick_params(axis="y", labelsize=fontsize)
#     ax.grid(axis="y", alpha=0.3)
#     ax.axhline(
#         df["sim_aa"].mean(),
#         color=colors[0],
#         linestyle="--",
#         alpha=0.6,
#         label=f"baseline mean {df['sim_aa'].mean():.3f}"
#     )
#     ax.legend(fontsize=fontsize, handlelength=4, handletextpad=1.0)
#     out = Path(out_dir) / filename
#     fig.tight_layout()
#     fig.savefig(out, format='pdf', bbox_inches='tight')
#     plt.close(fig)
#     print(f"✔ PDF saved at: {out}")
#     return out

# if __name__ == "__main__":
#     args = parse_args()
#     path_results = args.results_path

#     with open(path_results, 'r') as f:
#         results = [json.loads(line) for line in f]

#     sim_aa_aug, sim_bb_aug = [], []
#     sim_no_token_aug, sim_different_query_aug = [], []  

#     sim_A1_q12, sim_A2_q12, sim_A12_q12 = [], [], []


#     # Collect lists of floats (flattened, same count for both columns)
#     for chunk in results:
#         gt_list = [np.array(chunk['embedding_retrieved_ground_truth'][k]) for k in chunk['embedding_retrieved_ground_truth']]
#         aug_list = [np.array(chunk['embedding_retrieved_aug_auth'][k]) for k in chunk['embedding_retrieved_aug_auth']]
#         raw_query = np.array(chunk['embedding_query_rot_unauth'])
#         query_aug_auth = np.array(chunk['embedding_query_aug_auth'])


#         extra = np.zeros(20)  
#         extended_gt_list = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra]) for k in chunk['embedding_retrieved_ground_truth']]

#         user_index = chunk['user_index']
#         extra_ = np.zeros(20) 
#         extra_[(user_index + 1) % 20] = 1
#         different_query = np.concatenate([raw_query, extra_])

#         #### plot 2 : encoding multiple attributes

#         baseline = [np.array(chunk['embedding_retrieved_ground_truth'][k]) for k in chunk['embedding_retrieved_ground_truth']]

#         raw_query = np.array(chunk['embedding_query_rot_unauth'])

#         extra_A1 = np.zeros(20)
#         extra_A1[0] = 1
#         query_A1 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A1]) for k in chunk['embedding_retrieved_ground_truth']]

#         extra_A2 = np.zeros(20)
#         extra_A2[1] = 1
#         query_A2 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A2]) for k in chunk['embedding_retrieved_ground_truth']]

#         extra_A12 = np.zeros(20)
#         extra_A12[0] = 1
#         extra_A12[1] = 1
#         query_A12 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A12]) for k in chunk['embedding_retrieved_ground_truth']]

#         extra_q12 = np.zeros(20)
#         extra_q12[0] = 1
#         extra_q12[1] = 1
#         query_q12 = np.concatenate([raw_query, extra_q12])

#         sim_A1_q12.extend([cosine_similarity(query_A1[i], query_q12) for i in range(len(query_A1))])
#         sim_A2_q12.extend([cosine_similarity(query_A2[i], query_q12) for i in range(len(query_A2))])
#         sim_A12_q12.extend([cosine_similarity(query_A12[i], query_q12) for i in range(len(query_A12))])



#         sim_aa_aug.extend([cosine_similarity(raw_query, gt) for gt in gt_list])
#         sim_bb_aug.extend([cosine_similarity(query_aug_auth, aug) for aug in aug_list])
#         sim_no_token_aug.extend([cosine_similarity(query_aug_auth, aug) for aug in extended_gt_list])  
#         sim_different_query_aug.extend([cosine_similarity(different_query, aug) for aug in aug_list])

#     # Make the two columns equally long (truncate to min-length if needed)
#     min_len = min(len(sim_aa_aug), len(sim_bb_aug), len(sim_no_token_aug), len(sim_different_query_aug))
#     df_aug = pd.DataFrame({
#         'sim_aa': sim_aa_aug[:min_len],
#         'sim_bb': sim_bb_aug[:min_len],
#         'sim_no_token': sim_no_token_aug[:min_len],
#         'sim_different_query': sim_different_query_aug[:min_len]
#     })

#     output_folder = Path(os.path.join(os.getcwd(), 'plots', 'boxplots', "HV10"))
#     os.makedirs(output_folder, exist_ok=True)  # create output folder if needed

#     plot_sim_four_way(
#         df_aug,
#         output_folder,
#         "box_plot_augmentation_impact_HV_1.pdf",
#         title="Augmentation Impact on Similarity"
#     )



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional
import argparse
import json
import os

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-10 else 0.0

def plot_boxplots(
    data_arrays: List[np.ndarray],
    labels: List[str],
    colors: Optional[List[str]] = None,
    out_dir: Path = Path("plots/boxplots"),
    filename: str = "box_plot.pdf",
    title: str = "",
    fontsize: int = 22,
    figsize: tuple = (10, 6),
) -> Path:
    """
    Plot box plots for the given data arrays.

    Args:
        data_arrays: List of arrays, each containing the data for one box plot.
        labels: List of labels for each box plot.
        colors: List of colors for each box plot. If None, uses default colors.
        out_dir: Output directory for the plot.
        filename: Filename for the saved plot.
        title: Title for the plot.
        fontsize: Font size for labels and title.
        figsize: Figure size.
    """
    if colors is None:
        colors = ["#4472C4", "#ED7D31", "#70AD47", "#FFC000"]  # Default colors

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=figsize)

    bp = ax.boxplot(
        data_arrays,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel("Cosine Similarity", fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.grid(axis="y", alpha=0.3)

    # Add mean lines
    for i, (data, color, label) in enumerate(zip(data_arrays, colors, labels)):
        mean = np.mean(data)
        ax.axhline(
            mean,
            color=color,
            linestyle="--",
            alpha=0.6,
            label=f"{label} mean: {mean:.3f}",
        )

    if title:
        ax.set_title(title, fontsize=fontsize)

    # ax.legend(fontsize=fontsize, handlelength=4, handletextpad=1.0)
        # Place legend below the plot
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, -0.2),  # (x, y) position relative to the axes
        fontsize=fontsize-15,
        handlelength=3,
        handletextpad=1.0,
        ncol=1,  # Number of columns in the legend
    )

    out = Path(out_dir) / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    # fig.tight_layout()
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"✔ PDF saved at: {out}")
    return out


if __name__ == "__main__":
    # args = parse_args()
    # path_results = args.results_path
    path = "/home/tthuillier/project/experiment/heavy_exp/RAG_access_control_ex/results_experiment/raw_results_topk_10.jsonl"

    with open(path, 'r') as f:
        results = [json.loads(line) for line in f]

    sim_aa_aug, sim_bb_aug = [], []
    sim_no_token_aug, sim_different_query_aug = [], []  

    sim_A1_q12, sim_A2_q12, sim_A12_q12, sim_A2_q23 = [], [], [], []

    sim_A1_qprim, sim_A2_qprim, sim_A12_qprim = [], [], []

    chunk_encoding_value = 1
    query_encoding_value = 1


    # Collect lists of floats (flattened, same count for both columns)
    for chunk in results:
        gt_list = [np.array(chunk['embedding_retrieved_ground_truth'][k]) for k in chunk['embedding_retrieved_ground_truth']]
        aug_list = [np.array(chunk['embedding_retrieved_aug_auth'][k]) for k in chunk['embedding_retrieved_aug_auth']]
        raw_query = np.array(chunk['embedding_query_rot_unauth'])
        query_aug_auth = np.array(chunk['embedding_query_aug_auth'])


        extra = np.zeros(20)  
        extended_gt_list = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra]) for k in chunk['embedding_retrieved_ground_truth']]

        user_index = chunk['user_index']
        extra_ = np.zeros(20) 
        extra_[(user_index + 1) % 20] = 1
        different_query = np.concatenate([raw_query, extra_])

        sim_aa_aug.extend([cosine_similarity(raw_query, gt) for gt in gt_list])
        sim_bb_aug.extend([cosine_similarity(query_aug_auth, aug) for aug in aug_list])
        sim_no_token_aug.extend([cosine_similarity(query_aug_auth, aug) for aug in extended_gt_list])  
        sim_different_query_aug.extend([cosine_similarity(different_query, aug) for aug in aug_list])


        #### plot 2 : encoding multiple attributes

        baseline = [np.array(chunk['embedding_retrieved_ground_truth'][k]) for k in chunk['embedding_retrieved_ground_truth']]
        raw_query = np.array(chunk['embedding_query_rot_unauth'])

        extra_A1 = np.zeros(20)
        extra_A1[0] = chunk_encoding_value
        chunk_A1 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A1]) for k in chunk['embedding_retrieved_ground_truth']]

        extra_A2 = np.zeros(20)
        extra_A2[1] = chunk_encoding_value
        chunk_A2 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A2]) for k in chunk['embedding_retrieved_ground_truth']]

        extra_A12 = np.zeros(20)
        extra_A12[0] = chunk_encoding_value
        extra_A12[1] = chunk_encoding_value
        chunk_A12 = [np.concatenate([np.array(chunk['embedding_retrieved_ground_truth'][k]), extra_A12]) for k in chunk['embedding_retrieved_ground_truth']]

        extra_q12 = np.zeros(20)
        extra_q12[0] = query_encoding_value
        extra_q12[1] = query_encoding_value
        query_q12 = np.concatenate([raw_query, extra_q12])

        extra_q23 = np.zeros(20)
        extra_q23[1] = query_encoding_value
        extra_q23[2] = query_encoding_value
        query_q23 = np.concatenate([raw_query, extra_q23])

        sim_A1_q12.extend([cosine_similarity(chunk_A1[i], query_q12) for i in range(len(chunk_A1))])
        sim_A2_q12.extend([cosine_similarity(chunk_A2[i], query_q12) for i in range(len(chunk_A2))])
        sim_A12_q12.extend([cosine_similarity(chunk_A12[i], query_q12) for i in range(len(chunk_A12))])
        sim_A2_q23.extend([cosine_similarity(chunk_A2[i], query_q23) for i in range(len(chunk_A2))])
        # Test the security

        extra_qprim = np.zeros(20)
        extra_qprim[3] = query_encoding_value
        query_qprim = np.concatenate([raw_query, extra_qprim])
        sim_A1_qprim.extend([cosine_similarity(chunk_A1[i], query_qprim) for i in range(len(chunk_A1))])
        sim_A2_qprim.extend([cosine_similarity(chunk_A2[i], query_qprim) for i in range(len(chunk_A2))])
        sim_A12_qprim.extend([cosine_similarity(chunk_A12[i], query_qprim) for i in range(len(chunk_A12))])



    # Make the two columns equally long (truncate to min-length if needed)
    min_len = min(len(sim_aa_aug), len(sim_bb_aug), len(sim_no_token_aug), len(sim_different_query_aug))
    df_aug = pd.DataFrame({
        'sim_aa': sim_aa_aug[:min_len],
        'sim_bb': sim_bb_aug[:min_len],
        'sim_no_token': sim_no_token_aug[:min_len],
        'sim_different_query': sim_different_query_aug[:min_len]
    })

    min_len = min(len(sim_aa_aug), len(sim_A1_q12), len(sim_A2_q12), len(sim_A12_q12))
    df_A_q12 = pd.DataFrame({
        'sim_aa_aug': sim_aa_aug[:min_len],
        'sim_A1_q12': sim_A1_q12[:min_len],
        'sim_A2_q12': sim_A2_q12[:min_len],
        'sim_A12_q12': sim_A12_q12[:min_len],
        'sim_A2_q23': sim_A2_q23[:min_len]
    })

    min_len = min(len(sim_aa_aug), len(sim_A1_qprim), len(sim_A2_qprim), len(sim_A12_qprim))
    df_A_qprim = pd.DataFrame({
        'sim_aa_aug': sim_aa_aug[:min_len],
        'sim_A1_qprim': sim_A1_qprim[:min_len],
        'sim_A2_qprim': sim_A2_qprim[:min_len],
        'sim_A12_qprim': sim_A12_qprim[:min_len]
    })   




    output_folder = Path(os.path.join(os.getcwd(), 'plots', 'boxplots'))
    os.makedirs(output_folder, exist_ok=True)  # create output folder if needed

    # Plot augmenbtation impact
    plot_boxplots(
        data_arrays=[
            df_aug["sim_aa"].dropna().values,
            df_aug["sim_bb"].dropna().values,
            df_aug["sim_no_token"].dropna().values,
            df_aug["sim_different_query"].dropna().values,
        ],
        labels=[
            "(a)\nbaseline",
            "(b)\nboth",
            "(c)\nno token",
            "(d)\ndifferent token",
        ],
        colors=["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"],
        out_dir=output_folder,
        filename="box_plot_augmentation_impact_HV_1.pdf",
        title="Augmentation Impact on Similarity"
    )


    # Plot attribute encoding impact
    plot_boxplots(
        data_arrays=[
            df_aug["sim_aa"].dropna().values,
            df_A_q12["sim_A1_q12"].dropna().values,
            df_A_q12["sim_A2_q12"].dropna().values,
            df_A_q12["sim_A12_q12"].dropna().values,
            df_A_q12["sim_A2_q23"].dropna().values
        ],
        labels=[
            "(a)\nbaseline",
            "(b)\nA1 vs q12",
            "(c)\nA2 vs q12",
            "(d)\nA12 vs q12",
            "(e)\nA2 vs q23",
        ],
        colors=["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47", "#FFC000"],
        out_dir=output_folder,
        filename=f"box_plot_attribute_encoding_impact_HV_1_query_{query_encoding_value}.pdf",
        title=f"Attribute Encoding Impact on Similarity (with query encoding at {query_encoding_value})"
    )

    # Plot security test
    plot_boxplots(
        data_arrays=[
            df_aug["sim_aa"].dropna().values,
            df_A_qprim["sim_A1_qprim"].dropna().values,
            df_A_qprim["sim_A2_qprim"].dropna().values,
            df_A_qprim["sim_A12_qprim"].dropna().values,
        ],
        labels=[
            "(a)\nbaseline",
            "(b)\nA1 vs qprim",
            "(c)\nA2 vs qprim",
            "(d)\nA12 vs qprim",
        ],
        colors=["#4472C4", "#ED7D31", "#A5A5A5", "#70AD47"],
        out_dir=output_folder,
        filename=f"box_plot_security_test_HV_1_query_{query_encoding_value}.pdf",
        title=f"Security Test: Different Query Impact on Similarity (with query encoding at {query_encoding_value})"
    )


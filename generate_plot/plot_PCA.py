import argparse
import json
import sys
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

project_root = os.path.dirname(os.path.dirname(__file__))
helpers_path = os.path.join(project_root, "helpers")
sys.path.append(helpers_path)
from helper import *


# from helpers.helper import get_all_chunk_ids, get_list_id_targeted_chunk, get_id_untargeted_chunk, get_id_clusters, fetch_embeddings    


NUMBER_OF_COMPONENTS = 2  # Change to 3 for 3D PCA
NUMBER_OF_CLUSTERS_DISPLAYED = 3  # Limit the number of chunks displayed for clarity
NUMBER_OF_UNTARGETED_CHUNKS_DISPLAYED = 40  # Limit the number of untargeted chunks displayed for clarity
SEED = 42  # For reproducibility

DATA_PATH = os.path.join(os.getcwd(), "RAGBench_whole", "merged_id_triplets_with_metadata2.json")

COLLECTION_NAME_ROTATION = "rotated_experiment"
DIRECTORY_ROTATION_DB =  os.path.join(os.getcwd(), "./chroma_rotated_db")

COLLECTION_NAME_AUGMENTED ="augmented_db_norm_1000000000.0" # low value encode access control # "augmented_db_1000.0" large value encode security #"augmented_db"
DIRECTORY_AUGMENTED_DB = os.path.join(os.getcwd(), "./chroma_aug_db")

COLLECTION_NAME_BASELINE = "baseline_db"
DIRECTORY_BASELINE_DB = os.path.join(os.getcwd(), "./chroma_db")

SAVE_DIR = os.path.join(os.getcwd(), "plots")

# ensure directory for saving plots exists
os.makedirs(SAVE_DIR, exist_ok=True)










def parse_args():
    parser = argparse.ArgumentParser(description="Embedding cluster visualization experiment control.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility.")
    parser.add_argument("--num-clusters", type=int, default=NUMBER_OF_CLUSTERS_DISPLAYED, help="Number of clusters to display.")

    parser.add_argument("--collection-name", type=str, default=COLLECTION_NAME_AUGMENTED, help="Name of the ChromaDB collection to use.")
    parser.add_argument("--collection-dir", type=str, default=DIRECTORY_AUGMENTED_DB, help="Directory path where the ChromaDB collection is stored.")
    parser.add_argument("--plot-title", dest='plot_title', action='store_true', help="Include plot title text.")
    parser.add_argument("--no-plot-title", dest='plot_title', action='store_false', help="Omit plot title text.")
    parser.set_defaults(plot_title=True)

    parser.add_argument("--show-legend", dest='show_legend', action='store_true', help="Display a legend on plots.")
    parser.add_argument("--hide-legend", dest='show_legend', action='store_false', help="Do not display legend.")
    parser.set_defaults(show_legend=True)

    parser.add_argument("--show-centers", dest='plot_centers', action='store_true', help="Display centers of each cluster.")
    parser.add_argument("--hide-centers", dest='plot_centers', action='store_false', help="Do not display centers.")
    parser.set_defaults(plot_centers=True)

    parser.add_argument("--show-center-distances", dest="show_center_distances", action="store_true", help="Display distances between centers (cosine similarity).")
    parser.add_argument("--hide-center-distances", dest="show_center_distances", action="store_false", help="Do not display distances between centers.")
    parser.set_defaults(show_center_distances=True)

    parser.add_argument('--methods', type=str, nargs='+', default=['pca'], choices=["pca", "umap", "tsne", "all"],
                        help="Which reduction methods to use ('pca', 'umap', 'tsne', or 'all'). Default: pca")

    return parser.parse_args()

# Update plot_pca to receive show_legend and show_center_distances switches

def plot_pca(
    cluster_embeddings: List[List[np.ndarray]],
    title: str,
    n_components: int = 2,
    n_clusters_displayed: int = NUMBER_OF_CLUSTERS_DISPLAYED,
    method: str = "pca",
    save_path: Optional[str] = None,
    plot_centers: bool = True,
    embed_questions: Optional[List[np.ndarray]] = None,
    show_legend: bool = True,
    show_center_distances: bool = True,
):
    # Flatten points and build labels
    all_points = []
    cluster_labels = []
    for cluster_idx, cluster in enumerate(cluster_embeddings):
        for point in cluster:
            all_points.append(point)
            cluster_labels.append(cluster_idx)
        if plot_centers:
            all_points.append(np.mean(cluster, axis=0))
            cluster_labels.append(n_clusters_displayed + 1)  # Label for cluster center

    all_points = np.vstack(all_points)

    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "umap":
        reducer = umap.UMAP(n_components=n_components, random_state=SEED)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, random_state=SEED)
    else:
        raise ValueError(f"Unknown method: {method}")

    reduced = reducer.fit_transform(all_points)

    # Reduce embed_questions if provided
    if embed_questions is not None:
        embed_questions = np.vstack(embed_questions)
        reduced_questions = reducer.transform(embed_questions)

    n_clusters = len(cluster_embeddings)

    # Color map definition
    colors = ["red", "blue", "green", "yellow", "purple", "magenta", "yellow", "brown", "pink"]
    colors_used = colors[:n_clusters-1] + ["gray"]  # untargeted cluster in gray
    if plot_centers:
        colors_used = colors_used + ["black"]  # cluster centers in black
    custom_map = ListedColormap(colors_used)

    plt.figure(figsize=(10, 8))
    # Plot scatter
    if n_components == 2:
        scatter = plt.scatter(
            reduced[:, 0], reduced[:, 1],
            c=cluster_labels, cmap=custom_map, alpha=0.6
        )
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")

        # Plot embed_questions as white crosses
        if embed_questions is not None:
            plt.scatter(
                reduced_questions[:, 0], reduced_questions[:, 1],
                c="white", marker="x", s=100, linewidths=2, label="Questions"
            )
    else:
        ax = plt.axes(projection="3d")
        scatter = ax.scatter3D(
            reduced[:, 0], reduced[:, 1], reduced[:, 2],
            c=cluster_labels, cmap=custom_map, alpha=0.6
        )
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")

        # Plot embed_questions as white crosses (3D)
        if embed_questions is not None:
            ax.scatter3D(
                reduced_questions[:, 0], reduced_questions[:, 1], reduced_questions[:, 2],
                c="white", marker="x", s=100, linewidths=2, label="Questions"
            )

    plt.title(title)

    # Legend: Use same colors (by index) as the plotted points
    if plot_centers:
        legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"] + ["Cluster Center"]
    else:
        legend_labels = [f"Cluster {i}" for i in range(n_clusters - 1)] + ["Untargeted"]

    handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=custom_map(i), markersize=12, label=legend_labels[i]
        )
        for i in range(n_clusters_displayed + 2)
    ]

    if embed_questions is not None:
        handles.append(plt.Line2D([0], [0], marker='x', color='white', markersize=12, label="Questions"))

    if plot_centers and show_center_distances:
        distance_handle = plt.Line2D(
            [0], [0], color='black', linestyle='--', label='Distance : Cosine Similarity'
        )
        handles.append(distance_handle)
        legend_labels.append('Distance : Cosine Similarity')

    # Plot center distances only if enabled
    if plot_centers and show_center_distances:
        center_label = n_clusters_displayed + 1
        center_indices = [i for i, label in enumerate(cluster_labels) if label == center_label]
        cluster_centers_reduced = reduced[center_indices]
        for x, x_id in zip(cluster_centers_reduced, center_indices):
            for y, y_id in zip(cluster_centers_reduced, center_indices):
                if not np.array_equal(x, y):
                    if n_components == 2:
                        plt.plot([x[0], y[0]], [x[1], y[1]], color="grey", linestyle="--", alpha=0.5)
                        x_embedding = all_points[x_id]
                        y_embedding = all_points[y_id]
                        cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
                        plt.annotate(
                            f"Distance: {cos_sim:.9e}",
                            xy=((x[0] + y[0]) / 2, (x[1] + y[1]) / 2),
                            fontsize=8, color="black", weight="bold"
                        )
                    else:
                        ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], color="grey", linestyle="--", alpha=0.5)
                        x_embedding = all_points[x_id]
                        y_embedding = all_points[y_id]
                        cos_sim = float(cosine_similarity([x_embedding], [y_embedding]))
                        ax.text(
                            (x[0] + y[0]) / 2, (x[1] + y[1]) / 2, (x[2] + y[2]) / 2,
                            f"Distance: {cos_sim:.9e}",
                            fontsize=8, color="black", weight="bold"
                        )

    if show_legend:
        plt.legend(handles=handles, labels=legend_labels, title="Clusters")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

# Replace the usage block

if __name__ == "__main__":
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    list_of_chunk_ids = get_all_chunk_ids(data)
    list_of_targeted_chunk_ids = get_list_id_targeted_chunk(data)
    list_of_untargeted_chunk_ids = get_id_untargeted_chunk(data)
    list_of_cluster_ids, questions = get_id_clusters(data)
    list_of_cluster_embeddings = []
    for e in list_of_cluster_ids:
        embedding = fetch_embeddings(
            collection_path=args.collection_dir,
            collection_name=args.collection_name,
            chunk_ids=[e][0],
            is_rotated=False,
        )
        list_of_cluster_embeddings.append(embedding)

    list_of_targeted_chunk_embeddings = fetch_embeddings(
        collection_path=args.collection_dir,
        collection_name=args.collection_name,
        chunk_ids=list_of_untargeted_chunk_ids,
        is_rotated=False,
    )
    list_of_cluster_embeddings.append(list_of_targeted_chunk_embeddings)

    methods = ["pca", "umap", "tsne"] if "all" in args.methods else args.methods
    for method in methods:
        save_path = os.path.join(SAVE_DIR, f"{method}_clusters.png")
        plot_pca(
            list_of_cluster_embeddings,
            title=f"Clusters representation using {method.upper()}, seed : {SEED}" if args.plot_title else "",
            n_components=2,
            n_clusters_displayed=args.num_clusters,
            method=method,
            save_path=save_path,
            plot_centers=args.plot_centers,
            show_legend=args.show_legend,
            show_center_distances=args.show_center_distances  # new flag
            # Pass embed_questions arg if you wish
        )
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

from opti_experiment_raw_retrieve import RawQueryResults

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrieval_metrics.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Data Classes ---
from dataclasses import dataclass

# @dataclass
# class RawQueryResults:
#     user_index: int
#     query_index: str
#     high_value_encoding_chunk: int
#     high_value_encoding_query: int
#     top_k: int
#     number_user: int
#     distance_used: str
#     list_ground_truth: list[str]
#     list_retrieved_meta_auth: list[str]
#     list_retrieved_meta_unauth: list[str]
#     list_retrieved_rot_auth: list[str]
#     list_retrieved_rot_unauth: list[str]
#     list_retrieved_aug_auth: list[str]
#     list_retrieved_aug_unauth: list[str]
#     timestamp: str

# --- Helper Functions ---
def load_ground_truths(ground_truth_file: str) -> Dict[str, Dict]:
    """Load ground truth data from JSON file."""
    with open(ground_truth_file, 'r') as f:
        return json.load(f)

def load_chunk_ids(chunk_list_file: str) -> Set[str]:
    """Load all chunk IDs from text file."""
    with open(chunk_list_file, 'r') as f:
        return set(line.strip() for line in f)

def process_raw_query_results(
    raw_results_file: str,
    ground_truths: Dict[str, Dict],
    chunk_ids: Set[str],
    methods: List[str],
    batch_size: int = 1000
) -> pd.DataFrame:
    """
    Process RawQueryResults in batches, compute metrics for each query and method.
    """
    results = []
    with open(raw_results_file, 'rb') as f:
        while True:
            try:
                batch = pickle.load(f)
                for rr in batch:
                    try:
                        ground_truth = ground_truths.get(str(rr.query_index))
                        if not ground_truth:
                            logger.warning(f"Missing ground truth for query {rr.query_index}")
                            continue

                        target_chunk = set(ground_truth['targeted_chunk'])
                        untargeted = chunk_ids - set(ground_truth['reached_chunk'])

                        for method in methods:
                            chunk_sim = getattr(rr, f'list_retrieved_{method}', [])
                            if not chunk_sim:
                                continue
                            chunk_sim = set(chunk_sim)

                            VR = chunk_sim & (target_chunk | untargeted)
                            AAR = len(VR) / len(target_chunk) if target_chunk else np.nan
                            AAP = len(VR) / len(chunk_sim) if chunk_sim else np.nan

                            F = chunk_ids - (target_chunk | untargeted)
                            FA = chunk_sim & F

                            results.append({
                                'user_index': rr.user_index,
                                'query_index': rr.query_index,
                                'top_k': rr.top_k,
                                'method': method,
                                'VR': len(VR),
                                'AAR': AAR,
                                'AAP': AAP,
                                'F': len(F),
                                'FA': len(FA),
                            })
                    except Exception as e:
                        logger.error(f"Error processing query {rr.query_index}: {e}")
                        continue
            except EOFError:
                break
            except Exception as e:
                logger.error(f"Error loading batch: {e}")
                break
    return pd.DataFrame(results)

def aggregate_metrics(df: pd.DataFrame, output_file: str):
    """Aggregate metrics by method and top_k."""
    agg = df.groupby(['method', 'top_k']).agg({
        'VR': ['mean', 'std'],
        'AAR': ['mean', 'std'],
        'AAP': ['mean', 'std'],
        'F': ['mean', 'std'],
        'FA': ['mean', 'std'],
    }).reset_index()
    agg.columns = ['method', 'top_k', 'metric', 'mean', 'std']
    agg.to_csv(output_file, index=False)
    logger.info(f"Aggregated metrics saved to {output_file}")

def plot_metrics(df: pd.DataFrame, output_dir: str):
    """Plot metrics vs top_k for all methods."""
    os.makedirs(output_dir, exist_ok=True)
    for metric in ['VR', 'AAR', 'AAP', 'F', 'FA']:
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=df[df['metric'] == metric],
            x='top_k',
            y='mean',
            hue='method',
            style='method',
            markers=True,
            errorbar=('ci', 95)
        )
        plt.title(f'{metric} vs top_k')
        plt.xlabel('top_k')
        plt.ylabel(metric)
        plt.legend(title='Method')
        plt.savefig(f"{output_dir}/plot_{metric}.png")
        plt.close()
    logger.info(f"Plots saved to {output_dir}")

# --- Main Workflow ---
def main(
    top_k_values: List[int],
    methods: List[str],
    raw_results_prefix: str = "results_experiment_extra_dim/raw_results_topk",
    ground_truth_prefix: str = "results_experiment_extra_dim/GT_results/ground_truth_",
    chunk_list_file: str = "RAGBench_whole/list_chunks_id.json",
    output_dir: str = "retrieval_metrics_results"
):
    """
    Main function to process data, compute metrics, and generate plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    chunk_ids = load_chunk_ids(chunk_list_file)
    logger.info(f"Loaded {len(chunk_ids)} chunk IDs.")

    for top_k in top_k_values:
        raw_file = f"{raw_results_prefix}{top_k}.pkl"
        ground_file = f"{ground_truth_prefix}{top_k}.json"

        if not (os.path.exists(raw_file) and os.path.exists(ground_file)):
            logger.warning(f"Files not found for top_k={top_k}: {raw_file}, {ground_file}")
            continue

        logger.info(f"Processing top_k={top_k}")
        ground_truths = load_ground_truths(ground_file)

        # Process in batches
        df = process_raw_query_results(raw_file, ground_truths, chunk_ids, methods)
        logger.info(f"Processed {len(df)} queries for top_k={top_k}")

        # Save per-query metrics
        for method in methods:
            df_method = df[df['method'] == method]
            df_method.to_csv(f"{output_dir}/metrics_query_{top_k}_{method}.csv", index=False)
            logger.info(f"Saved per-query metrics for {method}, top_k={top_k}")

        # Aggregate and plot
        aggregate_metrics(df, f"{output_dir}/metrics_aggregate.csv")
        plot_metrics(df, output_dir)

    logger.info("All tasks completed!")

# --- Example Usage ---
if __name__ == "__main__":
    # --- Configuration ---
    top_k_values = [10]  # Adjust as needed
    methods = ['meta_auth', 'meta_unauth', 'rot_auth', 'rot_unauth', 'aug_auth', 'aug_unauth']

    # --- Run the analysis ---
    main(
        top_k_values=top_k_values,
        methods=methods,
        output_dir="retrieval_metrics_results"
    )
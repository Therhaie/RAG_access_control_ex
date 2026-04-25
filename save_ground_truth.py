import json 
import argparse
from pathlib import Path





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, help="Top-k value")
    args = parser.parse_args()

    with open("RAGBench_whole/merged_id_triplets_with_metadata2.json", "r") as f:
        data = json.load(f)

    # remove the unnecessary fields
    for item in data:
        item.pop("question", None)
        item.pop("response", None)
        item.pop("sentences", None)
    
    # Output path
    output_path = Path(f"results_experiment_extra_dim/GT_results/ground_truth_{args.top_k}.json")
    # Create parent directories if they do not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
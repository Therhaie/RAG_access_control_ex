from datasets import load_dataset
import json
from argparse import ArgumentParser


# def remove_duplicates_by_sentence_content(input_path, output_path):
#     with open(input_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     seen_contents = set()

#     # Assuming root structure is: { "sentences": [ [sentence, ...], ... ] }
#     unique_sentences = []

#     for doc in data:
#         document = doc["sentences"]
#         new_document = []
#         for sentences in document:
#             for sentence in sentences:
#                 content = sentence[-1]  # content is always at index 1
#                 if content not in seen_contents:
#                     seen_contents.add(content)
#                     new_document.append(sentence)
#             unique_sentences.append(new_document)

#     # Write output preserving the original structure
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump({"sentences": unique_sentences}, f, ensure_ascii=False, indent=2)



def remove_duplicates_by_sentence_content(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for doc in data:
        seen_contents = set()
        # doc["sentences"] is a list of lists (documents)
        new_documents = []
        for sentences in doc["sentences"]:
            new_sentences = []
            for sentence in sentences:
                content = sentence[1]  # sentence text is always at index 1
                if content not in seen_contents:
                    seen_contents.add(content)
                    new_sentences.append(sentence)
            if new_sentences:
                new_documents.append(new_sentences)
        doc["sentences"] = new_documents

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main(MAX_LENGTH=100):
    all_data = []

    for config_name in CONFIG_NAMES:
        dataset = load_dataset(DATASET_NAME, config_name, split="train")
        filtered_data = [
            {key: item[key] for key in desired_keys if key in item}
            for item in dataset
        ]
        
        for i, item in enumerate(filtered_data[:MAX_LENGTH]):
            item["id_triplets"] = dataset["id"][i]
            item["sentences"] = dataset["documents_sentences"][i]

            item.pop("id", None)
            item.pop("documents_sentences", None)
        # for data in dataset["id"]:
        #     data.pop("id", None)
        all_data.extend(filtered_data[:MAX_LENGTH])

    # Now all_data contains all your merged/filtered datapoints
    with open("RAGBench_whole/merged_dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"Merged and filtered dataset saved to RAGBench_whole/merged_dataset.json with {len(all_data)} items.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Merge and filter RAGBench datasets")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of items to process from each dataset")
    args = parser.parse_args()
    # # merge the 2 dataset together
    MAX_LENGTH  = args.max_length
    DATASET_NAME = "galileo-ai/ragbench"
    CONFIG_NAMES = ["covidqa", "msmarco"]
    desired_keys = [
        'id',
        'question',
        'response',
        'documents_sentences'
    ]
    main(MAX_LENGTH=MAX_LENGTH)
    remove_duplicates_by_sentence_content(input_path="RAGBench_whole/merged_dataset.json", output_path="RAGBench_whole/merged_dataset.json")


import json

with open('RAGBench_whole/merged_dataset.json', 'r') as f:
    data = json.load(f)

list_chunks_id = []
# print(data[0])

for item in data:
    id_triplets = item['id_triplets']
    for documents in item['sentences']:
        for document in documents:
            document_id = document[0][0]
            phrase_seq = document[0][1]
            id_chunks = f"{id_triplets}|{document_id}|{phrase_seq}"
            list_chunks_id.append(id_chunks)

# store the json
with open('RAGBench_whole/list_chunks_id.json', 'w') as f:
    json.dump(list_chunks_id, f)


   
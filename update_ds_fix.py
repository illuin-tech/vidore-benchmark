import sys

from datasets import Dataset, load_dataset

if len(sys.argv) != 2:
    print("usage python convert.py dataset")
    sys.exit()

source_dataset = sys.argv[1]
target_dataset = f'mixedbread-ai/{source_dataset.replace("/", "-")}'

ds = load_dataset(source_dataset)
# convert
name2image = {}
query2id = {}
for obj in ds["test"]:
    name2image[obj["image_filename"]] = obj["image"]
    query2id[obj["query"]] = len(query2id)

image2id = dict(zip(name2image.keys(), range(len(name2image.keys()))))

corpus = []
for name, image in name2image.items():
    corpus.append({"corpus-id": image2id[name], "image": image})
ds_corpus = Dataset.from_list(corpus)

queries = []
for query, idx in query2id.items():
    queries.append({"query-id": idx, "query": query})
ds_queries = Dataset.from_list(queries)

triples = []
for obj in ds["test"]:
    triples.append({"query-id": query2id[obj["query"]], "corpus-id": image2id[obj["image_filename"]], "score": 1.0})
ds_triple = Dataset.from_list(triples)

ds_corpus.push_to_hub(target_dataset, "corpus", private=True)
ds_queries.push_to_hub(target_dataset, "queries", private=True)
ds_triple.push_to_hub(target_dataset, "default", private=True)

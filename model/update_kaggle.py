# update_kaggle.py
import os
import json
from tqdm import tqdm
from pinecone import Pinecone
from paper import Paper
from helpers import pinecone_embedding_count

print("Preparing Kaggle dataset update...")

ARXIV_FILE_PATH = "arxiv-metadata-oai-snapshot.json"
EMBEDDING_FILE_PATH = "lhcb-arxiv-embeddings.json"
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]


if os.path.exists(EMBEDDING_FILE_PATH):
    num_kaggle = sum(1 for _ in open(EMBEDDING_FILE_PATH, "r", encoding="utf-8"))
else:
    num_kaggle = 0
    print("No existing embeddings file. Starting fresh.")
    num_pinecone = pinecone_embedding_count(INDEX_NAME)
num_new = num_pinecone - num_kaggle
print(f"{num_new} new embeddings potentially available to update Kaggle dataset.")

if num_new <= 0:
    print("No new embeddings to add. Exiting...")
    exit(0)

print("Loading LHCb metadata from arXiv snapshot...")
all_lhcb_papers = []
with open(ARXIV_FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        data_dict = json.loads(line)
        p = Paper(data_dict)
        if p.has_valid_id and p.is_lhcb_related:
            all_lhcb_papers.append(data_dict)

# naive approach: take the last `num_new` from this list
papers_to_update = all_lhcb_papers[-num_new:]

print("Fetching vectors from Pinecone for these new LHCb papers...")
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(INDEX_NAME)

chunk_size = 1000
for i in range(0, len(papers_to_update), chunk_size):
    chunk = papers_to_update[i : i + chunk_size]
    ids = [p["id"] for p in chunk]
    pinecone_resp = index.fetch(ids)
    vectors = pinecone_resp["vectors"]
    
    for paper_dict in chunk:
        paper_id = paper_dict["id"]
        if paper_id in vectors:
            paper_dict["embedding"] = vectors[paper_id]["values"]

print("Appending new LHCb papers' embeddings to local JSON file...")
with open(EMBEDDING_FILE_PATH, "a", encoding="utf-8") as out_file:
    for paper_dict in tqdm(papers_to_update):
        if "embedding" in paper_dict:
            out_file.write(json.dumps(paper_dict) + "\n")

print("✅ Kaggle dataset updated with new LHCb embeddings.")
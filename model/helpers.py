# helpers.py
import json
import os
from tqdm import tqdm
from paper import Paper
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

def count_lines(file_path):
    """Count total lines in a file for progress bar."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def load_data(file_path, pdf_dir=None, include_pdf=False, start_year=None):
    """
    Returns a generator over the papers contained in `file_path`.
    Args:
        file_path: Path to the arXiv metadata JSON file
        pdf_dir: Optional directory containing PDF files
        include_pdf: Whether to include PDF content in embeddings
        start_year: If set, only yield papers from that year onward
    """
    total_lines = count_lines(file_path)
    papers = []
    skipped = 0
    
    print("Loading papers from JSON file...")
    with open(file_path, "r", encoding="utf-8") as json_file:
        for line in tqdm(json_file, total=total_lines, desc="Loading papers"):
            try:
                data_dict = json.loads(line)
                paper = Paper(data_dict, pdf_dir=pdf_dir, include_pdf=include_pdf)
                
                if not paper.has_valid_id:
                    skipped += 1
                    continue
                    
                if start_year and paper.year < start_year:
                    skipped += 1
                    continue
                    
                papers.append(paper)
                
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON line: {str(e)}")
                continue
            except Exception as e:
                print(f"Error processing paper: {str(e)}")
                continue

    print(f"\nLoaded {len(papers)} papers, skipped {skipped} papers")
    return papers

def filter_lhcb_papers(papers):
    """
    Filter papers containing 'lhcb' in title or abstract.
    Now with progress bar.
    """
    print("\nFiltering LHCb papers...")
    lhcb_papers = []
    
    for paper in tqdm(papers, desc="Filtering papers"):
        if paper.is_lhcb_related:
            lhcb_papers.append(paper)
    
    print(f"Found {len(lhcb_papers)} LHCb papers out of {len(papers)} total papers")
    return lhcb_papers

def pinecone_embedding_count(index_name):
    """
    Helper function to get the total number of embeddings stored in the Pinecone
    index with the name specified in `index_name`.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    return stats["total_vector_count"]

def get_local_embeddings(texts, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32):
    """
    Embeds a list of texts locally with a Sentence Transformers model.
    Now with progress bar for batches.
    """
    model = SentenceTransformer(model_name)
    all_embeddings = []
    
    num_batches = (len(texts) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=len(texts), desc="Creating embeddings")
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch_texts)
        all_embeddings.extend(batch_embeddings)
        progress_bar.update(len(batch_texts))
    
    progress_bar.close()
    return np.array(all_embeddings)

def embed_and_upsert_lhcb(papers, index_name, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=50):
    """
    Locally embed LHCb-related papers and upsert the embeddings into Pinecone.
    Now with progress bars for batches.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    all_papers = list(papers)
    
    print("\nProcessing papers in batches...")
    for i in tqdm(range(0, len(all_papers), batch_size), desc="Processing batches"):
        chunk = all_papers[i : i + batch_size]
        texts = [p.embedding_text for p in chunk]
        
        # Get local embeddings
        embed_data = get_local_embeddings(texts, model_name=model_name, batch_size=batch_size)
        
        # Format for Pinecone upsert: (id, vector, metadata)
        pinecone_data = []
        for paper_obj, emb in zip(chunk, embed_data):
            pinecone_data.append((paper_obj.id, emb.tolist(), paper_obj.metadata))
            
        # Upsert to Pinecone
        index.upsert(vectors=pinecone_data)
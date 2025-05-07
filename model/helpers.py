# helpers.py
import json
import os
from tqdm import tqdm
from paper import Paper
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import gzip

def count_lines(file_path):
    """Count total lines in a file for progress bar."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def load_data(json_file_path, pdf_dir=None, include_pdf=False, start_year=None):
    """
    Load arXiv papers from JSON file and filter them.
    
    Args:
        json_file_path: Path to JSON file containing arXiv papers
        pdf_dir: Optional directory path containing PDF files
        include_pdf: Whether to include PDF content in embeddings
        start_year: Only include papers published after this year
        
    Yields:
        Paper objects
    """
    # Check if JSON file exists
    if not Path(json_file_path).exists():
        raise FileNotFoundError(f"JSON file not found at {json_file_path}")
    
    # First pass: filter for LHCb papers only (faster than loading all papers)
    print("First pass: filtering for LHCb papers only...")
    lhcb_ids = set()
    with gzip.open(json_file_path, 'rt', encoding='utf-8') if json_file_path.endswith('.gz') else open(json_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Scanning papers", unit="papers")):
            try:
                data_dict = json.loads(line)
                
                # Check if paper is related to LHCb
                title = data_dict.get("title", "").lower()
                abstract = data_dict.get("abstract", "").lower()
                paper_id = data_dict.get("id", "")
                
                # Quick filtering for LHCb papers
                if "lhcb" in title or "lhcb" in abstract:
                    # Apply year filter if specified
                    if start_year and 'update_date' in data_dict:
                        year = int(data_dict['update_date'].split('-')[0])
                        if year < int(start_year):
                            continue
                    
                    lhcb_ids.add(paper_id)
            except json.JSONDecodeError:
                continue
            except KeyError:
                continue
            except Exception as e:
                print(f"Error processing line {i}: {str(e)}")
                continue
    
    print(f"Found {len(lhcb_ids)} LHCb-related papers")
    
    # Second pass: load only the filtered LHCb papers
    print("Second pass: loading the filtered LHCb papers...")
    with gzip.open(json_file_path, 'rt', encoding='utf-8') if json_file_path.endswith('.gz') else open(json_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc="Loading papers", unit="papers")):
            try:
                data_dict = json.loads(line)
                paper_id = data_dict.get("id", "")
                
                # Only process papers in our filtered set
                if paper_id in lhcb_ids:
                    # Apply year filter (redundant check, but kept for safety)
                    if start_year and 'update_date' in data_dict:
                        year = int(data_dict['update_date'].split('-')[0])
                        if year < int(start_year):
                            continue
                    
                    paper = Paper(data_dict, pdf_dir=pdf_dir, include_pdf=include_pdf)
                    yield paper
            except json.JSONDecodeError:
                continue
            except KeyError:
                continue
            except Exception as e:
                print(f"Error processing line {i}: {str(e)}")
                continue

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
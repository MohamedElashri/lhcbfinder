import argparse
import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from dataset import ArxivDownloader, AdaptiveRateLimiter
from helpers import load_data, filter_lhcb_papers, pinecone_embedding_count
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from kaggle.api.kaggle_api_extended import KaggleApi

print("Starting script execution")

def download_new_pdfs(papers, output_dir, force_download: bool = False):
    print("Entering download_new_pdfs function")
    rate_limiter = AdaptiveRateLimiter(initial_delay=5, max_delay=300)
    downloader = ArxivDownloader(rate_limiter=rate_limiter)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    if force_download:
        print("Force download flag is set. Will download all PDFs.")
        papers_to_download = [{'id': paper.id} for paper in papers]
    else:
        # Get list of existing PDFs in the output directory
        existing_pdfs = {f.stem for f in Path(output_dir).glob('*.pdf')}
        print(f"Found {len(existing_pdfs)} existing PDFs in {output_dir}")
        
        # Filter papers that don't have PDFs yet
        papers_to_download = []
        for paper in papers:
            safe_paper_id = paper.id.replace('/', '_')
            if safe_paper_id not in existing_pdfs:
                papers_to_download.append({'id': paper.id})
    
    if not papers_to_download and not force_download:
        print("No new papers to download - all PDFs exist locally.")
        return [], []
    
    print(f"Found {len(papers_to_download)} papers to download.")
    print("Starting download of papers...")
    
    return downloader.process_batch(papers_to_download, output_dir, batch_size=10)

def verify_pdf_downloads(papers, pdf_dir):
    """Verify downloaded PDFs exist and have content."""
    print("\nVerifying PDF downloads...")
    missing_pdfs = []
    empty_pdfs = []
    
    for paper in papers:
        safe_paper_id = paper.id.replace('/', '_')
        pdf_path = Path(pdf_dir) / f"{safe_paper_id}.pdf"
        
        if not pdf_path.exists():
            missing_pdfs.append(paper)
        elif pdf_path.stat().st_size == 0:
            empty_pdfs.append(paper)
            pdf_path.unlink()  # Remove empty PDF
    
    if missing_pdfs or empty_pdfs:
        print(f"Found {len(missing_pdfs)} missing and {len(empty_pdfs)} empty PDFs")
        print("Attempting to download these papers again...")
        papers_to_retry = missing_pdfs + empty_pdfs
        new_successful, new_failed = download_new_pdfs(papers_to_retry, pdf_dir)
        print(f"Successfully downloaded {len(new_successful)} PDFs")
        if new_failed:
            print(f"Failed to download {len(new_failed)} PDFs: {new_failed}")
    else:
        print("All PDFs verified successfully")

def create_embeddings(
    papers: List,
    batch_size: int = 32
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Create embeddings for papers using BAAI/bge-large-en-v1.5 model.
    
    Args:
        papers: List of paper objects to create embeddings for
        batch_size: Number of papers to process in each batch (default: 32)
        
    Returns:
        List of tuples containing (paper_id, embedding_vector, metadata)
    """
    print("Entering create_embeddings function")
    
    model_name = "BAAI/bge-large-en-v1.5"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"Successfully loaded {model_name}")
    
    embedding_data = []
    
    # Calculate and display PDF content statistics
    papers_with_pdf = sum(1 for p in papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
    print(f"\nPDF Content Statistics:")
    print(f"Total papers: {len(papers)}")
    print(f"Papers with PDF content: {papers_with_pdf}")
    
    # Display sample content for verification
    if papers:
        sample_paper = papers[0]
        print("\nSample paper content structure:")
        print(f"Paper ID: {sample_paper.id}")
        print(f"Has PDF content: {bool(getattr(sample_paper, '_cleaned_pdf_content', None))}")
        print(f"Embedding text length: {len(sample_paper.embedding_text)}")
        print(f"Content preview (first 200 chars):")
        print(sample_paper.embedding_text[:200] + "...")
    
    # Process papers in batches
    for i in range(0, len(papers), batch_size):
        batch = papers[i:i + batch_size]
        texts = [p.embedding_text for p in batch]
        
        # Display statistics for first batch
        if i == 0:
            print(f"\nFirst batch statistics:")
            print(f"Batch size: {len(texts)}")
            print(f"Average text length: {sum(len(t) for t in texts) / len(texts):.0f} chars")
        
        try:
            embeddings = model.encode(texts, show_progress_bar=True)
            
            # Store embeddings with metadata
            for paper, embedding in zip(batch, embeddings):
                embedding_data.append((
                    paper.id,
                    embedding.tolist(),
                    paper.metadata
                ))
                
            print(f"Processed {min(i + batch_size, len(papers))}/{len(papers)} papers")
            
        except Exception as e:
            print(f"Error processing batch {i//batch_size}: {str(e)}")
            print("Skipping problematic batch and continuing...")
            continue
    
    print(f"\nCreated {len(embedding_data)} embeddings")
    return embedding_data

def get_existing_embeddings(kaggle_file: str) -> set:
    """Get set of paper IDs that already have embeddings stored locally."""
    existing_ids = set()
    if os.path.exists(kaggle_file):
        with open(kaggle_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    paper_dict = json.loads(line)
                    existing_ids.add(paper_dict['id'])
                except json.JSONDecodeError:
                    continue
    return existing_ids

def get_papers_needing_embeddings(papers: List, index_name: str, kaggle_file: str, force_embeddings: bool = False) -> Tuple[List, bool]:
    print("\nChecking existing embeddings...")
    
    pinecone_count = pinecone_embedding_count(index_name)
    print(f"Found {pinecone_count} embeddings in Pinecone index")
    
    local_ids = get_existing_embeddings(kaggle_file)
    print(f"Found {len(local_ids)} embeddings in local file")
    
    # Detect if this is a new/empty index
    is_new_index = pinecone_count == 0 and len(local_ids) > 0
    
    if force_embeddings:
        print("Force embeddings flag is set. Will reprocess all papers.")
        papers_to_process = papers
    elif is_new_index:
        print("Detected new/empty Pinecone index. Will sync all existing embeddings.")
        papers_to_process = papers
    else:
        papers_to_process = [paper for paper in papers if paper.id not in local_ids]
    
    print(f"Found {len(papers_to_process)} papers to process")
    return papers_to_process, is_new_index

def store_embeddings(embedding_data: List[Tuple[str, List[float], Dict]], index_name: str, kaggle_file: str, batch_size: int = 50):
    """Store embeddings both in Pinecone and locally for Kaggle."""
    if not embedding_data:
        print("No new embeddings to store.")
        return
        
    print("Entering store_embeddings function")
    total_vectors = len(embedding_data)
    
    # First: Store locally for Kaggle
    print("Saving embeddings locally...")
    try:
        with open(kaggle_file, 'a', encoding='utf-8') as f:
            for id_, embedding, metadata in embedding_data:
                paper_dict = {
                    "id": id_,
                    "embedding": embedding,
                    **metadata
                }
                f.write(json.dumps(paper_dict) + '\n')
        print(f"Successfully saved {total_vectors} embeddings locally")
    except Exception as e:
        print(f"Error saving embeddings locally: {str(e)}")
        raise
    
    # Second: Upload to Pinecone in batches
    print("Uploading embeddings to Pinecone...")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name)
    
    vectors = [(id_, emb, meta) for id_, emb, meta in embedding_data]
    
    successful_uploads = 0
    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i + batch_size]
        try:
            index.upsert(vectors=batch)
            successful_uploads += len(batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(total_vectors + batch_size - 1)//batch_size} "
                  f"({successful_uploads}/{total_vectors} vectors)")
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")
            continue
    
    if successful_uploads < total_vectors:
        print(f"Warning: Only uploaded {successful_uploads}/{total_vectors} vectors to Pinecone")
    else:
        print(f"Successfully uploaded all {total_vectors} vectors to Pinecone")

def main():
    print("Starting script execution")
    required_env_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing_vars = [var for var in required_env_vars if var not in os.environ]
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("Please ensure these are set in your .env file")
        return
    
    print("Environment variables verified...")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-confirmation", action="store_true")
    parser.add_argument("--start-year", type=int, default=None)
    parser.add_argument("--pdf-dir", type=str, default="lhcb_pdfs")
    parser.add_argument("--download-pdfs", action="store_true",
                       help="Download PDFs for papers")
    parser.add_argument("--include-pdf", action="store_true",
                       help="Include PDF content in embeddings")
    parser.add_argument("--force-arxiv-download", action="store_true",
                       help="Force download of new arXiv metadata JSON even if it exists")
    parser.add_argument("--force-embeddings", action="store_true",
                       help="Force reprocessing of all papers even if they have embeddings")
    parser.add_argument("--force-pdf-download", action="store_true",
                       help="Force download of PDFs even if they exist locally")
    args = parser.parse_args()

    print(f"Arguments parsed: {vars(args)}")

    JSON_FILE_PATH = "arxiv-metadata-oai-snapshot.json"
    INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
    KAGGLE_FILE = "lhcb-arxiv-embeddings.json"
    PDF_DIR = args.pdf_dir

    print(f"Checking if JSON file exists at: {JSON_FILE_PATH}")
    if not os.path.exists(JSON_FILE_PATH) or args.force_arxiv_download:
        action = "JSON file not found" if not os.path.exists(JSON_FILE_PATH) else "Forced download requested"
        print(f"{action}. Setting up Kaggle and downloading dataset...")
        try:
            # Run setup_kaggle.py
            subprocess.run(["python", "setup_kaggle.py"], check=True)
            
            # Download dataset using Kaggle API
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            print("Downloading arXiv dataset from Kaggle...")
            print("This is a large file (~2.5GB) and may take several minutes...")
            
            start_time = time.time()
            
            # Start download
            api.dataset_download_file(
                dataset='Cornell-University/arxiv',
                file_name='arxiv-metadata-oai-snapshot.json',
                path='.'
            )
            
            print("\nDownload completed!")
            final_size_mb = os.path.getsize(JSON_FILE_PATH) / (1024 * 1024)
            total_time = time.time() - start_time
            print(f"Downloaded {final_size_mb:.1f}MB in {total_time:.1f} seconds")
            
        except Exception as e:
            print(f"Error setting up Kaggle or downloading dataset: {str(e)}")
            print("Please ensure you have set KAGGLE_USERNAME and KAGGLE_API_KEY in your environment")
            return
            
        if not os.path.exists(JSON_FILE_PATH):
            print(f"Error: Failed to download the dataset")
            return

    # Load and filter data
    print("\nStep 1: Loading and filtering papers...")
    try:
        all_papers = list(load_data(
            JSON_FILE_PATH, 
            pdf_dir=PDF_DIR if args.include_pdf else None,
            start_year=args.start_year
        ))
        print(f"Loaded {len(all_papers)} total papers")
        
        lhcb_papers = list(filter_lhcb_papers(all_papers))
        print(f"Found {len(lhcb_papers)} LHCb papers total")
        
        if len(lhcb_papers) == 0:
            print("Error: No LHCb papers found in the dataset")
            return
            
        if not os.path.exists(KAGGLE_FILE):
            print(f"No existing embedding file found at {KAGGLE_FILE}")
            print("All papers should need embeddings")
            papers_to_process = lhcb_papers
            is_new_index = True
        else:
            papers_to_process, is_new_index = get_papers_needing_embeddings(
                lhcb_papers, 
                INDEX_NAME, 
                KAGGLE_FILE,
                force_embeddings=args.force_embeddings
            )
        
        print(f"\nCurrent status:")
        print(f"- Total LHCb papers found: {len(lhcb_papers)}")
        print(f"- Papers needing embeddings: {len(papers_to_process)}")
        print(f"- Using new/empty Pinecone index: {is_new_index}")
        
        if not papers_to_process and not args.force_embeddings and not is_new_index:
            print("\nNo new papers to process. Use --force-embeddings to override.")
            return
        else:
            print("\nProceeding with paper processing...")
            
    except Exception as e:
        print(f"Error checking papers needing embeddings: {str(e)}")
        raise

    if not args.no_confirmation:
        print(f"\nReady to process {len(papers_to_process)} papers")
        print(f"PDF content will{' not' if not args.include_pdf else ''} be included in embeddings.")
        confirm = input("Type 'yes' if you wish to continue: ")
        assert confirm == "yes"

    # Download and verify PDFs if needed
    if args.download_pdfs:
        print("\nStep 2: Checking for missing PDFs...")
        try:
            new_successful, new_failed = download_new_pdfs(
                papers_to_process, 
                PDF_DIR,
                force_download=args.force_pdf_download
            )
            if new_failed:
                print(f"Failed to download {len(new_failed)} papers")
            print("PDF download check complete")
            
            # Verify all downloads
            verify_pdf_downloads(papers_to_process, PDF_DIR)
        except Exception as e:
            print(f"Error during PDF downloads: {str(e)}")
            raise
    
    # Create embeddings for the papers
    print(f"\nStep 3: Creating embeddings for {len(papers_to_process)} papers...")
    try:
        embedding_data = create_embeddings(papers_to_process)
        print(f"Created {len(embedding_data)} embeddings")
        
        print("\nStep 4: Storing embeddings...")
        store_embeddings(embedding_data, INDEX_NAME, KAGGLE_FILE, batch_size=50)
        print("Embeddings stored successfully")
    except Exception as e:
        print(f"Error during embedding creation/storage: {str(e)}")
        raise
    
    print("\n✅ Pipeline complete successfully")
    
if __name__ == "__main__":
    print("Starting script execution")
    main()
    print("Script finished")
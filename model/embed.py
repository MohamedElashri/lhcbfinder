import argparse
import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from paper import Paper, PDFCleaner
from dataset import ArxivDownloader, AdaptiveRateLimiter, download_arxiv_metadata
from helpers import load_data, filter_lhcb_papers, pinecone_embedding_count
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from kaggle.api.kaggle_api_extended import KaggleApi
import PyPDF2
import gc
import sys


def download_new_pdfs(papers, output_dir, force_download: bool = False):
    print("Entering download_new_pdfs function")
    rate_limiter = AdaptiveRateLimiter(initial_delay=5, max_delay=300)
    downloader = ArxivDownloader(rate_limiter=rate_limiter)
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Track paper objects along with IDs to maintain object references
    paper_map = {paper.id: paper for paper in papers}
    
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
    
    total_to_download = len(papers_to_download)
    print(f"Found {total_to_download} papers to download.")
    
    # If we need to download many PDFs, split into smaller batches for better progress tracking
    # and to avoid memory issues
    if total_to_download > 100:
        print(f"Large number of PDFs to download, using batched approach")
        batch_size = 50
        successful_ids = []
        failed_ids = []
        
        for i in range(0, total_to_download, batch_size):
            batch = papers_to_download[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(total_to_download+batch_size-1)//batch_size}")
            print(f"Downloading {len(batch)} papers ({i+1}-{min(i+batch_size, total_to_download)} of {total_to_download})")
            
            batch_successful, batch_failed = downloader.process_batch(batch, output_dir, batch_size=10)
            successful_ids.extend(batch_successful)
            failed_ids.extend(batch_failed)
            
            # After each successful batch, reload the PDFs for these papers
            for paper_id in batch_successful:
                if paper_id in paper_map:
                    paper = paper_map[paper_id]
                    print(f"Loading PDF content for {paper_id}")
                    # Force reload PDF content 
                    paper._load_and_clean_pdf(output_dir)
                    
            # Free memory after each batch
            gc.collect()
        
        print(f"Download complete. Successfully downloaded {len(successful_ids)} PDFs, failed to download {len(failed_ids)} PDFs.")
        return successful_ids, failed_ids
    else:
        print("Starting download of papers...")
        return downloader.process_batch(papers_to_download, output_dir, batch_size=10)


def verify_pdf_downloads(papers, pdf_dir, retry_download=True):
    """
    Verify downloaded PDFs exist and have content.
    
    Args:
        papers: List of paper objects to verify
        pdf_dir: Directory containing PDFs
        retry_download: Whether to attempt to download missing PDFs
        
    Returns:
        Tuple of (papers_with_pdf, papers_without_pdf)
    """
    print("\nVerifying PDF downloads...")
    missing_pdfs = []
    empty_pdfs = []
    papers_with_pdf = []
    papers_without_pdf = []
    
    # Check for directory existence
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        print(f"PDF directory {pdf_dir} does not exist. Creating it now.")
        pdf_path.mkdir(exist_ok=True)
    
    for paper in papers:
        safe_paper_id = paper.id.replace('/', '_')
        pdf_path = Path(pdf_dir) / f"{safe_paper_id}.pdf"
        
        if not pdf_path.exists():
            print(f"PDF not found for paper {paper.id}")
            missing_pdfs.append(paper)
            papers_without_pdf.append(paper)
        elif pdf_path.stat().st_size < 1000:  # Files <1KB are likely invalid
            print(f"Empty/tiny PDF for paper {paper.id}")
            empty_pdfs.append(paper)
            pdf_path.unlink()  # Remove empty PDF
            papers_without_pdf.append(paper)
        else:
            # Verify PDF can be opened and has extractable content
            try:
                with open(pdf_path, 'rb') as f:
                    try:
                        reader = PyPDF2.PdfReader(f)
                        if len(reader.pages) == 0:
                            print(f"PDF has 0 pages for paper {paper.id}")
                            empty_pdfs.append(paper)
                            papers_without_pdf.append(paper)
                            continue
                            
                        # Try to extract at least some text from first page
                        first_page_text = reader.pages[0].extract_text()
                        if not first_page_text or len(first_page_text) < 50:
                            print(f"PDF appears to be unreadable for paper {paper.id}")
                            empty_pdfs.append(paper)
                            papers_without_pdf.append(paper)
                            continue
                            
                        # PDF seems valid
                        papers_with_pdf.append(paper)
                    except Exception as e:
                        print(f"Error reading PDF for {paper.id}: {str(e)}")
                        empty_pdfs.append(paper)
                        papers_without_pdf.append(paper)
                        # Delete corrupt PDF
                        pdf_path.unlink()
            except Exception as e:
                print(f"Error opening PDF file for {paper.id}: {str(e)}")
                empty_pdfs.append(paper)
                papers_without_pdf.append(paper)
    
    if (missing_pdfs or empty_pdfs) and retry_download:
        print(f"Found {len(missing_pdfs)} missing and {len(empty_pdfs)} empty/corrupt PDFs")
        print("Attempting to download these papers again...")
        
        # Set up a fresh downloader with more aggressive settings for retries
        rate_limiter = AdaptiveRateLimiter(initial_delay=2, max_delay=60)
        downloader = ArxivDownloader(rate_limiter=rate_limiter)
        
        # Create papers dict for retry
        papers_to_retry = missing_pdfs + empty_pdfs
        papers_dict_to_retry = [{'id': p.id} for p in papers_to_retry]
        
        # Use smaller batch size and more retries for recovery attempt
        new_successful, new_failed = downloader.process_batch(
            papers_dict_to_retry, 
            pdf_dir,
            batch_size=5  # Smaller batch for retries
        )
        
        # Move successfully retried papers to the with_pdf list
        if new_successful:
            print(f"Successfully downloaded {len(new_successful)} PDFs on retry")
            
            # Update our lists
            newly_successful_papers = []
            for paper in papers_without_pdf[:]:  # Copy the list for iteration
                if paper.id in new_successful:
                    papers_without_pdf.remove(paper)
                    papers_with_pdf.append(paper)
                    newly_successful_papers.append(paper)
            
            # Reload the PDFs for these papers
            for paper in newly_successful_papers:
                print(f"Reloading PDF content for {paper.id}")
                paper._load_and_clean_pdf(pdf_dir)
        
        if new_failed:
            print(f"Failed to download {len(new_failed)} PDFs on retry")
    
    # Final summary
    print(f"\nPDF verification complete:")
    print(f"- Papers with PDFs: {len(papers_with_pdf)}")
    print(f"- Papers without PDFs: {len(papers_without_pdf)}")
    
    return papers_with_pdf, papers_without_pdf


def create_embeddings(
    papers: List,
    batch_size: int = 32,
    chunk_mode: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Tuple[str, List[float], Dict[str, Any]]]:
    """
    Create embeddings for papers using BAAI/bge-large-en-v1.5 model.
    
    Args:
        papers: List of paper objects to create embeddings for
        batch_size: Number of papers to process in each batch (default: 32)
        chunk_mode: Whether to create separate embeddings for chunks of the paper content
        chunk_size: Maximum number of words per chunk when chunk_mode is True
        chunk_overlap: Number of words to overlap between chunks
        
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
    print(f"Using chunking: {chunk_mode}")
    
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
        
        # Handle chunking mode
        if chunk_mode:
            chunk_texts = []
            chunk_paper_ids = []
            chunk_metadata = []
            
            for paper in batch:
                if paper._cleaned_pdf_content:
                    # Get base metadata for all chunks
                    base_metadata = paper.metadata
                    
                    # Create chunks from the paper content
                    chunks = PDFCleaner.chunk_content(
                        paper._cleaned_pdf_content, 
                        chunk_size=chunk_size, 
                        overlap=chunk_overlap
                    )
                    
                    for idx, chunk in enumerate(chunks):
                        # Prepare text for embedding with metadata but just this chunk
                        # Authors are excluded from embedding text but kept in metadata
                        chunk_text = (
                            f"Title: {paper.title} "
                            f"Year: {paper.year} "
                            f"Abstract: {paper.abstract} "
                            f"Content chunk {idx+1}/{len(chunks)}: {chunk}"
                        )
                        chunk_texts.append(chunk_text)
                        
                        # Create unique ID for each chunk
                        chunk_id = f"{paper.id}_chunk_{idx}"
                        chunk_paper_ids.append(chunk_id)
                        
                        # Add chunk info to metadata
                        chunk_metadata.append({
                            **base_metadata,
                            "chunk_id": idx,
                            "total_chunks": len(chunks),
                            "is_chunk": True,
                            "parent_id": paper.id
                        })
                else:
                    # For papers without PDF content, use standard embedding
                    chunk_texts.append(paper.embedding_text)
                    chunk_paper_ids.append(paper.id)
                    chunk_metadata.append({
                        **paper.metadata,
                        "is_chunk": False
                    })
            
            try:
                # Create embeddings for all chunks
                if chunk_texts:
                    print(f"\nProcessing batch with {len(chunk_texts)} chunks")
                    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)
                    
                    # Store chunk embeddings with metadata
                    for paper_id, embedding, metadata in zip(chunk_paper_ids, chunk_embeddings, chunk_metadata):
                        embedding_data.append((
                            paper_id,
                            embedding.tolist(),
                            metadata
                        ))
            except Exception as e:
                print(f"Error processing chunk batch: {str(e)}")
                print("Skipping problematic chunks and continuing...")
                continue
                
        else:
            # Standard mode (no chunking)
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
    
    for env_var in required_env_vars:
        if env_var not in os.environ and f"{env_var}=" not in os.environ:
            print(f"Error: Environment variable {env_var} is not set.")
            print("Please set it and try again.")
            sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create embeddings for LHCb papers")
    parser.add_argument("--include-pdf", action="store_true", help="Include PDF content in embeddings")
    parser.add_argument("--download-pdfs", action="store_true", help="Download new PDFs")
    parser.add_argument("--pdf-dir", type=str, default="lhcb_pdfs", help="Directory to store PDFs")
    parser.add_argument("--force-arxiv-download", action="store_true", help="Force download of new arXiv metadata")
    parser.add_argument("--force-embeddings", action="store_true", help="Force reprocessing of all papers")
    parser.add_argument("--force-pdf-download", action="store_true", help="Force download of all PDFs")
    parser.add_argument("--start-year", type=int, help="Start year for papers")
    parser.add_argument("--no-confirmation", action="store_true", help="Skip confirmation prompts")
    parser.add_argument("--chunk-mode", action="store_true", help="Chunk PDF content for more precise search")
    parser.add_argument("--chunk-size", type=int, default=500, help="Maximum words per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Words to overlap between chunks")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with limited papers")
    parser.add_argument("--limit", type=int, default=10, help="Limit number of papers to process in test mode")
    
    args = parser.parse_args()
    
    print("Environment variables verified...")
    print("Args parsed:")
    print(f"- include_pdf: {args.include_pdf}")
    print(f"- pdf_dir: {args.pdf_dir}")
    print(f"- chunk_mode: {args.chunk_mode}")
    if args.chunk_mode:
        print(f"- chunk_size: {args.chunk_size}")
        print(f"- chunk_overlap: {args.chunk_overlap}")
    if args.test_mode:
        print(f"- test_mode: {args.test_mode}")
        print(f"- limit: {args.limit}")
    
    # First, check if we have the arXiv JSON file
    JSON_FILE_PATH = "arxiv-metadata-oai-snapshot.json"
    print(f"Checking if JSON file exists at: {JSON_FILE_PATH}")
    
    if not os.path.exists(JSON_FILE_PATH) or args.force_arxiv_download:
        print("Starting arXiv data download using Kaggle API...")
        download_arxiv_metadata()
    
    # Set up directory for PDFs if needed
    if args.include_pdf:
        os.makedirs(args.pdf_dir, exist_ok=True)
    
    print("\nStep 1: Loading and filtering papers...")
    
    # Note for include_pdf
    if args.include_pdf:
        print("Using include_pdf=True for loading papers")
        
    # Load and filter papers
    paper_generator = load_data(
        JSON_FILE_PATH,
        pdf_dir=args.pdf_dir if args.include_pdf else None,
        include_pdf=args.include_pdf,
        start_year=args.start_year
    )
    
    # If in test mode, limit the number of papers
    if args.test_mode:
        print(f"TEST MODE: Limiting to {args.limit} papers")
        all_papers = []
        for i, paper in enumerate(paper_generator):
            all_papers.append(paper)
            if i >= args.limit - 1:  # -1 because i starts at 0
                break
    else:
        all_papers = list(paper_generator)
        
    print(f"Loaded {len(all_papers)} total papers")
    
    if args.include_pdf:
        # Verify PDF downloads
        papers_with_pdf, papers_without_pdf = verify_pdf_downloads(all_papers, args.pdf_dir)
        
        # Show PDF content stats
        pdf_content_count = sum(1 for p in all_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
        pdf_content_ratio = pdf_content_count / len(all_papers) * 100 if all_papers else 0
        
        print(f"\nPDF Content Summary:")
        print(f"- Papers with extractable PDF content: {pdf_content_count}/{len(all_papers)} ({pdf_content_ratio:.1f}%)")
        
        # Add diagnostic for empty PDF content after cleaning
        if pdf_content_count < len(papers_with_pdf):
            cleaned_empty_count = sum(1 for p in all_papers 
                              if hasattr(p, '_cleaned_pdf_content') and 
                                 (not p._cleaned_pdf_content or len(p._cleaned_pdf_content) < 200))
            print(f"- Papers where PDF content cleaning removed too much text: {cleaned_empty_count}")
            
            # Attempt recovery of these papers with minimal cleaning
            recovered = 0
            for paper in all_papers:
                if (hasattr(paper, '_cleaned_pdf_content') and 
                   (not paper._cleaned_pdf_content or len(paper._cleaned_pdf_content) < 200) and
                   hasattr(paper, '_pdf_content') and
                   paper._pdf_content and
                   len(paper._pdf_content) >= 200):
                    # Try minimal cleaning
                    paper._cleaned_pdf_content = PDFCleaner._fallback_cleaning(
                        paper._pdf_content, paper.title, paper.abstract
                    )
                    if paper._cleaned_pdf_content and len(paper._cleaned_pdf_content) >= 200:
                        recovered += 1
                        
            if recovered > 0:
                print(f"- Successfully recovered PDF content for {recovered} papers using minimal cleaning")
                # Update PDF content count after recovery
                pdf_content_count = sum(1 for p in all_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content and len(p._cleaned_pdf_content) >= 200)
                pdf_content_ratio = pdf_content_count / len(all_papers) * 100 if all_papers else 0
                print(f"- Updated papers with usable PDF content: {pdf_content_count}/{len(all_papers)} ({pdf_content_ratio:.1f}%)")
    
    # Download PDFs if requested
    if args.download_pdfs:
        print("\nStep 2: Downloading PDFs...")
        download_new_pdfs(all_papers, args.pdf_dir, force_download=args.force_pdf_download)
    
    lhcb_papers = list(filter_lhcb_papers(all_papers))
    print(f"Found {len(lhcb_papers)} LHCb papers total")
    
    # Recheck PDF content after filtering
    papers_with_pdf = sum(1 for p in lhcb_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
    print(f"LHCb papers with PDF content: {papers_with_pdf}")
    
    if len(lhcb_papers) == 0:
        print("Error: No LHCb papers found in the dataset")
        return
            
    if not os.path.exists("lhcb-arxiv-embeddings.json"):
        print(f"No existing embedding file found at lhcb-arxiv-embeddings.json")
        print("All papers should need embeddings")
        papers_to_process = lhcb_papers
        is_new_index = True
    else:
        papers_to_process, is_new_index = get_papers_needing_embeddings(
            lhcb_papers, 
            os.environ["PINECONE_INDEX_NAME"], 
            "lhcb-arxiv-embeddings.json",
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
            
    if not args.no_confirmation:
        print(f"\nReady to process {len(papers_to_process)} papers")
        print(f"PDF content will{' not' if not args.include_pdf else ''} be included in embeddings.")
        confirm = input("Type 'yes' if you wish to continue: ")
        assert confirm == "yes"

    # Create embeddings for the papers
    print(f"\nStep 3: Creating embeddings for {len(papers_to_process)} papers...")
    try:
        embedding_data = create_embeddings(
            papers_to_process,
            chunk_mode=args.chunk_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        print(f"Created {len(embedding_data)} embeddings")
        
        print("\nStep 4: Storing embeddings...")
        store_embeddings(embedding_data, os.environ["PINECONE_INDEX_NAME"], "lhcb-arxiv-embeddings.json", batch_size=50)
        print("Embeddings stored successfully")
    except Exception as e:
        print(f"Error during embedding creation/storage: {str(e)}")
        raise
    
    print("\n✅ Pipeline complete successfully")
    
if __name__ == "__main__":
    main()
    print("Script finished")
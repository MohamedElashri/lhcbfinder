import argparse
import os
import json
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm
from colorama import Fore, Back, Style, init
from datetime import datetime
from paper import Paper, PDFCleaner

# Initialize colorama
init(autoreset=True)
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
    # Start timing the embedding process
    start_time = time.time()
    
    print(f"{Fore.CYAN}╔════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     EMBEDDING CREATION STARTED        ║")
    print(f"{Fore.CYAN}╚════════════════════════════════════════╝")
    print(f"{Fore.GREEN}  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    model_name = "BAAI/bge-large-en-v1.5"
    print(f"{Fore.YELLOW} Loading model: {Fore.WHITE}{model_name}")
    model_load_start = time.time()
    model = SentenceTransformer(model_name)
    model_load_time = time.time() - model_load_start
    print(f"{Fore.GREEN} Model loaded successfully in {model_load_time:.2f} seconds")
    
    embedding_data = []
    
    # Calculate and display PDF content statistics
    papers_with_pdf = sum(1 for p in papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
    print(f"\n{Fore.CYAN} PDF Content Statistics:")
    print(f"{Fore.WHITE} Total papers: {Fore.YELLOW}{len(papers)}")
    print(f"{Fore.WHITE} Papers with PDF content: {Fore.YELLOW}{papers_with_pdf} ({papers_with_pdf/len(papers)*100:.1f}%)")
    print(f"{Fore.WHITE} Using chunking: {Fore.YELLOW}{chunk_mode}")
    
    if chunk_mode:
        print(f"{Fore.WHITE} Chunk size: {Fore.YELLOW}{chunk_size} words")
        print(f"{Fore.WHITE} Chunk overlap: {Fore.YELLOW}{chunk_overlap} words")
    
    # Display memory usage
    memory_info = ""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        memory_info = f"{Fore.WHITE} Current memory usage: {Fore.YELLOW}{memory_mb:.1f} MB"
    except ImportError:
        pass
    
    if memory_info:
        print(memory_info)
    
    # Display sample content for verification
    if papers:
        sample_paper = papers[0]
        print(f"\n{Fore.CYAN} Sample Paper Content:")
        print(f"{Fore.WHITE} Paper ID: {Fore.YELLOW}{sample_paper.id}")
        print(f"{Fore.WHITE} Has PDF content: {Fore.GREEN if bool(getattr(sample_paper, '_cleaned_pdf_content', None)) else Fore.RED}{bool(getattr(sample_paper, '_cleaned_pdf_content', None))}")
        print(f"{Fore.WHITE} Embedding text length: {Fore.YELLOW}{len(sample_paper.embedding_text)} chars")
        
        if hasattr(sample_paper, '_cleaned_pdf_content') and sample_paper._cleaned_pdf_content:
            pdf_text_ratio = len(sample_paper._cleaned_pdf_content) / len(sample_paper.embedding_text) * 100
            print(f"{Fore.WHITE} PDF content proportion: {Fore.YELLOW}{pdf_text_ratio:.1f}% of embedding text")
        
        print(f"{Fore.WHITE}  Content preview (first 200 chars):")
        print(f"{Fore.CYAN}{sample_paper.embedding_text[:200]}...")
        
        # Debug information about cleaning process if PDF exists
        if hasattr(sample_paper, '_pdf_content') and sample_paper._pdf_content:
            original_len = len(sample_paper._pdf_content)
            cleaned_len = len(sample_paper._cleaned_pdf_content) if hasattr(sample_paper, '_cleaned_pdf_content') else 0
            if original_len > 0:
                cleaning_ratio = cleaned_len / original_len * 100
                print(f"{Fore.WHITE} PDF cleaning preserved: {Fore.YELLOW}{cleaning_ratio:.1f}% of original content")
                print(f"{Fore.WHITE} Original PDF size: {Fore.YELLOW}{original_len} chars")
                print(f"{Fore.WHITE} Cleaned PDF size: {Fore.YELLOW}{cleaned_len} chars")
    
    # Process papers in batches
    print(f"\n{Fore.CYAN} Processing {Fore.YELLOW}{len(papers)}{Fore.CYAN} papers:")
    
    # Setup tqdm progress bar for papers
    progress_bar = tqdm(
        total=len(papers),
        desc=f"{Fore.GREEN}Embedding papers",
        unit="paper",
        bar_format="{l_bar}{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )
    
    # Process all batches with enhanced tracking
    embedding_start_time = time.time()
    total_chunks = 0
    errors = 0
    
    for i in range(0, len(papers), batch_size):
        batch_start_time = time.time()
        batch = papers[i:i + batch_size]
        
        batch_num = i // batch_size + 1
        total_batches = (len(papers) + batch_size - 1) // batch_size
        print(f"\n{Fore.YELLOW} Processing batch {Fore.WHITE}{batch_num}/{total_batches} {Fore.YELLOW}with {Fore.WHITE}{len(batch)}{Fore.YELLOW} papers")
        
        # Handle chunking mode
        if chunk_mode:
            chunk_texts = []
            chunk_paper_ids = []
            chunk_metadata = []
            papers_with_chunks = 0
            
            # First pass to count total chunks for better progress tracking
            batch_chunks_count = 0
            
            # Process each paper in the batch
            for paper in batch:
                paper_start_time = time.time()
                
                if paper._cleaned_pdf_content:
                    # Get base metadata for all chunks
                    base_metadata = paper.metadata
                    
                    try:
                        # Create chunks from the paper content
                        chunks = PDFCleaner.chunk_content(
                            paper._cleaned_pdf_content, 
                            chunk_size=chunk_size, 
                            overlap=chunk_overlap
                        )
                        
                        # Debugging for chunk creation
                        paper_chunk_count = len(chunks)
                        batch_chunks_count += paper_chunk_count
                        if paper_chunk_count > 0:
                            papers_with_chunks += 1
                            print(f"{Fore.WHITE}  - Paper {Fore.CYAN}{paper.id}{Fore.WHITE}: {Fore.YELLOW}{paper_chunk_count}{Fore.WHITE} chunks created")
                        
                        for idx, chunk in enumerate(chunks):
                            # Prepare text for embedding with metadata but just this chunk
                            # Authors are excluded from embedding text but kept in metadata (per user preference)
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
                    except Exception as e:
                        errors += 1
                        print(f"{Fore.RED} Error creating chunks for paper {paper.id}: {str(e)}")
                else:
                    # For papers without PDF content, use standard embedding
                    chunk_texts.append(paper.embedding_text)
                    chunk_paper_ids.append(paper.id)
                    chunk_metadata.append({
                        **paper.metadata,
                        "is_chunk": False
                    })
                
                # Track paper completion time
                paper_time = time.time() - paper_start_time
                if paper_time > 1.0:  # Only show times over 1 second
                    print(f"{Fore.WHITE}  - Paper processing time: {Fore.YELLOW}{paper_time:.2f}s")
            
            # Update the total chunk counter
            total_chunks += batch_chunks_count
            
            try:
                # Create embeddings for all chunks
                if chunk_texts:
                    print(f"{Fore.GREEN} Creating embeddings for {Fore.YELLOW}{len(chunk_texts)}{Fore.GREEN} chunks from {Fore.YELLOW}{papers_with_chunks}{Fore.GREEN} papers with content")
                    
                    # Use our own tqdm progress instead of the default one
                    encode_start = time.time()
                    chunk_embeddings = model.encode(chunk_texts, show_progress_bar=False)
                    encode_time = time.time() - encode_start
                    
                    print(f"{Fore.GREEN} Embeddings created in {Fore.YELLOW}{encode_time:.2f}s {Fore.GREEN}({Fore.YELLOW}{len(chunk_texts)/encode_time:.1f}{Fore.GREEN} chunks/sec)")
                    
                    # Store chunk embeddings with metadata
                    for paper_id, embedding, metadata in zip(chunk_paper_ids, chunk_embeddings, chunk_metadata):
                        embedding_data.append((
                            paper_id,
                            embedding.tolist(),
                            metadata
                        ))
            except Exception as e:
                errors += 1
                print(f"{Fore.RED} Error processing chunk batch: {str(e)}")
        
        # Standard non-chunking mode
        else:
            batch_texts = []
            batch_ids = []
            batch_metadata = []
            
            # Process each paper in batch
            for paper in batch:
                paper_start = time.time()
                try:
                    batch_texts.append(paper.embedding_text)
                    batch_ids.append(paper.id)
                    batch_metadata.append(paper.metadata)
                    
                    paper_time = time.time() - paper_start
                    if paper_time > 0.5:  # Only log slow papers
                        print(f"{Fore.WHITE}  - Paper {Fore.CYAN}{paper.id}{Fore.WHITE}: processed in {Fore.YELLOW}{paper_time:.2f}s")
                except Exception as e:
                    errors += 1
                    print(f"{Fore.RED} Error processing paper {paper.id}: {str(e)}")
                
            try:
                # Create embeddings for the batch
                if batch_texts:
                    print(f"{Fore.GREEN} Creating embeddings for {Fore.YELLOW}{len(batch_texts)}{Fore.GREEN} papers")
                    
                    # Use our own tqdm progress instead of the default one
                    encode_start = time.time()
                    batch_embeddings = model.encode(batch_texts, show_progress_bar=False) 
                    encode_time = time.time() - encode_start
                    
                    print(f"{Fore.GREEN} Embeddings created in {Fore.YELLOW}{encode_time:.2f}s {Fore.GREEN}({Fore.YELLOW}{len(batch_texts)/encode_time:.1f}{Fore.GREEN} papers/sec)")
                    
                    # Store embeddings with their metadata
                    for paper_id, embedding, metadata in zip(batch_ids, batch_embeddings, batch_metadata):
                        embedding_data.append((
                            paper_id,
                            embedding.tolist(),
                            metadata
                        ))
            except Exception as e:
                errors += 1
                print(f"{Fore.RED} Error processing standard batch: {str(e)}")
        
        # Update progress bar and show batch timing
        progress_bar.update(len(batch))
        batch_time = time.time() - batch_start_time
        print(f"{Fore.CYAN}  Batch processing time: {Fore.YELLOW}{batch_time:.2f}s {Fore.CYAN}({Fore.YELLOW}{len(batch)/batch_time:.1f}{Fore.CYAN} papers/sec)")
    
    # Close the progress bar
    progress_bar.close()
    
    # Show final statistics for the whole embedding process
    total_embedding_time = time.time() - embedding_start_time
    papers_per_second = len(papers) / total_embedding_time if total_embedding_time > 0 else 0
    
    print(f"\n{Fore.CYAN} Embedding Process Summary:")
    print(f"{Fore.WHITE}  Total time: {Fore.YELLOW}{total_embedding_time:.2f} seconds")
    print(f"{Fore.WHITE}Processing speed: {Fore.YELLOW}{papers_per_second:.2f} papers/second")
    print(f"{Fore.WHITE} Total papers processed: {Fore.YELLOW}{len(papers)}")
    
    if chunk_mode:
        print(f"{Fore.WHITE} Total chunks created: {Fore.YELLOW}{total_chunks}")
        print(f"{Fore.WHITE} Average chunks per paper: {Fore.YELLOW}{total_chunks/len(papers):.1f}")
    
    if errors > 0:
        print(f"{Fore.RED} Encountered {errors} errors during processing")
        
    embeddings_created = len(embedding_data)
    print(f"{Fore.GREEN} Successfully created {Fore.YELLOW}{embeddings_created}{Fore.GREEN} embeddings")
    
    # Calculate and display memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"{Fore.WHITE} Final memory usage: {Fore.YELLOW}{memory_mb:.1f} MB")
    except ImportError:
        pass
        
    # Track total function execution time
    total_time = time.time() - start_time
    print(f"{Fore.GREEN}  Total embedding creation time: {Fore.YELLOW}{total_time:.2f} seconds")
    
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


def store_embeddings(embedding_data: List[Tuple[str, List[float], Dict]], index_name: str, kaggle_file: str, batch_size: int = 50, test_mode: bool = False):
    """Store embeddings both in Pinecone and locally for Kaggle.
    
    Args:
        embedding_data: List of tuples (id, embedding_vector, metadata)
        index_name: Name of Pinecone index
        kaggle_file: Path to local file for storing embeddings
        batch_size: Batch size for Pinecone uploads
        test_mode: If True, embeddings are not stored to avoid overwriting production data
    """
    if not embedding_data:
        print("No new embeddings to store.")
        return
        
    print("Entering store_embeddings function")
    total_vectors = len(embedding_data)
    
    if test_mode:
        print(f"{Fore.MAGENTA} TEST MODE: Saving embeddings to test file instead of production data")
        test_file = "embeddings_test.json"
        print(f"{Fore.MAGENTA} Saving {total_vectors} embeddings to {test_file}")
        try:
            with open(test_file, 'w', encoding='utf-8') as f:  # Using 'w' to overwrite previous test data
                for id_, embedding, metadata in embedding_data:
                    paper_dict = {
                        "id": id_,
                        "embedding": embedding,
                        **metadata
                    }
                    f.write(json.dumps(paper_dict) + '\n')
            print(f"{Fore.GREEN} Successfully saved {total_vectors} embeddings to test file")
        except Exception as e:
            print(f"{Fore.RED} Error saving embeddings to test file: {str(e)}")
        print(f"{Fore.MAGENTA} Skipping upload to Pinecone in test mode")
        return
    
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
    # Start timing the entire process
    main_start_time = time.time()
    start_datetime = datetime.now()
    
    # Print fancy header
    print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║        LHCb FINDER - EMBEDDING PIPELINE        ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════╝")
    print(f"{Fore.GREEN}  Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check environment variables with better formatting
    print(f"\n{Fore.YELLOW} Checking environment variables...")
    required_env_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    
    for env_var in required_env_vars:
        if env_var not in os.environ and f"{env_var}=" not in os.environ:
            print(f"{Fore.RED}❌ Error: Environment variable {env_var} is not set.")
            print(f"{Fore.RED}Please set it and try again.")
            sys.exit(1)
        else:
            print(f"{Fore.GREEN}✓ {env_var}: {'*' * 8 if 'KEY' in env_var else os.environ.get(env_var)}")
    
    print(f"{Fore.GREEN} All required environment variables verified!")
    
    # Initialize memory tracking if available
    try:
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"{Fore.WHITE} Initial memory usage: {Fore.YELLOW}{initial_memory:.1f} MB")
    except ImportError:
        initial_memory = None
    
    # Parse command line arguments
    print(f"\n{Fore.CYAN} Parsing command line arguments...")
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
    
    # Print arguments with colorful formatting
    print(f"\n{Fore.CYAN} Command Line Arguments:")
    # Common arguments
    print(f"{Fore.WHITE} Include PDF: {Fore.YELLOW}{args.include_pdf}")
    print(f"{Fore.WHITE} PDF directory: {Fore.YELLOW}{args.pdf_dir}")
    print(f"{Fore.WHITE} Download PDFs: {Fore.YELLOW}{args.download_pdfs}")
    print(f"{Fore.WHITE} Force PDF download: {Fore.YELLOW}{args.force_pdf_download}")
    print(f"{Fore.WHITE} Force ArXiv download: {Fore.YELLOW}{args.force_arxiv_download}")
    print(f"{Fore.WHITE} Force embeddings: {Fore.YELLOW}{args.force_embeddings}")
    
    # Chunking options
    print(f"{Fore.WHITE} Chunk mode: {Fore.YELLOW}{args.chunk_mode}")
    if args.chunk_mode:
        print(f"{Fore.WHITE} Chunk size: {Fore.YELLOW}{args.chunk_size} words")
        print(f"{Fore.WHITE} Chunk overlap: {Fore.YELLOW}{args.chunk_overlap} words")
    
    # Test mode options
    if args.test_mode:
        print(f"{Fore.WHITE} Test mode: {Fore.YELLOW}{args.test_mode}")
        print(f"{Fore.WHITE} Paper limit: {Fore.YELLOW}{args.limit}")
    
    # Start year if provided
    if args.start_year:
        print(f"{Fore.WHITE} Start year: {Fore.YELLOW}{args.start_year}")
        
    print(f"{Fore.WHITE} Skip confirmation: {Fore.YELLOW}{args.no_confirmation}")
    
    # Print section header for arXiv download
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 1: DATA PREPARATION     ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")
    
    # First, check if we have the ArXiv JSON file
    JSON_FILE_PATH = "arxiv-metadata-oai-snapshot.json"
    print(f"{Fore.YELLOW} Checking if ArXiv dataset exists at: {Fore.WHITE}{JSON_FILE_PATH}")
    
    if not os.path.exists(JSON_FILE_PATH) or args.force_arxiv_download:
        if args.force_arxiv_download:
            print(f"{Fore.YELLOW} Force download flag set, downloading fresh ArXiv dataset")
        else:
            print(f"{Fore.YELLOW} Dataset not found, downloading ArXiv data")
            
        print(f"{Fore.CYAN} Starting Kaggle download of ArXiv metadata...")
        download_start = time.time()
        
        # Show a spinner or progress indicator since this can take a while
        try:
            from yaspin import yaspin
            from yaspin.spinners import Spinners
            with yaspin(Spinners.moon, text="Downloading ArXiv dataset from Kaggle") as sp:
                download_arxiv_metadata()
                sp.text = "Download complete!"
                sp.ok("")
        except ImportError:
            # Fallback if yaspin is not available
            print(f"{Fore.YELLOW} Downloading ArXiv dataset... (this may take a while)")
            download_arxiv_metadata()
            print(f"{Fore.GREEN} ArXiv dataset download complete!")
            
        download_time = time.time() - download_start
        print(f"{Fore.GREEN} ArXiv dataset download completed in {Fore.YELLOW}{download_time:.1f} seconds {Fore.GREEN}({Fore.YELLOW}{download_time/60:.1f} minutes)")
        
        # Check file size
        if os.path.exists(JSON_FILE_PATH):
            file_size_bytes = os.path.getsize(JSON_FILE_PATH)
            file_size_gb = file_size_bytes / (1024 ** 3)  # Convert to GB
            print(f"{Fore.GREEN} Downloaded file size: {Fore.YELLOW}{file_size_gb:.2f} GB")
    else:
        # File exists and no force download
        file_size_bytes = os.path.getsize(JSON_FILE_PATH)
        file_size_gb = file_size_bytes / (1024 ** 3)  # Convert to GB
        print(f"{Fore.GREEN} ArXiv dataset found! {Fore.WHITE}({Fore.YELLOW}{file_size_gb:.2f} GB{Fore.WHITE})")
    
    # Set up directory for PDFs if needed
    if args.include_pdf:
        print(f"{Fore.YELLOW} Setting up PDF directory: {Fore.WHITE}{args.pdf_dir}")
        os.makedirs(args.pdf_dir, exist_ok=True)
    
    print(f"\n{Fore.CYAN} Loading and filtering papers...")
    load_start_time = time.time()
    
    # Note for include_pdf
    if args.include_pdf:
        print(f"{Fore.YELLOW} Including PDF content in embeddings")
    
    # Create a spinner or progress indicator for paper loading    
    print(f"{Fore.YELLOW} Loading papers from ArXiv dataset...")
    
    # Load and filter papers with timing
    loading_start = time.time()
    paper_generator = load_data(
        JSON_FILE_PATH,
        pdf_dir=args.pdf_dir if args.include_pdf else None,
        include_pdf=args.include_pdf,
        start_year=args.start_year
    )
    
    # If in test mode, limit the number of papers with progress indicator
    if args.test_mode:
        print(f"{Fore.MAGENTA} TEST MODE: Limiting to {args.limit} papers")
        all_papers = []
        with tqdm(total=args.limit, desc=f"{Fore.GREEN}Loading papers", unit="paper") as pbar:
            for i, paper in enumerate(paper_generator):
                all_papers.append(paper)
                pbar.update(1)
                if i >= args.limit - 1:  # -1 because i starts at 0
                    break
    else:
        # For full mode, we can't know the total count in advance, use simple progress indicator
        print(f"{Fore.YELLOW} Loading all papers from dataset (this may take a while)...")
        all_papers = []
        for i, paper in enumerate(paper_generator):
            all_papers.append(paper)
            # Print progress every 50,000 papers
            if (i + 1) % 50000 == 0:
                print(f"{Fore.GREEN} Loaded {i + 1} papers so far...")
    
    # Calculate loading time and speed
    loading_time = time.time() - loading_start
    papers_per_second = len(all_papers) / loading_time if loading_time > 0 else 0
    
    print(f"{Fore.GREEN} Loaded {Fore.YELLOW}{len(all_papers):,}{Fore.GREEN} total papers in {Fore.YELLOW}{loading_time:.1f}s {Fore.GREEN}({Fore.YELLOW}{papers_per_second:.1f}{Fore.GREEN} papers/sec)")
    
    # Print some information about the dataset
    years = {}
    categories = {}
    for paper in all_papers[:1000]:  # Sample first 1000 papers for quick stats
        year = getattr(paper, 'year', 0)
        years[year] = years.get(year, 0) + 1
        
        for category in getattr(paper, 'categories', []):
            categories[category] = categories.get(category, 0) + 1
    
    if years:
        print(f"{Fore.CYAN} Sample Data Statistics (first 1000 papers):")
        print(f"{Fore.WHITE} Years: {Fore.YELLOW}{sorted(years.keys())[:5]}...")
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"{Fore.WHITE} Top categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}")

    
    if args.include_pdf:
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
        print(f"{Fore.CYAN}║     STAGE 2: PDF VERIFICATION     ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")
        
        # Verify PDF downloads with timing
        print(f"{Fore.YELLOW} Verifying PDF downloads...")
        pdf_verify_start = time.time()
        papers_with_pdf, papers_without_pdf = verify_pdf_downloads(all_papers, args.pdf_dir)
        pdf_verify_time = time.time() - pdf_verify_start
        
        print(f"{Fore.GREEN} PDF verification completed in {Fore.YELLOW}{pdf_verify_time:.1f}s")
        
        # Show PDF content stats with enhanced formatting
        pdf_content_count = sum(1 for p in all_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
        pdf_content_ratio = pdf_content_count / len(all_papers) * 100 if all_papers else 0
        
        print(f"\n{Fore.CYAN} PDF Content Summary:")
        pdf_color = Fore.GREEN if pdf_content_ratio > 70 else (Fore.YELLOW if pdf_content_ratio > 30 else Fore.RED)
        print(f"{Fore.WHITE} Papers with extractable PDF content: {pdf_color}{pdf_content_count}/{len(all_papers)} ({pdf_content_ratio:.1f}%)")
        
        # Calculate average cleaning effectiveness 
        pdf_sizes = [(len(getattr(p, '_pdf_content', "")), len(getattr(p, '_cleaned_pdf_content', ""))) 
                   for p in all_papers if hasattr(p, '_pdf_content') and p._pdf_content]
        
        if pdf_sizes:
            avg_original_size = sum(orig for orig, _ in pdf_sizes) / len(pdf_sizes)
            avg_cleaned_size = sum(cleaned for _, cleaned in pdf_sizes) / len(pdf_sizes)
            avg_preservation = (avg_cleaned_size / avg_original_size * 100) if avg_original_size > 0 else 0
            
            preservation_color = Fore.GREEN if avg_preservation > 60 else (Fore.YELLOW if avg_preservation > 30 else Fore.RED)
            print(f"{Fore.WHITE} Average PDF cleaning preservation: {preservation_color}{avg_preservation:.1f}%")
            print(f"{Fore.WHITE} Average original PDF size: {Fore.YELLOW}{avg_original_size:.0f} chars")
            print(f"{Fore.WHITE} Average cleaned PDF size: {Fore.YELLOW}{avg_cleaned_size:.0f} chars")
        
        # Add diagnostic for empty PDF content after cleaning
        if pdf_content_count < len(papers_with_pdf):
            cleaned_empty_count = sum(1 for p in all_papers 
                              if hasattr(p, '_cleaned_pdf_content') and 
                                 (not p._cleaned_pdf_content or len(p._cleaned_pdf_content) < 200))
            
            print(f"{Fore.RED} Papers where PDF cleaning removed too much text: {cleaned_empty_count}")
            
            # Attempt recovery of these papers with improved cleaning approaches
            print(f"{Fore.YELLOW} Attempting recovery of papers with insufficient content...")
            recovery_start = time.time()
            
            with tqdm(total=cleaned_empty_count, desc=f"{Fore.YELLOW}Recovering papers", unit="paper") as pbar:
                recovered = 0
                for paper in all_papers:
                    if (hasattr(paper, '_cleaned_pdf_content') and 
                      (not paper._cleaned_pdf_content or len(paper._cleaned_pdf_content) < 200) and
                      hasattr(paper, '_pdf_content') and
                      paper._pdf_content and
                      len(paper._pdf_content) >= 200):
                        
                        # First try basic cleaning
                        try:
                            paper._cleaned_pdf_content = PDFCleaner._basic_cleaning(
                                paper._pdf_content, paper.title, paper.abstract
                            )
                            
                            # If basic cleaning doesn't give enough content, fall back to minimal cleaning
                            if not paper._cleaned_pdf_content or len(paper._cleaned_pdf_content) < 200:
                                paper._cleaned_pdf_content = PDFCleaner._fallback_cleaning(
                                    paper._pdf_content, paper.title, paper.abstract
                                )
                                
                            if paper._cleaned_pdf_content and len(paper._cleaned_pdf_content) >= 200:
                                recovered += 1
                                
                        except Exception as e:
                            # Silently continue if recovery fails
                            pass
                            
                        pbar.update(1)
            
            recovery_time = time.time() - recovery_start
            
            if recovered > 0:
                print(f"{Fore.GREEN} Successfully recovered PDF content for {Fore.YELLOW}{recovered}{Fore.GREEN} papers in {Fore.YELLOW}{recovery_time:.1f}s")
                
                # Update PDF content count after recovery
                pdf_content_count = sum(1 for p in all_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content and len(p._cleaned_pdf_content) >= 200)
                pdf_content_ratio = pdf_content_count / len(all_papers) * 100 if all_papers else 0
                pdf_color = Fore.GREEN if pdf_content_ratio > 70 else (Fore.YELLOW if pdf_content_ratio > 30 else Fore.RED)
                print(f"{Fore.WHITE} Updated papers with usable PDF content: {pdf_color}{pdf_content_count}/{len(all_papers)} ({pdf_content_ratio:.1f}%)")
            else:
                print(f"{Fore.RED} No papers could be recovered with improved cleaning methods")
        
        # Add statistics on which cleaning method was most effective
        basic_count = 0
        full_count = 0
        fallback_count = 0
        
        for paper in all_papers:
            if not hasattr(paper, '_pdf_content') or not paper._pdf_content:
                continue
                
            orig_size = len(paper._pdf_content)
            cleaned_size = len(getattr(paper, '_cleaned_pdf_content', ""))
            
            if cleaned_size < 200:
                continue
                
            ratio = cleaned_size / orig_size if orig_size > 0 else 0
            
            # Estimate which cleaning method was likely used based on preservation ratio
            if ratio > 0.7:
                fallback_count += 1  # Very high preservation suggests fallback cleaning
            elif ratio > 0.4:
                basic_count += 1     # Medium preservation suggests basic cleaning
            else:
                full_count += 1      # Low preservation suggests full cleaning
        
        print(f"\n{Fore.CYAN} Cleaning Method Statistics:")
        print(f"{Fore.WHITE} Estimated usage of cleaning methods:")
        print(f"{Fore.WHITE} - Basic cleaning: {Fore.YELLOW}{basic_count} papers")
        print(f"{Fore.WHITE} - Full cleaning: {Fore.YELLOW}{full_count} papers")
        print(f"{Fore.WHITE} - Fallback cleaning: {Fore.YELLOW}{fallback_count} papers")
    
    # Download PDFs if requested
    if args.download_pdfs:
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
        print(f"{Fore.CYAN}║     STAGE 3: PDF DOWNLOAD      ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")
        download_start = time.time()
        successful, failed = download_new_pdfs(all_papers, args.pdf_dir, force_download=args.force_pdf_download)
        download_time = time.time() - download_start
        print(f"{Fore.GREEN} PDF download completed in {Fore.YELLOW}{download_time:.1f}s {Fore.GREEN}({Fore.YELLOW}{download_time/60:.1f} minutes)")
        print(f"{Fore.WHITE} Successfully downloaded: {Fore.GREEN}{len(successful)}{Fore.WHITE} PDFs")
        print(f"{Fore.WHITE} Failed downloads: {Fore.RED}{len(failed)}{Fore.WHITE} PDFs")
    
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 4: LHCb FILTERING     ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")
    
    print(f"{Fore.YELLOW} Filtering for LHCb papers...")
    filtering_start = time.time()
    lhcb_papers = list(filter_lhcb_papers(all_papers))
    filtering_time = time.time() - filtering_start
    
    # Stats for LHCb papers
    lhcb_ratio = len(lhcb_papers) / len(all_papers) * 100 if all_papers else 0
    papers_color = Fore.GREEN if len(lhcb_papers) > 50 else (Fore.YELLOW if len(lhcb_papers) > 10 else Fore.RED)
    
    print(f"{Fore.GREEN} Found {papers_color}{len(lhcb_papers)}{Fore.GREEN} LHCb papers in {Fore.YELLOW}{filtering_time:.2f}s {Fore.WHITE}({Fore.YELLOW}{lhcb_ratio:.2f}%{Fore.WHITE} of total papers)")
    
    # Recheck PDF content after filtering
    papers_with_pdf = sum(1 for p in lhcb_papers if hasattr(p, '_cleaned_pdf_content') and p._cleaned_pdf_content)
    pdf_ratio = papers_with_pdf / len(lhcb_papers) * 100 if lhcb_papers else 0
    pdf_color = Fore.GREEN if pdf_ratio > 70 else (Fore.YELLOW if pdf_ratio > 30 else Fore.RED)
    
    print(f"{Fore.WHITE} LHCb papers with PDF content: {pdf_color}{papers_with_pdf}/{len(lhcb_papers)} ({pdf_ratio:.1f}%)")
    
    if len(lhcb_papers) == 0:
        print(f"{Fore.RED} Error: No LHCb papers found in the dataset")
        
        # Show completion time before exiting
        total_time = time.time() - main_start_time
        print(f"\n{Fore.RED} Pipeline terminated due to no LHCb papers found")
        print(f"{Fore.WHITE}⏱ Total execution time: {Fore.YELLOW}{total_time:.1f}s {Fore.WHITE}({Fore.YELLOW}{total_time/60:.1f} minutes)")
        return
    
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 5: EMBEDDING PREPARATION     ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")
            
    if not os.path.exists("lhcb-arxiv-embeddings.json"):
        print(f"{Fore.YELLOW} No existing embedding file found at lhcb-arxiv-embeddings.json")
        print(f"{Fore.YELLOW} All papers will need embeddings")
        papers_to_process = lhcb_papers
        is_new_index = True
    else:
        # Get file stats for embeddings file
        embedding_file_size = os.path.getsize("lhcb-arxiv-embeddings.json") / (1024 * 1024)  # MB
        print(f"{Fore.GREEN} Found existing embedding file ({Fore.YELLOW}{embedding_file_size:.2f} MB{Fore.GREEN})")
        
        print(f"{Fore.YELLOW} Checking for papers needing embeddings...")
        check_start = time.time()
        papers_to_process, is_new_index = get_papers_needing_embeddings(
            lhcb_papers, 
            os.environ["PINECONE_INDEX_NAME"], 
            "lhcb-arxiv-embeddings.json",
            force_embeddings=args.force_embeddings
        )
        check_time = time.time() - check_start
        print(f"{Fore.GREEN} Embedding check completed in {Fore.YELLOW}{check_time:.2f}s")
        
    papers_to_process_ratio = len(papers_to_process) / len(lhcb_papers) * 100 if lhcb_papers else 0
    need_embeddings_color = Fore.GREEN if papers_to_process_ratio < 30 else (Fore.YELLOW if papers_to_process_ratio < 70 else Fore.RED)
    
    print(f"\n{Fore.CYAN} Current Status:")
    print(f"{Fore.WHITE} Total LHCb papers found: {papers_color}{len(lhcb_papers)}")
    print(f"{Fore.WHITE} Papers needing embeddings: {need_embeddings_color}{len(papers_to_process)} {Fore.WHITE}({need_embeddings_color}{papers_to_process_ratio:.1f}%{Fore.WHITE})")
    print(f"{Fore.WHITE} Using new/empty Pinecone index: {Fore.YELLOW}{is_new_index}")
        
    if not papers_to_process and not args.force_embeddings and not is_new_index:
        print(f"\n{Fore.GREEN} No new papers to process.")
        print(f"{Fore.WHITE} Hint: Use --force-embeddings to override and process all papers again.")
        
        # Show completion time before exiting
        total_time = time.time() - main_start_time
        print(f"\n{Fore.GREEN} Pipeline completed successfully (no new papers to process)")
        print(f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{total_time:.1f}s {Fore.WHITE}({Fore.YELLOW}{total_time/60:.1f} minutes)")
        return
    else:
        print(f"\n{Fore.GREEN} Found {need_embeddings_color}{len(papers_to_process)}{Fore.GREEN} papers to process")
            
    if not args.no_confirmation:
        print(f"\n{Fore.YELLOW}❗ Ready to process {Fore.WHITE}{len(papers_to_process)}{Fore.YELLOW} papers")
        print(f"{Fore.WHITE} PDF content will{' '+Fore.RED+'not' if not args.include_pdf else ' '+Fore.GREEN} be included in embeddings.")
        confirm = input(f"{Fore.YELLOW}❓ Type 'yes' if you wish to continue: ")
        if confirm.lower() != "yes":
            print(f"{Fore.RED} Embedding process cancelled by user")
            return
        print(f"{Fore.GREEN} Confirmation received")

    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 6: EMBEDDING CREATION      ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")
    
    # Create embeddings for the papers
    embedding_start = time.time()
    
    try:
        # Call the enhanced create_embeddings function
        embedding_data = create_embeddings(
            papers_to_process,
            chunk_mode=args.chunk_mode,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        embedding_time = time.time() - embedding_start
        print(f"{Fore.GREEN} Created {Fore.YELLOW}{len(embedding_data)}{Fore.GREEN} embeddings in {Fore.YELLOW}{embedding_time:.1f}s {Fore.GREEN}({Fore.YELLOW}{embedding_time/60:.1f} minutes)")
        
        # Storage phase
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════╗")
        print(f"{Fore.CYAN}║     STAGE 7: EMBEDDING STORAGE      ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")
        
        storage_start = time.time()
        store_embeddings(embedding_data, os.environ["PINECONE_INDEX_NAME"], "lhcb-arxiv-embeddings.json", batch_size=50, test_mode=args.test_mode)
        storage_time = time.time() - storage_start
        
        print(f"{Fore.GREEN} Embeddings stored successfully in {Fore.YELLOW}{storage_time:.1f}s {Fore.GREEN}({Fore.YELLOW}{storage_time/60:.1f} minutes)")
        
        # Check the final file size of the embeddings
        if os.path.exists("lhcb-arxiv-embeddings.json"):
            final_size_mb = os.path.getsize("lhcb-arxiv-embeddings.json") / (1024 * 1024)
            print(f"{Fore.GREEN} Final embeddings file size: {Fore.YELLOW}{final_size_mb:.2f} MB")
    except Exception as e:
        print(f"{Fore.RED} Error during embedding creation/storage: {str(e)}")
        # Print stack trace for debugging
        import traceback
        print(f"{Fore.RED}{traceback.format_exc()}")
        raise
    
    # Show overall execution time and summary
    total_time = time.time() - main_start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    
    print(f"\n{Fore.GREEN} Pipeline completed successfully ")
    print(f"{Fore.CYAN}╔═══════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║            EXECUTION SUMMARY           ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")
    
    print(f"{Fore.WHITE} Started at: {Fore.YELLOW}{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{Fore.WHITE} Finished at: {Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if hours > 0:
        print(f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        print(f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{minutes}m {seconds:.1f}s")
    else:
        print(f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{seconds:.1f}s")
        
    # Final statistics summary
    print(f"\n{Fore.CYAN} Final Statistics:")
    print(f"{Fore.WHITE} Total papers processed: {Fore.YELLOW}{len(all_papers):,}")
    print(f"{Fore.WHITE} LHCb papers found: {papers_color}{len(lhcb_papers)} {Fore.WHITE}({papers_color}{lhcb_ratio:.2f}%{Fore.WHITE})")
    print(f"{Fore.WHITE} Papers with PDF content: {pdf_color}{papers_with_pdf} {Fore.WHITE}({pdf_color}{pdf_ratio:.1f}%{Fore.WHITE})")
    print(f"{Fore.WHITE} Embeddings created: {Fore.GREEN}{len(embedding_data)}")
    
    try:
        import psutil
        process = psutil.Process(os.getpid())
        final_memory = process.memory_info().rss / 1024 / 1024
        if initial_memory is not None:
            memory_diff = final_memory - initial_memory
            print(f"{Fore.WHITE} Peak memory usage: {Fore.YELLOW}{final_memory:.1f} MB {Fore.WHITE}(+{Fore.YELLOW}{memory_diff:.1f} MB{Fore.WHITE})")
        else:
            print(f"{Fore.WHITE} Final memory usage: {Fore.YELLOW}{final_memory:.1f} MB")
    except (ImportError, NameError):
        pass
    
    # Reminder about embedding formatting     
    print(f"\n{Fore.GREEN} Note: Author information is kept in metadata but excluded from embedding text.")
    print(f"{Fore.GREEN} The embeddings focus on title, year, abstract, and content chunks for better semantic search.")
    
    
if __name__ == "__main__":
    main()
    print("Script finished")
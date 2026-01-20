import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List
from itertools import islice

from colorama import Fore, init, Style
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import gc
import torch

# Disable ChromaDB telemetry to prevent connection hangs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from paper import Paper
from dataset import ArxivDownloader, AdaptiveRateLimiter, download_arxiv_metadata
from helpers import load_data

# Initialize colorama
init(autoreset=True)


def get_device():
    """Detect and return the best available device (GPU/CPU)"""
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_count = torch.cuda.device_count()
        return device, gpu_name, gpu_count
    else:
        return "cpu", None, 0


def download_new_content(
    papers, html_dir, pdf_dir, output_dir=None, redownload_content: bool = False
):
    print(f"Entering unified download_new_content function")
    rate_limiter = AdaptiveRateLimiter(initial_delay=5, max_delay=300)
    downloader = ArxivDownloader(
        rate_limiter=rate_limiter, data_dir=output_dir if output_dir else "."
    )

    parents_map = {paper.id: paper for paper in papers}

    # Filter papers that need downloading if not forcing
    papers_to_download = []
    if redownload_content:
        print(f"Redownload content flag is set. Will download all content with immediate PDF fallback.")
        papers_to_download = [{"id": paper.id} for paper in papers]
    else:
        existing_html = {f.stem for f in Path(html_dir).glob("*.html")}
        existing_pdf = {f.stem for f in Path(pdf_dir).glob("*.pdf")}
        print(f"Found {len(existing_html)} existing HTML files and {len(existing_pdf)} existing PDF files")
        
        for paper in papers:
            safe_paper_id = paper.id.replace("/", "_")
            # Only download if neither format exists
            if safe_paper_id not in existing_html and safe_paper_id not in existing_pdf:
                papers_to_download.append({"id": paper.id})

    if not papers_to_download:
        print(f"No new papers to download.")
        # Still need to reload content for existing papers
        for paper in papers:
            safe_paper_id = paper.id.replace("/", "_")
            if safe_paper_id in {f.stem for f in Path(html_dir).glob("*.html")}:
                paper.reload_content(html_dir=html_dir)
            elif safe_paper_id in {f.stem for f in Path(pdf_dir).glob("*.pdf")}:
                paper.reload_content(pdf_dir=pdf_dir)
        return [], []

    print(f"Downloading content for {len(papers_to_download)} papers (HTML with immediate PDF fallback)...")
    
    # Use the new unified download with immediate fallback
    successful_html_ids, successful_pdf_ids, failed_ids = downloader.process_with_fallback(
        papers_to_download, html_dir, pdf_dir, batch_size=10
    )

    # Reload content for successfully downloaded papers
    for paper_id in successful_html_ids:
        if paper_id in parents_map:
            parents_map[paper_id].reload_content(html_dir=html_dir)
    
    for paper_id in successful_pdf_ids:
        if paper_id in parents_map:
            parents_map[paper_id].reload_content(pdf_dir=pdf_dir)

    all_successful_ids = list(set(successful_html_ids + successful_pdf_ids))
    return all_successful_ids, failed_ids


# Removed verify_pdf_downloads as we are moving to HTML.
# Can implement verify_html_downloads later if needed.


def stream_and_embed_papers(
    paper_generator,
    model,
    abstract_collection,
    content_collection,
    html_dir: str,
    pdf_dir: str,
    output_dir: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    batch_size: int = 32,
    redownload_content: bool = False,
    with_content: bool = False,
):
    """
    Stream papers from generator and embed them incrementally.
    Model and collections are pre-initialized and passed in.
    """
    from html_parser import ArxivHTMLParser
    
    # Initialize downloader if content is requested
    downloader = None
    if with_content:
        rate_limiter = AdaptiveRateLimiter(initial_delay=5, max_delay=300)
        downloader = ArxivDownloader(
            rate_limiter=rate_limiter, 
            data_dir=output_dir
        )
    
    # Start timing for embedding work only
    start_time = time.time()
    
    # Statistics
    total_papers_seen = 0
    lhcb_papers_found = 0
    total_chunks = 0
    errors = 0
    
    # Progress bar (will be created after first paper arrives)
    pbar = None
    
    # Batch accumulation for efficient embedding
    abstract_batch_texts = []
    abstract_batch_ids = []
    abstract_batch_metadatas = []
    abstract_batch_documents = []
    
    content_batch_texts = []
    content_batch_ids = []
    content_batch_metadatas = []
    content_batch_documents = []
    
    def flush_batches():
        """Helper to flush accumulated batches to ChromaDB"""
        nonlocal abstract_batch_texts, abstract_batch_ids, abstract_batch_metadatas, abstract_batch_documents
        nonlocal content_batch_texts, content_batch_ids, content_batch_metadatas, content_batch_documents
        
        # Flush abstracts
        if abstract_batch_texts:
            embeddings = model.encode(
                abstract_batch_texts,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=32  # Explicit batch size for GPU
            ).tolist()
            
            valid_indices = [
                idx for idx, emb in enumerate(embeddings)
                if all(abs(x) < 1e10 for x in emb) and any(abs(x) > 1e-10 for x in emb)
            ]
            
            if valid_indices:
                abstract_collection.upsert(
                    ids=[abstract_batch_ids[i] for i in valid_indices],
                    embeddings=[embeddings[i] for i in valid_indices],
                    metadatas=[abstract_batch_metadatas[i] for i in valid_indices],
                    documents=[abstract_batch_documents[i] for i in valid_indices],
                )
            
            abstract_batch_texts = []
            abstract_batch_ids = []
            abstract_batch_metadatas = []
            abstract_batch_documents = []
        
        # Flush content - process in smaller sub-batches to avoid GPU memory issues
        if content_batch_texts:
            # Process content in sub-batches of 64 to prevent GPU OOM
            sub_batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(content_batch_texts), sub_batch_size):
                sub_batch = content_batch_texts[i:i+sub_batch_size]
                sub_embeddings = model.encode(
                    sub_batch,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=32
                )
                all_embeddings.extend(sub_embeddings.tolist())
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            valid_content_indices = [
                idx for idx, emb in enumerate(all_embeddings)
                if all(abs(x) < 1e10 for x in emb) and any(abs(x) > 1e-10 for x in emb)
            ]
            
            if valid_content_indices:
                content_collection.upsert(
                    ids=[content_batch_ids[i] for i in valid_content_indices],
                    embeddings=[all_embeddings[i] for i in valid_content_indices],
                    metadatas=[content_batch_metadatas[i] for i in valid_content_indices],
                    documents=[content_batch_documents[i] for i in valid_content_indices],
                )
            
            content_batch_texts = []
            content_batch_ids = []
            content_batch_metadatas = []
            content_batch_documents = []
    
    # Process papers (load_data already filtered them)
    for paper in paper_generator:
        # Create progress bar after filtering completes (first paper arrives)
        if pbar is None:
            pbar = tqdm(desc="Embedding papers", unit=" papers", colour="green")
        
        total_papers_seen += 1
        lhcb_papers_found += 1  # All papers from generator are LHCb papers
        pbar.update(1)
        
        try:
            # Download content if requested
            if with_content and downloader:
                safe_paper_id = paper.id.replace("/", "_")
                html_path = Path(html_dir) / f"{safe_paper_id}.html"
                pdf_path = Path(pdf_dir) / f"{safe_paper_id}.pdf"
                
                # Check if we need to download
                if redownload_content or (not html_path.exists() and not pdf_path.exists()):
                    # Download with immediate fallback
                    html_ids, pdf_ids, failed = downloader.process_with_fallback(
                        [{"id": paper.id}],
                        html_dir,
                        pdf_dir,
                        batch_size=1
                    )
                    
                    # Reload content after download
                    if paper.id in html_ids:
                        paper.reload_content(html_dir=html_dir)
                    elif paper.id in pdf_ids:
                        paper.reload_content(pdf_dir=pdf_dir)
                else:
                    # Load existing content
                    if html_path.exists():
                        paper.reload_content(html_dir=html_dir)
                    elif pdf_path.exists():
                        paper.reload_content(pdf_dir=pdf_dir)
            
            # Prepare abstract embedding
            abs_text = paper.embedding_text_abstract
            abstract_batch_texts.append(abs_text)
            abstract_batch_ids.append(paper.id)
            abstract_batch_metadatas.append(paper.metadata)
            abstract_batch_documents.append(abs_text)
            
            # Prepare content embeddings if available
            if paper.has_content:
                chunks = ArxivHTMLParser.chunk_content(
                    paper._content, chunk_size, chunk_overlap
                )
                
                for idx, chunk in enumerate(chunks):
                    content_batch_texts.append(chunk)
                    content_batch_ids.append(f"{paper.id}_chunk_{idx}")
                    content_batch_metadatas.append({
                        "chunk_index": idx,
                        "parent_id": paper.id,
                        "total_chunks": len(chunks),
                    })
                    content_batch_documents.append(chunk)
                
                total_chunks += len(chunks)
            
            # Flush batches when they reach batch_size or content exceeds threshold
            # Keep content batches smaller to avoid GPU memory issues
            if len(abstract_batch_texts) >= batch_size or len(content_batch_texts) >= 200:
                flush_batches()
                # Update progress bar with statistics
                elapsed = time.time() - start_time
                rate = lhcb_papers_found / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    'chunks': f"{total_chunks:,}",
                    'rate': f"{rate:.1f}/s",
                    'errors': errors
                })
        
        except Exception as e:
            errors += 1
            pbar.write(f"{Fore.RED}Error processing paper {paper.id}: {e}")
    
    # Flush remaining batches
    flush_batches()
    
    # Close progress bar (if it was created)
    if pbar is not None:
        pbar.close()
    
    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}Streaming Processing Complete!")
    print(f"{Fore.GREEN}{'='*60}")
    print(f"{Fore.CYAN}Statistics:")
    print(f"  - LHCb papers processed: {lhcb_papers_found:,}")
    print(f"  - Total content chunks: {total_chunks:,}")
    print(f"  - Errors encountered: {errors}")
    print(f"  - Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  - Papers per minute: {lhcb_papers_found/(elapsed/60):.1f}")
    
    # Get final collection counts
    abs_count = abstract_collection.count()
    content_count = content_collection.count()
    print(f"\n{Fore.CYAN}Final Collection Sizes:")
    print(f"  - Abstracts: {abs_count:,} documents")
    print(f"  - Contents: {content_count:,} chunks")
    if abs_count > 0:
        print(f"  - Average chunks per paper: {content_count/abs_count:.1f}")
    print(f"{Fore.GREEN}{'='*60}\n")
    
    return lhcb_papers_found, total_chunks, errors


def create_and_store_embeddings(
    papers: List,
    chroma_client=None,
    batch_size: int = 32,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    output_dir: str = ".",
):
    """
    DEPRECATED: Use stream_and_embed_papers for better memory efficiency.
    
    Create embeddings and store them in ChromaDB.
    Maintains two collections:
    1. 'lhcb_abstracts': One embedding per paper (Title + Abstract)
    2. 'lhcb_contents': Multiple embeddings per paper (Full content chunks)
    """
    start_time = time.time()

    model_name = "BAAI/bge-large-en-v1.5"
    print(f"{Fore.YELLOW} Loading model: {Fore.WHITE}{model_name}")
    
    # Detect device
    device, gpu_name, gpu_count = get_device()
    if device == "cuda":
        print(f"{Fore.GREEN} GPU detected: {Fore.YELLOW}{gpu_name} {Fore.WHITE}(Count: {gpu_count})")
    else:
        print(f"{Fore.YELLOW} No GPU detected, using CPU")
    
    model = SentenceTransformer(model_name, device=device)

    # Initialize ChromaDB Collections and client
    print(f"{Fore.CYAN} Initializing ChromaDB client...")
    db_path = str(Path(output_dir) / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)

    # Optimized HNSW parameters for cosine similarity
    # M: Number of bi-directional links (16 is good balance of speed/quality)
    # ef_construction: Size of dynamic candidate list (200 for high quality)
    # space: cosine for normalized BAAI embeddings
    hnsw_config = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,  # Higher = better quality, slower build
        "hnsw:M": 16,  # Links per node (8-64 typical, 16 is balanced)
    }
    
    print(f"{Fore.CYAN} HNSW Configuration:")
    print(f"  - Distance metric: cosine (optimal for BAAI embeddings)")
    print(f"  - M (links): {hnsw_config['hnsw:M']}")
    print(f"  - ef_construction: {hnsw_config['hnsw:construction_ef']}")
    
    abstract_collection = chroma_client.get_or_create_collection(
        name="lhcb_abstracts",
        metadata=hnsw_config
    )
    content_collection = chroma_client.get_or_create_collection(
        name="lhcb_contents",
        metadata=hnsw_config
    )

    print(f"{Fore.CYAN} Processing {len(papers)} papers for embeddings...")

    total_chunks = 0
    errors = 0

    for i in range(0, len(papers), batch_size):
        batch = papers[i : i + batch_size]
        print(
            f"Processing batch {i // batch_size + 1}/{(len(papers) + batch_size - 1) // batch_size}"
        )

        abstract_texts = []
        abstract_ids = []
        abstract_metadatas = []

        content_texts = []
        content_ids = []
        content_metadatas = []

        for paper in batch:
            try:
                # 1. Prepare Abstract Embedding
                # Use a specific property for abstract embedding to ensure consistency
                abs_text = paper.embedding_text_abstract
                abstract_texts.append(abs_text)
                abstract_ids.append(paper.id)
                abstract_metadatas.append(paper.metadata)

                # 2. Prepare Content Embeddings (if content exists)
                if paper.has_content:
                    # Chunk the content
                    # We can use the static method from ArxivHTMLParser even if content came from PDF (fallback)
                    # Use the parser's chunk_content which is text-based
                    from html_parser import ArxivHTMLParser

                    chunks = ArxivHTMLParser.chunk_content(
                        paper._content, chunk_size, chunk_overlap
                    )

                    for idx, chunk in enumerate(chunks):
                        content_texts.append(
                            chunk
                        )  # Embedding vector will be made from this
                        content_ids.append(f"{paper.id}_chunk_{idx}")

                        # MINIMAL metadata for chunks - avoid duplication
                        # Full metadata stored in abstract collection, referenced by parent_id
                        meta = {
                            "chunk_index": idx,
                            "parent_id": paper.id,
                            "total_chunks": len(chunks),
                        }
                        content_metadatas.append(meta)

                    total_chunks += len(chunks)

            except Exception as e:
                errors += 1
                print(f"Error preparing paper {paper.id}: {e}")

        # Generate Embeddings & Store

        # A. Abstracts
        if abstract_texts:
            # Generate embeddings with normalization (important for cosine similarity)
            embeddings = model.encode(
                abstract_texts, 
                show_progress_bar=False,
                normalize_embeddings=True,  # Explicit normalization for cosine
                batch_size=32  # Explicit batch size for GPU
            ).tolist()
            
            # Validate embeddings (check for NaN, Inf, or zero vectors)
            valid_indices = []
            for idx, emb in enumerate(embeddings):
                if all(abs(x) < 1e10 for x in emb) and any(abs(x) > 1e-10 for x in emb):
                    valid_indices.append(idx)
                else:
                    print(f"{Fore.YELLOW}Warning: Invalid embedding for {abstract_ids[idx]}")
            
            if valid_indices:
                abstract_collection.upsert(
                    ids=[abstract_ids[i] for i in valid_indices],
                    embeddings=[embeddings[i] for i in valid_indices],
                    metadatas=[abstract_metadatas[i] for i in valid_indices],
                    documents=[abstract_texts[i] for i in valid_indices],
                )
                print(f"{Fore.GREEN}  ✓ Stored {len(valid_indices)}/{len(abstract_texts)} abstract embeddings")

        # B. Content Chunks - process in sub-batches for GPU memory management
        if content_texts:
            # Process in smaller sub-batches to avoid GPU OOM
            sub_batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(content_texts), sub_batch_size):
                sub_batch = content_texts[i:i+sub_batch_size]
                sub_embeddings = model.encode(
                    sub_batch, 
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    batch_size=32
                )
                all_embeddings.extend(sub_embeddings.tolist())
                
                # Clear GPU cache periodically
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Validate content embeddings
            valid_content_indices = []
            for idx, emb in enumerate(all_embeddings):
                if all(abs(x) < 1e10 for x in emb) and any(abs(x) > 1e-10 for x in emb):
                    valid_content_indices.append(idx)
                else:
                    print(f"{Fore.YELLOW}Warning: Invalid content embedding for {content_ids[idx]}")
            
            if valid_content_indices:
                content_collection.upsert(
                    ids=[content_ids[i] for i in valid_content_indices],
                    embeddings=[all_embeddings[i] for i in valid_content_indices],
                    metadatas=[content_metadatas[i] for i in valid_content_indices],
                    documents=[content_texts[i] for i in valid_content_indices],
                )
                print(f"{Fore.GREEN}  ✓ Stored {len(valid_content_indices)}/{len(content_texts)} content chunk embeddings")

    # Final summary with detailed statistics
    elapsed = time.time() - start_time
    print(f"\n{Fore.GREEN}{'='*60}")
    print(f"{Fore.GREEN}Embedding Complete!")
    print(f"{Fore.GREEN}{'='*60}")
    print(f"{Fore.CYAN}Statistics:")
    print(f"  - Total papers processed: {len(papers)}")
    print(f"  - Total content chunks: {total_chunks}")
    print(f"  - Errors encountered: {errors}")
    print(f"  - Time elapsed: {elapsed/60:.1f} minutes")
    print(f"  - Papers per minute: {len(papers)/(elapsed/60):.1f}")
    
    # Get final collection counts
    abs_count = abstract_collection.count()
    content_count = content_collection.count()
    print(f"\n{Fore.CYAN}Final Collection Sizes:")
    print(f"  - Abstracts: {abs_count:,} documents")
    print(f"  - Contents: {content_count:,} chunks")
    if abs_count > 0:
        print(f"  - Average chunks per paper: {content_count/abs_count:.1f}")
    print(f"{Fore.GREEN}{'='*60}\n")


def main():
    # Start timing the entire process
    main_start_time = time.time()
    start_datetime = datetime.now()

    # Print fancy header
    print(f"\n{Fore.CYAN}╔══════════════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║        LHCb FINDER - EMBEDDING PIPELINE              ║")
    print(f"{Fore.CYAN}╚══════════════════════════════════════════════════════╝")
    print(f"{Fore.GREEN}  Started at: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check environment variables with better formatting
    print(f"\n{Fore.YELLOW} Checking environment variables...")

    print(f"{Fore.GREEN} Environment ready!")

    # Initialize memory tracking if available
    try:
        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(
            f"{Fore.WHITE} Initial memory usage: {Fore.YELLOW}{initial_memory:.1f} MB"
        )
    except ImportError:
        initial_memory = None

    # Parse command line arguments
    print(f"\n{Fore.CYAN} Parsing command line arguments...")
    parser = argparse.ArgumentParser(description="Create embeddings for LHCb papers")
    parser.add_argument(
        "--with-content",
        action="store_true",
        help="Include paper content in embeddings (loads from existing HTML/PDF files)",
    )
    parser.add_argument(
        "--html-dir",
        type=str,
        default="lhcb_html",
        help="Directory to store HTML files (relative to output-dir)",
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default="lhcb_pdfs",
        help="Directory to store PDFs (relative to output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Base directory for all output (HTML, PDFs, DB, logs)",
    )
    parser.add_argument(
        "--redownload-metadata",
        action="store_true",
        help="Force re-downloading arXiv metadata from source",
    )
    parser.add_argument(
        "--redownload-content",
        action="store_true",
        help="Force re-downloading content (HTML/PDF) from arXiv",
    )
    parser.add_argument(
        "--rebuild-embeddings",
        action="store_true",
        help="Force recreating all embeddings (ignores existing)",
    )
    parser.add_argument("--start-year", type=int, help="Start year for papers")
    parser.add_argument(
        "--no-confirmation", action="store_true", help="Skip confirmation prompts"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500, help="Maximum words per chunk"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100, help="Words to overlap between chunks"
    )
    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode with limited papers"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Limit number of papers to process in test mode",
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve subdirectories relative to output_dir
    html_dir = output_dir / args.html_dir
    pdf_dir = output_dir / args.pdf_dir

    # Print arguments with colorful formatting
    print(f"\n{Fore.CYAN} Command Line Arguments:")
    # Common arguments
    print(f"{Fore.WHITE} Output Directory: {Fore.YELLOW}{output_dir}")
    print(f"{Fore.WHITE} Include Content: {Fore.YELLOW}{args.with_content}")
    print(f"{Fore.WHITE} HTML directory: {Fore.YELLOW}{html_dir}")
    print(f"{Fore.WHITE} PDF directory: {Fore.YELLOW}{pdf_dir}")
    print(f"{Fore.WHITE} Redownload content: {Fore.YELLOW}{args.redownload_content}")
    print(f"{Fore.WHITE} Redownload metadata: {Fore.YELLOW}{args.redownload_metadata}")
    print(f"{Fore.WHITE} Rebuild embeddings: {Fore.YELLOW}{args.rebuild_embeddings}")

    # Chunking options
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
    print(f"{Fore.CYAN}║     STAGE 1: DATA PREPARATION         ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")

    # First, check if we have the ArXiv JSON file
    JSON_FILE_PATH = "arxiv-metadata-oai-snapshot.json"
    print(
        f"{Fore.YELLOW} Checking if ArXiv dataset exists at: {Fore.WHITE}{JSON_FILE_PATH}"
    )

    # If redownload_metadata is set, we need to ensure dataset checks happen
    if args.redownload_metadata:
        # logic handled in check_arxiv_dataset or before call
        pass
    if not os.path.exists(JSON_FILE_PATH) or args.redownload_metadata:
        if args.redownload_metadata:
            print(
                f"{Fore.YELLOW} Redownload flag set, downloading fresh ArXiv dataset"
            )
        else:
            print(f"{Fore.YELLOW} Dataset not found, downloading ArXiv data")

        print(f"{Fore.CYAN} Starting Kaggle download of ArXiv metadata...")
        download_start = time.time()

        # Show a spinner or progress indicator since this can take a while
        # Fallback if yaspin is removed/not available
        print(
            f"{Fore.YELLOW} Downloading ArXiv dataset... (this may take a while if manual download is needed)"
        )
        download_arxiv_metadata()
        print(f"{Fore.GREEN} ArXiv dataset check/download complete!")

        download_time = time.time() - download_start
        print(
            f"{Fore.GREEN} ArXiv dataset download completed in {Fore.YELLOW}{download_time:.1f} seconds {Fore.GREEN}({Fore.YELLOW}{download_time / 60:.1f} minutes)"
        )

        # Check file size
        if os.path.exists(JSON_FILE_PATH):
            file_size_bytes = os.path.getsize(JSON_FILE_PATH)
            file_size_gb = file_size_bytes / (1024**3)  # Convert to GB
            print(
                f"{Fore.GREEN} Downloaded file size: {Fore.YELLOW}{file_size_gb:.2f} GB"
            )
    else:
        # File exists and no force download
        file_size_bytes = os.path.getsize(JSON_FILE_PATH)
        file_size_gb = file_size_bytes / (1024**3)  # Convert to GB
        print(
            f"{Fore.GREEN} ArXiv dataset found! {Fore.WHITE}({Fore.YELLOW}{file_size_gb:.2f} GB{Fore.WHITE})"
        )

    # Set up directories if downloading content
    if args.with_content:
        print(f"{Fore.YELLOW} Setting up directories in {output_dir}...")
        html_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

    # STAGE 2: Initialize model and ChromaDB
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 2: INITIALIZATION & FILTERING           ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════╝")
    
    model_name = "BAAI/bge-large-en-v1.5"
    print(f"{Fore.YELLOW} Loading embedding model: {Fore.WHITE}{model_name}")
    
    # Detect device
    device, gpu_name, gpu_count = get_device()
    if device == "cuda":
        print(f"{Fore.GREEN} ✓ GPU detected: {Fore.YELLOW}{gpu_name}")
        if gpu_count > 1:
            print(f"{Fore.CYAN}   Available GPUs: {Fore.YELLOW}{gpu_count}")
        # Show GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"{Fore.CYAN}   GPU Memory: {Fore.YELLOW}{gpu_memory:.1f} GB")
    else:
        print(f"{Fore.YELLOW} ⚠ No GPU detected, using CPU (this will be slower)")
    
    model = SentenceTransformer(model_name, device=device)
    print(f"{Fore.GREEN} ✓ Model loaded on {Fore.YELLOW}{device.upper()}")
    
    print(f"{Fore.CYAN} Initializing ChromaDB client...")
    db_path = str(output_dir / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)
    
    hnsw_config = {
        "hnsw:space": "cosine",
        "hnsw:construction_ef": 200,
        "hnsw:M": 16,
    }
    
    print(f"{Fore.CYAN} HNSW Configuration:")
    print(f"  - Distance metric: cosine (optimal for BAAI embeddings)")
    print(f"  - M (links): {hnsw_config['hnsw:M']}")
    print(f"  - ef_construction: {hnsw_config['hnsw:construction_ef']}")
    
    abstract_collection = chroma_client.get_or_create_collection(
        name="lhcb_abstracts",
        metadata=hnsw_config
    )
    content_collection = chroma_client.get_or_create_collection(
        name="lhcb_contents",
        metadata=hnsw_config
    )
    
    print(f"{Fore.GREEN} Initialization complete!\n")
    
    # Create paper generator (load_data will filter and show progress bars)
    print(f"{Fore.CYAN} Filtering arXiv dataset for LHCb papers...")
    pass_pdf_dir = str(pdf_dir) if args.with_content else None
    pass_html_dir = str(html_dir) if args.with_content else None
    include_content = args.with_content

    paper_generator = load_data(
        JSON_FILE_PATH,
        html_dir=pass_html_dir,
        pdf_dir=pass_pdf_dir,
        include_content=include_content,
        start_year=args.start_year,
    )
    
    # Limit generator if in test mode
    if args.test_mode:
        print(f"{Fore.MAGENTA} TEST MODE: Will process only first {args.limit} papers\n")
        paper_generator = islice(paper_generator, args.limit)


    # STAGE 3: Embedding (will start after generator completes filtering)
    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 3: EMBEDDING PIPELINE                   ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════════╝")
    print(f"{Fore.GREEN} Starting embedding process...\n")
    
    try:
        lhcb_count, chunk_count, error_count = stream_and_embed_papers(
            paper_generator=paper_generator,
            model=model,
            abstract_collection=abstract_collection,
            content_collection=content_collection,
            html_dir=str(html_dir),
            pdf_dir=str(pdf_dir),
            output_dir=str(output_dir),
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            batch_size=32,
            redownload_content=args.redownload_content,
            with_content=args.with_content,
        )

    except Exception as e:
        print(f"{Fore.RED} Error during streaming pipeline: {str(e)}")
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
    print(f"{Fore.CYAN}║            EXECUTION SUMMARY                  ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")

    print(
        f"{Fore.WHITE} Started at: {Fore.YELLOW}{start_datetime.strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print(
        f"{Fore.WHITE} Finished at: {Fore.YELLOW}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if hours > 0:
        print(
            f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{hours}h {minutes}m {seconds:.1f}s"
        )
    elif minutes > 0:
        print(
            f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{minutes}m {seconds:.1f}s"
        )
    else:
        print(f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{seconds:.1f}s")

    # Final statistics summary
    print(f"\n{Fore.CYAN} Final Statistics:")
    print(f"{Fore.WHITE} LHCb papers found & embedded: {Fore.YELLOW}{lhcb_count:,}")
    print(f"{Fore.WHITE} Content chunks created: {Fore.YELLOW}{chunk_count:,}")
    print(f"{Fore.WHITE} Errors encountered: {Fore.YELLOW}{error_count}")
    print(f"{Fore.WHITE} Embeddings creation process finished.")

    try:
        import psutil

        process = psutil.Process(os.getpid())
        final_memory = process.memory_info().rss / 1024 / 1024
        if initial_memory is not None:
            memory_diff = final_memory - initial_memory
            print(
                f"{Fore.WHITE} Peak memory usage: {Fore.YELLOW}{final_memory:.1f} MB {Fore.WHITE}(+{Fore.YELLOW}{memory_diff:.1f} MB{Fore.WHITE})"
            )
        else:
            print(
                f"{Fore.WHITE} Final memory usage: {Fore.YELLOW}{final_memory:.1f} MB"
            )
    except (ImportError, NameError):
        pass

    # Reminder about embedding formatting
    print(
        f"\n{Fore.GREEN} Note: Author information is kept in metadata but excluded from embedding text."
    )
    print(
        f"{Fore.GREEN} The embeddings focus on title, year, abstract, and content chunks for better semantic search."
    )


if __name__ == "__main__":
    main()
    print("Script finished")

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List

from colorama import Fore, init, Style
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import gc

# Disable ChromaDB telemetry to prevent connection hangs
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from paper import Paper
from dataset import ArxivDownloader, AdaptiveRateLimiter, download_arxiv_metadata
from helpers import load_data, filter_lhcb_papers

# Initialize colorama
init(autoreset=True)


def download_new_content(
    papers, html_dir, pdf_dir, output_dir=None, force_content: bool = False
):
    print(f"Entering unified download_new_content function")
    rate_limiter = AdaptiveRateLimiter(initial_delay=5, max_delay=300)
    downloader = ArxivDownloader(
        rate_limiter=rate_limiter, data_dir=output_dir if output_dir else "."
    )

    parents_map = {paper.id: paper for paper in papers}

    # Filter papers that need downloading if not forcing
    papers_to_download = []
    if force_content:
        print(f"Force content flag is set. Will download all content with immediate PDF fallback.")
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


def create_and_store_embeddings(
    papers: List,
    chroma_client=None,
    batch_size: int = 32,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    output_dir: str = ".",
):
    """
    Create embeddings and store them in ChromaDB.
    Maintains two collections:
    1. 'lhcb_abstracts': One embedding per paper (Title + Abstract)
    2. 'lhcb_contents': Multiple embeddings per paper (Full content chunks)
    """
    start_time = time.time()

    model_name = "BAAI/bge-large-en-v1.5"
    print(f"{Fore.YELLOW} Loading model: {Fore.WHITE}{model_name}")
    model = SentenceTransformer(model_name)

    # Initialize ChromaDB Collections and client
    print(f"{Fore.CYAN} Initializing ChromaDB client...")
    db_path = str(Path(output_dir) / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=db_path)

    abstract_collection = chroma_client.get_or_create_collection(name="lhcb_abstracts")
    content_collection = chroma_client.get_or_create_collection(name="lhcb_contents")

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

                        # Content metadata includes parent ID to link back
                        meta = paper.metadata.copy()
                        meta.update(
                            {
                                "chunk_index": idx,
                                "parent_id": paper.id,
                                "total_chunks": len(chunks),
                            }
                        )
                        content_metadatas.append(meta)

                    total_chunks += len(chunks)

            except Exception as e:
                errors += 1
                print(f"Error preparing paper {paper.id}: {e}")

        # Generate Embeddings & Store

        # A. Abstracts
        if abstract_texts:
            embeddings = model.encode(abstract_texts, show_progress_bar=False).tolist()
            abstract_collection.upsert(
                ids=abstract_ids,
                embeddings=embeddings,
                metadatas=abstract_metadatas,
                documents=abstract_texts,  # Optional: Store text in DB too
            )

        # B. Content Chunks
        if content_texts:
            content_embeddings = model.encode(
                content_texts, show_progress_bar=False
            ).tolist()
            content_collection.upsert(
                ids=content_ids,
                embeddings=content_embeddings,
                metadatas=content_metadatas,
                documents=content_texts,
            )

    print(f"\n{Fore.GREEN} Embdedding complete!")
    print(f"Abstracts stored: {len(papers)}")
    print(f"Content chunks stored: {total_chunks}")
    if errors:
        print(f"Errors: {errors}")


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
        "--download-content",
        action="store_true",
        help="Download content (HTML with PDF fallback)",
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
        "--force-metadata",
        action="store_true",
        help="Force download of new arXiv metadata",
    )
    parser.add_argument(
        "--force-content",
        action="store_true",
        help="Force download of content (HTML/PDF)",
    )
    parser.add_argument(
        "--force-embeddings",
        action="store_true",
        help="Force reprocessing of all papers",
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
    print(f"{Fore.WHITE} Download Content: {Fore.YELLOW}{args.download_content}")
    print(f"{Fore.WHITE} HTML directory: {Fore.YELLOW}{html_dir}")
    print(f"{Fore.WHITE} PDF directory: {Fore.YELLOW}{pdf_dir}")
    print(f"{Fore.WHITE} Force content: {Fore.YELLOW}{args.force_content}")
    print(f"{Fore.WHITE} Force metadata: {Fore.YELLOW}{args.force_metadata}")
    print(f"{Fore.WHITE} Force embeddings: {Fore.YELLOW}{args.force_embeddings}")

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

    # If force_metadata is set, we need to ensure dataset checks happen
    if args.force_metadata:
        # logic handled in check_arxiv_dataset or before call
        pass
    if not os.path.exists(JSON_FILE_PATH) or args.force_metadata:
        if args.force_metadata:
            print(
                f"{Fore.YELLOW} Force download flag set, downloading fresh ArXiv dataset"
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

    # Set up directories
    # Set up directories if downloading content
    # Set up directories if downloading content
    if args.download_content:
        print(f"{Fore.YELLOW} Setting up directories in {output_dir}...")
        html_dir.mkdir(parents=True, exist_ok=True)
        pdf_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{Fore.CYAN} Loading and filtering papers...")
    load_start_time = time.time()

    # Note for include_pdf
    if args.download_content:
        print(f"{Fore.YELLOW} Including content in embeddings")

    # Create a spinner or progress indicator for paper loading
    print(f"{Fore.YELLOW} Loading papers from ArXiv dataset...")

    # Load and filter papers with timing
    loading_start = time.time()

    # Pass both dirs if they exist/are requested
    # Variables html_dir and pdf_dir are already resolved Paths above
    pass_pdf_dir = str(pdf_dir) if args.download_content else None
    pass_html_dir = str(html_dir) if args.download_content else None

    # Determine if we should include content in embeddings
    include_content = args.download_content

    paper_generator = load_data(
        JSON_FILE_PATH,
        html_dir=pass_html_dir,
        pdf_dir=pass_pdf_dir,
        include_content=include_content,
        start_year=args.start_year,
    )

    # If in test mode, limit the number of papers with progress indicator
    if args.test_mode:
        print(f"{Fore.MAGENTA} TEST MODE: Limiting to {args.limit} papers")
        all_papers = []
        with tqdm(
            total=args.limit, desc=f"{Fore.GREEN}Loading papers", unit="paper"
        ) as pbar:
            for i, paper in enumerate(paper_generator):
                all_papers.append(paper)
                pbar.update(1)
                if i >= args.limit - 1:  # -1 because i starts at 0
                    break
    else:
        # For full mode, we can't know the total count in advance, use simple progress indicator
        print(
            f"{Fore.YELLOW} Loading all papers from dataset (this may take a while)..."
        )
        all_papers = []
        for i, paper in enumerate(paper_generator):
            all_papers.append(paper)
            # Print progress every 50,000 papers
            if (i + 1) % 50000 == 0:
                print(f"{Fore.GREEN} Loaded {i + 1} papers so far...")

    # Calculate loading time and speed
    loading_time = time.time() - loading_start
    papers_per_second = len(all_papers) / loading_time if loading_time > 0 else 0

    print(
        f"{Fore.GREEN} Loaded {Fore.YELLOW}{len(all_papers):,}{Fore.GREEN} total papers in {Fore.YELLOW}{loading_time:.1f}s {Fore.GREEN}({Fore.YELLOW}{papers_per_second:.1f}{Fore.GREEN} papers/sec)"
    )

    # Print some information about the dataset
    years = {}
    categories = {}
    for paper in all_papers[:1000]:  # Sample first 1000 papers for quick stats
        year = getattr(paper, "year", 0)
        years[year] = years.get(year, 0) + 1

        for category in getattr(paper, "categories", []):
            categories[category] = categories.get(category, 0) + 1

    if years:
        print(f"{Fore.CYAN} Sample Data Statistics (first 1000 papers):")
        print(f"{Fore.WHITE} Years: {Fore.YELLOW}{sorted(years.keys())[:5]}...")
        top_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[
            :5
        ]
        print(
            f"{Fore.WHITE} Top categories: {', '.join([f'{cat} ({count})' for cat, count in top_categories])}"
        )

    # Download content if requested
    if args.download_content:
        print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
        print(f"{Fore.CYAN}║     STAGE 3: CONTENT DOWNLOAD         ║")
        print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")
        download_start = time.time()

        download_new_content(
            all_papers,
            str(html_dir),
            str(pdf_dir),
            output_dir=str(output_dir),
            force_content=args.force_content,
        )

        download_time = time.time() - download_start
        print(
            f"{Fore.GREEN} Content download completed in {Fore.YELLOW}{download_time:.1f}s"
        )

    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 4: LHCb FILTERING           ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════╝")

    print(f"{Fore.YELLOW} Filtering for LHCb papers...")
    filtering_start = time.time()
    lhcb_papers = list(filter_lhcb_papers(all_papers))
    filtering_time = time.time() - filtering_start

    # Stats for LHCb papers
    lhcb_ratio = len(lhcb_papers) / len(all_papers) * 100 if all_papers else 0
    papers_color = (
        Fore.GREEN
        if len(lhcb_papers) > 50
        else (Fore.YELLOW if len(lhcb_papers) > 10 else Fore.RED)
    )

    print(
        f"{Fore.GREEN} Found {papers_color}{len(lhcb_papers)}{Fore.GREEN} LHCb papers in {Fore.YELLOW}{filtering_time:.2f}s {Fore.WHITE}({Fore.YELLOW}{lhcb_ratio:.2f}%{Fore.WHITE} of total papers)"
    )

    if len(lhcb_papers) == 0:
        print(f"{Fore.RED} Error: No LHCb papers found in the dataset")

        # Show completion time before exiting
        total_time = time.time() - main_start_time
        print(f"\n{Fore.RED} Pipeline terminated due to no LHCb papers found")
        print(
            f"{Fore.WHITE}⏱ Total execution time: {Fore.YELLOW}{total_time:.1f}s {Fore.WHITE}({Fore.YELLOW}{total_time / 60:.1f} minutes)"
        )
        return

    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 5: EMBEDDING PREPARATION            ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")

    print(f"{Fore.YELLOW} Checking for existing embeddings...")
    # NOTE: Chroma DB handles duplicates if IDs match, but we can verify counts.
    # We can skip complex checking for now and let Chroma upsert handle updates.
    # TODO: implement a simple check function later.
    papers_to_process = lhcb_papers
    is_new_index = (
        True  # Assume we want to process everything or rely on Chroma's upsert
    )

    # Optional: logic to skip if already in DB (not implemented for Chroma yet in this script,
    # relying on upsert efficiency or user using --force-embeddings)
    # TODO: Implement check_existing_embeddings function later.

    papers_to_process_ratio = (
        len(papers_to_process) / len(lhcb_papers) * 100 if lhcb_papers else 0
    )
    need_embeddings_color = (
        Fore.GREEN
        if papers_to_process_ratio < 30
        else (Fore.YELLOW if papers_to_process_ratio < 70 else Fore.RED)
    )

    print(f"\n{Fore.CYAN} Current Status:")
    print(f"{Fore.WHITE} Total LHCb papers found: {papers_color}{len(lhcb_papers)}")
    print(
        f"{Fore.WHITE} Papers needing embeddings: {need_embeddings_color}{len(papers_to_process)} {Fore.WHITE}({need_embeddings_color}{papers_to_process_ratio:.1f}%{Fore.WHITE})"
    )
    print(f"{Fore.WHITE} Using new/empty Pinecone index: {Fore.YELLOW}{is_new_index}")

    if not papers_to_process and not args.force_embeddings and not is_new_index:
        print(f"\n{Fore.GREEN} No new papers to process.")
        print(
            f"{Fore.WHITE} Hint: Use --force-embeddings to override and process all papers again."
        )

        # Show completion time before exiting
        total_time = time.time() - main_start_time
        print(
            f"\n{Fore.GREEN} Pipeline completed successfully (no new papers to process)"
        )
        print(
            f"{Fore.WHITE} Total execution time: {Fore.YELLOW}{total_time:.1f}s {Fore.WHITE}({Fore.YELLOW}{total_time / 60:.1f} minutes)"
        )
        return
    else:
        print(f"\n{Fore.GREEN} Found {len(papers_to_process)} papers to process")

        print(
            f"\n{Fore.YELLOW}❗ Ready to process {Fore.WHITE}{len(papers_to_process)}{Fore.YELLOW} papers"
        )
        include_status = (
            Fore.GREEN + "will" if args.download_content else Fore.RED + "will not"
        )
        print(
            f"{Fore.WHITE} Content (HTML/PDF) {include_status} be included in embeddings."
        )

        # User requested no interaction, so we skip confirmation
        print(f"{Fore.GREEN} Proceeding automatically...")

    print(f"\n{Fore.CYAN}╔═══════════════════════════════════════════════╗")
    print(f"{Fore.CYAN}║     STAGE 6: EMBEDDING CREATION               ║")
    print(f"{Fore.CYAN}╚═══════════════════════════════════════════════╝")

    # Create embeddings for the papers
    embedding_start = time.time()

    try:
        # Call the create_and_store_embeddings function
        create_and_store_embeddings(
            papers_to_process,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            output_dir=str(output_dir),
        )

    except Exception as e:
        print(f"{Fore.RED} Error during embedding creation/storage: {str(e)}")
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
    print(f"{Fore.WHITE} Total papers processed: {Fore.YELLOW}{len(all_papers):,}")
    print(
        f"{Fore.WHITE} LHCb papers found: {papers_color}{len(lhcb_papers)} {Fore.WHITE}({papers_color}{lhcb_ratio:.2f}%{Fore.WHITE})"
    )
    # Calculate content source statistics
    html_count = sum(
        1 for p in lhcb_papers if getattr(p, "_content_source", None) == "html"
    )
    pdf_count = sum(
        1 for p in lhcb_papers if getattr(p, "_content_source", None) == "pdf"
    )
    no_content_count = len(lhcb_papers) - html_count - pdf_count

    print(f"{Fore.WHITE} Content Distribution:")
    print(
        f"  • {Fore.GREEN}HTML: {html_count} papers ({html_count / len(lhcb_papers) * 100:.1f}%)"
    )
    print(
        f"  • {Fore.YELLOW}PDF (Fallback): {pdf_count} papers ({pdf_count / len(lhcb_papers) * 100:.1f}%)"
    )
    if no_content_count > 0:
        print(
            f"  • {Fore.RED}No Content: {no_content_count} papers ({no_content_count / len(lhcb_papers) * 100:.1f}%)"
        )
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

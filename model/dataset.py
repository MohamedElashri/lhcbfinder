# dataset.py
import json
import os
import argparse
import requests
import time
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
from collections import deque
import math
import gzip
import re


class AdaptiveRateLimiter:
    def __init__(self, initial_delay=5, window_size=10, max_delay=300):
        self.current_delay = initial_delay
        self.max_delay = max_delay
        self.window_size = window_size
        self.success_history = deque(maxlen=window_size)
        self.last_request_time = None
        self.consecutive_failures = 0

    def wait(self):
        """Wait appropriate amount of time before next request."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.current_delay:
                time.sleep(self.current_delay - elapsed)
        self.last_request_time = time.time()

    def update(self, success, status_code=None, retry_after=None):
        """Update rate limiting based on request success/failure."""
        self.success_history.append(success)

        # If we get a 429, use the Retry-After header
        if status_code == 429 and retry_after:
            self.current_delay = max(self.current_delay, float(retry_after))
            self.consecutive_failures += 1
            logging.warning(f"Rate limit hit. New delay: {self.current_delay}s")
            return

        if success:
            self.consecutive_failures = 0
            # If we have a full window of successes, try decreasing the delay
            if len(self.success_history) == self.window_size and all(
                self.success_history
            ):
                self.current_delay = max(5, self.current_delay * 0.8)
        else:
            self.consecutive_failures += 1
            # Exponential backoff on failures
            backoff_multiplier = min(math.pow(2, self.consecutive_failures), 10)
            self.current_delay = min(
                self.current_delay * backoff_multiplier, self.max_delay
            )
            logging.warning(f"Request failed. New delay: {self.current_delay}s")


class ArxivDownloader:
    def __init__(self, rate_limiter=None, data_dir="."):
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter()
        self.data_dir = Path(data_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (compatible; LHCbDatasetBuilder/1.0; mailto:arxiv@elashri.com)"
            }
        )
        self.error_counts = {}
        self.removed_papers = []  # Track removed/404 papers

        # Ensure the removed_papers.json file exists
        removed_path = self.data_dir / "removed_papers.json"
        if removed_path.exists():
            try:
                with open(removed_path, "r") as f:
                    self.removed_papers = json.load(f)
                logging.info(
                    f"Loaded {len(self.removed_papers)} known removed paper IDs from file"
                )
            except json.JSONDecodeError:
                logging.warning(
                    "Error loading removed_papers.json, starting with empty list"
                )

    def download_content(self, paper_id, output_dir, format="html"):
        """
        Download paper content (HTML or PDF) with adaptive rate limiting.
        format: 'html' or 'pdf'
        """
        # Check if paper is in known removed list
        if paper_id in self.removed_papers:
            logging.info(f"Skipping {paper_id} - known to be removed/retracted")
            return True

        # Replace slashes with underscores in the filename
        safe_paper_id = paper_id.replace("/", "_")

        # Determine extension and URL prefix
        if format == "html":
            extension = ".html"
            url_prefix = "https://arxiv.org/html"
            # For HTML, ArXiv typically serves a folder structure or a single page.
            # The new ArXiv HTML standard: https://arxiv.org/html/2312.12345
        else:
            extension = ".pdf"
            url_prefix = "https://arxiv.org/pdf"

        file_path = Path(output_dir) / f"{safe_paper_id}{extension}"

        # Skip if already downloaded successfully
        if file_path.exists() and file_path.stat().st_size > 0:
            logging.info(f"Skipping {paper_id} - already downloaded")
            return True

        # Normalize the paper ID format
        # Modern arXiv IDs look like 2010.12345, older ones like 0704.0001
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", paper_id):
            try:
                # Try to normalize legacy arXiv ID format
                if "/" in paper_id:
                    normalized_id = paper_id
                else:
                    match = re.match(r"^(\d{7})$", paper_id)
                    if match:
                        yymm = paper_id[:4]
                        number = paper_id[4:]
                        normalized_id = f"{yymm}.{number}"
                    else:
                        normalized_id = paper_id
            except Exception as e:
                logging.error(f"Error normalizing paper ID {paper_id}: {str(e)}")
                normalized_id = paper_id
        else:
            normalized_id = paper_id

        logging.info(f"Normalized ID: {paper_id} -> {normalized_id}")

        # Construct URLs to try
        urls_to_try = []
        if format == "html":
            urls_to_try = [
                f"{url_prefix}/{normalized_id}v1",  # Try specific version first? Or just base?
                f"{url_prefix}/{normalized_id}",
            ]
        else:
            urls_to_try = [
                f"{url_prefix}/{normalized_id}.pdf",
                f"{url_prefix}/{normalized_id}",
            ]

        max_retries = 3

        for attempt in range(max_retries):
            self.rate_limiter.wait()

            current_url = urls_to_try[0]  # Simple approach for now
            logging.info(
                f"Attempt {attempt + 1}/{max_retries} for {paper_id} ({format}) using URL: {current_url}"
            )

            try:
                # Add different user agents on retries
                if attempt > 0:
                    user_agents = [
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
                    ]
                    self.session.headers.update(
                        {"User-Agent": user_agents[attempt % len(user_agents)]}
                    )

                response = self.session.get(current_url, stream=True, timeout=30)

                if response.status_code == 404:
                    if format == "html":
                        logging.debug(f"HTML Content for {paper_id} not found (404).")
                        # Don't add to removed_papers purely on HTML 404, might just be PDF-only paper
                        return False
                    else:
                        logging.debug(f"Paper {paper_id} not found (404)")
                        self.removed_papers.append(paper_id)
                        # Save updated removed list
                        with open(self.data_dir / "removed_papers.json", "w") as f:
                            json.dump(self.removed_papers, f)
                        return True

                response.raise_for_status()

                # Check formatted content
                # For HTML, we dump the text response
                with open(
                    file_path,
                    "wb" if format == "pdf" else "w",
                    encoding=None if format == "pdf" else "utf-8",
                ) as f:
                    if format == "pdf":
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    else:
                        f.write(response.text)

                # Verify
                if file_path.stat().st_size > 100:
                    self.rate_limiter.update(True)
                    logging.info(f"Successfully downloaded {paper_id} {format}")
                    return True
                else:
                    self.rate_limiter.update(False)
                    return False

            except Exception as e:
                self.rate_limiter.update(False)
                logging.error(f"Error downloading {paper_id} {format}: {str(e)}")
                if attempt == max_retries - 1:
                    return False
        return False

    def download_with_fallback(self, paper_id, html_dir, pdf_dir):
        """Try HTML first, immediately fallback to PDF if HTML fails."""
        # Check if paper is in known removed list
        if paper_id in self.removed_papers:
            logging.info(f"Skipping {paper_id} - known to be removed/retracted")
            return "skipped", None

        safe_paper_id = paper_id.replace("/", "_")
        html_path = Path(html_dir) / f"{safe_paper_id}.html"
        pdf_path = Path(pdf_dir) / f"{safe_paper_id}.pdf"

        # Check if either format already exists
        if html_path.exists() and html_path.stat().st_size > 0:
            logging.info(f"Skipping {paper_id} - HTML already exists")
            return "html", paper_id
        
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logging.info(f"Skipping {paper_id} - PDF already exists")
            return "pdf", paper_id

        # Try HTML first
        if self.download_content(paper_id, html_dir, format="html"):
            return "html", paper_id
        
        # Immediate PDF fallback
        logging.info(f"HTML failed for {paper_id}, attempting PDF fallback...")
        if self.download_content(paper_id, pdf_dir, format="pdf"):
            return "pdf", paper_id
        
        # Both failed
        return "failed", paper_id

    def process_batch(self, papers, output_dir, batch_size=10, format="html"):
        """Process a batch of papers with adaptive rate limiting."""
        successful = []
        failed = []

        # Create a single progress bar for all papers
        with tqdm(total=len(papers), desc=f"Processing papers ({format})") as pbar:
            for i in range(0, len(papers), batch_size):
                batch = papers[i : i + batch_size]
                logging.info(f"\nProcessing mini-batch {i // batch_size + 1}")

                for paper in batch:
                    paper_id = paper.get("id")
                    if not paper_id:
                        pbar.update(1)
                        continue

                    if self.download_content(paper_id, output_dir, format=format):
                        successful.append(paper_id)
                    else:
                        failed.append(paper_id)
                    
                    pbar.update(1)

                # After each mini-batch, check if we need to adjust our rate limiting
                if self.rate_limiter.consecutive_failures > 3:
                    sleep_time = min(300, self.rate_limiter.current_delay * 2)
                    logging.info(f"Too many failures, pausing for {sleep_time}s...")
                    time.sleep(sleep_time)

                # Save progress after each batch
                self._save_progress(successful, failed, format)

        return successful, failed

    def process_with_fallback(self, papers, html_dir, pdf_dir, batch_size=10):
        """Process papers with immediate PDF fallback for failed HTML downloads."""
        successful_html = []
        successful_pdf = []
        failed = []

        # Create a single progress bar for all papers
        with tqdm(total=len(papers), desc="Processing papers (HTML→PDF fallback)") as pbar:
            for i in range(0, len(papers), batch_size):
                batch = papers[i : i + batch_size]
                
                for paper in batch:
                    paper_id = paper.get("id")
                    if not paper_id:
                        pbar.update(1)
                        continue

                    result, pid = self.download_with_fallback(paper_id, html_dir, pdf_dir)
                    
                    if result == "html":
                        successful_html.append(pid)
                    elif result == "pdf":
                        successful_pdf.append(pid)
                    elif result == "failed":
                        failed.append(pid)
                    # "skipped" doesn't get added to any list
                    
                    pbar.update(1)

                # After each mini-batch, check if we need to adjust our rate limiting
                if self.rate_limiter.consecutive_failures > 3:
                    sleep_time = min(300, self.rate_limiter.current_delay * 2)
                    logging.info(f"Too many failures, pausing for {sleep_time}s...")
                    time.sleep(sleep_time)

                # Save progress after each batch
                if successful_html:
                    self._update_json_file("successful_downloads.json", successful_html, "html")
                if successful_pdf:
                    self._update_json_file("successful_downloads.json", successful_pdf, "pdf")
                if failed:
                    self._update_json_file("failed_downloads.json", failed, "both")

        logging.info(f"\nDownload Summary:")
        logging.info(f"  HTML: {len(successful_html)} papers")
        logging.info(f"  PDF: {len(successful_pdf)} papers")
        logging.info(f"  Failed: {len(failed)} papers")
        
        return successful_html, successful_pdf, failed

    def _update_json_file(self, filename, new_items, format_key):
        """Update a JSON file with new items under a specific format key."""
        file_path = self.data_dir / filename
        data = {"html": [], "pdf": []}

        if file_path.exists():
            try:
                with open(file_path, "r") as f:
                    content = json.load(f)
                    if isinstance(content, dict):
                        data = content
            except Exception as e:
                logging.warning(f"Could not load {filename}: {e}")

        # Ensure keys exist
        if format_key not in data:
            data[format_key] = []

        # Merge
        existing = set(data[format_key])
        existing.update(new_items)
        data[format_key] = list(existing)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    def _save_progress(self, successful, failed, format="html"):
        """Save download progress to files."""
        # Ensure directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)

        if successful:
            self._update_json_file("successful_downloads.json", successful, format)
        if failed:
            self._update_json_file("failed_downloads.json", failed, format)


def download_arxiv_metadata():
    """Download the arXiv metadata JSON file using Kaggle API if not already present."""
    metadata_file = "arxiv-metadata-oai-snapshot.json"

    if os.path.exists(metadata_file):
        logging.info(f"Found existing metadata file: {metadata_file}")
        return metadata_file

    logging.info("Checking for Kaggle credentials (loading .env if present)...")
    try:
        from dotenv import load_dotenv

        load_dotenv()  # Load environment variables from .env file

        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
        if username and key:
            logging.info(
                f"Environment variables found: KAGGLE_USERNAME={username[:2]}***, KAGGLE_KEY=***"
            )
        else:
            logging.warning(
                "KAGGLE_USERNAME or KAGGLE_KEY not found in environment variables."
            )

        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        logging.info("Downloading arXiv metadata from Kaggle...")
        api.dataset_download_file(
            dataset="Cornell-University/arxiv", file_name=metadata_file, path="."
        )
        return metadata_file

    except Exception as e:
        error_msg = (
            f"Failed to download metadata from Kaggle: {e}\n"
            "Please ensure you have set up your Kaggle API credentials (kaggle.json).\n"
            "Alternatively, download 'arxiv-metadata-oai-snapshot.json' manually from\n"
            "https://www.kaggle.com/datasets/Cornell-University/arxiv and place it in this directory."
        )
        logging.error(error_msg)
        raise FileNotFoundError(error_msg)


def filter_lhcb_papers(metadata_file):
    """Filter papers containing 'lhcb' in title or abstract."""
    lhcb_papers = []
    total_papers = 0
    matched_papers = 0
    file_reader = None

    logging.info(f"Reading metadata file: {metadata_file}")

    try:
        # Handle both gzipped and regular files
        if metadata_file.endswith(".gz"):
            file_reader = gzip.open(metadata_file, "rt")
        else:
            file_reader = open(metadata_file, "r")

        with file_reader as f:
            for line in tqdm(f, desc="Filtering LHCb papers"):
                total_papers += 1
                try:
                    paper = json.loads(line)
                    title = paper.get("title", "").lower()
                    abstract = paper.get("abstract", "").lower()

                    if "lhcb" in title or "lhcb" in abstract:
                        lhcb_papers.append(paper)
                        matched_papers += 1

                        if matched_papers % 100 == 0:
                            logging.info(
                                f"Found {matched_papers} LHCb papers so far..."
                            )

                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON line: {str(e)}")
                    continue

        logging.info(f"Processed {total_papers} papers total")
        logging.info(f"Found {matched_papers} papers containing 'lhcb'")

        if len(lhcb_papers) == 0:
            logging.warning(
                "No LHCb papers found! This might indicate an issue with the metadata file."
            )

    except Exception as e:
        logging.error(f"Error reading metadata file: {str(e)}")
        raise

    return lhcb_papers


def load_paper_lists():
    """Load lists of successful and failed downloads if they exist."""
    successful = []
    failed = []

    if os.path.exists("successful_downloads.json"):
        with open("successful_downloads.json", "r") as f:
            successful = json.load(f)

    if os.path.exists("failed_downloads.json"):
        with open("failed_downloads.json", "r") as f:
            failed = json.load(f)

    return successful, failed


def save_filtered_metadata(papers, output_file="lhcb_papers.json"):
    """Save filtered papers to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(papers, f, indent=2)
    logging.info(f"Saved {len(papers)} LHCb papers to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Download and filter arXiv papers related to LHCb"
    )
    parser.add_argument(
        "--download-content",
        action="store_true",
        help="Download content (HTML w/ PDF fallback)",
    )
    parser.add_argument(
        "--initial-delay",
        type=float,
        default=5.0,
        help="Initial delay between requests (seconds)",
    )
    parser.add_argument(
        "--max-delay",
        type=float,
        default=300.0,
        help="Maximum delay between requests (seconds)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Size of mini-batches"
    )
    parser.add_argument(
        "--retry-failed", action="store_true", help="Retry previously failed downloads"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to store all output files",
    )
    args = parser.parse_args()

    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "arxiv_download.log"),
            logging.StreamHandler(),
        ],
    )

    # Download and process metadata
    metadata_file = download_arxiv_metadata()

    # Check if filtered metadata already exists
    lhcb_file = output_dir / "lhcb_papers.json"
    if lhcb_file.exists():
        logging.info("Loading existing filtered metadata...")
        with open(lhcb_file, "r") as f:
            lhcb_papers = json.load(f)
    else:
        logging.info("Filtering papers for LHCb content...")
        lhcb_papers = filter_lhcb_papers(metadata_file)
        save_filtered_metadata(lhcb_papers, str(lhcb_file))

    # Handle Content Download (Unified with immediate fallback)
    if args.download_content:
        output_dir_html = output_dir / "lhcb_html"
        output_dir_html.mkdir(exist_ok=True)
        output_dir_pdf = output_dir / "lhcb_pdfs"
        output_dir_pdf.mkdir(exist_ok=True)

        rate_limiter = AdaptiveRateLimiter(
            initial_delay=args.initial_delay, max_delay=args.max_delay
        )
        downloader = ArxivDownloader(rate_limiter=rate_limiter, data_dir=output_dir)

        logging.info(f"Downloading content for {len(lhcb_papers)} papers (HTML with immediate PDF fallback)")
        successful_html, successful_pdf, failed = downloader.process_with_fallback(
            lhcb_papers, output_dir_html, output_dir_pdf, batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()

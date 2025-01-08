# dataset.py
import json
import os
import argparse
import requests
import time
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import deque
import math
import gzip

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
            if len(self.success_history) == self.window_size and all(self.success_history):
                self.current_delay = max(5, self.current_delay * 0.8)
        else:
            self.consecutive_failures += 1
            # Exponential backoff on failures
            backoff_multiplier = min(math.pow(2, self.consecutive_failures), 10)
            self.current_delay = min(self.current_delay * backoff_multiplier, self.max_delay)
            logging.warning(f"Request failed. New delay: {self.current_delay}s")

class ArxivDownloader:
    def __init__(self, rate_limiter=None):
        self.rate_limiter = rate_limiter or AdaptiveRateLimiter()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; LHCbDatasetBuilder/1.0; mailto:arxiv@elashri.com)'
        })
        self.error_counts = {}
        self.removed_papers = []  # Track removed/404 papers

    def download_pdf(self, paper_id, output_dir):
        """Download a single PDF with adaptive rate limiting."""
        # Replace slashes with underscores in the filename to avoid folder creation
        safe_paper_id = paper_id.replace('/', '_')
        pdf_path = Path(output_dir) / f"{safe_paper_id}.pdf"
        
        # Skip if already downloaded successfully
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logging.info(f"Skipping {paper_id} - already downloaded")
            return True

        url = f"https://arxiv.org/pdf/{paper_id}.pdf"
        max_retries = 5
        
        for attempt in range(max_retries):
            # Wait according to rate limiter
            self.rate_limiter.wait()
            
            try:
                response = self.session.get(url, stream=True, timeout=30)
                
                # Handle 404 errors (removed/retracted papers)
                if response.status_code == 404:
                    logging.warning(f"Paper {paper_id} not found (404) - likely removed or retracted")
                    self.removed_papers.append(paper_id)
                    # Save removed papers list
                    with open('removed_papers.json', 'w') as f:
                        json.dump(self.removed_papers, f, indent=2)
                    return True  # Consider it "successful" to avoid retries
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = response.headers.get('Retry-After', 60)
                    self.rate_limiter.update(False, status_code=429, retry_after=retry_after)
                    logging.warning(f"Rate limited. Retry-After: {retry_after}s")
                    if attempt < max_retries - 1:
                        continue
                    return False
                
                response.raise_for_status()
                
                # Save the PDF
                with open(pdf_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verify download
                if pdf_path.stat().st_size > 0:
                    self.rate_limiter.update(True)
                    logging.info(f"Successfully downloaded {paper_id}")
                    return True
                else:
                    self.rate_limiter.update(False)
                    logging.error(f"Downloaded file is empty for {paper_id}")
                    pdf_path.unlink()
                    return False
            
            except requests.exceptions.RequestException as e:
                # Special handling for 404 errors that might be raised as exceptions
                if isinstance(e, requests.exceptions.HTTPError) and e.response.status_code == 404:
                    logging.warning(f"Paper {paper_id} not found (404) - likely removed or retracted")
                    self.removed_papers.append(paper_id)
                    # Save removed papers list
                    with open('removed_papers.json', 'w') as f:
                        json.dump(self.removed_papers, f, indent=2)
                    return True  # Consider it "successful" to avoid retries
                
                self.rate_limiter.update(False)
                logging.error(f"Attempt {attempt + 1} failed for {paper_id}: {str(e)}")
                if pdf_path.exists():
                    pdf_path.unlink()
                
                # Track error types for adaptive behavior
                error_type = type(e).__name__
                self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
                
                # If we're getting too many errors, increase the delay more aggressively
                if self.error_counts[error_type] > 5:
                    self.rate_limiter.current_delay = min(
                        self.rate_limiter.current_delay * 2,
                        self.rate_limiter.max_delay
                    )
                
                if attempt < max_retries - 1:
                    continue
                return False
                
            except Exception as e:
                self.rate_limiter.update(False)
                logging.error(f"Unexpected error for {paper_id}: {str(e)}")
                if pdf_path.exists():
                    pdf_path.unlink()
                return False
        
        return False
    
    def process_batch(self, papers, output_dir, batch_size=10):
        """Process a batch of papers with adaptive rate limiting."""
        successful = []
        failed = []
        
        for i in range(0, len(papers), batch_size):
            batch = papers[i:i + batch_size]
            logging.info(f"\nProcessing mini-batch {i//batch_size + 1}")
            
            for paper in tqdm(batch, desc="Processing papers"):
                paper_id = paper.get('id')
                if not paper_id:
                    continue
                
                if self.download_pdf(paper_id, output_dir):
                    successful.append(paper_id)
                else:
                    failed.append(paper_id)
            
            # After each mini-batch, check if we need to adjust our rate limiting
            if self.rate_limiter.consecutive_failures > 3:
                sleep_time = min(300, self.rate_limiter.current_delay * 2)
                logging.info(f"Too many failures, pausing for {sleep_time}s...")
                time.sleep(sleep_time)
            
            # Save progress after each batch
            self._save_progress(successful, failed)
        
        return successful, failed

    def _save_progress(self, successful, failed):
        """Save download progress to files."""
        with open('successful_downloads.json', 'w') as f:
            json.dump(successful, f, indent=2)
        
        with open('failed_downloads.json', 'w') as f:
            json.dump(failed, f, indent=2)

def download_arxiv_metadata():
    """Download the arXiv metadata JSON file using Kaggle API if not already present."""
    metadata_file = 'arxiv-metadata-oai-snapshot.json'
    
    if os.path.exists(metadata_file):
        logging.info(f"Found existing metadata file: {metadata_file}")
        return metadata_file
        
    logging.info("Authenticating with Kaggle...")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    logging.info("Downloading arXiv metadata from Kaggle...")
    api.dataset_download_file(
        dataset='Cornell-University/arxiv',
        file_name=metadata_file,
        path='.'
    )
    return metadata_file

def filter_lhcb_papers(metadata_file):
    """Filter papers containing 'lhcb' in title or abstract."""
    lhcb_papers = []
    total_papers = 0
    matched_papers = 0
    file_reader = None
    
    logging.info(f"Reading metadata file: {metadata_file}")
    
    try:
        # Handle both gzipped and regular files
        if metadata_file.endswith('.gz'):
            file_reader = gzip.open(metadata_file, 'rt')
        else:
            file_reader = open(metadata_file, 'r')
        
        with file_reader as f:
            for line in tqdm(f, desc="Filtering LHCb papers"):
                total_papers += 1
                try:
                    paper = json.loads(line)
                    title = paper.get('title', '').lower()
                    abstract = paper.get('abstract', '').lower()
                    
                    if 'lhcb' in title or 'lhcb' in abstract:
                        lhcb_papers.append(paper)
                        matched_papers += 1
                        
                        if matched_papers % 100 == 0:
                            logging.info(f"Found {matched_papers} LHCb papers so far...")
                            
                except json.JSONDecodeError as e:
                    logging.error(f"Error parsing JSON line: {str(e)}")
                    continue
                
        logging.info(f"Processed {total_papers} papers total")
        logging.info(f"Found {matched_papers} papers containing 'lhcb'")
        
        if len(lhcb_papers) == 0:
            logging.warning("No LHCb papers found! This might indicate an issue with the metadata file.")
            
    except Exception as e:
        logging.error(f"Error reading metadata file: {str(e)}")
        raise
    
    return lhcb_papers

def load_paper_lists():
    """Load lists of successful and failed downloads if they exist."""
    successful = []
    failed = []
    
    if os.path.exists('successful_downloads.json'):
        with open('successful_downloads.json', 'r') as f:
            successful = json.load(f)
    
    if os.path.exists('failed_downloads.json'):
        with open('failed_downloads.json', 'r') as f:
            failed = json.load(f)
    
    return successful, failed

def save_filtered_metadata(papers, output_file='lhcb_papers.json'):
    """Save filtered papers to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(papers, f, indent=2)
    logging.info(f"Saved {len(papers)} LHCb papers to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Download and filter arXiv papers related to LHCb')
    parser.add_argument('--download-pdfs', action='store_true', help='Download PDFs for filtered papers')
    parser.add_argument('--initial-delay', type=float, default=5.0, help='Initial delay between requests (seconds)')
    parser.add_argument('--max-delay', type=float, default=300.0, help='Maximum delay between requests (seconds)')
    parser.add_argument('--batch-size', type=int, default=10, help='Size of mini-batches')
    parser.add_argument('--retry-failed', action='store_true', help='Retry previously failed downloads')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('arxiv_download.log'),
            logging.StreamHandler()
        ]
    )

    # Download and process metadata
    metadata_file = download_arxiv_metadata()
    
    # Check if filtered metadata already exists
    if os.path.exists('lhcb_papers.json'):
        logging.info("Loading existing filtered metadata...")
        with open('lhcb_papers.json', 'r') as f:
            lhcb_papers = json.load(f)
    else:
        logging.info("Filtering papers for LHCb content...")
        lhcb_papers = filter_lhcb_papers(metadata_file)
        save_filtered_metadata(lhcb_papers)

    if args.download_pdfs:
        output_dir = Path('lhcb_pdfs')
        output_dir.mkdir(exist_ok=True)
        
        rate_limiter = AdaptiveRateLimiter(
            initial_delay=args.initial_delay,
            max_delay=args.max_delay
        )
        downloader = ArxivDownloader(rate_limiter=rate_limiter)
        
        # Load existing progress
        successful, failed = load_paper_lists()
        logging.info(f"Found {len(successful)} previously successful and {len(failed)} failed downloads")
        
        # Verify the content of lhcb_papers
        if not lhcb_papers:
            logging.error("No LHCb papers loaded from metadata! Rerunning filter...")
            lhcb_papers = filter_lhcb_papers(metadata_file)
            save_filtered_metadata(lhcb_papers)
        
        logging.info(f"Total LHCb papers available: {len(lhcb_papers)}")
        
        # Filter out already downloaded papers
        if args.retry_failed:
            if not failed:
                logging.info("No failed downloads to retry.")
                return
            remaining_papers = [paper for paper in lhcb_papers 
                              if paper.get('id') in failed]
            logging.info(f"Retrying {len(remaining_papers)} previously failed downloads")
        else:
            remaining_papers = [paper for paper in lhcb_papers 
                              if paper.get('id') not in successful]
            logging.info(f"Found {len(remaining_papers)} new papers to download")
        
        logging.info(f"Attempting to download {len(remaining_papers)} papers")
        
        try:
            new_successful, new_failed = downloader.process_batch(
                remaining_papers, 
                output_dir, 
                batch_size=args.batch_size
            )
            
            # Update the full lists
            successful.extend(new_successful)
            failed = list(set(failed + new_failed))  # Remove duplicates
            
            # Save final progress
            downloader._save_progress(successful, failed)
            
            logging.info(f"\nDownload complete:")
            logging.info(f"Total successful: {len(successful)}")
            logging.info(f"Total failed: {len(failed)}")
            
        except KeyboardInterrupt:
            logging.info("\nDownload interrupted. Progress has been saved.")
            logging.info("You can resume by running the script again.")

if __name__ == "__main__":
    main()
import json
import os
from pathlib import Path
import logging
from dataset import ArxivDownloader, AdaptiveRateLimiter
import re

def load_metadata(output_dir='output', metadata_file='lhcb_papers.json'):
    """Load the LHCb papers metadata."""
    metadata_path = Path(output_dir) / metadata_file
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Metadata file {metadata_path} not found!")
        return []

def get_downloaded_pdfs(output_dir='output', pdf_subdir='lhcb_pdfs'):
    """Get list of downloaded PDFs, handling different ID formats."""
    pdf_dir = Path(output_dir) / pdf_subdir
    if not pdf_dir.exists():
        return set()
    
    # Get all PDF files
    pdfs = set()
    for pdf in pdf_dir.glob('*.pdf'):
        # Convert filename back to arxiv ID format
        paper_id = pdf.stem.replace('_', '/')
        pdfs.add(paper_id)
    
    return pdfs

def parse_log_for_404s(output_dir='output', log_file='arxiv_download.log'):
    """Parse the log file to find papers that returned 404."""
    removed_papers = set()
    log_path = Path(output_dir) / log_file
    
    try:
        with open(log_path, 'r') as f:
            for line in f:
                if '404' in line and 'likely removed or retracted' in line:
                    # Extract paper ID using regex
                    match = re.search(r'Paper ([\w.-]+/?\d+) not found', line)
                    if match:
                        removed_papers.add(match.group(1))
    except FileNotFoundError:
        logging.warning(f"Log file {log_path} not found!")
    
    return removed_papers

def check_removed_papers_json(output_dir='output', json_file='removed_papers.json'):
    """Check the removed_papers.json file if it exists."""
    json_path = Path(output_dir) / json_file
    try:
        with open(json_path, 'r') as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def download_missing_papers(missing_papers, output_base_dir='output'):
    """Download missing papers using ArxivDownloader with immediate PDF fallback."""
    # Create paper objects in the format expected by ArxivDownloader
    papers_to_download = [{'id': paper_id} for paper_id in missing_papers]
    
    # Initialize the downloader
    rate_limiter = AdaptiveRateLimiter(initial_delay=5.0, max_delay=300.0)
    output_base = Path(output_base_dir)
    downloader = ArxivDownloader(rate_limiter=rate_limiter, data_dir=output_base)
    
    # Ensure output directories exist under output_base_dir
    html_dir = output_base / 'lhcb_html'
    pdf_dir = output_base / 'lhcb_pdfs'
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAttempting to download {len(papers_to_download)} missing papers (HTML with PDF fallback)...")
    
    # Process the downloads with immediate fallback
    successful_html, successful_pdf, failed = downloader.process_with_fallback(
        papers_to_download,
        html_dir,
        pdf_dir,
        batch_size=10
    )
    
    return successful_html + successful_pdf, failed

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Default output directory
    output_dir = 'output'

    # Load metadata
    papers = load_metadata(output_dir=output_dir)
    if not papers:
        return
    
    total_papers = len(papers)
    paper_ids = {paper['id'] for paper in papers}
    
    # Get downloaded PDFs
    downloaded_pdfs = get_downloaded_pdfs(output_dir=output_dir)
    
    # Get removed papers from both log and json
    removed_papers_log = parse_log_for_404s(output_dir=output_dir)
    removed_papers_json = check_removed_papers_json(output_dir=output_dir)
    removed_papers = removed_papers_log | removed_papers_json  # Union of both sets
    
    # Find missing papers (excluding removed ones)
    missing_papers = paper_ids - downloaded_pdfs - removed_papers
    
    # Print statistics
    print("\nDataset Statistics:")
    print(f"Total papers in metadata: {total_papers}")
    print(f"Total PDFs downloaded: {len(downloaded_pdfs)}")
    print(f"Total removed/retracted papers: {len(removed_papers)}")
    print(f"Missing papers: {len(missing_papers)}")
    
    # Calculate and explain any discrepancy
    expected_total = len(downloaded_pdfs) + len(removed_papers) + len(missing_papers)
    if expected_total != total_papers:
        print(f"\nWARNING: Discrepancy detected!")
        print(f"Sum of (downloaded + removed + missing) = {expected_total}")
        print(f"This differs from total in metadata ({total_papers})")
        
    # If there are missing papers, offer to download them
    if missing_papers:
        print("\nMissing papers:")
        for paper_id in sorted(missing_papers):
            print(f"  {paper_id}")
        
        response = input("\nWould you like to download missing papers now? (y/n): ")
        if response.lower() == 'y':
            successful, failed = download_missing_papers(missing_papers, output_base_dir=output_dir)
            
            print("\nDownload Results:")
            print(f"Successfully downloaded: {len(successful)}")
            print(f"Failed downloads: {len(failed)}")
            
            if failed:
                print("\nFailed papers:")
                for paper_id in failed:
                    print(f"  {paper_id}")

if __name__ == "__main__":
    main()
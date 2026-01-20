"""
Canonical LHCb paper filtering functions.

This module provides a single source of truth for filtering LHCb-related papers
from arXiv metadata or Paper objects.
"""
import json
import gzip
import logging
from tqdm import tqdm


def is_lhcb_related(title: str, abstract: str) -> bool:
    """
    Check if a paper is LHCb-related based on title and abstract.
    
    Args:
        title: Paper title (case-insensitive)
        abstract: Paper abstract (case-insensitive)
        
    Returns:
        True if "lhcb" appears in title or abstract
    """
    title_lower = title.lower() if title else ""
    abstract_lower = abstract.lower() if abstract else ""
    return "lhcb" in title_lower or "lhcb" in abstract_lower


def filter_lhcb_papers_from_file(metadata_file: str) -> list:
    """
    Filter LHCb papers directly from arXiv metadata JSON file.
    
    This function scans the metadata file line-by-line to avoid loading
    the entire dataset into memory.
    
    Args:
        metadata_file: Path to arXiv metadata JSON file (supports .gz)
        
    Returns:
        List of paper dictionaries that match LHCb criteria
    """
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
                    title = paper.get("title", "")
                    abstract = paper.get("abstract", "")

                    if is_lhcb_related(title, abstract):
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


def filter_lhcb_papers_from_objects(papers: list) -> list:
    """
    Filter LHCb papers from a list of Paper objects.
    
    Args:
        papers: List of Paper objects with is_lhcb_related() method
        
    Returns:
        List of Paper objects that match LHCb criteria
    """
    print("\nFiltering LHCb papers...")
    lhcb_papers = []

    for paper in tqdm(papers, desc="Filtering papers"):
        if paper.is_lhcb_related():
            lhcb_papers.append(paper)

    print(f"Found {len(lhcb_papers)} LHCb papers out of {len(papers)} total papers")
    return lhcb_papers

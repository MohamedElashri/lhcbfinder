# helpers.py
import json
from tqdm import tqdm
from paper import Paper
from pathlib import Path
import gzip


def count_lines(file_path):
    """Count total lines in a file for progress bar."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_data(
    json_file_path, html_dir=None, pdf_dir=None, include_content=False, start_year=None
):
    """
    Load arXiv papers from JSON file and filter them.

    Args:
        json_file_path: Path to JSON file containing arXiv papers
        html_dir: Optional directory path containing HTML files
        pdf_dir: Optional directory path containing PDF files (Legacy)
        include_content: Whether to include full content in embeddings (HTML or PDF)
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
    with (
        gzip.open(json_file_path, "rt", encoding="utf-8")
        if json_file_path.endswith(".gz")
        else open(json_file_path, "r", encoding="utf-8") as f
    ):
        for i, line in enumerate(tqdm(f, desc="Scanning papers", unit="papers")):
            try:
                data_dict = json.loads(line)


            # The following "lhcb" papers filtering logic is probably is the most stupid logic here 
            # But it's done this way to avoid loading all papers into memory first, which is infeasible for large datasets.
            # And it surprisingly works well enough for our use case.  
            # TODO: Any improvements are welcome.   
                # Check if paper is related to LHCb
                title = data_dict.get("title", "").lower()
                abstract = data_dict.get("abstract", "").lower()
                paper_id = data_dict.get("id", "")

                # Quick filtering for LHCb papers
                if "lhcb" in title or "lhcb" in abstract:
                    # Apply year filter if specified
                    if start_year and "update_date" in data_dict:
                        year = int(data_dict["update_date"].split("-")[0])
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
    with (
        gzip.open(json_file_path, "rt", encoding="utf-8")
        if json_file_path.endswith(".gz")
        else open(json_file_path, "r", encoding="utf-8") as f
    ):
        for i, line in enumerate(tqdm(f, desc="Loading papers", unit=" papers")):
            try:
                data_dict = json.loads(line)
                paper_id = data_dict.get("id", "")

                # Only process papers in our filtered set
                if paper_id in lhcb_ids:
                    # Apply year filter (redundant check, but kept for safety)
                    if start_year and "update_date" in data_dict:
                        year = int(data_dict["update_date"].split("-")[0])
                        if year < int(start_year):
                            continue

                    paper = Paper(
                        data_dict,
                        html_dir=html_dir,
                        pdf_dir=pdf_dir,
                        include_content=include_content,
                    )
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
        if paper.is_lhcb_related():
            lhcb_papers.append(paper)

    print(f"Found {len(lhcb_papers)} LHCb papers out of {len(papers)} total papers")
    return lhcb_papers

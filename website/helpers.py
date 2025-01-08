# helpers.py
import json
from paper import Paper
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from diskcache import Cache
import validators

# Initialize file-based cache for arXiv abstracts
cache = Cache("/tmp/arxiv_cache")

def fetch_abstract(url):
    """
    Fetches abstract from arXiv URL using requests and BeautifulSoup with caching.
    """
    if url in cache:
        return cache[url]
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        # Find the abstract block
        abstract_block = soup.find('blockquote', class_='abstract')
        if abstract_block:
            # Remove the "Abstract: " prefix if it exists
            abstract_text = abstract_block.get_text().strip()
            if abstract_text.lower().startswith('abstract:'):
                abstract_text = abstract_text[9:].strip()
            cache[url] = abstract_text  # Cache the abstract
            return abstract_text
        else:
            raise ValueError("Abstract not found on page")
    except Exception as e:
        print(f"Error fetching abstract: {e}")
        raise ValueError("Could not fetch abstract from arXiv")

def avg_score(papers):
    avg_score = sum([p.score for p in papers]) / len(papers)
    return round(avg_score, 2)

def get_matches(index, k, vector=None, id=None, exclude=None, per_page=10, page=1):
    assert vector is not None or id is not None
    if vector is not None:
        top_k = index.query(vector=vector, top_k=k, include_metadata=True)
    else:
        top_k = index.query(id=id, top_k=k, include_metadata=True)
    
    matches = top_k["matches"]
    papers = [Paper(match) for match in matches if match["id"] != exclude]
    total_results = min(len(papers), 50)  # Cap total results at 50
    papers = papers[:total_results]  # Limit to first 50 papers
    
    # Calculate pagination
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_papers = papers[start_idx:end_idx]
    
    # Calculate total pages based on total_results and per_page
    total_pages = (total_results + per_page - 1) // per_page
    
    # Get authors for the current page
    authors = get_authors(paginated_papers)
    
    # Convert papers to dict for JSON serialization
    papers_dict = [paper.__dict__ for paper in paginated_papers]
    
    return json.dumps({
        "papers": papers_dict,
        "authors": authors,
        "pagination": {
            "current_page": page,
            "total_pages": total_pages,
            "total_results": total_results,
            "per_page": per_page
        }
    })
    
def get_authors(papers):
    authors = defaultdict(list)
    for paper in papers:
        for author in paper.authors_parsed:
            authors[author].append(paper)
    authors = [{"author": author,
                "papers": [paper.__dict__ for paper in papers],
                "avg_score": avg_score(papers)}
               for author, papers in authors.items()]
    authors = sorted(authors, key=lambda e: e["avg_score"], reverse=True)
    authors = sorted(authors, key=lambda e: len(e["papers"]), reverse=True)
    return authors[:10]

def error(msg):
    return json.dumps({"error": msg})

def parse_arxiv_identifier(query):
    """Parse different forms of arXiv identifiers."""
    # Remove any whitespace
    query = query.strip()
    
    # Full URL pattern
    if validators.url(query):
        return query.split("/")[-1]
    
    # Just the ID pattern (e.g., 2409.03496)
    if len(query.split(".")) == 2:
        try:
            year, number = query.split(".")
            if len(year) == 4 and year.isdigit() and number.isdigit():
                return query
        except ValueError:
            pass
    
    return None


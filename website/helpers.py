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

def get_authors(papers):
    authors = defaultdict(list)
    for paper in papers:
        for author in paper.authors_parsed:
            authors[author].append(paper)

    # Convert Paper objects to dictionaries in the 'papers' list
    authors_dict = {
        author: {
            "papers": [paper.__dict__ for paper in papers],  # Convert to dict
            "avg_score": avg_score(papers)
        }
        for author, papers in authors.items()
    }

    authors = [{"author": author, **data} for author, data in authors_dict.items()]
    authors = sorted(authors, key=lambda e: e["avg_score"], reverse=True)
    authors = sorted(authors, key=lambda e: len(e["papers"]), reverse=True)
    return authors[:10]

def error(msg):
    return json.dumps({"error": msg})

def parse_arxiv_identifier(query: str) -> str | None:
    """
    Parse different forms of arXiv identifiers.
    
    Handles:
    - Full URLs: https://arxiv.org/abs/2511.02619v1
    - IDs with version: 2511.02619v1
    - IDs without version: 2511.02619
    
    Args:
        query: The input string to parse
        
    Returns:
        The arXiv ID if valid, None otherwise
    """
    import re
    
    # Remove any whitespace
    query = query.strip()
    
    # Full URL pattern - extract the ID from the URL
    if validators.url(query):
        arxiv_id = query.split("/")[-1]
    else:
        arxiv_id = query
    
    # arXiv ID pattern: YYMM.NNNNN or YYMM.NNNNNvN (with optional version)
    # Modern format (post-2007): YYMM.NNNNN[vN]
    pattern = r'^(\d{4})\.(\d{4,5})(v\d+)?$'
    match = re.match(pattern, arxiv_id)
    
    if match:
        year, number, version = match.groups()
        # Validate year is reasonable (arXiv started in 1991, format YYMM)
        year_num = int(year[:2])
        month_num = int(year[2:])
        if 91 <= year_num <= 99 or 0 <= year_num <= 50:  # 1991-1999 or 2000-2050
            if 1 <= month_num <= 12:  # Valid month
                return arxiv_id  # Return the full ID including version if present
    
    return None

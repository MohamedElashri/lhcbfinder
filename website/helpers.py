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

def get_matches_initial(index, k, vector=None, id=None, exclude=None, min_score=0.0):
    assert vector is not None or id is not None

    if vector is not None:
        top_k = index.query(vector=vector, top_k=k, include_metadata=True)
    else:
        top_k = index.query(id=id, top_k=k, include_metadata=True)

    matches = top_k["matches"]
    
    # Filter matches by minimum similarity score
    matches = [match for match in matches if match["score"] >= min_score and match["id"] != exclude]
    
    # Convert to Paper objects and limit results
    papers = [Paper(match) for match in matches]
    total_results = min(len(papers), 50)
    papers = papers[:total_results]

    # Convert papers to dict for JSON serialization
    papers_dict = [paper.__dict__ for paper in papers]

    return json.dumps({
        "papers": papers_dict,
        "total_results": total_results
    })
    
            
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

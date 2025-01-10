import json
from paper import Paper
import requests
from bs4 import BeautifulSoup
from collections import defaultdict
from diskcache import Cache
import validators
from vector_store import create_vector_store
import os

# Initialize file-based cache for arXiv abstracts
cache = Cache("/tmp/arxiv_cache")

def get_vector_store():
    """Get configured vector store instance"""
    use_pinecone = os.getenv("USE_PINECONE", "true").lower() == "true"
    use_qdrant = os.getenv("USE_QDRANT", "false").lower() == "true"
    pinecone_index = os.getenv("PINECONE_INDEX_NAME")
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "lhcb_papers")
    
    return create_vector_store(
        use_pinecone=use_pinecone,
        use_qdrant=use_qdrant,
        pinecone_index=pinecone_index,
        qdrant_collection=qdrant_collection
    )

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

def get_matches_initial(store, k, vector=None, id=None, exclude=None):
    """
    Get matches from vector store with adapter for different backends
    """
    assert vector is not None or id is not None
    
    # Adapt the query based on store type
    if hasattr(store, 'query'):  # Pinecone style
        if vector is not None:
            top_k = store.query(vector=vector, top_k=k, include_metadata=True)
        else:
            top_k = store.query(id=id, top_k=k, include_metadata=True)
        matches = top_k["matches"]
    else:  # Qdrant style
        if vector is not None:
            matches = store.search(
                collection_name=store.collection_name,
                query_vector=vector,
                limit=k
            )
        else:
            # Adapt for Qdrant's ID-based search
            matches = store.retrieve(
                collection_name=store.collection_name,
                ids=[id_hash(id)]
            )
    
    # Convert matches to common format
    papers = [Paper(match) for match in matches if match["id"] != exclude]
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
            "papers": [paper.__dict__ for paper in papers],
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

def id_hash(id_str: str) -> int:
    """Convert string ID to uint64 for Qdrant"""
    return hash(id_str) & 0xFFFFFFFFFFFFFFFF
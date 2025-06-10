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

    # Ensure we return exactly k unique papers (not chunks)
    max_papers = k  # Target number of unique papers
    unique_papers = {}
    
    # Start with a reasonable batch size to avoid too many API calls
    initial_fetch = max(max_papers * 3, 50)  # Start with 3x or at least 50
    max_fetch = 1000  # Maximum to fetch in a single query
    
    # Parameters for pagination if needed
    fetch_size = initial_fetch
    last_score = None
    paper_ids_seen = set()  # Track paper IDs we've already seen
    
    # Keep fetching until we have enough unique papers or no more results
    while len(unique_papers) < max_papers:
        # Prepare query parameters
        if last_score is not None:
            # For pagination: filter scores less than the last one we've seen
            filter = {"score": {"$lt": last_score}}
        else:
            filter = None
            
        # Execute query
        if vector is not None:
            results = index.query(vector=vector, top_k=fetch_size, include_metadata=True, filter=filter)
        else:
            results = index.query(id=id, top_k=fetch_size, include_metadata=True, filter=filter)
            
        matches = results["matches"]
        
        # Break if no more results
        if not matches:
            break
            
        # Update last score for pagination
        last_score = matches[-1]["score"]
        
        # Filter by minimum score and excluded ID
        matches = [match for match in matches if match["score"] >= min_score and match["id"] != exclude]
        
        # Process matches and deduplicate
        for match in matches:
            # Extract real paper ID (removing chunk suffix if present)
            paper_id = match["id"]
            if "_chunk_" in paper_id:
                parent_id = paper_id.split("_chunk_")[0]
            else:
                parent_id = paper_id
                
            # Skip if we've already seen this paper
            if parent_id in paper_ids_seen:
                continue
                
            paper_ids_seen.add(parent_id)
            
            # Keep the highest scoring match for each paper
            if parent_id not in unique_papers or match["score"] > unique_papers[parent_id]["score"]:
                # Store highest scoring match for this paper
                unique_papers[parent_id] = match
                
                # If this is a chunk, make sure metadata is properly set
                if "_chunk_" in paper_id:
                    if "parent_id" not in match["metadata"]:
                        match["metadata"]["parent_id"] = parent_id
                    if "is_chunk" not in match["metadata"]:
                        match["metadata"]["is_chunk"] = True
                        
            # Break early if we have enough unique papers
            if len(unique_papers) >= max_papers:
                break
                
        # If we didn't get enough papers but processed all available matches,
        # increase fetch size for next iteration (up to max_fetch)
        if len(unique_papers) < max_papers and len(matches) < fetch_size:
            # We've exhausted results at this fetch size
            if fetch_size >= max_fetch:
                # We've hit our maximum fetch limit, so no point in continuing
                break
            else:
                # Double the fetch size for next iteration
                fetch_size = min(fetch_size * 2, max_fetch)
    
    # Sort papers by score (highest first)
    sorted_papers = sorted(unique_papers.values(), key=lambda p: p["score"], reverse=True)
    
    # Convert to Paper objects
    papers = [Paper(match) for match in sorted_papers[:max_papers]]
    total_results = len(papers)
    
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
    """Parse different forms of arXiv identifiers.

    The original implementation only handled bare identifiers like
    ``2409.03496`` or URLs pointing to the arXiv page.  It failed when a
    version suffix (e.g. ``v1``) or the ``arXiv:`` prefix was present.
    This caused valid identifiers such as ``arXiv:2409.03496v1`` or
    ``https://arxiv.org/abs/2409.03496v2`` to be treated as free text
    queries.  The search endpoint would then incorrectly embed these
    strings instead of fetching the paper directly.

    The new logic normalises common forms of arXiv IDs by stripping
    prefixes, handling version components and supporting the older
    ``category/number`` notation.
    """

    # Remove whitespace and ``arXiv:`` prefix
    query = query.strip()
    if query.lower().startswith("arxiv:"):
        query = query[6:]

    # Extract ID from URL forms
    if validators.url(query):
        query = query.split("/")[-1]
        if query.lower().endswith(".pdf"):
            query = query[:-4]

    # Drop optional version suffix (e.g. v1, v2)
    query = re.sub(r'v\d+$', '', query)

    # Old style IDs with category prefix (e.g. hep-ex/9909055)
    if "/" in query:
        prefix, number = query.split("/", 1)
        if prefix and number and number.isdigit():
            return f"{prefix}/{number}"

    # New style IDs ``YYYY.NNNNN``
    parts = query.split(".")
    if len(parts) == 2:
        year, number = parts
        if len(year) == 4 and year.isdigit() and number.isdigit():
            return f"{year}.{number}"

    return None

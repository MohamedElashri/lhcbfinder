import flask
import os
import chromadb
import time
from flask import render_template, request
from sentence_transformers import SentenceTransformer
from helpers import error
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import lru_cache
import hashlib

from query_processor import QueryProcessor

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)

# Initialize rate limiter with configurable storage
if os.getenv("FLASK_ENV") == "development":
    # Use in-memory storage for local development
    limiter = Limiter(app=app, key_func=get_remote_address, storage_uri="memory://")
else:
    # Use Redis storage for production
    redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
    try:
        limiter = Limiter(app=app, key_func=get_remote_address, storage_uri=redis_url)
    except Exception as e:
        print(f"Warning: Redis connection failed, falling back to memory storage: {e}")
        limiter = Limiter(app=app, key_func=get_remote_address, storage_uri="memory://")


# Rate limit configurations
@limiter.limit("1/30seconds", error_message="Too many requests. Slow down!")
@limiter.limit("5/3minutes", exempt_when=lambda: False, deduct_when=lambda: True)
@limiter.request_filter
def exempt_limits():
    return False


@app.errorhandler(429)
def ratelimit_error(e):
    return {"error": "Too many requests. Try again later."}, 429


# Initialize sentence transformer model and query processor
MODEL_NAME = "BAAI/bge-large-en-v1.5"
print(f"Loading embedding model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)
# Optimize for inference speed
model.max_seq_length = 512  # Limit sequence length for faster encoding
print("✅ Embedding model loaded")
query_processor = QueryProcessor()

# Simple in-memory cache for search results (LRU with max 100 queries)
@lru_cache(maxsize=100)
def get_query_embedding(query_text: str):
    """Cache query embeddings to avoid recomputing for same queries."""
    return tuple(model.encode([query_text], show_progress_bar=False, normalize_embeddings=True)[0].tolist())


def get_chroma_db_path():
    """
    Professional path resolution for ChromaDB with fallback strategy.
    Supports both development and production environments.
    """
    # Priority 1: Explicit environment variable
    env_path = os.environ.get("CHROMA_DB_PATH")
    if env_path:
        # Expand user home directory if present
        env_path = os.path.expanduser(env_path)
        # Convert to absolute path if relative
        if not os.path.isabs(env_path):
            env_path = os.path.abspath(env_path)
        return env_path
    
    # Priority 2: Check if running in Docker (common production pattern)
    docker_path = "/app/chroma_db"
    if os.path.exists(docker_path):
        return docker_path
    
    # Priority 3: Development fallback - relative to project root
    # Try to find the model/output/chroma_db directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Check parent directory (if website is a subdirectory)
    parent_model_path = os.path.join(current_dir, "..", "model", "output", "chroma_db")
    if os.path.exists(parent_model_path):
        return os.path.abspath(parent_model_path)
    
    # Check same level
    sibling_model_path = os.path.join(current_dir, "model", "output", "chroma_db")
    if os.path.exists(sibling_model_path):
        return os.path.abspath(sibling_model_path)
    
    # Priority 4: Default to environment variable or relative path
    # This allows the app to start even if DB doesn't exist yet (for initial setup)
    default_path = os.path.join(current_dir, "..", "model", "output", "chroma_db")
    return os.path.abspath(default_path)


# Initialize ChromaDB client
try:
    chroma_path = get_chroma_db_path()
    print(f"ChromaDB path: {chroma_path}")
    
    if not os.path.exists(chroma_path):
        raise FileNotFoundError(
            f"ChromaDB path does not exist: {chroma_path}\n"
            f"Please ensure the database is created or set CHROMA_DB_PATH environment variable."
        )
    
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    print("✅ ChromaDB client initialized successfully")
except Exception as e:
    print(f"❌ Error initializing ChromaDB: {e}")
    chroma_client = None


def get_collections():
    """Get or create the LHCb paper collections with proper distance metric."""
    if chroma_client is None:
        raise RuntimeError("ChromaDB client not initialized.")
    # Use cosine similarity for optimal performance with BAAI embeddings
    abstracts = chroma_client.get_or_create_collection(
        name="lhcb_abstracts",
        metadata={"hnsw:space": "cosine"}
    )
    contents = chroma_client.get_or_create_collection(
        name="lhcb_contents",
        metadata={"hnsw:space": "cosine"}
    )
    return abstracts, contents


@app.route("/")
def home():
    app.logger.info("Request received for home route")
    try:
        return render_template("index.html")
    except Exception:
        app.logger.error("Error rendering home template")
        return "Internal Server Error", 500


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/search")
def search():
    search_start = time.time()
    query = request.args.get("query")
    include_content = request.args.get("include_content", "0") == "1"
    
    if not query:
        return error("Query cannot be empty.")

    # Get search limits from environment variables
    ABSTRACT_K = int(os.environ.get("SEARCH_ABSTRACT_LIMIT", "20"))
    CONTENT_K = int(os.environ.get("SEARCH_CONTENT_LIMIT", "100"))
    MAX_RESULTS = int(os.environ.get("SEARCH_MAX_RESULTS", "20"))
    
    abstract_collection, content_collection = get_collections()

    # 1. Generate query embedding (with caching)
    t1 = time.time()
    clean_query = query_processor.clean_query(query)
    try:
        query_embed = list(get_query_embedding(clean_query))
    except Exception as e:
        print(f"Embedding error: {e}")
        return error("Error processing query.")
    print(f"⏱️  Embedding: {time.time() - t1:.2f}s")

    # 2. Query abstracts collection
    try:
        t2 = time.time()
        abs_results = abstract_collection.query(
            query_embeddings=[query_embed],
            n_results=ABSTRACT_K,
            include=["metadatas", "distances", "documents"],  # Include documents to extract abstracts
        )
        print(f"⏱️  Abstract query: {time.time() - t2:.2f}s")

        # 3. Process abstract results
        scores = {}  # paper_id -> weighted_score
        paper_metadata = {}  # paper_id -> metadata
        abstract_paper_ids = set()  # Track which papers were found

        if abs_results["ids"] and abs_results["ids"][0]:
            ids = abs_results["ids"][0]
            dists = abs_results["distances"][0]
            metas = abs_results["metadatas"][0]
            docs = abs_results.get("documents", [[]])[0]  # Get stored documents

            for idx, (pid, dist, meta) in enumerate(zip(ids, dists, metas)):
                # ChromaDB with L2/cosine distance
                sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                
                # Extract abstract from stored document if not in metadata
                if "abstract" not in meta or not meta.get("abstract"):
                    if idx < len(docs) and docs[idx]:
                        # Document format: "Title: ... Year: ... Abstract: ..."
                        doc = docs[idx]
                        if "Abstract: " in doc:
                            abstract = doc.split("Abstract: ", 1)[1]
                            meta["abstract"] = abstract
                        else:
                            meta["abstract"] = ""
                    else:
                        meta["abstract"] = ""
                
                # Ensure month field exists
                if "month" not in meta or not meta.get("month"):
                    if "date" in meta:
                        # Extract month from date (format: YYYY-MM)
                        try:
                            date_parts = meta["date"].split("-")
                            if len(date_parts) >= 2:
                                meta["month"] = date_parts[1]
                            else:
                                meta["month"] = "01"
                        except:
                            meta["month"] = "01"
                    else:
                        meta["month"] = "01"
                
                # If content search is disabled, abstract is 100% of score
                # If enabled, abstract is 60% of score
                weight = 1.0 if not include_content else 0.6
                weighted_score = sim * weight
                scores[pid] = weighted_score
                paper_metadata[pid] = meta
                abstract_paper_ids.add(pid)

        # Step 2: Optional content search (only for papers found in abstracts)
        if include_content and abstract_paper_ids:
            t3 = time.time()
            print(f"🔍 Smart content search: filtering {len(abstract_paper_ids)} papers from 86K chunks")
            
            # Create a filter to only search chunks from papers in abstract results
            # This dramatically reduces search space from 86K to ~500-1000 chunks
            paper_best_chunk_score = {}
            
            # Query content but we'll filter results to only matching papers
            # We fetch more results to ensure we get chunks from our target papers
            content_results = content_collection.query(
                query_embeddings=[query_embed],
                n_results=CONTENT_K,
                include=["metadatas", "distances"],
            )
            print(f"⏱️  Content query: {time.time() - t3:.2f}s")

            if content_results["ids"] and content_results["ids"][0]:
                ids = content_results["ids"][0]
                dists = content_results["distances"][0]
                metas = content_results["metadatas"][0]

                for cid, dist, meta in zip(ids, dists, metas):
                    parent_id = meta.get("parent_id")
                    if not parent_id:
                        continue
                    
                    # SMART FILTER: Only process chunks from papers found in abstract search
                    if parent_id not in abstract_paper_ids:
                        continue

                    sim = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                    if (
                        parent_id not in paper_best_chunk_score
                        or sim > paper_best_chunk_score[parent_id]
                    ):
                        paper_best_chunk_score[parent_id] = sim

                # Add content scores (40% weight)
                for pid, best_sim in paper_best_chunk_score.items():
                    weighted_score = best_sim * 0.4
                    scores[pid] = scores.get(pid, 0) + weighted_score
                    
                    # NOTE: No need to store metadata from chunks
                    # Content chunks have minimal metadata (only parent_id, chunk_index)
                    # Full metadata already stored from abstract collection

        # 4. Sort and Format
        t4 = time.time()
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Format for JSON response - Return top N results based on config
        # Flatten metadata into paper object for frontend compatibility
        papers = []
        for pid in sorted_ids[:MAX_RESULTS]:
            meta = paper_metadata.get(pid, {})
            
            # FALLBACK: If abstract missing (old embeddings), fetch from ChromaDB documents
            abstract_text = meta.get("abstract", "")
            if not abstract_text:
                try:
                    # Get the document text which contains abstract
                    doc_result = abstract_collection.get(ids=[pid], include=["documents"])
                    if doc_result and doc_result["documents"]:
                        # Document format: "Title. Abstract"
                        doc_text = doc_result["documents"][0]
                        # Extract abstract (everything after first sentence/title)
                        if ". " in doc_text:
                            abstract_text = doc_text.split(". ", 1)[1]
                        else:
                            abstract_text = doc_text
                except Exception as e:
                    print(f"⚠️  Could not fetch abstract for {pid}: {e}")
                    abstract_text = "Abstract not available"
            
            # Flatten the structure - merge metadata fields with id and score
            paper_dict = {
                "id": pid,
                "score": round(scores[pid], 2),
                # Extract all metadata fields
                "title": meta.get("title", ""),
                "authors": meta.get("authors", ""),
                "abstract": abstract_text,
                "year": meta.get("year", ""),
                "month": meta.get("month", ""),
                "has_pdf_content": meta.get("has_pdf_content", False),
                "pdf_preview": meta.get("pdf_preview", ""),
                "pdf_content": meta.get("pdf_content", ""),
                "chunks": meta.get("chunks", []),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
            }
            papers.append(paper_dict)
            
        print(f"⏱️  Processing: {time.time() - t4:.2f}s")
        print(f"⏱️  TOTAL: {time.time() - search_start:.2f}s (content={'ON' if include_content else 'OFF'})")
        print(f"📊 Returning {len(papers)} results")

        # Return JSON response for AJAX
        return {
            "papers": papers,
            "total_results": len(papers)
        }

    except Exception as e:
        print(f"Search error: {e}")
        return error("An error occurred during search.")


@app.route("/robots.txt")
def robots():
    with open("static/robots.txt", "r") as f:
        content = f.read()
    return content


@app.route("/health")
def health():
    return {"status": "healthy"}, 200


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)

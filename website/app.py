import flask
import os
import chromadb
from flask import render_template, request
from sentence_transformers import SentenceTransformer
from helpers import error
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

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
model = SentenceTransformer(MODEL_NAME)
query_processor = QueryProcessor()


# Initialize ChromaDB client
try:
    chroma_client = chromadb.PersistentClient(
        path=os.environ.get("CHROMA_DB_PATH", "model/chroma_db")
    )
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    chroma_client = None


def get_collections():
    if chroma_client is None:
        raise RuntimeError("ChromaDB client not initialized.")
    abstracts = chroma_client.get_or_create_collection(name="lhcb_abstracts")
    contents = chroma_client.get_or_create_collection(name="lhcb_contents")
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
    query = request.args.get("query")
    if not query:
        return error("Query cannot be empty.")

    K = 50  # Results to fetch per collection
    abstract_collection, content_collection = get_collections()

    # 1. Generate query embedding
    clean_query = query_processor.clean_query(query)
    try:
        query_embed = model.encode([clean_query])[0].tolist()
    except Exception as e:
        print(f"Embedding error: {e}")
        return error("Error processing query.")

    # 2. Query both collections
    try:
        # Abstract search (60% weight)
        abs_results = abstract_collection.query(
            query_embeddings=[query_embed],
            n_results=K,
            include=[
                "metadatas",
                "distances",
            ],  # Chroma returns distances, so score = 1 - distance (approx for cosine) in some spaces, or just use distance
            # Note: Chroma default is L2 squared distance? Or Cosine?
            # BAAI model is normalized, so dot product / cosine is best.
            # If default is l2, lower is better.
            # Let's for now assume we need to convert distance to similarity.
        )

        # Content search (40% weight)
        content_results = content_collection.query(
            query_embeddings=[query_embed],
            n_results=K,
            include=["metadatas", "distances"],
        )

        # 3. Process and merge results
        scores = {}  # paper_id -> weighted_score
        paper_metadata = {}  # paper_id -> metadata

        # Process Abstract Results
        if abs_results["ids"] and abs_results["ids"][0]:
            ids = abs_results["ids"][0]
            dists = abs_results["distances"][0]
            metas = abs_results["metadatas"][0]

            for pid, dist, meta in zip(ids, dists, metas):
                # Convert distance to similarity score
                # Assuming simple conversion or use distance directly if we want output to be distance
                # For weighted average, we need similarity (higher is better)
                # If distance is cosine distance (0 to 2), sim = 1 - dist
                # If L2, it's unbounded but usually small for normalized vectors.
                # Let's assume standard behavior: similarity = 1 - distance (clamped 0-1) mechanism is safer
                sim = max(0, 1.0 - dist)

                weighted_score = sim * 0.6
                scores[pid] = scores.get(pid, 0) + weighted_score
                paper_metadata[pid] = meta

        # Process Content Results
        if content_results["ids"] and content_results["ids"][0]:
            ids = content_results["ids"][0]
            dists = content_results["distances"][0]
            metas = content_results["metadatas"][0]

            # Content results are chunks. We need to aggregate them to the paper level.
            # Strategy: Take the Max chunk score for a paper, then apply weight.
            paper_best_chunk_score = {}  # parent_id -> max_sim

            for cid, dist, meta in zip(ids, dists, metas):
                parent_id = meta.get("parent_id")
                if not parent_id:
                    continue

                sim = max(0, 1.0 - dist)
                if (
                    parent_id not in paper_best_chunk_score
                    or sim > paper_best_chunk_score[parent_id]
                ):
                    paper_best_chunk_score[parent_id] = sim
                    # We might want to retain metadata from the best chunk if abstract missing?
                    # Usually abstract metadata is authoritative for our use case.

            # Add to main scores
            for pid, best_sim in paper_best_chunk_score.items():
                weighted_score = best_sim * 0.4
                scores[pid] = scores.get(pid, 0) + weighted_score

                # Fallback metadata if not in abstract results (rare)
                if pid not in paper_metadata:
                    # We need to find one chunk's metadata to use
                    # Just looping again to find it is inefficient but simple for now
                    for m in metas:
                        if m.get("parent_id") == pid:
                            paper_metadata[pid] = m
                            break

        # 4. Sort and Format
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Format for template
        matches = []
        for pid in sorted_ids[:K]:
            meta = paper_metadata.get(pid, {})
            matches.append(
                {
                    "id": pid,
                    "metadata": meta,
                    "score": scores[pid],
                    # Add any other fields template expects
                }
            )

        return render_template("index.html", matches=matches, query=query)

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

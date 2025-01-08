# app.py
import flask
import os
from pinecone import Pinecone
import validators
from flask import render_template, request
from sentence_transformers import SentenceTransformer
from helpers import get_matches_initial, fetch_abstract, error, parse_arxiv_identifier
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
# For Flask-Limiter v2, the storage classes are imported directly
import redis
from limits.storage import MemoryStorage, RedisStorage

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)

# Initialize rate limiter with configurable storage
if os.getenv('FLASK_ENV') == 'development':
    # Use in-memory storage for local development
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri="memory://"
    )
else:
    # Use Redis storage for production
    redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')
    try:
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri=redis_url
        )
    except Exception as e:
        print(f"Warning: Redis connection failed, falling back to memory storage: {e}")
        limiter = Limiter(
            app=app,
            key_func=get_remote_address,
            storage_uri="memory://"
        )

# Rate limit configurations
@limiter.limit("1/30seconds", error_message="Too many requests. Slow down!")
@limiter.limit("5/3minutes", exempt_when=lambda: False, deduct_when=lambda: True)
@limiter.request_filter
def exempt_limits():
    return False

@app.errorhandler(429)
def ratelimit_error(e):
    return {"error": "Too many requests. Try again later."}, 429

# Initialize sentence transformer model
MODEL_NAME = "BAAI/bge-large-en-v1.5"
model = SentenceTransformer(MODEL_NAME)

# Pinecone connection (singleton pattern)
def get_pinecone_index():
    if not hasattr(app, '_pinecone'):
        app._pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        app._index = app._pinecone.Index(os.environ["PINECONE_INDEX_NAME"])
    return app._index

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
    K = 50  # Maximum total results capped at 50
    index = get_pinecone_index()

    # Check if query is an arXiv identifier
    arxiv_id = parse_arxiv_identifier(query)
    if arxiv_id:
        matches = index.fetch([arxiv_id])["vectors"]
        if len(matches) == 0:
            abstract = fetch_abstract(f"https://arxiv.org/abs/{arxiv_id}")
            embed = model.encode([abstract])[0].tolist()
            return get_matches_initial(index, K, vector=embed, exclude=arxiv_id)
        return get_matches_initial(index, K, id=arxiv_id, exclude=arxiv_id)

    # Rest of your validation logic
    if len(query) > 200:
        return error("Sorry! The length of your query cannot exceed 200 characters.")

    try:
        embed = model.encode([query])[0].tolist()
    except Exception as e:
        print(f"Encountered error when creating embedding: {e}", flush=True)
        return error("Error creating embedding. Try again in a few minutes.")

    try:
        return get_matches_initial(index, K, vector=embed)  # Use the new function
    except Exception as e:
        print(f"Encountered error when fetching matches from Pinecone: {e}", flush=True)
        return error("Pinecone not responding. Try again in a few minutes.")
        
@app.route("/robots.txt")
def robots():
    with open("static/robots.txt", "r") as f:
        content = f.read()
    return content

@app.route("/health")
def health():
    return {"status": "healthy"}, 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)

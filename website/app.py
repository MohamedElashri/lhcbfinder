# app.py
import flask
import os
from pinecone import Pinecone
import validators
from flask import render_template, request
from sentence_transformers import SentenceTransformer
from helpers import get_matches, fetch_abstract, error, parse_arxiv_identifier
from dotenv import load_dotenv
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables from .env file
load_dotenv()

app = flask.Flask(__name__)



# Initialize rate limiter with app and key function
limiter = Limiter(key_func=get_remote_address)
limiter.init_app(app)


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
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/search")
def search():
    query = request.args.get("query")
    results_per_page = int(request.args.get("per_page", 10))  # Default 10, no maximum limit
    page = int(request.args.get("page", 1))  # Current page number
    K = 50  # Maximum total results capped at 50
    index = get_pinecone_index()

    # Check if query is an arXiv identifier
    arxiv_id = parse_arxiv_identifier(query)
    if arxiv_id:
        matches = index.fetch([arxiv_id])["vectors"]
        if len(matches) == 0:
            abstract = fetch_abstract(f"https://arxiv.org/abs/{arxiv_id}")
            embed = model.encode([abstract])[0].tolist()
            return get_matches(index, K, vector=embed, exclude=arxiv_id, per_page=results_per_page, page=page)
        return get_matches(index, K, id=arxiv_id, exclude=arxiv_id, per_page=results_per_page, page=page)

    # Rest of your validation logic
    if len(query) > 200:
        return error("Sorry! The length of your query cannot exceed 200 characters.")

    try:
        embed = model.encode([query])[0].tolist()
    except Exception as e:
        print(f"Encountered error when creating embedding: {e}", flush=True)
        return error("Error creating embedding. Try again in a few minutes.")

    try:
        return get_matches(index, K, vector=embed, per_page=results_per_page, page=page)
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

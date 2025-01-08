# run_local.py
from dotenv import load_dotenv
import os

# Load environment variables before importing app
load_dotenv()

# Verify environment variables are loaded
required_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
missing_vars = [var for var in required_vars if var not in os.environ]
if missing_vars:
    print("Error: Missing required environment variables:")
    for var in missing_vars:
        print(f"- {var}")
    print("\nPlease check your .env file contains these variables.")
    exit(1)

# Import app only after environment variables are loaded
from app import app

if __name__ == "__main__":
    print("Starting server with environment variables:")
    print(f"PINECONE_INDEX_NAME: {os.environ['PINECONE_INDEX_NAME']}")
    print(f"PINECONE_API_KEY: {'*' * len(os.environ['PINECONE_API_KEY'])}")
    app.run(debug=True, port=8000)
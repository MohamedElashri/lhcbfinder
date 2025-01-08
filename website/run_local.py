from dotenv import load_dotenv
import os
import sys

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
    sys.exit(1)

# Set environment variable to indicate local development
os.environ['FLASK_ENV'] = 'development'
os.environ['FLASK_DEBUG'] = '1'

# Import app only after environment variables are loaded
try:
    from app import app
    
    if __name__ == "__main__":
        print("Starting local development server with environment variables:")
        print(f"PINECONE_INDEX_NAME: {os.environ['PINECONE_INDEX_NAME']}")
        print(f"PINECONE_API_KEY: {'*' * len(os.environ['PINECONE_API_KEY'])}")
        print("\nUsing in-memory storage for rate limiting (development only)")
        
        app.run(debug=True, port=8000)
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("\nMake sure you have installed all required packages:")
    print("pip install -r requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"Error starting the application: {e}")
    sys.exit(1)
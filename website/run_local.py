from dotenv import load_dotenv
import os
import sys

def check_vector_store_config():
    """Check and validate vector store configuration"""
    use_pinecone = os.getenv('USE_PINECONE', 'true').lower() == 'true'
    use_qdrant = os.getenv('USE_QDRANT', 'false').lower() == 'true'
    
    if not use_pinecone and not use_qdrant:
        print("Warning: No vector store enabled. Falling back to Pinecone.")
        os.environ['USE_PINECONE'] = 'true'
        use_pinecone = True
    
    missing_vars = []
    
    if use_pinecone:
        pinecone_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
        missing_vars.extend([var for var in pinecone_vars if var not in os.environ])
    
    if use_qdrant:
        if 'QDRANT_COLLECTION' not in os.environ:
            os.environ['QDRANT_COLLECTION'] = 'lhcb_papers'
            print("Info: Using default Qdrant collection name 'lhcb_papers'")
    
    return use_pinecone, use_qdrant, missing_vars

def main():
    # Load environment variables
    load_dotenv()
    
    # Set development environment
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    # Check vector store configuration
    use_pinecone, use_qdrant, missing_vars = check_vector_store_config()
    
    if missing_vars:
        print("Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease check your .env file contains these variables.")
        sys.exit(1)
    
    # Import app only after environment variables are loaded
    try:
        from app import app
        
        print("\nStarting local development server with configuration:")
        print("\nVector Store Configuration:")
        print(f"- Pinecone: {'Enabled' if use_pinecone else 'Disabled'}")
        if use_pinecone:
            print(f"  - Index: {os.environ['PINECONE_INDEX_NAME']}")
            print(f"  - API Key: {'*' * len(os.environ['PINECONE_API_KEY'])}")
        
        print(f"- Qdrant: {'Enabled' if use_qdrant else 'Disabled'}")
        if use_qdrant:
            print(f"  - Collection: {os.environ['QDRANT_COLLECTION']}")
            print("  - Storage: Local (.qdrant directory)")
        
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

if __name__ == "__main__":
    main()
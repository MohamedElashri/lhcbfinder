# test_setup.py
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

def test_environment():
    print("Testing environment setup...")
    
    # Test environment variables
    load_dotenv()
    required_vars = ['PINECONE_API_KEY', 'PINECONE_INDEX_NAME']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    print("✅ Environment variables loaded successfully")
    
    # Test Sentence Transformer model
    try:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        test_embedding = model.encode(["Test sentence"])
        print("✅ Sentence Transformer model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Sentence Transformer model: {e}")
        return False
    
    # Test Pinecone connection
    try:
        pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))
        stats = index.describe_index_stats()
        print(f"✅ Connected to Pinecone index: {stats['total_vector_count']} vectors found")
    except Exception as e:
        print(f"❌ Error connecting to Pinecone: {e}")
        return False
    
    return True

if __name__ == "__main__":
    if test_environment():
        print("\n🎉 All systems ready! You can now run the Flask app.")
    else:
        print("\n❌ Please fix the errors above before running the Flask app.")

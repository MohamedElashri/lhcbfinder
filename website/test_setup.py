# test_setup.py
import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb

def get_chroma_db_path():
    """Get ChromaDB path with same logic as app.py."""
    env_path = os.environ.get("CHROMA_DB_PATH")
    if env_path:
        env_path = os.path.expanduser(env_path)
        if not os.path.isabs(env_path):
            env_path = os.path.abspath(env_path)
        return env_path
    
    docker_path = "/app/chroma_db"
    if os.path.exists(docker_path):
        return docker_path
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_model_path = os.path.join(current_dir, "..", "model", "output", "chroma_db")
    if os.path.exists(parent_model_path):
        return os.path.abspath(parent_model_path)
    
    sibling_model_path = os.path.join(current_dir, "model", "output", "chroma_db")
    if os.path.exists(sibling_model_path):
        return os.path.abspath(sibling_model_path)
    
    default_path = os.path.join(current_dir, "..", "model", "output", "chroma_db")
    return os.path.abspath(default_path)

def test_environment():
    print("Testing LHCb Finder environment setup...\n")
    print("="*60)
    
    # Load environment variables
    load_dotenv()
    print("✅ Environment variables loaded")
    
    # Test Sentence Transformer model
    print("\n📦 Testing Sentence Transformer model...")
    try:
        MODEL_NAME = "BAAI/bge-large-en-v1.5"
        print(f"   Loading model: {MODEL_NAME}")
        model = SentenceTransformer(MODEL_NAME)
        test_embedding = model.encode(["Test sentence"])
        print(f"   Embedding dimension: {len(test_embedding)}")
        print("✅ Sentence Transformer model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading Sentence Transformer model: {e}")
        return False
    
    # Test ChromaDB connection
    print("\n🗄️  Testing ChromaDB connection...")
    try:
        chroma_path = get_chroma_db_path()
        print(f"   ChromaDB path: {chroma_path}")
        
        if not os.path.exists(chroma_path):
            print(f"❌ ChromaDB path does not exist: {chroma_path}")
            print("   Please run the embedding pipeline first or set CHROMA_DB_PATH")
            return False
        
        client = chromadb.PersistentClient(path=chroma_path)
        print("✅ ChromaDB client initialized")
        
        # Test collections
        print("\n📚 Testing collections...")
        abstracts = client.get_or_create_collection(name="lhcb_abstracts")
        contents = client.get_or_create_collection(name="lhcb_contents")
        
        abs_count = abstracts.count()
        content_count = contents.count()
        
        print(f"   - lhcb_abstracts: {abs_count} documents")
        print(f"   - lhcb_contents: {content_count} documents")
        
        if abs_count == 0 and content_count == 0:
            print("⚠️  Warning: Collections are empty. Have you run the embedding pipeline?")
        else:
            print("✅ Collections loaded successfully")
        
    except Exception as e:
        print(f"❌ Error with ChromaDB: {e}")
        return False
    
    # Test Redis (if configured)
    print("\n🔴 Testing Redis connection (optional)...")
    redis_url = os.getenv("REDIS_URL")
    if redis_url and redis_url != "memory://":
        try:
            import redis
            r = redis.from_url(redis_url)
            r.ping()
            print(f"✅ Redis connection successful: {redis_url}")
        except ImportError:
            print("⚠️  Redis package not installed (optional for rate limiting)")
        except Exception as e:
            print(f"⚠️  Redis not available: {e}")
            print("   App will use in-memory rate limiting")
    else:
        print("   Using in-memory rate limiting (development mode)")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*60)
    print("LHCb Finder - Environment Setup Test")
    print("="*60 + "\n")
    
    if test_environment():
        print("\n" + "="*60)
        print("🎉 All systems ready! You can now run the Flask app.")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("❌ Please fix the errors above before running the Flask app.")
        print("="*60)
        sys.exit(1)

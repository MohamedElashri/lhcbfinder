#!/usr/bin/env python3
"""
Delete and recreate ChromaDB collections with cosine similarity.
This script will delete the old L2-based collections and the next
embedding run will create them with cosine.
"""
import chromadb
import sys

db_path = '../model/output/chroma_db'
print(f"Connecting to ChromaDB at: {db_path}\n")

try:
    client = chromadb.PersistentClient(path=db_path)
    
    print("=" * 60)
    print("DELETING OLD COLLECTIONS")
    print("=" * 60)
    
    # Delete abstract collection
    try:
        client.delete_collection('lhcb_abstracts')
        print('Deleted lhcb_abstracts (will be recreated with cosine)')
    except Exception as e:
        print(f'Could not delete lhcb_abstracts: {e}')
    
    # Delete content collection
    try:
        client.delete_collection('lhcb_contents')
        print('Deleted lhcb_contents (will be recreated with cosine)')
    except Exception as e:
        print(f'Could not delete lhcb_contents: {e}')

    print("\n" + "=" * 60)
    print("READY FOR RE-EMBEDDING")
    print("=" * 60)
    print("\nNext steps:")
    print("1. cd /data/home/melashri/LLM/lhcbfinder/model")
    print("2. ./run.sh --output-dir output --with-content --rebuild-embeddings")
    print("\nBoth collections will be created with cosine similarity.")
    
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

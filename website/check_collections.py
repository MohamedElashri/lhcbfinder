#!/usr/bin/env python3
import chromadb
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model'))

db_path = '../model/output/chroma_db'
print(f"Checking ChromaDB at: {db_path}\n")

client = chromadb.PersistentClient(path=db_path)

# Get collections
abs_col = client.get_collection('lhcb_abstracts')
content_col = client.get_collection('lhcb_contents')

print("=" * 60)
print("CURRENT COLLECTION STATUS")
print("=" * 60)
print(f"\n📊 Abstract Collection:")
print(f"   - Metadata: {abs_col.metadata}")
print(f"   - Document count: {abs_col.count():,}")

print(f"\n📊 Content Collection:")
print(f"   - Metadata: {content_col.metadata}")
print(f"   - Document count: {content_col.count():,}")

print("\n" + "=" * 60)
print("RE-EMBEDDING NEEDED?")
print("=" * 60)

# Check if cosine is already configured
abs_space = abs_col.metadata.get('hnsw:space', 'l2') if abs_col.metadata else 'l2'
content_space = content_col.metadata.get('hnsw:space', 'l2') if content_col.metadata else 'l2'

if abs_space == 'cosine' and content_space == 'cosine':
    print("NO RE-EMBEDDING NEEDED")
    print("   Both collections already use cosine distance")
elif abs_space == 'l2' or content_space == 'l2':
    print(" OPTIONAL: Collections use L2 distance")
    print(f"   - Abstracts: {abs_space}")
    print(f"   - Contents: {content_space}")
    print("\n   Options:")
    print("   1. Keep as-is (will still work, just slightly less optimal)")
    print("   2. Re-create collections with cosine metric")
else:
    print(f"ℹ️  Current metrics: abstracts={abs_space}, contents={content_space}")

print("\n" + "=" * 60)

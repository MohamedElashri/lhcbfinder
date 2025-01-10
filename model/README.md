# LHCb Papers Embedding Pipeline

## Prerequisites

1. Environment Setup
```bash
# Create and edit .env file with your credentials
touch .env
# Add these variables to .env:
# Vector Store Selection (at least one must be true)
USE_PINECONE=true
USE_QDRANT=false

# Pinecone Configuration (required if USE_PINECONE=true)
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_pinecone_index

# Qdrant Configuration (required if USE_QDRANT=true)
QDRANT_COLLECTION=lhcb_papers
QDRANT_DATA_PATH=/app/data/qdrant

# Kaggle Configuration (for dataset download)
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Create Data Directories (if using Qdrant)
```bash
mkdir -p data/qdrant
```

## Vector Store Options

The pipeline supports two vector store backends:

### Pinecone (Cloud-based)
- Requires API key and index name
- Suitable for production deployments
- Set `USE_PINECONE=true` in .env

### Qdrant (Self-hosted)
- Local development: Uses embedded mode
- Data stored in `./data/qdrant/`
- Set `USE_QDRANT=true` in .env

You can use either or both vector stores simultaneously.

## Basic Usage

The pipeline can be run in different configurations using the run.sh script.

### Simple Embedding Without PDFs
```bash
# Using Pinecone only (default)
./run.sh

# Using Qdrant only
USE_PINECONE=false USE_QDRANT=true ./run.sh

# Using both stores
USE_PINECONE=true USE_QDRANT=true ./run.sh
```

### Including PDFs
```bash
./run.sh --include-pdf --download-pdfs
```

### Time-Based Filtering
```bash
./run.sh --start-year 2020
```

## Force Flags

The pipeline includes three force flags that can be used independently or together:

### Force arXiv Download
```bash
./run.sh --force-arxiv-download
```

### Force Embeddings
```bash
./run.sh --force-embeddings
```

### Force PDF Downloads
```bash
./run.sh --force-pdf-download
```

### Combining Force Flags
```bash
# Force everything with both vector stores
USE_PINECONE=true USE_QDRANT=true ./run.sh --force-arxiv-download --force-embeddings --force-pdf-download --include-pdf --download-pdfs

# Force embeddings with PDF content from 2020 using Qdrant only
USE_PINECONE=false USE_QDRANT=true ./run.sh --force-embeddings --include-pdf --download-pdfs --start-year 2020
```

## Common Scenarios

### First Time Setup
```bash
# Make run.sh executable
chmod +x run.sh

# Run with PDF content using both stores
USE_PINECONE=true USE_QDRANT=true ./run.sh --include-pdf --download-pdfs
```

### Updating Existing Embeddings
```bash
./run.sh --include-pdf --download-pdfs
```

### Moving to New Vector Store
1. Update configuration in .env
2. Run with force embeddings:
```bash
# Switch to new Pinecone index
./run.sh --force-embeddings --include-pdf --download-pdfs

# Switch to Qdrant
USE_PINECONE=false USE_QDRANT=true ./run.sh --force-embeddings --include-pdf --download-pdfs

# Add Qdrant alongside existing Pinecone
USE_PINECONE=true USE_QDRANT=true ./run.sh --force-embeddings --include-pdf --download-pdfs
```

### Refreshing All Data
```bash
./run.sh --force-arxiv-download --force-embeddings --force-pdf-download --include-pdf --download-pdfs
```

### Migrating Between Vector Stores
To migrate existing embeddings from localfile to Qdrant :
```bash
python migrate_embeddings.py
```
**TODO**: Add migration from Pinecone to Qdrant and vice versa

## All Available Options
```bash
--include-pdf             Include PDF content in embeddings
--download-pdfs          Download new PDFs
--force-arxiv-download   Force download of new arXiv metadata
--force-embeddings       Force reprocessing of all papers
--force-pdf-download     Force download of all PDFs
--start-year YEAR        Process papers from this year onwards
--pdf-dir DIR           Specify PDF directory (default: lhcb_pdfs)
```

## Directory Structure
```
./
├── data/
│   ├── qdrant/          # Qdrant embedded storage
│   └── qdrant_storage/  # Qdrant service storage (if using Docker)
├── lhcb_pdfs/          # Downloaded PDFs
├── arxiv-metadata-oai-snapshot.json  # arXiv metadata
└── lhcb-arxiv-embeddings.json       # Generated embeddings
```

## Vector Store Management

### Checking Vector Counts
```python
from vector_store import create_vector_store

# Get store instance
store = create_vector_store(
    use_pinecone=True, 
    use_qdrant=True,
    pinecone_index="your_index_name",
    qdrant_collection="lhcb_papers"
)

# Check total vectors
print(f"Total vectors: {store.get_total_vectors()}")
```

### Clearing Vector Store

If using Qdrant, clear the data directory before running the pipeline again:

```bash
# Clear Qdrant data
rm -rf data/qdrant/*
```

For Pinecone, create a new index is the easiest way to clear all data (and remove the old index).


## Error Handling and Logging

### Log Files
- `arxiv_download.log`: Download progress and errors
- `successful_downloads.json`: Successfully downloaded papers
- `failed_downloads.json`: Failed download attempts
- `removed_papers.json`: List of removed/retracted papers




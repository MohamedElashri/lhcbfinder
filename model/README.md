# LHCb Papers Embedding Pipeline

## Prerequisites

1. Environment Setup
```bash
# Create and edit .env file with your credentials
touch .env

# Add these variables to .env:
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX_NAME=your_pinecone_index
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Basic Usage

The pipeline can be run in different configurations using the run.sh script.

### Simple Embedding Without PDFs
To only create embeddings without downloading or including PDF content:
```bash
./run.sh
```

### Including PDFs
To download and include PDF content in embeddings:
```bash
./run.sh --include-pdf --download-pdfs
```

### Time-Based Filtering
Process papers from a specific year onwards:
```bash
./run.sh --start-year 2020
```

## Force Flags

The pipeline includes three force flags that can be used independently or together:

### Force arXiv Download
Forces a fresh download of the arXiv metadata:
```bash
./run.sh --force-arxiv-download
```

### Force Embeddings
Reprocesses all papers, even those that already have embeddings:
```bash
./run.sh --force-embeddings
```

### Force PDF Downloads
Re-downloads PDFs even if they exist locally:
```bash
./run.sh --force-pdf-download
```

### Combining Force Flags
You can combine any of the force flags:
```bash
# Force everything
./run.sh --force-arxiv-download --force-embeddings --force-pdf-download --include-pdf --download-pdfs

# Force embeddings with PDF content from 2020
./run.sh --force-embeddings --include-pdf --download-pdfs --start-year 2020
```

## Common Scenarios

### First Time Setup
For first-time setup, run:
```bash
# Make run.sh executable
chmod +x run.sh

# Run with PDF content
./run.sh --include-pdf --download-pdfs
```

### Updating Existing Embeddings
To process only new papers:
```bash
./run.sh --include-pdf --download-pdfs
```

### Moving to New Pinecone Index
When switching to a new Pinecone index:
1. Update PINECONE_INDEX_NAME in .env
2. Run:
```bash
./run.sh --force-embeddings --include-pdf --download-pdfs
```

### Refreshing All Data
To completely refresh all data:
```bash
./run.sh --force-arxiv-download --force-embeddings --force-pdf-download --include-pdf --download-pdfs
```

## All Available Options

```bash
--include-pdf             Include PDF content in embeddings
--download-pdfs           Download new PDFs
--force-arxiv-download    Force download of new arXiv metadata
--force-embeddings        Force reprocessing of all papers
--force-pdf-download      Force download of all PDFs
--start-year YEAR         Process papers from this year onwards
--pdf-dir DIR            Specify PDF directory (default: lhcb_pdfs)
```

## Output Files

- `arxiv-metadata-oai-snapshot.json`: arXiv metadata
- `lhcb-arxiv-embeddings.json`: Generated embeddings
- `lhcb_pdfs/`: Directory containing downloaded PDFs

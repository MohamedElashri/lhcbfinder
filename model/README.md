# LHCb Papers Embedding Pipeline

This pipeline creates vector embeddings for LHCb papers from the arXiv dataset, with optional PDF content inclusion for improved search results.

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

# For enhanced progress visualization (optional but recommended)
pip install colorama psutil yaspin
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

## Content Chunking Options

For more precise searching, you can enable content chunking which creates separate embeddings for sections of papers.

### Basic Chunking
Enable chunking with default settings (500 words per chunk, 100 words overlap):
```bash
./run.sh --include-pdf --download-pdfs --chunk-mode
```

### Custom Chunk Size
Control chunk size and overlap for fine-tuning:
```bash
./run.sh --include-pdf --download-pdfs --chunk-mode --chunk-size 300 --chunk-overlap 50
```

## Test Mode

You can run in test mode with a limited number of papers to verify your setup quickly:
```bash
# Test with 10 papers (default)
./run.sh --test-mode

# Test with custom paper count
./run.sh --test-mode --limit 20 --include-pdf --download-pdfs
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


### Skip Confirmation
To run without confirmation prompts (useful for automated scripts):
```bash
./run.sh --include-pdf --download-pdfs --no-confirmation
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

### Creating Chunked Embeddings
For more fine-grained search with content chunks:
```bash
./run.sh --include-pdf --download-pdfs --chunk-mode --chunk-size 400 --chunk-overlap 80
```

### Moving to New Pinecone Index
When switching to a new Pinecone index:
1. Update PINECONE_INDEX_NAME in .env
2. Run:
```bash
./run.sh --force-embeddings --include-pdf --download-pdfs
```

### Quick Testing
For testing your setup without processing the entire dataset:
```bash
./run.sh --test-mode --limit 5 --include-pdf --download-pdfs
```

## Embedding Format and Metadata

### Embedding Content Structure
The embedding text focuses on the most semantically relevant content:
- **Title**: The paper title
- **Year**: Publication year
- **Abstract**: Paper abstract
- **Content**: PDF content (when using --include-pdf) or content chunks (when using --chunk-mode)

### Metadata Fields
Each embedding includes these metadata fields:
- id: The paper ID
- title: Paper title
- authors: Authors list (preserved in metadata)
- year: Publication year
- abstract: Paper abstract
- categories: arXiv categories
- is_chunk: Whether this is a content chunk (when using --chunk-mode)
- chunk_id: The chunk identifier (when applicable)
- parent_id: ID of the parent paper (for chunks)

### Refreshing All Data
To completely refresh all data:
```bash
./run.sh --force-arxiv-download --force-embeddings --force-pdf-download --include-pdf --download-pdfs
```


## Available Options

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

## Production Deployment

### lhcbfinder.net Production Setup
To generate the complete production-ready version as deployed on lhcbfinder.net, use:
```bash
./run.sh --force-arxiv-download --force-embeddings --include-pdf --download-pdfs --chunk-mode --chunk-size 500 --chunk-overlap 100 --no-confirmation
```
Note: This command assumes a clean slate and will rebuild the entire dataset and embeddings from scratch. It requires significant processing time and disk space.

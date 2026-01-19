# LHCb Papers Embedding Pipeline

This pipeline ingests LHCb papers from arXiv, processes them (prioritizing HTML content with PDF processing fallback), and creates vector embeddings stored in a local **ChromaDB** instance.

## Features
- **Dual Content Source**: Attempts to download high-quality HTML from arXiv. If unavailable (e.g., 404), falls back to PDF processing.
- **Weighted Embeddings**: Generates separate embeddings for:
  - **Abstracts**: For broad semantic matching (60% weight).
  - **Content Chunks**: For precise detail matching (40% weight).
- **Vector Database**: Uses **ChromaDB** (local persistence) for storing and querying embeddings.

## Command Line Flags
The `embed.py` script and `run.sh` wrapper support the following flags:
- **--download-content**: Download content (HTML with PDF fallback) for filtered papers (Recommended).
- **--force-metadata**: Force re-download of the arXiv metadata file from Kaggle.
- **--force-content**: Force re-download of content (HTML/PDF) even if files exist.
- **--force-embeddings**: Force regeneration of embeddings even if they exist.
- **--start-year YYYY**: Process papers published from the specified year onwards.
- **--test-mode**: Enable test mode, processing a small, fixed number of papers.
- **--limit N**: Limit the number of papers processed (useful with `--test-mode`).
- **--output-dir**: Base directory for all output (HTML, PDFs, DB, logs).

## Prerequisites

1.  **Environment Setup**
    Ensure you are using **Python 3.11** (recommended to avoid dependency issues with `onnxruntime` on newer Python versions).

    ```bash
    # Create valid (.env) file
    touch .env
    
    # Add optional configuration (if needed)
    # CHROMA_DB_PATH=chroma_db
    
    # KAGGLE Credentials (Required for automatic dataset download)
    # KAGGLE_USERNAME=your_username
    # KAGGLE_KEY=your_key
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The pipeline is controlled via `run.sh` or directly via `python embed.py`.

### 1. Standard Ingestion (Recommended)
Downloads HTML/PDFs and creates embeddings:
```bash
./run.sh --download-content
```
*Note: `--download-content` attempts to download HTML first, then falls back to PDF if HTML is unavailable.*

### 2. Time-Based Filtering
Process only recent papers (e.g., fro 2024 onwards):
```bash
./run.sh --start-year 2024 --download-content
```

### 3. Force Refresh
To re-download and re-embed everything:
```bash
./run.sh --force-embeddings --force-content --download-content
```
*(`--force-content` re-fetches content for all papers).*

### 4. Test Mode (Development)
Process a small subset (e.g., 5 papers) to verify setup:
```bash
./run.sh --test-mode --limit 5 --download-content
```

## Docker

Build and run the pipeline in a container:
```bash
# Build
docker build -t lhcb-embedder .

# Run
docker run -v $(pwd)/chroma_db:/app/chroma_db lhcb-embedder
```
*Note: We mount the `chroma_db` directory to persist the vector database outside the container.*

## Output

- `chroma_db/`: Directory containing the ChromaDB database files.
- `lhcb_html/`: Cache of downloaded HTML content.
- `lhcb_pdfs/`: Cache of downloaded PDF files.
- `arxiv-metadata-oai-snapshot.json`: Metadata source file.

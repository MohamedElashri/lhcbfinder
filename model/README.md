# LHCb Papers Embedding Pipeline

This pipeline ingests LHCb papers from arXiv, processes them (prioritizing HTML content with PDF processing fallback), and creates vector embeddings stored in a local **ChromaDB** instance.

## Features
- **Dual Content Source**: Attempts to download high-quality HTML from arXiv. If unavailable (e.g., 404), falls back to PDF processing.
- **Streaming Processing**: Processes papers incrementally (stream → filter → download → embed) for minimal memory usage
- **Weighted Embeddings**: Generates separate embeddings for:
  - **Abstracts**: For broad semantic matching (60% weight).
  - **Content Chunks**: For precise detail matching (40% weight).
- **Vector Database**: Uses **ChromaDB** (local persistence) for storing and querying embeddings.

## Pipeline Order

The pipeline runs these logical steps (in order):

1. Download arXiv metadata (from Kaggle) if not present or when `--force-metadata` is used.
2. Filter the metadata for LHCb-related papers (quick first-pass scan to avoid loading the full dataset).
3. Download content for the filtered LHCb papers (HTML preferred, PDF fallback) when `--download-content` is used.
4. Parse content / extract text and chunk as configured.
5. Create embeddings and store them in the local ChromaDB instance.

This ordering keeps the expensive content download and parsing steps limited to the filtered LHCb subset.

## Command Line Flags
The `embed.py` script and `run.sh` wrapper support the following flags:
- **--with-content**: Include paper content in embeddings (loads from existing HTML/PDF files)
- **--redownload-metadata**: Force re-downloading arXiv metadata from source
- **--redownload-content**: Force re-downloading content (HTML/PDF) from arXiv
- **--rebuild-embeddings**: Force recreating all embeddings (ignores existing)
- **--start-year YYYY**: Process papers published from the specified year onwards
- **--test-mode**: Enable test mode, processing a small, fixed number of papers
- **--limit N**: Limit the number of papers processed (useful with `--test-mode`)
- **--output-dir**: Base directory for all output (HTML, PDFs, DB, logs)
- **--chunk-size**: Maximum words per chunk (default: 500)
- **--chunk-overlap**: Words to overlap between chunks (default: 100)

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
Load existing HTML/PDFs and create embeddings:
```bash
./run.sh --with-content
```
*Note: `--with-content` loads existing files. Use `--redownload-content` to force fresh downloads.*

### 2. Time-Based Filtering
Process only recent papers (e.g., from 2025 onwards):
```bash
./run.sh --start-year 2025 --with-content
```

These papers from 2025 onwards will probably have HTML versions available for all of them. Older papers may still require PDF fallback.

### 3. Force Refresh
To re-download and re-embed everything:
```bash
./run.sh --rebuild-embeddings --redownload-content --with-content
```
*(`--redownload-content` re-fetches content for all papers).*

### 4. Test Mode (Development)
Process a small subset (e.g., 5 papers) to verify setup:
```bash
./run.sh --test-mode --limit 5 --with-content
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

#!/bin/bash
# Force immediate output
export PYTHONUNBUFFERED=1
# Exit on error
set -e

echo "Starting pipeline..."

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo "Error: .env file not found"
    echo "Please create a .env file with the following variables:"
    echo "PINECONE_API_KEY=your_api_key"
    echo "PINECONE_INDEX_NAME=your_index_name"
    exit 1
fi

# Parse command line arguments
INCLUDE_PDF=false
DOWNLOAD_PDFS=false
FORCE_ARXIV=false
FORCE_EMBEDDINGS=false
FORCE_PDF=false
START_YEAR=""
PDF_DIR="lhcb_pdfs"
CHUNK_MODE=false
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TEST_MODE=false
LIMIT=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --include-pdf)
            INCLUDE_PDF=true
            shift
            ;;
        --download-pdfs)
            DOWNLOAD_PDFS=true
            shift
            ;;
        --force-arxiv-download)
            FORCE_ARXIV=true
            shift
            ;;
        --force-embeddings)
            FORCE_EMBEDDINGS=true
            shift
            ;;
        --force-pdf-download)
            FORCE_PDF=true
            shift
            ;;
        --start-year)
            START_YEAR="$2"
            shift 2
            ;;
        --pdf-dir)
            PDF_DIR="$2"
            shift 2
            ;;
        --chunk-mode)
            CHUNK_MODE=true
            shift
            ;;
        --chunk-size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --chunk-overlap)
            CHUNK_OVERLAP="$2"
            shift 2
            ;;
        --test-mode)
            TEST_MODE=true
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Available options:"
            echo "  --include-pdf              Include PDF content in embeddings"
            echo "  --download-pdfs            Download new PDFs"
            echo "  --force-arxiv-download     Force download of new arXiv metadata"
            echo "  --force-embeddings         Force reprocessing of all papers"
            echo "  --force-pdf-download       Force download of all PDFs"
            echo "  --start-year YEAR          Process papers from this year onwards"
            echo "  --pdf-dir DIR              Specify PDF directory"
            echo "  --chunk-mode               Enable chunking of PDF content for better search"
            echo "  --chunk-size SIZE          Maximum number of words per chunk (default: 500)"
            echo "  --chunk-overlap OVERLAP    Number of words to overlap between chunks (default: 100)"
            echo "  --test-mode                Enable test mode to process a small batch of papers"
            echo "  --limit N                  Limit number of papers to process (default: 10 in test mode)" 
            exit 1
            ;;
    esac
done

# Create PDF directory if it doesn't exist
if [ "$INCLUDE_PDF" = true ] || [ "$DOWNLOAD_PDFS" = true ]; then
    if [ ! -d "$PDF_DIR" ]; then
        echo "Creating PDF directory: $PDF_DIR"
        mkdir -p "$PDF_DIR"
    fi
fi

# Show configuration
echo "Running with configuration:"
echo "- PDF Directory: $PDF_DIR"
echo "- Include PDF content: $INCLUDE_PDF"
echo "- Download PDFs: $DOWNLOAD_PDFS"
echo "- Force arXiv download: $FORCE_ARXIV"
echo "- Force embeddings: $FORCE_EMBEDDINGS"
echo "- Force PDF download: $FORCE_PDF"
echo "- Start Year: $START_YEAR"
echo "- Chunk mode: $CHUNK_MODE"
if [ "$CHUNK_MODE" = true ]; then
    echo "- Chunk size: $CHUNK_SIZE"
    echo "- Chunk overlap: $CHUNK_OVERLAP"
fi
if [ "$TEST_MODE" = true ]; then
    echo "- Test mode: enabled"
    echo "- Paper limit: $LIMIT"
fi
echo

# Build command string
CMD="python3 -u embed.py --no-confirmation --pdf-dir \"$PDF_DIR\""

if [ ! -z "$START_YEAR" ]; then
    CMD="$CMD --start-year $START_YEAR"
fi

if $INCLUDE_PDF; then
    CMD="$CMD --include-pdf"
fi

if $DOWNLOAD_PDFS; then
    CMD="$CMD --download-pdfs"
fi

if $FORCE_ARXIV; then
    CMD="$CMD --force-arxiv-download"
fi

if $FORCE_EMBEDDINGS; then
    CMD="$CMD --force-embeddings"
fi

if $FORCE_PDF; then
    CMD="$CMD --force-pdf-download"
fi

if $CHUNK_MODE; then
    CMD="$CMD --chunk-mode --chunk-size $CHUNK_SIZE --chunk-overlap $CHUNK_OVERLAP"
fi

if $TEST_MODE; then
    CMD="$CMD --test-mode --limit $LIMIT"
fi

echo "Running command: $CMD"
echo "----------------------------------------"

# Run the pipeline with unbuffered output
eval $CMD

echo "----------------------------------------"
echo "✅ Pipeline complete"
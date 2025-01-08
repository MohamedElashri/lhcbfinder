#!/bin/bash
# Force immediate output
export PYTHONUNBUFFERED=1
# Exit on error
set -e

echo "Starting script execution..."

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

echo "Running command: $CMD"
echo "----------------------------------------"

# Run the pipeline with unbuffered output
eval $CMD

echo "----------------------------------------"
echo "✅ Pipeline complete"
#!/bin/bash
# Force immediate output
export PYTHONUNBUFFERED=1
# Exit on error
set -e

# Function to show help message
show_help() {
    echo "LHCbFinder Embedding Pipeline"
    echo "Available options:"
    echo "  -h, --help                 Show this help message"
    echo "  --download-content         Download content (HTML w/ PDF fallback)"
    echo "  --force-metadata           Force download of new arXiv metadata"
    echo "  --force-embeddings         Force reprocessing of all papers"
    echo "  --force-content            Force download of content (HTML/PDF)"
    echo "  --output-dir DIR           Directory to store all output files (default: output)"
    echo "  --start-year YEAR          Process papers from this year onwards"
    echo "  --html-dir DIR             Specify HTML directory"
    echo "  --pdf-dir DIR              Specify PDF directory"
    echo "  --chunk-mode               Enable chunking of PDF content for better search"
    echo "  --chunk-size SIZE          Maximum number of words per chunk (default: 500)"
    echo "  --chunk-overlap OVERLAP    Number of words to overlap between chunks (default: 100)"
    echo "  --test-mode                Enable test mode to process a small batch of papers"
    echo "  --limit N                  Limit number of papers to process (default: 10 in test mode)"
    echo "  --no-confirmation          Skip confirmation prompts"
    exit 0
}

# Check for help flags first before showing any startup messages
for arg in "$@"; do
    if [ "$arg" = "-h" ] || [ "$arg" = "--help" ]; then
        show_help
    fi
done

echo "Starting pipeline..."
# Change to the directory of this script to ensure relative paths work
cd "$(dirname "$0")"

# Load environment variables
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    set -a  # automatically export all variables
    source .env
    set +a
else
    echo "No .env file found (using defaults or environment variables)"
fi

# Parse command line arguments
DOWNLOAD_CONTENT=false
FORCE_METADATA=false
FORCE_EMBEDDINGS=false
FORCE_CONTENT=false
OUTPUT_DIR="output"
START_YEAR=""
PDF_DIR="lhcb_pdfs"
HTML_DIR="lhcb_html"
CHUNK_MODE=false
CHUNK_SIZE=500
CHUNK_OVERLAP=100
TEST_MODE=false
LIMIT=10



while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            # This is handled earlier, but included here for completeness
            show_help
            ;;
        --download-content)
            DOWNLOAD_CONTENT=true
            shift
            ;;
        --force-metadata)
            FORCE_METADATA=true
            shift
            ;;
        --force-embeddings)
            FORCE_EMBEDDINGS=true
            shift
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --force-content)
            FORCE_CONTENT=true
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
        --html-dir)
            HTML_DIR="$2"
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
        --no-confirmation)
            # Handled automatically, just consume arg
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help to see available options"
            exit 1
            ;;
    esac
done

# Create directories if needed
if [ "$DOWNLOAD_CONTENT" = true ]; then
    if [ ! -d "$PDF_DIR" ]; then
        echo "Creating PDF directory: $PDF_DIR"
        mkdir -p "$PDF_DIR"
    fi
     if [ ! -d "$HTML_DIR" ]; then
        echo "Creating HTML directory: $HTML_DIR"
        mkdir -p "$HTML_DIR"
    fi
fi

# Show configuration
echo "Running with configuration:"
echo "- Python Env: $PYTHON_PATH"
echo "- Output Directory: $OUTPUT_DIR"
echo "- HTML Directory: $HTML_DIR"
echo "- PDF Directory: $PDF_DIR"
echo "- Download Content: $DOWNLOAD_CONTENT"
echo "- Force metadata: $FORCE_METADATA"
echo "- Force embeddings: $FORCE_EMBEDDINGS"
echo "- Force content: $FORCE_CONTENT"
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

# Determine Python Interpreter
# Check for .venv in root or current dir
if [ -f "../.venv/bin/python" ]; then
    PYTHON_PATH="../.venv/bin/python"
elif [ -f ".venv/bin/python" ]; then
    PYTHON_PATH=".venv/bin/python"
elif [ -f "venv/bin/python" ]; then
    PYTHON_PATH="venv/bin/python"
elif [ ! -z "$VIRTUAL_ENV" ]; then
     PYTHON_PATH="python" # Already activated
else
    PYTHON_PATH="python3"
fi

# Build command string
CMD="$PYTHON_PATH -u embed.py --no-confirmation --output-dir \"$OUTPUT_DIR\" --pdf-dir \"$PDF_DIR\" --html-dir \"$HTML_DIR\""

if [ ! -z "$START_YEAR" ]; then
    CMD="$CMD --start-year $START_YEAR"
fi

if $DOWNLOAD_CONTENT; then
    CMD="$CMD --download-content"
fi

if $FORCE_METADATA; then
    CMD="$CMD --force-metadata"
fi

if $FORCE_EMBEDDINGS; then
    CMD="$CMD --force-embeddings"
fi

if $FORCE_CONTENT; then
    CMD="$CMD --force-content"
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
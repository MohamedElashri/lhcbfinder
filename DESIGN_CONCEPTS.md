# LHCb Finder - Design Concepts and Coding Guidelines

## Core Design Principles

### 1. Semantic Search Architecture

The LHCb Finder project implements a semantic search engine specifically tailored for research papers from the LHCb experiment (can be tailored to other experiments). The architecture is designed around these key principles:

#### 1.1 Vector Embedding Strategy

- **Content Focus**: Embeddings are generated from the most semantically relevant content:
  - Paper title
  - Publication year
  - Abstract
  - Paper content (chunked for longer papers)

- **Author Exclusion**: Author information is intentionally excluded from the embedding text as author names don't contribute meaningful semantic concepts to the embeddings. However, author data is preserved in the metadata for retrieval and display purposes.

- **Chunking Strategy**: For longer documents, content is divided into overlapping chunks to:
  - Maintain semantic coherence
  - Improve retrieval precision
  - Enable more specific matching to queries
  - Default chunk size is 500 words with 100 word overlap

#### 1.2 Vector Database Integration

- **Pinecone Integration**: Using Pinecone as the primary vector database for efficient similarity search
- **Local Storage**: Maintaining a local copy of embeddings for backup and Kaggle dataset updates
- **Upsert Batching**: Optimized batching for database operations to minimize API calls

### 2. System Architecture

#### 2.1 Two-Component Design

The system is divided into two main components:

1. **Model Pipeline** (model/):
   - Responsible for data acquisition, processing, and embedding generation
   - Operates as a batch process to update the search index
   - Designed for efficient processing of large document collections

2. **Web Application** (website/):
   - Provides the user interface for searching
   - Handles query processing and result presentation
   - Implements rate limiting and caching for performance

#### 2.2 Separation of Concerns

- **Data Processing**: Isolated from search interface
- **Configuration**: Environment-based configuration for different deployment scenarios
- **Error Handling**: Graceful degradation at component boundaries

## Code Design Patterns

### 1. Object-Oriented Design

#### 1.1 Paper Class

The `Paper` class serves as the core data model with responsibilities:
- Storing metadata (title, abstract, year, etc.)
- Managing PDF content loading and cleaning
- Generating embedding text
- Providing metadata for storage

```python
class Paper:
    def __init__(self, data_dict, pdf_dir=None, include_pdf=False):
        # Initialize from metadata dictionary
        
    def embedding_text(self) -> str:
        # Return text for embedding - excluding author information
        # Focus on title, year, abstract, and content
        
    def metadata(self) -> dict:
        # Return metadata including author information for retrieval
```

#### 1.2 Utility Classes

Specialized classes for specific functionality:
- `PDFCleaner`: Handles PDF text extraction and cleaning
- `ArxivDownloader`: Manages downloads with rate limiting
- `AdaptiveRateLimiter`: Implements backoff strategies

### 2. Function Design

#### 2.1 Pure Functions

Where possible, functions should be pure (no side effects) and focused on a single responsibility:

```python
def clean_pdf_content(content: str, title: str, abstract: str) -> str:
    """Clean PDF content by removing redundant information and formatting artifacts."""
    # Implementation
```

#### 2.2 Batch Processing Functions

For operations on collections, implement batch processing with progress tracking:

```python
def process_batch(items: List[Dict], output_dir: str, batch_size: int = 10) -> Tuple[List[str], List[str]]:
    """Process a batch of items with progress tracking."""
    # Implementation with tqdm for progress
```

## Coding Style Guidelines

### 1. Python Conventions

#### 1.1 Naming Conventions

- **Classes**: CapWords convention (`Paper`, `PDFCleaner`)
- **Functions/Methods**: snake_case (`create_embeddings`, `process_batch`)
- **Variables**: snake_case (`paper_id`, `embedding_data`)
- **Constants**: UPPER_CASE (`MAX_BATCH_SIZE`, `DEFAULT_CHUNK_SIZE`)

#### 1.2 Documentation

- **Docstrings**: Google style docstrings for all public functions and classes
- **Type Hints**: Use type hints for function parameters and return values
- **Comments**: Add comments for complex logic, not obvious functionality
- **Todo**: Don't add TODOs in the code, no one actually will do them in real life. Instead use issue tracker for tracking TODOs

```python
def create_embeddings(
    papers: List[Paper],
    batch_size: int = 32,
    chunk_mode: bool = False,
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Tuple[str, List[float], Dict]]:
    """Create embeddings for papers using the specified model.
    
    Args:
        papers: List of paper objects to create embeddings for
        batch_size: Number of papers to process in each batch
        chunk_mode: Whether to create separate embeddings for chunks
        chunk_size: Maximum number of words per chunk when chunk_mode is True
        chunk_overlap: Number of words to overlap between chunks
        
    Returns:
        List of tuples containing (paper_id, embedding_vector, metadata)
    """
    # Implementation
```

### 2. Code Organization

#### 2.1 File Structure

- One class per file for major classes
- Group related functions in a single file
- Keep files focused on a single responsibility

#### 2.2 Import Order

1. Standard library imports
2. Third-party library imports
3. Local application imports

```python
# Standard library
import os
import json
from typing import List, Dict

# Third-party libraries
import numpy as np
from tqdm import tqdm

# Local application
from paper import Paper
from helpers import load_data
```

## Error Handling and Resilience

### 1. Error Handling Strategy

- **Specific Exceptions**: Use specific exception types
- **Graceful Degradation**: Fall back to simpler methods when optimal approaches fail
- **Contextual Error Messages**: Include relevant context in error messages

### 2. Resilience Patterns

- **Rate Limiting**: Implement adaptive rate limiting for external APIs
- **Retries with Backoff**: Use exponential backoff for retrying failed operations
- **Resource Management**: Close resources properly with context managers

```python
def _extract_with_pypdf2(self, pdf_path):
    """Extract text using PyPDF2 with improved handling for different PDF quality scenarios."""
    try:
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            # Processing
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {str(e)}")
        # Fall back to alternative method
        return self._extract_with_fallback(pdf_path)
```

## Performance Considerations

### 1. Resource Management

- **Memory Efficiency**: Process large datasets in batches
- **Garbage Collection**: Explicitly call gc.collect() after large operations
- **Progress Tracking**: Use tqdm for progress tracking of long-running operations

### 2. Optimization Techniques

- **Vectorization**: Use numpy for vector operations
- **Parallelization**: Use multiprocessing for CPU-bound tasks
- **Caching**: Cache expensive computation results

## Documentation Standards

### 1. Code Documentation

- **Module Docstrings**: Include purpose and usage for each module
- **Function Docstrings**: Document parameters, return values, and exceptions
- **Complex Logic**: Add comments for complex algorithms

### 2. Project Documentation

- **README**: Overview, installation, and basic usage
- **CONTRIBUTING**: Guide for contributors
- **API Documentation**: Detailed API reference
- **Architecture Documentation**: System design and component interactions

# LHCb Finder - Contribution Guidelines

## Project Overview

LHCb Finder is a specialized search engine for research papers related to the Large Hadron Collider beauty experiment (LHCb). The project uses natural language processing and vector embeddings to enable semantic search across a corpus of scientific papers. The goal is to later develop integration with LLMs and provide framework for RAG for LHC experiments.

The project consists of two main components:
1. **Model Pipeline**: Handles data collection, processing, and embedding generation
2. **Web Application**: Provides a user interface for semantic search functionality

## Project Structure

```
lhcbfinder/
├── model/                # Data processing and embedding generation
│   ├── dataset.py        ## Data loading and ArXiv paper downloading
│   ├── embed.py          ## Embedding generation and management
│   ├── paper.py          ## Paper class and PDF content cleaning
│   ├── helpers.py        ## Utility functions
│   └── requirements.txt  ## Dependencies for model pipeline
└── website/              # Web interface
    ├── app.py            ## Flask web application
    ├── query_processor.py ## Search functionality
    ├── static/           ## CSS, JavaScript, and other static assets
    ├── templates/        ## HTML templates
    └── requirements.txt  ## Dependencies for web application
```

## Design Concepts

### 1. Separation of Concerns

The project maintains a clear separation between:
- **Data Processing** (model directory): Handles all aspects of acquiring and processing research paper data
- **User Interface** (website directory): Focuses on providing a clean, usable search experience

### 2. Paper Processing Pipeline

The paper processing workflow follows these steps:
1. **Collection**: Download metadata and PDFs from arXiv
2. **Cleaning**: Process PDF content to remove noise and normalize text
3. **Embedding**: Generate vector representations of papers for semantic search
4. **Storage**: Save embeddings to both local storage and vector database (Pinecone)


### 3. Embedding Strategy

The embedding approach:
- Focuses on `title`, `year`, `abstract`, and `content chunks` for semantic representation
- Excludes `author` information from embeddings to avoid skewing semantic relationships
- Stores `author` data as metadata for retrieval and display purposes
- Uses `chunking` for longer documents to maintain quality and relevance

## Coding Standards

### Python Style Guide

1. **PEP 8 Compliance**: Follow PEP 8 style guide for Python code
   - 4-space indentation
   - 79-character line length limit (flexible up to 100 characters when necessary)
   - Use clear, descriptive variable and function names

2. **Type Hints**: Use Python type hints for function parameters and return values
   ```python
   def process_data(data: List[Dict[str, Any]]) -> Tuple[List[Paper], List[str]]:
       # Function implementation
   ```

3. **Docstrings**: Include docstrings for all functions and classes following Google style
   ```python
   def function_name(param1: type, param2: type) -> return_type:
       """Brief description of the function.
       
       More detailed description if needed.
       
       Args:
           param1: Description of param1
           param2: Description of param2
           
       Returns:
           Description of return value
           
       Raises:
           ExceptionType: When and why this exception is raised
       """
       # Function implementation
   ```

### Error Handling

1. Use specific exception types rather than broad exception catching
2. Include informative error messages
3. Log errors appropriately for debugging
4. Implement graceful degradation for non-critical failures

### Code Organization

1. **Modularity**: Organize code into logical modules with clear responsibilities
2. **Classes vs. Functions**: Use classes for complex objects with state, functions for stateless operations
3. **File Structure**: Keep related functionality in the same file, split unrelated functionality

## Contributing Workflow

### Setting Up Development Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/MohamedElashri/lhcbfinder.git
   cd lhcbfinder
   ```

2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   # For model development
   cd model
   pip install -r requirements.txt
   
   # For website development
   cd website
   pip install -r requirements.txt
   ```

### Making Changes

1. Create a new branch for your feature or fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the coding standards
3. Write or update tests as needed
4. Ensure all tests pass
5. Submit a pull request with a clear description of changes

### Pull Request Guidelines

1. Provide a clear, descriptive title
2. Reference any related issues
3. Include a summary of changes and the problem they solve
4. List any dependencies or breaking changes
5. Include any necessary documentation updates

## Documentation

1. Update docstrings for any modified code
2. Update README.md if adding new features or changing usage
3. Document any new environment variables or configuration options

## Contact

For questions or clarifications about contribution guidelines, please contact the project maintainer.

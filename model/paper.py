# paper.py
import os
import PyPDF2
import re
from typing import Optional

class PDFCleaner:
    @staticmethod
    def clean_pdf_content(content: str, title: str, abstract: str) -> str:
        """
        Clean PDF content by removing redundant information and formatting artifacts.
        """
        if not content:
            return ""
            
        # Save original content length for comparison
        original_length = len(content)
            
        # Try to detect scrambled PDF content
        if len(content) > 100:
            # Calculate ratio of non-ASCII characters to detect PDFs with encoding issues
            non_ascii_count = sum(1 for char in content if ord(char) > 127)
            if non_ascii_count / len(content) > 0.3:  # More than 30% non-ASCII
                print(f"Detected possible encoding issues in PDF content")
                # Try to clean up encoding issues
                content = re.sub(r'[^\x00-\x7F]+', ' ', content)  # Remove non-ASCII chars
            
            # Check for unusual symbol density (potentially a sign of math formulas that didn't extract well)
            symbol_count = sum(1 for char in content if char in '{}\\^$&%#@!~*()<>[]|')
            if symbol_count / len(content) > 0.1:  # More than 10% symbols
                print(f"High density of special symbols in PDF content, may be poorly extracted math")
                # Try to clean up excessive symbols but keep equation structures
                content = re.sub(r'[\{\}\\\^\$\&\%\#\@\!\~\*\(\)\<\>\[\]\|]{3,}', ' [EQUATION] ', content)
                
        # Convert multiple spaces to single space
        content = re.sub(r'\s+', ' ', content)
        
        # Instead of using regex patterns which can be too aggressive, 
        # only remove exact matches for title and abstract
        if title in content:
            content = content.replace(title, '')
        if abstract in content:
            content = content.replace(abstract, '')
        
        # Apply less aggressive cleaning
        content = PDFCleaner._remove_headers_footers(content)
        content = PDFCleaner._remove_references_section(content)
        content = PDFCleaner._remove_affiliations_acknowledgments(content)
        content = PDFCleaner._clean_general_content(content)
        
        cleaned_content = content.strip()
        
        # If cleaning removed more than 90% of content, something went wrong
        # Use fallback cleaning instead
        if len(cleaned_content) < 0.1 * original_length:
            print(f"Warning: Cleaning removed {original_length - len(cleaned_content)} chars ({(original_length - len(cleaned_content))/original_length*100:.1f}% of content)")
            print(f"Using fallback minimal cleaning instead")
            cleaned_content = PDFCleaner._fallback_cleaning(content, title, abstract)
            
        return cleaned_content
        
    @staticmethod
    def _fallback_cleaning(content: str, title: str, abstract: str) -> str:
        """Apply less aggressive cleaning when standard cleaning yields too little content."""
        if not content:
            return ""
            
        # Simply preserve the original content with minimal modification
        # This ensures we have text to work with even if it contains some noise
        
        # Just do basic whitespace normalization
        content = re.sub(r'\s+', ' ', content)
        
        # Remove only exact matches for title and abstract if they appear multiple times
        # (once is likely okay to keep for context)
        if title in content and content.count(title) > 1:
            # Keep the first occurrence, remove others
            first_pos = content.find(title)
            cleaned_content = content[:first_pos + len(title)]
            cleaned_content += content[first_pos + len(title):].replace(title, '')
            content = cleaned_content
            
        if abstract in content and content.count(abstract) > 1:
            # Keep the first occurrence, remove others
            first_pos = content.find(abstract)
            cleaned_content = content[:first_pos + len(abstract)]
            cleaned_content += content[first_pos + len(abstract):].replace(abstract, '')
            content = cleaned_content
        
        # Remove excessive newlines but preserve paragraph structure
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Remove non-printable characters that might interfere with embedding
        content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
        
        return content.strip()

    @staticmethod
    def _remove_headers_footers(content: str) -> str:
        """Remove page numbers, running headers, and footers."""
        # Remove page numbers (various formats)
        content = re.sub(r'\b\d+\s*(?:of|/)\s*\d+\b', '', content)
        content = re.sub(r'\bpage\s+\d+\b', '', content, flags=re.IGNORECASE)
        
        # Remove arXiv headers
        content = re.sub(r'arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w\-\.]+\].*', '', content)
        
        # Remove common headers/footers in physics papers
        patterns = [
            r'Submitted to.*\n?',
            r'Preprint typeset using.*\n?',
            r'©.*\d{4}.*\n?',
            r'This work is licensed under.*\n?',
            r'Prepared for submission to.*\n?',
            r'To appear in.*\n?',
            r'Published in.*\n?',
            r'Proceedings of.*\n?',
            r'CERN-.*\n?',
            r'LHCb-.*\n?',
            r'EUROPEAN ORGANIZATION FOR NUCLEAR RESEARCH.*\n?',
            r'LHCB-.*\n?',
            r'Submitted to:.*\n?',
            r'Conference:.*\n?',
            r'Journal:.*\n?',
            r'Keywords:.*\n?'
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
        return content

    @staticmethod
    def _remove_references_section(content: str) -> str:
        """Remove the references section."""
        # Try to find and remove the references section - common in physics papers
        patterns = [
            r'References.*?\n(?:(?:[A-Z][\w\s,\.\-\(\)]+\[\d+\].*?\n)+|(?:\[\d+\].*?\n)+)',
            r'Bibliography.*?\n(?:(?:[A-Z][\w\s,\.\-\(\)]+\[\d+\].*?\n)+|(?:\[\d+\].*?\n)+)',
            r'REFERENCES.*?(?=\n\n|\Z)',  # Until next double newline or end of text
            r'References\s+\[\d+\].*?(?=\n\n|\Z)',  # Physics style references
            r'\d+\.\s+References.*?(?=\n\n|\Z)',
            r'\[\d+\]\s+[A-Z].*?(?:\n\s+[a-z].*?)+(?=\n\n|\[\d+\]|\Z)', # Common reference format
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        return content

    @staticmethod
    def _remove_affiliations_acknowledgments(content: str) -> str:
        """Remove author affiliations and acknowledgments sections."""
        # Remove affiliations - common in physics papers
        patterns = [
            r'(?:Department|University|Institute|Laboratory|CERN)[\w\s,\-\.]+\n',
            r'E-mail:.*?\n',
            r'(?:Received|Accepted):.*?\d{4}.*?\n',
            r'Acknowledgments?.*?\n.*?\n\n',
            r'Acknowledgements?.*?\n.*?\n\n',
            r'Abstract[\s\n]+.*?\n\n',  # Remove duplicated abstract
            r'Author contributions:.*?\n\n',
            r'Corresponding author:.*?\n',
            r'\*?Corresponding author\.?.*?\n',
            r'Speaker\.?.*?\n',
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        return content

    @staticmethod
    def _clean_general_content(content: str) -> str:
        """Clean general content issues."""
        # Remove URLs
        content = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', content)
        
        # Remove email addresses
        content = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', content)
        
        # Remove figure and table captions
        content = re.sub(r'Figure \d+[.:]\s*.*?\n', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'Table \d+[.:]\s*.*?\n', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'Fig\.\s*\d+[.:]\s*.*?\n', '\n', content, flags=re.IGNORECASE)
        content = re.sub(r'Tab\.\s*\d+[.:]\s*.*?\n', '\n', content, flags=re.IGNORECASE)
        
        # Fix spacing around punctuation
        content = re.sub(r'\s+([,\.;:\?\!])', r'\1', content)
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'\s{2,}', ' ', content)
        
        return content

    @staticmethod
    def truncate_content(content: str, max_words: int = 1000) -> str:
        """Truncate content to a maximum number of words while preserving complete sentences."""
        words = content.split()
        if len(words) <= max_words:
            return content
            
        # Join first max_words and find the last complete sentence
        truncated = ' '.join(words[:max_words])
        last_sentence = re.search(r'^.*?[.!?](?=\s|$)', truncated[::-1])
        
        if last_sentence:
            # Reverse back the found last sentence and the whole text
            last_sentence_pos = len(truncated) - last_sentence.end()
            return truncated[:last_sentence_pos + 1]
        
        return truncated
        
    @staticmethod
    def chunk_content(content: str, chunk_size: int = 500, overlap: int = 100) -> list:
        """
        Split content into overlapping chunks for efficient embedding and retrieval.
        
        Args:
            content: The text content to chunk
            chunk_size: Maximum number of words per chunk
            overlap: Number of overlapping words between consecutive chunks
            
        Returns:
            List of text chunks
        """
        if not content:
            return []
            
        words = content.split()
        if len(words) <= chunk_size:
            return [content]
            
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            if len(chunk_words) < 50:  # Skip very small chunks at the end
                break
                
            # Try to end each chunk at a sentence boundary if possible
            chunk_text = ' '.join(chunk_words)
            sentence_end = re.search(r'[.!?]\s+[A-Z]', chunk_text[::-1])
            
            if sentence_end and sentence_end.start() < len(chunk_text) // 3:  # Only if reasonably close to end
                end_pos = len(chunk_text) - sentence_end.start()
                chunk_text = chunk_text[:end_pos]
                
            chunks.append(chunk_text)
            
        return chunks

class Paper:
    def __init__(self, data_dict, pdf_dir: Optional[str] = None, include_pdf: bool = False):
        """
        Initialize a Paper object from a dictionary and optionally load PDF content.
        Args:
            data_dict: Dictionary containing paper metadata
            pdf_dir: Optional directory path containing PDF files
            include_pdf: Whether to include PDF content in embeddings
        """
        self.id = data_dict["id"]
        self.categories = data_dict["categories"].lower().split()
        
        # Replace various LaTeX encodings and normalize whitespace in title
        self.title = data_dict["title"].replace('\n', ' ')
        self.title = " ".join(self.title.split())
        
        # Same for abstract
        abstract = data_dict.get("abstract", "").replace('\n', ' ')
        self.abstract = " ".join(abstract.split())
        
        # Get update date for determining year and month
        update_date = data_dict.get("update_date", "")
        if update_date and len(update_date) >= 7:  # Format: YYYY-MM-DD
            try:
                date_parts = update_date.split('-')
                self.year = int(date_parts[0])
                self.month = date_parts[1]  # Keep as string: "01", "02", etc.
            except (IndexError, ValueError):
                self.year = 0
                self.month = "01"  # Default to January
        else:
            # Fallback: Try to extract from versions if available
            try:
                if "versions" in data_dict and data_dict["versions"]:
                    created_date = data_dict["versions"][0].get("created", "")
                    if created_date:
                        parts = created_date.split()
                        if len(parts) >= 3:
                            self.month = parts[2]  # Month name
                            self.year = int(parts[3])  # Year
                        else:
                            self.month = "Jan" 
                            self.year = 0
                    else:
                        self.month = "Jan"
                        self.year = 0
                else:
                    self.month = "Jan" 
                    self.year = 0
            except Exception:
                self.month = "Jan"
                self.year = 0
        
        # Parse authors (needed for embedding)
        try:
            authors_parsed = data_dict.get("authors_parsed", [])
            if authors_parsed:
                authors = [author[::-1][1:] for author in authors_parsed]  # Reverses [Last, First, Middle]
                authors = [" ".join(author).strip() for author in authors]
                self.authors_string = ", ".join(authors)
            else:
                # Fallback to raw authors string if parsed not available
                self.authors_string = data_dict.get("authors", "Unknown authors")
        except Exception as e:
            print(f"Error parsing authors for {self.id}: {str(e)}")
            self.authors_string = "Unknown authors"
            
        # Store raw and cleaned PDF content
        self._pdf_content = ""  # Store raw PDF content
        self._cleaned_pdf_content = ""  # Store cleaned PDF content
        
        # No need to set has_valid_id - it's a property that will be calculated when accessed
        
        # Process PDF content if needed
        if include_pdf and pdf_dir:
            self._load_and_clean_pdf(pdf_dir)
            
    def has_category(self, categories):
        """Checks if this paper is in at least one category from `categories`."""
        for category in categories:
            if category in self.categories:
                return True
        return False

    @property
    def has_valid_id(self):
        """Returns True if "this paper has a valid ID."""
        # Check if the ID is in the expected format
        if not self.id:
            return False
            
        # Valid arxiv IDs can be in formats like:
        # - New format: YYMM.NNNNN (e.g., 2101.12345)
        # - Old format: category/YYMMNNN (e.g., hep-ex/0101001)
        # - Old format without category: YYMMNNN (e.g., 0101001)
        
        if '/' in self.id:  # Category prefix format
            return True
            
        # Check for new format YYMM.NNNNN
        if re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', self.id):
            return True
            
        # Check for 7-digit format YYMMNNN
        if re.match(r'^\d{7}(v\d+)?$', self.id):
            return True
            
        return False

    @property
    def is_lhcb_related(self):
        """Returns True if "lhcb" is found in either the title or abstract (case-insensitive)."""
        return ("lhcb" in self.title.lower()) or ("lhcb" in self.abstract.lower())

    def _load_and_clean_pdf(self, pdf_dir: str) -> bool:
        """
        Load and clean the PDF content for this paper.
        
        Args:
            pdf_dir: Directory containing the PDF files
            
        Returns:
            bool: True if PDF was loaded successfully, False otherwise
        """
        if not pdf_dir:
            self._cleaned_pdf_content = ""
            return False
            
        # Replace slashes with underscores in paper ID for filename
        safe_paper_id = self.id.replace('/', '_')
        pdf_path = os.path.join(pdf_dir, f"{safe_paper_id}.pdf")
        
        if not os.path.exists(pdf_path):
            # Don't print error message here - the script handles missing PDFs properly now
            return False
            
        if os.path.getsize(pdf_path) < 1000:  # Skip tiny PDFs (likely corrupted)
            print(f"PDF file for {self.id} is too small, likely corrupted.")
            return False
            
        try:
            # Try extraction methods in order of preference
            extraction_methods = [
                self._extract_with_pypdf2,
                self._extract_with_fallback
            ]
            
            for method in extraction_methods:
                try:
                    cleaned_text = method(pdf_path)
                    if cleaned_text and len(cleaned_text) >= 200:
                        self._cleaned_pdf_content = cleaned_text
                        return True
                except Exception as e:
                    print(f"Error with extraction method {method.__name__} for {self.id}: {str(e)}")
                    continue
                    
            print(f"All extraction methods failed for {self.id}")
            return False
            
        except Exception as e:
            print(f"Unexpected error processing PDF for {self.id}: {str(e)}")
            return False
            
    def _extract_with_pypdf2(self, pdf_path):
        """Extract text using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                    print(f"PDF file for {self.id} has 0 pages.")
                    return None
                    
                # Extract text from all pages
                text = ""
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        print(f"Error extracting page {page_num} for {self.id}: {str(e)}")
                        continue
                
                if not text or len(text) < 200:
                    print(f"Failed to extract meaningful text from PDF for {self.id}")
                    return None
                
                # Store raw content before cleaning
                self._pdf_content = text
                
                # Clean the extracted text    
                cleaned_text = PDFCleaner.clean_pdf_content(text, self.title, self.abstract)
                
                # If cleaned text is too short, use the fallback cleaning
                if not cleaned_text or len(cleaned_text) < 200:
                    print(f"Cleaned PDF text for {self.id} is too short, trying fallback cleaning")
                    cleaned_text = PDFCleaner._fallback_cleaning(text, self.title, self.abstract)
                    
                    # If fallback cleaning also produces too little content, use the raw text as a last resort
                    if not cleaned_text or len(cleaned_text) < 200:
                        print(f"Fallback cleaning also produced short text, using raw PDF content for {self.id}")
                        # Just do minimal cleaning on raw text to ensure usability
                        raw_content = text.replace('\n', ' ')
                        raw_content = re.sub(r'\s+', ' ', raw_content)
                        cleaned_text = raw_content[:8000]  # Limit to 8000 chars to avoid too large embeddings
                
                return cleaned_text
                
        except Exception as e:
            print(f"PyPDF2 extraction error for {self.id}: {str(e)}")
            return None
        
    def _extract_with_fallback(self, pdf_path):
        """Fallback method using minimal cleaning to preserve content."""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                if len(reader.pages) == 0:
                    return None
                
                # Get only raw text with minimal processing
                text = ""
                for page_num in range(len(reader.pages)):
                    try:
                        page = reader.pages[page_num]
                        text += page.extract_text(0) + "\n"  # Extract with simpler settings
                    except:
                        pass
                        
                if not text or len(text) < 200:
                    return None
                    
                # Apply very minimal cleaning
                text = re.sub(r'\s+', ' ', text)  # Remove excess whitespace
                
                # Ensure text is not too large for embedding (max ~8000 words)
                words = text.split()
                if len(words) > 8000:
                    text = ' '.join(words[:8000])
                    
                return text
                
        except Exception as e:
            print(f"Fallback extraction error for {self.id}: {str(e)}")
            return None

    @property
    def has_pdf_content(self):
        """Check if the paper has PDF content loaded."""
        return self._cleaned_pdf_content is not None and len(self._cleaned_pdf_content) > 0

    @property
    def embedding_text(self):
        """Text used for embedding, combining metadata with cleaned PDF content if available."""
        text = [
            f"Title: {self.title}",
            f"Authors: {self.authors_string}",
            f"Year: {self.year}",
            f"Month: {self.month}",
            f"Abstract: {self.abstract}"
        ]
        
        if self._cleaned_pdf_content:
            text.append(f"Content: {self._cleaned_pdf_content}")
            
        return " ".join(text)

    @property
    def metadata(self):
        """Metadata dict that you can store in a DB or Pinecone."""
        return {
            "title": self.title,
            "authors": self.authors_string,
            "abstract": self.abstract,
            "year": self.year,
            "month": self.month,
            "has_pdf_content": bool(self._cleaned_pdf_content)
        }
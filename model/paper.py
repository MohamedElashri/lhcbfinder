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
            
        # Convert multiple spaces to single space
        content = re.sub(r'\s+', ' ', content)
        
        # Remove title and abstract as they're already in metadata
        # Use case-insensitive removal to catch variations
        title_pattern = re.escape(title.strip())
        abstract_pattern = re.escape(abstract.strip())
        content = re.sub(rf'{title_pattern}', '', content, flags=re.IGNORECASE)
        content = re.sub(rf'{abstract_pattern}', '', content, flags=re.IGNORECASE)
        
        # Remove common headers and footers
        content = PDFCleaner._remove_headers_footers(content)
        
        # Remove references section (often at the end)
        content = PDFCleaner._remove_references_section(content)
        
        # Remove author affiliations and acknowledgments
        content = PDFCleaner._remove_affiliations_acknowledgments(content)
        
        # Clean remaining content
        content = PDFCleaner._clean_general_content(content)
        
        return content.strip()
    
    @staticmethod
    def _remove_headers_footers(content: str) -> str:
        """Remove page numbers, running headers, and footers."""
        # Remove page numbers (various formats)
        content = re.sub(r'\b\d+\s*(?:of|/)\s*\d+\b', '', content)
        content = re.sub(r'\bpage\s+\d+\b', '', content, flags=re.IGNORECASE)
        
        # Remove arXiv headers
        content = re.sub(r'arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w\-\.]+\].*', '', content)
        
        # Remove common headers/footers
        patterns = [
            r'Submitted to.*\n?',
            r'Preprint typeset using.*\n?',
            r'©.*\d{4}.*\n?',
            r'This work is licensed under.*\n?'
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
            
        return content

    @staticmethod
    def _remove_references_section(content: str) -> str:
        """Remove the references section."""
        # Try to find and remove the references section
        patterns = [
            r'References.*?\n(?:(?:[A-Z][\w\s,\.\-\(\)]+\[\d+\].*?\n)+|(?:\[\d+\].*?\n)+)',
            r'Bibliography.*?\n(?:(?:[A-Z][\w\s,\.\-\(\)]+\[\d+\].*?\n)+|(?:\[\d+\].*?\n)+)',
            r'REFERENCES.*?(?=\n\n|\Z)',  # Until next double newline or end of text
        ]
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.DOTALL)
        return content

    @staticmethod
    def _remove_affiliations_acknowledgments(content: str) -> str:
        """Remove author affiliations and acknowledgments sections."""
        # Remove affiliations
        patterns = [
            r'(?:Department|University|Institute|Laboratory)[\w\s,\-\.]+\n',
            r'E-mail:.*?\n',
            r'(?:Received|Accepted):.*?\d{4}.*?\n',
            r'Acknowledgments?.*?\n.*?\n\n',
            r'Acknowledgements?.*?\n.*?\n\n'
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
        
        # Remove line breaks and excess whitespace in titles
        title = data_dict["title"].replace("\n", " ")
        self.title = " ".join(title.split())
        
        # Remove line breaks and excess whitespace in abstracts
        abstract = data_dict["abstract"].replace("\n", " ")
        self.abstract = " ".join(abstract.split())
        
        # Parse date from first version
        self.month = data_dict["versions"][0]["created"].split()[2]
        self.year = int(data_dict["versions"][0]["created"].split()[3])
        
        # Parse authors
        authors_parsed = data_dict["authors_parsed"]
        authors = [author[::-1][1:] for author in authors_parsed]  # Reverses [Last, First, Middle]
        authors = [" ".join(author).strip() for author in authors]
        self.authors_string = ", ".join(authors)

        # PDF content
        self._pdf_content = None
        self._cleaned_pdf_content = None
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
        """Checks if the ID is valid (not all uppercase or all lowercase)."""
        invalid_id = self.id.isupper() or self.id.islower()
        return not invalid_id

    @property
    def is_lhcb_related(self):
        """Returns True if "lhcb" is found in either the title or abstract (case-insensitive)."""
        return ("lhcb" in self.title.lower()) or ("lhcb" in self.abstract.lower())

    def _load_and_clean_pdf(self, pdf_dir: str) -> None:
        """Load and clean PDF content."""
        pdf_path = os.path.join(pdf_dir, f"{self.id}.pdf")
        if not os.path.exists(pdf_path):
            return

        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                content = []
                for page in pdf_reader.pages:
                    content.append(page.extract_text())
                self._pdf_content = " ".join(content)
                
                # Clean and store the content
                self._cleaned_pdf_content = self._clean_pdf_content(self._pdf_content)
                
        except Exception as e:
            print(f"Error processing PDF for {self.id}: {str(e)}")
            self._pdf_content = None
            self._cleaned_pdf_content = None

    def _clean_pdf_content(self, content: str) -> str:
        """Clean PDF content by removing redundant information and formatting artifacts."""
        if not content:
            return ""
            
        # Convert multiple spaces to single space
        content = re.sub(r'\s+', ' ', content)
        
        # Remove title and abstract as they're in metadata
        content = re.sub(re.escape(self.title), '', content, flags=re.IGNORECASE)
        content = re.sub(re.escape(self.abstract), '', content, flags=re.IGNORECASE)
        
        # Remove headers and footers
        content = self._remove_headers_footers(content)
        
        # Truncate to manageable size (~1500 words)
        return self._truncate_content(content, max_words=1500)

    def _remove_headers_footers(self, content: str) -> str:
        """Remove page numbers, headers, and footers."""
        patterns = [
            r'\b\d+\s*(?:of|/)\s*\d+\b',  # Page numbers
            r'\bpage\s+\d+\b',
            r'arXiv:\d{4}\.\d{4,5}v\d+\s+\[[\w\-\.]+\].*',  # arXiv headers
            r'Submitted to.*\n?',
            r'Preprint typeset using.*\n?',
            r'©.*\d{4}.*\n?',
            r'This work is licensed under.*\n?',
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',  # URLs
            r'[\w\.-]+@[\w\.-]+\.\w+'  # Email addresses
        ]
        
        for pattern in patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        return content

    def _truncate_content(self, content: str, max_words: int = 1500) -> str:
        """Truncate content to a maximum number of words while preserving complete sentences."""
        words = content.split()
        if len(words) <= max_words:
            return content
            
        # Join first max_words and find the last complete sentence
        truncated = ' '.join(words[:max_words])
        last_sentence = re.search(r'^.*?[.!?](?=\s|$)', truncated[::-1])
        
        if last_sentence:
            last_sentence_pos = len(truncated) - last_sentence.end()
            return truncated[:last_sentence_pos + 1]
        
        return truncated

    @property
    def embedding_text(self):
        """Text used for embedding, combining metadata with cleaned PDF content if available."""
        text = [
            f"Title: {self.title}",
            f"Authors: {self.authors_string}",
            f"Year: {self.year}",
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
import re
from bs4 import BeautifulSoup
import logging


class ArxivHTMLParser:
    """Parses ArXiv HTML content to extract clean text for embedding."""

    @staticmethod
    def clean_html_content(html_content: str) -> str:
        """
        Clean ArXiv HTML content by removing navigation, references, and formatting.
        Returns extracted text.
        """
        if not html_content:
            return ""

        soup = BeautifulSoup(html_content, "html.parser")

        # 1. Remove Navigation and Header/Footer
        for element in soup.select(
            ".ltx_page_navbar, .ltx_page_header, .ltx_page_footer, .ltx_bibliography, .ltx_ref, .ltx_cite"
        ):
            element.decompose()

        # 2. Extract Main Content
        # ArXiv HTML usually puts the main paper content in specific containers
        main_content = (
            soup.find("div", class_="ltx_page_main")
            or soup.find("div", class_="ltx_document")
            or soup
        )

        # 3. Process common structures

        # Remove MathML (often too verbose for embedding, keep separate alt text if available)
        # Note: Mathjax/MathML might be present. For embedding, we might want to keep the text representation or equation numbers.
        # Decisions: Let's remove detailed mathml blocks but keep the text flow.
        for math in main_content.select("math"):
            math.decompose()

        # 4. Extract Text
        text = main_content.get_text(separator=" ", strip=True)

        # 5. Post-process Text
        # Identify section headers?
        # For now, just basic cleaning of whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove Reference section markers if they leaked through
        # (References usually decompose above, but sometimes headers remain)
        text = re.sub(r"References\s*\[.*?\]", "", text)

        return text.strip()

    @staticmethod
    def chunk_content(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
        """Same chunking logic as before, but operating on the cleaner HTML-derived text."""
        if not text:
            return []

        words = text.split()
        if len(words) <= chunk_size:
            return [text]

        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) < 50:
                break

            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)

        return chunks

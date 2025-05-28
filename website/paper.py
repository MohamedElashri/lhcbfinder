#paper.py
class Paper(dict):
    def __init__(self, match):
        super().__init__()
        
        self.id = match["id"]
        self.score = round(match["score"], 2)
        
        metadata = match["metadata"]
        self.title = metadata["title"]
        self.authors = metadata["authors"]
        self.abstract = metadata["abstract"]
        self.year = metadata["year"]
        
        # Set parent_id if this is a chunk
        if "_chunk_" in self.id:
            self.parent_id = metadata.get("parent_id", self.id.split("_chunk_")[0])
        else:
            self.parent_id = None
        
        # Convert numeric month to month name
        month_raw = metadata["month"]
        self.month = self._convert_month_to_name(month_raw)
        
        # Extract PDF content if available
        self.has_pdf_content = "pdf_content" in metadata and bool(metadata["pdf_content"])
        # Store full PDF content
        self.pdf_content = metadata.get("pdf_content", "")
        # Store preview content (first 1000 chars for display purposes)
        if self.has_pdf_content and self.pdf_content:
            self.pdf_preview = self.pdf_content[:1000] + ("..." if len(self.pdf_content) > 1000 else "")
        else:
            self.pdf_preview = ""
        
        # Parse chunks if available (for chunked content)
        self.chunks = metadata.get("chunks", [])
        if self.chunks:
            # Store the chunk index for the current match (if it exists)
            self.chunk_index = metadata.get("chunk_index", 0)
            # Store total chunk count (if available)
            self.total_chunks = metadata.get("chunk_count", len(self.chunks))
        else:
            self.chunk_index = 0
            self.total_chunks = 0
            
        authors_parsed = self.authors.split(",")
        self.authors_parsed = [author.strip() for author in authors_parsed]
        
    def _convert_month_to_name(self, month):
        """Convert numeric or abbreviated month to full month name."""
        month_names = {
            "01": "Jan", "1": "Jan", "Jan": "Jan", "January": "Jan",
            "02": "Feb", "2": "Feb", "Feb": "Feb", "February": "Feb",
            "03": "Mar", "3": "Mar", "Mar": "Mar", "March": "Mar",
            "04": "Apr", "4": "Apr", "Apr": "Apr", "April": "Apr",
            "05": "May", "5": "May", "May": "May",
            "06": "Jun", "6": "Jun", "Jun": "Jun", "June": "Jun",
            "07": "Jul", "7": "Jul", "Jul": "Jul", "July": "Jul",
            "08": "Aug", "8": "Aug", "Aug": "Aug", "August": "Aug",
            "09": "Sep", "9": "Sep", "Sep": "Sep", "September": "Sep",
            "10": "Oct", "Oct": "Oct", "October": "Oct",
            "11": "Nov", "Nov": "Nov", "November": "Nov",
            "12": "Dec", "Dec": "Dec", "December": "Dec"
        }
        # Default to first month if unknown format
        return month_names.get(str(month), "Jan")
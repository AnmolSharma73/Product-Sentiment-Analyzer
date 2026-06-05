import io
import os
import mimetypes
from typing import List, Union, Any
import pandas as pd
import json
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import pdfplumber
import docx
import nltk
from utils.constants import TEXT_COLUMN_KEYWORDS
from utils.helpers import setup_logger

# Ensure NLTK punkt is downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = setup_logger(__name__)

class FileParseError(Exception):
    """Custom exception for file parsing errors."""
    pass

class FileParser:
    """
    Universal file ingestion module for Product Sentiment Analyzer.
    Supports CSV, TSV, Excel, JSON, JSONL, TXT, PDF, DOCX, XML, Parquet.
    """
    
    def __init__(self):
        self.supported_extensions = {
            '.csv', '.tsv', '.xlsx', '.xls', '.json', '.jsonl', 
            '.txt', '.pdf', '.docx', '.xml', '.parquet'
        }
    
    def _detect_file_type(self, file_obj: Any, filename: str) -> str:
        """Detects file type using extension and MIME type if available."""
        ext = os.path.splitext(filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(filename)
        
        # Streamlit UploadedFile has 'type' attribute
        if hasattr(file_obj, 'type') and file_obj.type:
            mime_type = file_obj.type
            
        if ext in self.supported_extensions:
            return ext
            
        raise FileParseError(
            f"Unsupported file format: {ext} (MIME: {mime_type}). "
            f"Supported formats are: {', '.join(self.supported_extensions)}"
        )
        
    def _find_text_column(self, columns: List[str]) -> str:
        """Auto-detects the review text column in tabular data."""
        for col in columns:
            col_lower = str(col).lower()
            for keyword in TEXT_COLUMN_KEYWORDS:
                if keyword in col_lower:
                    return col
        # Fallback to the first string column or just the first column
        return columns[0]

    def _parse_dataframe(self, df: pd.DataFrame) -> List[str]:
        """Extracts review texts from a DataFrame."""
        if df.empty:
            return []
        
        text_col = self._find_text_column(df.columns.tolist())
        logger.info(f"Auto-detected text column: '{text_col}'")
        
        # Drop missing values and convert to string
        texts = df[text_col].dropna().astype(str).tolist()
        # Filter out empty strings
        return [t.strip() for t in texts if t.strip()]

    def _split_unstructured_text(self, text: str) -> List[str]:
        """Splits unstructured text into individual reviews."""
        if not text.strip():
            return []
            
        # Try double newline first (paragraphs)
        parts = [p.strip() for p in text.split('\n\n') if p.strip()]
        if len(parts) > 1:
            return parts
            
        # Fallback to sentence tokenization
        sentences = nltk.sent_tokenize(text)
        return sentences

    def parse(self, file_obj: Any, filename: str) -> List[str]:
        """
        Main parse method to extract a list of review strings from various file formats.
        
        Args:
            file_obj: A file path (str), file-like object, or Streamlit UploadedFile.
            filename: The name of the file (used for extension and MIME detection).
            
        Returns:
            List[str]: A list of clean review strings.
            
        Raises:
            FileParseError: If the file format is unsupported or parsing fails.
        """
        ext = self._detect_file_type(file_obj, filename)
        logger.info(f"Parsing file: {filename} with extension {ext}")
        
        # Handle file paths vs file-like objects
        is_path = isinstance(file_obj, str)
        
        try:
            if ext == '.csv':
                df = pd.read_csv(file_obj)
                return self._parse_dataframe(df)
                
            elif ext == '.tsv':
                df = pd.read_csv(file_obj, sep='\t')
                return self._parse_dataframe(df)
                
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_obj)
                return self._parse_dataframe(df)
                
            elif ext == '.parquet':
                df = pd.read_parquet(file_obj)
                return self._parse_dataframe(df)
                
            elif ext == '.json':
                # Can be an array of objects or a single object with a data array
                df = pd.read_json(file_obj)
                return self._parse_dataframe(df)
                
            elif ext == '.jsonl':
                df = pd.read_json(file_obj, lines=True)
                return self._parse_dataframe(df)
                
            elif ext == '.txt':
                if is_path:
                    with open(file_obj, 'r', encoding='utf-8') as f:
                        text = f.read()
                else:
                    text = file_obj.read()
                    if isinstance(text, bytes):
                        text = text.decode('utf-8', errors='replace')
                return self._split_unstructured_text(text)
                
            elif ext == '.pdf':
                text = ""
                # pdfplumber expects a path or file-like object
                with pdfplumber.open(file_obj) as pdf:
                    for page in pdf.pages:
                        extracted = page.extract_text()
                        if extracted:
                            text += extracted + "\n\n"
                return self._split_unstructured_text(text)
                
            elif ext == '.docx':
                # python-docx can take path or file-like object
                doc = docx.Document(file_obj)
                text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                return self._split_unstructured_text(text)
                
            elif ext == '.xml':
                if is_path:
                    with open(file_obj, 'r', encoding='utf-8') as f:
                        content = f.read()
                else:
                    content = file_obj.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8', errors='replace')
                
                soup = BeautifulSoup(content, 'xml')
                # Extract all text nodes
                text = " ".join(soup.stripped_strings)
                # Split roughly into sentences since XML structure might be arbitrary
                return self._split_unstructured_text(text)
                
            else:
                raise FileParseError(f"Extension {ext} is in supported list but missing parser implementation.")
                
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {str(e)}")
            raise FileParseError(f"Failed to parse the file '{filename}'. Error: {str(e)}")

if __name__ == "__main__":
    # Test block
    parser = FileParser()
    
    # 1. Test CSV
    import io
    csv_data = "id,review_text,date\n1,Great battery life!,2023-01-01\n2,Terrible screen.,2023-01-02\n"
    csv_file = io.StringIO(csv_data)
    try:
        reviews = parser.parse(csv_file, "test.csv")
        print("CSV Parsed:", reviews)
    except Exception as e:
        print("CSV Error:", e)

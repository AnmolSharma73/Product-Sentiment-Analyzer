import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from typing import List, Optional, Any
import contractions
from utils.constants import NEGATION_WORDS
from utils.helpers import setup_logger

# Ensure NLTK punkt is downloaded for sentence tokenization
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

logger = setup_logger(__name__)

class Preprocessor:
    """
    Text preprocessing module for tokenization, cleaning, and normalization.
    Uses spaCy for robust NLP tasks and NLTK for sentence tokenization.
    """
    
    def __init__(self, nlp=None):
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                # Load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy en_core_web_sm model not found. Attempting to download...")
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
        # Get default spaCy stopwords
        self.default_stopwords = self.nlp.Defaults.stop_words
        
    def clean(self, text: Optional[str]) -> str:
        """
        Cleans text: lowercases, removes HTML, expands contractions,
        removes URLs, special chars, and fixes repeated characters.
        """
        if text is None or not isinstance(text, str):
            return ""
            
        text = text.lower().strip()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Expand contractions (e.g., can't -> cannot)
        text = contractions.fix(text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters except sentence punctuation (. ! ? , ')
        text = re.sub(r'[^a-z0-9\.\!\?\,\'\s]', ' ', text)
        
        # Fix repeated characters (e.g., sooooo -> soo)
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def tokenize(self, text: str) -> List[Any]:
        """
        Tokenizes text using spaCy and returns a list of spaCy Token objects.
        This provides POS tags and dependencies.
        """
        if not text:
            return []
        # Return the parsed document tokens
        return list(self.nlp(text))
        
    def get_sentences(self, text: str) -> List[str]:
        """Splits text into sentences using NLTK."""
        if not text:
            return []
        return sent_tokenize(text)
        
    def remove_stopwords(self, tokens: List[Any]) -> List[str]:
        """
        Removes stopwords but keeps critical negation words.
        Args:
            tokens: List of spaCy tokens or string tokens.
        Returns:
            List[str]: Filtered token strings.
        """
        filtered = []
        for token in tokens:
            # Handle both spaCy tokens and strings
            word = token.text if hasattr(token, 'text') else str(token)
            word_lower = word.lower()
            
            # Keep if it's a negation word OR not a stopword OR it's a punctuation
            if word_lower in NEGATION_WORDS or word_lower not in self.default_stopwords or not word.isalpha():
                if word.strip():
                    filtered.append(word)
        return filtered

    def batch_process(self, reviews: List[str]) -> pd.DataFrame:
        """
        Processes a list of reviews and returns a DataFrame with analysis columns.
        """
        results = []
        
        for idx, original in enumerate(reviews):
            if original is None or not isinstance(original, str) or not original.strip():
                continue
                
            cleaned = self.clean(original)
            sentences = self.get_sentences(cleaned)
            doc_tokens = self.tokenize(cleaned)
            
            # Extract simple string tokens for dataframe storage
            token_strs = [t.text for t in doc_tokens]
            filtered_tokens = self.remove_stopwords(doc_tokens)
            
            has_negation = any(t.lower() in NEGATION_WORDS for t in token_strs)
            
            results.append({
                'original': original,
                'cleaned': cleaned,
                'tokens': filtered_tokens, # Storing filtered words as strings
                'sentences': sentences,
                'token_count': len(token_strs),
                'has_negation': has_negation
            })
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    # Test block
    processor = Preprocessor()
    sample_text = "I REALLY love this phone!!!! But the battery can't last a day. It's sooooo bad... <br> Also see http://test.com"
    
    print("Original:", sample_text)
    cleaned = processor.clean(sample_text)
    print("Cleaned:", cleaned)
    
    sentences = processor.get_sentences(cleaned)
    print("Sentences:", sentences)
    
    tokens = processor.tokenize(cleaned)
    print("POS Tags:", [(t.text, t.pos_) for t in tokens[:5]])
    
    filtered = processor.remove_stopwords(tokens)
    print("Without Stopwords (keeping negations):", filtered)
    
    df = processor.batch_process([sample_text, "Great product. Highly recommend.", None, ""])
    print("\nBatch Processing DataFrame:")
    print(df.head())

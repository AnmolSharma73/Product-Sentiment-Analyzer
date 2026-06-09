import spacy
from typing import Dict, List, Any
from utils.constants import ASPECT_KEYWORDS
from models.absa_model import ABSAModel
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class AspectExtractor:
    """
    Extracts aspects and related opinion phrases from text.
    Uses a rule-based dependency parsing approach (spaCy) and 
    an optional BERT layer for advanced ABSA.
    """
    
    def __init__(self, nlp=None, use_bert=False):
        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                spacy.cli.download("en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
            
        # Build keyword → aspect mapping for n-gram matching
        self.aspect_mapping = {}
        self.all_keywords = set()
        self.max_keyword_len = 1  # track max n-gram size needed
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                kw_lower = kw.lower()
                self.aspect_mapping[kw_lower] = aspect
                self.all_keywords.add(kw_lower)
                word_count = len(kw_lower.split())
                if word_count > self.max_keyword_len:
                    self.max_keyword_len = word_count
                
        self.use_bert = use_bert
        self.absa_model = None
        
        if self.use_bert:
            logger.info("BERT is enabled. Initializing ABSA Model...")
            self.absa_model = ABSAModel()
            if not self.absa_model.load():
                logger.warning("Failed to load BERT ABSA model. Falling back to rule-based.")
                self.use_bert = False

    def _extract_opinion_phrases(self, doc: Any, aspect_token: Any) -> str:
        """
        Extracts the opinion phrase associated with an aspect token
        using spaCy dependency parsing to capture the full clause.
        """
        # Find the root verb of the clause containing the aspect
        clause_root = aspect_token
        while clause_root.pos_ not in ["VERB", "AUX"] and clause_root.head != clause_root:
            # Prevent going too far up the tree if we hit a conjunction that might link two different aspects
            if clause_root.dep_ == "ccomp" or clause_root.dep_ == "conj":
                break
            clause_root = clause_root.head
            
        # Get all tokens in this clause's subtree, ignoring other conjuncts 
        # to avoid pulling in sentiments for different aspects
        clause_tokens = []
        for t in clause_root.subtree:
            if t.dep_ == "conj" and t != clause_root and t != aspect_token:
                continue
            clause_tokens.append(t)
            
        # Sort tokens by their position in the document
        clause_tokens = sorted(clause_tokens, key=lambda x: x.i)
        phrase = " ".join([t.text for t in clause_tokens])
        
        return phrase if phrase else aspect_token.text

    def _generate_ngrams(self, words: List[str], n: int) -> List[str]:
        """Generates n-grams from a list of words."""
        return [" ".join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Rule-based extraction of aspects and their opinion phrases
        using n-gram matching for multi-word keyword support.
        
        Args:
            text (str): The review text.
            
        Returns:
            Dict[str, List[str]]: Mapping of standard aspect to list of phrases.
        """
        if not text or not isinstance(text, str):
            return {}
            
        text_lower = text.lower()
        doc = self.nlp(text_lower)
        aspects_found: Dict[str, List[str]] = {}
        
        # Split text into words for n-gram generation
        words = text_lower.split()
        
        # Track which word positions have been matched to avoid duplicates
        matched_positions = set()
        
        # Check n-grams from longest to shortest (greedy matching)
        for n in range(self.max_keyword_len, 0, -1):
            ngrams = self._generate_ngrams(words, n)
            for i, ngram in enumerate(ngrams):
                # Skip if any position in this n-gram is already matched
                positions = set(range(i, i + n))
                if positions & matched_positions:
                    continue
                    
                if ngram in self.aspect_mapping:
                    standard_aspect = self.aspect_mapping[ngram]
                    matched_positions.update(positions)
                    
                    # Find a representative token in the spaCy doc for opinion extraction
                    # Use the last word of the n-gram as the anchor token
                    anchor_word = ngram.split()[-1]
                    phrase = ngram  # default phrase is the keyword itself
                    
                    for token in doc:
                        if token.text == anchor_word and token.i >= i and token.i < i + n + 2:
                            phrase = self._extract_opinion_phrases(doc, token)
                            break
                    
                    if standard_aspect not in aspects_found:
                        aspects_found[standard_aspect] = []
                    
                    if phrase and phrase not in aspects_found[standard_aspect]:
                        aspects_found[standard_aspect].append(phrase)
                        
        return aspects_found

    def get_bert_sentiment(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Gets BERT-based sentiment for a specific aspect if enabled.
        """
        if self.use_bert and self.absa_model:
            return self.absa_model.predict(text, aspect)
        return {}

if __name__ == "__main__":
    extractor = AspectExtractor()
    sample = "The camera quality is amazing, but the battery drains very fast and the price is too high."
    print("Review:", sample)
    
    extracted = extractor.extract(sample)
    print("Extracted Aspects & Phrases:", extracted)
    
    if extractor.use_bert:
        print("\nBERT Sentiments:")
        for asp in extracted.keys():
            print(f"{asp}:", extractor.get_bert_sentiment(sample, asp))

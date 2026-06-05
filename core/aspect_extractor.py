import spacy
from typing import Dict, List, Any
from utils.constants import ASPECT_KEYWORDS, USE_BERT
from models.absa_model import ABSAModel
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class AspectExtractor:
    """
    Extracts aspects and related opinion phrases from text.
    Uses a rule-based dependency parsing approach (spaCy) and 
    an optional BERT layer for advanced ABSA.
    """
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
            
        self.aspect_mapping = {}
        for aspect, keywords in ASPECT_KEYWORDS.items():
            for kw in keywords:
                self.aspect_mapping[kw.lower()] = aspect
                
        self.use_bert = USE_BERT
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
        using spaCy dependency parsing.
        """
        opinion_words = []
        
        # Check children for adjectives/adverbs modifying the aspect
        for child in aspect_token.children:
            if child.dep_ in ['amod', 'advmod', 'acomp', 'xcomp']:
                opinion_words.append(child.text)
                # Check for negations attached to the modifier
                for sub_child in child.children:
                    if sub_child.dep_ == 'neg':
                        opinion_words.insert(0, sub_child.text)
        
        # Check head for verbs
        head = aspect_token.head
        if head.pos_ in ['VERB', 'ADJ']:
            opinion_words.append(head.text)
            for child in head.children:
                if child.dep_ == 'neg':
                    opinion_words.insert(0, child.text)
                elif child.dep_ in ['advmod', 'acomp'] and child != aspect_token:
                    opinion_words.append(child.text)
                    
        # Construct the phrase
        if opinion_words:
            # Simple heuristic: modifier + aspect
            return f"{' '.join(opinion_words)} {aspect_token.text}"
        
        return aspect_token.text

    def extract(self, text: str) -> Dict[str, List[str]]:
        """
        Rule-based extraction of aspects and their opinion phrases.
        
        Args:
            text (str): The review text.
            
        Returns:
            Dict[str, List[str]]: Mapping of standard aspect to list of phrases.
        """
        if not text or not isinstance(text, str):
            return {}
            
        doc = self.nlp(text.lower())
        aspects_found: Dict[str, List[str]] = {}
        
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN']:
                word = token.text
                if word in self.aspect_mapping:
                    standard_aspect = self.aspect_mapping[word]
                    phrase = self._extract_opinion_phrases(doc, token)
                    
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

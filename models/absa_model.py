import os
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.constants import BERT_MODEL_NAME
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class ABSAModel:
    """
    Wrapper for HuggingFace BERT-based Aspect-Based Sentiment Analysis.
    Uses yangheng/deberta-v3-base-absa-v1.1 or similar models.
    """
    
    def __init__(self, model_name: str = BERT_MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.is_loaded = False
        
    def load(self) -> bool:
        """Loads the tokenizer and model. Returns True if successful."""
        if self.is_loaded:
            return True
            
        try:
            logger.info(f"Loading ABSA model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Use TextClassificationPipeline
            self.classifier = pipeline(
                "text-classification", 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            self.is_loaded = True
            logger.info("ABSA model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load ABSA model '{self.model_name}': {e}")
            self.is_loaded = False
            return False
            
    def predict(self, text: str, aspect: str) -> Dict[str, Any]:
        """
        Predicts sentiment for a specific aspect in the text.
        Format typically expected by the selected DeBERTa model:
        text + '[SEP]' + aspect (or handled by the tokenizer's text_pair)
        """
        if not self.is_loaded:
            if not self.load():
                return {"label": "Neutral", "score": 0.0}
                
        try:
            # For yangheng/deberta-v3-base-absa-v1.1, the format is often:
            # "[CLS] text [SEP] aspect [SEP]"
            input_text = f"{text} [SEP] {aspect}"
            
            # Predict
            result = self.classifier(input_text)[0]
            
            # Map labels to our standard
            label = result['label'].lower()
            if "positive" in label:
                standard_label = "positive"
                score = result['score']
            elif "negative" in label:
                standard_label = "negative"
                score = -result['score']
            else:
                standard_label = "neutral"
                score = 0.0
                
            return {
                "sentiment": standard_label,
                "score": score,
                "confidence": result['score']
            }
            
        except Exception as e:
            logger.error(f"ABSA prediction error for aspect '{aspect}': {e}")
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}

if __name__ == "__main__":
    # Test block
    model = ABSAModel()
    if model.load():
        text = "The screen is amazing but the battery life is terrible."
        print(f"Text: {text}")
        print("Aspect: screen ->", model.predict(text, "screen"))
        print("Aspect: battery ->", model.predict(text, "battery"))

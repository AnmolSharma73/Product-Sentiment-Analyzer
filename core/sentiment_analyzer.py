import spacy
import pandas as pd
from typing import Dict, Any, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from core.aspect_extractor import AspectExtractor
from core.preprocessor import Preprocessor
from utils.constants import NEGATION_WORDS
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class SentimentAnalyzer:
    """
    Performs Aspect-Based Sentiment Analysis (ABSA) by combining 
    rule-based extraction, VADER sentiment scoring, and optional BERT scoring.
    """
    
    def __init__(self, use_bert=False):
        self.vader = SentimentIntensityAnalyzer()
        
        # Create ONE shared spaCy instance for all NLP components
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            nlp = spacy.load("en_core_web_sm")
        
        self.preprocessor = Preprocessor(nlp=nlp)
        self.extractor = AspectExtractor(nlp=nlp, use_bert=use_bert)
        
        # Basic emotion mapping
        self.emotion_keywords = {
            "frustration": ["annoyed", "frustrated", "angry", "hate", "terrible", "worst", "bad", "disgusting", "upset"],
            "satisfaction": ["satisfied", "happy", "love", "amazing", "great", "excellent", "perfect", "good"],
            "disappointment": ["disappointed", "sad", "let down", "unfortunate", "poor", "lacking"],
            "excitement": ["excited", "wow", "incredible", "can't wait", "awesome", "fantastic"]
        }

    def _get_emotions(self, text: str) -> List[str]:
        """Tags basic emotions based on keywords."""
        text_lower = text.lower()
        emotions = []
        for emotion, keywords in self.emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                emotions.append(emotion)
        return emotions

    def _get_vader_score(self, text: str) -> Dict[str, Any]:
        """Gets VADER sentiment score and standardizes it."""
        scores = self.vader.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
            
        return {
            "sentiment": sentiment,
            "score": compound,
            "confidence": abs(compound) if abs(compound) > 0.1 else 0.5
        }

    def analyze_review(self, review_text: str) -> Dict[str, Any]:
        """
        Analyzes a single review for overall sentiment and aspect-based sentiment.
        
        Args:
            review_text (str): The raw review text.
            
        Returns:
            Dict: Analysis results including overall sentiment and per-aspect sentiments.
        """
        if not review_text or not str(review_text).strip():
            return {
                "overall_sentiment": "neutral",
                "overall_score": 0.0,
                "confidence": 0.0,
                "aspects": {},
                "emotions": [],
                "has_negation": False
            }
            
        text = str(review_text)
        cleaned_text = self.preprocessor.clean(text)
        tokens = cleaned_text.split()
        
        # Edge Cases
        is_short = len(tokens) < 5
        is_all_caps = text.isupper() and len(text) > 5
        
        # Determine overall sentiment using cleaned text
        overall_vader = self._get_vader_score(cleaned_text)
        
        # Extract aspects and opinion phrases using cleaned text
        extracted_aspects = self.extractor.extract(cleaned_text)
        
        aspects_analysis = {}
        for aspect, phrases in extracted_aspects.items():
            # Create a context string from phrases for scoring
            context = " ".join(phrases) if phrases else cleaned_text
            
            # Primary score: VADER on context
            vader_result = self._get_vader_score(context)
            
            # Secondary score: BERT if available
            bert_result = self.extractor.get_bert_sentiment(cleaned_text, aspect)
            
            if bert_result:
                # Weighted combination if BERT is active
                final_score = (vader_result['score'] * 0.4) + (bert_result['score'] * 0.6)
                final_sentiment = "positive" if final_score >= 0.05 else ("negative" if final_score <= -0.05 else "neutral")
                final_confidence = bert_result['confidence']
            else:
                final_score = vader_result['score']
                final_sentiment = vader_result['sentiment']
                final_confidence = vader_result['confidence']
                
            aspects_analysis[aspect] = {
                "sentiment": final_sentiment,
                "score": round(final_score, 2),
                "phrases": phrases
            }

        return {
            "overall_sentiment": overall_vader['sentiment'],
            "overall_score": overall_vader['score'],
            "confidence": round(overall_vader['confidence'], 2),
            "aspects": aspects_analysis,
            "emotions": self._get_emotions(text),
            "has_negation": any(word in NEGATION_WORDS for word in tokens),
            "is_short": is_short,
            "is_all_caps": is_all_caps
        }

    def analyze_batch(self, reviews: List[str]) -> pd.DataFrame:
        """
        Analyzes a batch of reviews and returns a DataFrame.
        """
        results = []
        for idx, review in enumerate(reviews):
            analysis = self.analyze_review(review)
            
            # Flatten aspect dictionary for dataframe rows
            row = {
                "review_id": idx,
                "review_text": review,
                "overall_sentiment": analysis["overall_sentiment"],
                "overall_score": analysis["overall_score"],
                "confidence": analysis["confidence"],
                "emotions": ", ".join(analysis["emotions"]),
                "has_negation": analysis["has_negation"]
            }
            
            # Add aspect scores directly as columns
            for aspect, data in analysis["aspects"].items():
                row[f"aspect_{aspect}_sentiment"] = data["sentiment"]
                row[f"aspect_{aspect}_score"] = data["score"]
                row[f"aspect_{aspect}_phrases"] = " | ".join(data["phrases"])
                
            results.append(row)
            
        return pd.DataFrame(results)

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    samples = [
        "The camera is amazing but the battery life is terrible. Not happy with the price either.",
        "WOW I LOVE IT",
        "It's okay. Nothing special. Screen is fine.",
        "Worst phone ever. Complete garbage. Laggy interface."
    ]
    
    df = analyzer.analyze_batch(samples)
    print("Batch Analysis DataFrame:")
    print(df.to_string())

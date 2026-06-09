"""
Product Sentiment Analyzer — Core Package
==========================================
Contains the main processing modules for sentiment analysis:
  - file_parser: Universal file ingestion
  - preprocessor: Text cleaning and tokenization
  - aspect_extractor: Aspect-based keyword and BERT extraction
  - sentiment_analyzer: Polarity classification per aspect
  - insights_engine: Rule-based insights generation
  - report_generator: PDF / Excel / DOCX / JSON export
"""

from core.file_parser import FileParser
from core.preprocessor import Preprocessor
from core.aspect_extractor import AspectExtractor
from core.sentiment_analyzer import SentimentAnalyzer
from core.insights_engine import InsightsEngine
from core.report_generator import ReportGenerator

__all__ = [
    "FileParser",
    "Preprocessor",
    "AspectExtractor",
    "SentimentAnalyzer",
    "InsightsEngine",
    "ReportGenerator",
]

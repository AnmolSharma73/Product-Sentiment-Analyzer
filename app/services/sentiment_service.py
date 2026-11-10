"""
Sentiment service — thin wrapper around core/ modules.
Keeps FastAPI routes clean and business logic testable.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import pandas as pd

from core.sentiment_analyzer import SentimentAnalyzer
from app.api.schemas.schemas import AnalyzeResponse, AspectResult


# Module-level singletons to avoid reloading models on every request
_vader_analyzer: SentimentAnalyzer | None = None
_bert_analyzer: SentimentAnalyzer | None = None


def _get_analyzer(use_bert: bool = False) -> SentimentAnalyzer:
    global _vader_analyzer, _bert_analyzer
    if use_bert:
        if _bert_analyzer is None:
            _bert_analyzer = SentimentAnalyzer(use_bert=True)
        return _bert_analyzer
    else:
        if _vader_analyzer is None:
            _vader_analyzer = SentimentAnalyzer(use_bert=False)
        return _vader_analyzer


def analyze_single(review: str, use_bert: bool = False) -> AnalyzeResponse:
    """Analyze a single review and return a structured response."""
    start = time.perf_counter()
    analyzer = _get_analyzer(use_bert)
    result = analyzer.analyze_review(review)
    elapsed_ms = (time.perf_counter() - start) * 1000

    aspects: List[AspectResult] = []
    for aspect_name, data in result.get("aspects", {}).items():
        aspects.append(AspectResult(
            aspect=aspect_name,
            sentiment=data.get("sentiment", "neutral"),
            score=round(float(data.get("score", 0.0)), 4),
            confidence=round(float(data.get("confidence", 0.0)), 4),
            phrases=data.get("phrases", []),
        ))

    return AnalyzeResponse(
        review=review,
        overall_sentiment=result.get("overall_sentiment", "neutral"),
        overall_score=round(float(result.get("overall_score", 0.0)), 4),
        confidence=round(float(result.get("confidence", 0.0)), 4),
        emotions=result.get("emotions", []),
        aspects=aspects,
        processing_time_ms=round(elapsed_ms, 2),
    )


def analyze_batch(reviews: List[str], use_bert: bool = False) -> pd.DataFrame:
    """Analyze a batch of reviews and return raw DataFrame."""
    analyzer = _get_analyzer(use_bert)
    return analyzer.analyze_batch(reviews)

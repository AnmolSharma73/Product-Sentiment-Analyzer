"""
pytest suite — Unit tests for core ML modules and services.
Optimized with module-scoped fixtures for fast execution.
"""
from __future__ import annotations

import io
import pandas as pd
import pytest
from pydantic import ValidationError


# ── Shared Module Fixtures (loads once) ───────────────────────────────────────

@pytest.fixture(scope="module")
def shared_analyzer():
    from core.sentiment_analyzer import SentimentAnalyzer
    return SentimentAnalyzer(use_bert=False)


@pytest.fixture(scope="module")
def preprocessor(shared_analyzer):
    return shared_analyzer.preprocessor


@pytest.fixture(scope="module")
def aspect_extractor(shared_analyzer):
    return shared_analyzer.extractor


# ── Preprocessing ────────────────────────────────────────────────────────────

def test_preprocessor_basic(preprocessor):
    assert preprocessor.clean("  Hello World!  ") != ""


def test_preprocessor_empty(preprocessor):
    result = preprocessor.clean("")
    assert isinstance(result, str)


def test_preprocessor_contraction(preprocessor):
    res = preprocessor.clean("I can't wait for this!")
    assert "cannot" in res or "can" in res


# ── Aspect Extraction ─────────────────────────────────────────────────────────

def test_aspect_extractor_battery(aspect_extractor):
    result = aspect_extractor.extract("The battery lasts all day but the screen is dim.")
    assert "battery" in result or "display" in result


def test_aspect_extractor_empty(aspect_extractor):
    result = aspect_extractor.extract("")
    assert result == {}


def test_aspect_extractor_long_review(aspect_extractor):
    long_text = "Great product. " * 50
    result = aspect_extractor.extract(long_text)
    assert isinstance(result, dict)


def test_aspect_extractor_multiple_aspects(aspect_extractor):
    text = "Battery is great, camera is terrible, screen looks decent."
    result = aspect_extractor.extract(text)
    assert len(result) >= 2


def test_aspect_extractor_no_known_aspects(aspect_extractor):
    result = aspect_extractor.extract("This thing arrived in a box.")
    assert isinstance(result, dict)


# ── Sentiment Analyzer ────────────────────────────────────────────────────────

def test_sentiment_positive(shared_analyzer):
    result = shared_analyzer.analyze_review("I absolutely love this product! It's amazing.")
    assert result["overall_sentiment"] == "positive"


def test_sentiment_negative(shared_analyzer):
    result = shared_analyzer.analyze_review("This is terrible, completely broken garbage.")
    assert result["overall_sentiment"] == "negative"


def test_sentiment_empty_review(shared_analyzer):
    result = shared_analyzer.analyze_review("")
    assert "overall_sentiment" in result


def test_sentiment_very_short(shared_analyzer):
    result = shared_analyzer.analyze_review("ok")
    assert result["overall_sentiment"] in {"positive", "negative", "neutral"}


def test_batch_analysis_returns_dataframe(shared_analyzer):
    reviews = ["Great product", "Terrible quality", "It is what it is"]
    df = shared_analyzer.analyze_batch(reviews)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert "overall_sentiment" in df.columns


def test_batch_analysis_all_sentiments_valid(shared_analyzer):
    reviews = ["Amazing!", "Garbage.", "Meh."]
    df = shared_analyzer.analyze_batch(reviews)
    assert all(s in {"positive", "negative", "neutral"} for s in df["overall_sentiment"])


# ── Insights Engine ───────────────────────────────────────────────────────────

def test_insights_basic():
    from core.insights_engine import InsightsEngine
    engine = InsightsEngine()
    df = pd.DataFrame({
        "overall_sentiment": ["positive", "positive", "negative"],
        "aspect_battery_sentiment": ["positive", "positive", None],
        "aspect_camera_sentiment": [None, "negative", "negative"],
    })
    result = engine.generate_insights(df, "Test Product")
    assert "executive_summary" in result
    assert len(result["executive_summary"]) > 10


def test_insights_empty_df():
    from core.insights_engine import InsightsEngine
    engine = InsightsEngine()
    df = pd.DataFrame()
    result = engine.generate_insights(df, "Empty Product")
    assert "executive_summary" in result


# ── Schemas ────────────────────────────────────────────────────────────────────

def test_analyze_request_valid():
    from app.api.schemas.schemas import AnalyzeRequest
    req = AnalyzeRequest(review="Good product!", product_name="Phone")
    assert req.review == "Good product!"


def test_analyze_request_empty_fails():
    from app.api.schemas.schemas import AnalyzeRequest
    with pytest.raises(ValidationError):
        AnalyzeRequest(review="   ")


def test_analyze_request_too_long_fails():
    from app.api.schemas.schemas import AnalyzeRequest
    with pytest.raises(ValidationError):
        AnalyzeRequest(review="a" * 6000)


# ── File Parser ────────────────────────────────────────────────────────────────

def test_file_parser_csv():
    from core.file_parser import FileParser
    csv_content = b"review\nGreat product!\nTerrible quality\n"
    reviews = FileParser().parse(io.BytesIO(csv_content), "test.csv")
    assert len(reviews) == 2


def test_file_parser_txt():
    from core.file_parser import FileParser
    txt_content = b"Great product!\n\nTerrible quality!"
    reviews = FileParser().parse(io.BytesIO(txt_content), "test.txt")
    assert len(reviews) >= 1


def test_file_parser_unsupported_format():
    from core.file_parser import FileParser, FileParseError
    with pytest.raises(FileParseError):
        FileParser().parse(io.BytesIO(b"data"), "test.xyz")


# ── Data Validator ─────────────────────────────────────────────────────────────

def test_data_validator():
    from app.data.validation.data_validator import DataValidator
    df = pd.DataFrame({
        "feedback": ["Great quality", "Great quality", "", None, "Good value"],
        "rating": [5, 5, 1, 2, 4]
    })
    clean, report = DataValidator.validate_dataframe(df)
    assert len(clean) == 2
    assert report.duplicate_reviews == 1
    assert report.empty_reviews == 2


# ── Dataset Store (SQLite) ─────────────────────────────────────────────────────

def test_dataset_store():
    from app.data.storage import dataset_store
    dataset_store.save_dataset_meta("test_ds_unit", "test.csv", "Widget", "csv", 5)
    ds = dataset_store.get_dataset("test_ds_unit")
    assert ds is not None
    assert ds["name"] == "test.csv"
    assert dataset_store.delete_dataset("test_ds_unit") is True


# ── RAG Service ────────────────────────────────────────────────────────────────

def test_rag_no_index_returns_empty():
    from app.services import rag_service
    rag_service._faiss_index = None
    rag_service._review_store = []
    results = rag_service.semantic_search("battery life")
    assert results == []


def test_rag_query_no_index_returns_response():
    from app.services import rag_service
    rag_service._faiss_index = None
    rag_service._review_store = []
    response = rag_service.query("What do customers say about battery?")
    assert "Insufficient evidence" in response.answer or isinstance(response.answer, str)

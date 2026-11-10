"""FastAPI integration tests using TestClient."""
from __future__ import annotations
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from app.api.main import app
    return TestClient(app)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ready_ok(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert "checks" in r.json()


def test_analyze_sentiment(client):
    r = client.post("/api/v1/analyze-sentiment", json={
        "review": "The battery is absolutely amazing.",
        "product_name": "Smartphone"
    })
    assert r.status_code == 200
    data = r.json()
    assert "overall_sentiment" in data
    assert data["overall_sentiment"] in {"positive", "negative", "neutral"}


def test_analyze_sentiment_empty_fails(client):
    r = client.post("/api/v1/analyze-sentiment", json={"review": "   "})
    assert r.status_code == 422


def test_analyze_batch(client):
    r = client.post("/api/v1/analyze-batch", json={
        "reviews": ["Great!", "Terrible!", "Okay."],
        "product_name": "Widget"
    })
    assert r.status_code == 200
    assert r.json()["total"] == 3


def test_rag_query_no_data(client):
    r = client.post("/api/v1/rag/query", json={
        "question": "What do customers say about battery?"
    })
    assert r.status_code == 200
    assert "answer" in r.json()


def test_insights_generate(client):
    r = client.post("/api/v1/insights/generate", json={
        "reviews": [
            "Battery life is amazing!",
            "Camera is blurry.",
            "Great value for money."
        ],
        "product_name": "Test Phone"
    })
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert len(data["summary"]) > 5

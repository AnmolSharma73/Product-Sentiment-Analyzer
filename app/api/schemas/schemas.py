"""
Pydantic schemas for FastAPI request/response validation.
All API contracts are defined here.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


# ── Sentiment Analysis ──────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    review: str = Field(..., min_length=1, max_length=5000, description="Review text to analyze")
    product_name: str = Field(default="Unknown Product", max_length=200)
    use_bert: bool = Field(default=False, description="Use DeBERTa model instead of VADER")

    @field_validator("review")
    @classmethod
    def review_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Review text cannot be empty or whitespace only.")
        return v.strip()


class AspectResult(BaseModel):
    aspect: str
    sentiment: str  # positive | negative | neutral
    score: float
    confidence: float
    phrases: List[str] = []


class AnalyzeResponse(BaseModel):
    review: str
    overall_sentiment: str
    overall_score: float
    confidence: float
    emotions: List[str] = []
    aspects: List[AspectResult] = []
    processing_time_ms: float = 0.0


class BatchAnalyzeRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1, max_length=1000)
    product_name: str = Field(default="Unknown Product", max_length=200)
    use_bert: bool = False


# ── Insights ────────────────────────────────────────────────────────────────

class InsightFinding(BaseModel):
    aspect: str
    sentiment: str
    evidence_count: int
    insight: str


class Recommendation(BaseModel):
    priority: str  # high | medium | low
    recommendation: str
    evidence: List[str] = []


class InsightsResponse(BaseModel):
    summary: str
    key_findings: List[InsightFinding] = []
    recommendations: List[Recommendation] = []
    source: str = "rule_based"  # rule_based | llm


# ── RAG ─────────────────────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=1000)
    dataset_id: Optional[str] = None
    top_k: int = Field(default=10, ge=1, le=50)
    product_filter: Optional[str] = None


class CitedReview(BaseModel):
    review_id: str
    review_text: str
    similarity_score: float


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitedReview] = []
    evidence_count: int
    confidence: str = "low"  # low | medium | high
    source: str = "rag"


# ── Datasets ─────────────────────────────────────────────────────────────────

class DatasetMeta(BaseModel):
    dataset_id: str
    name: str
    created_at: datetime
    review_count: int
    file_type: str


class DatasetListResponse(BaseModel):
    datasets: List[DatasetMeta] = []


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


class ReadyResponse(BaseModel):
    status: str
    checks: Dict[str, Any] = {}


# ── Jobs ─────────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str  # pending | running | done | failed
    progress: int = 0
    total: int = 0
    created_at: datetime
    result_available: bool = False
    error: Optional[str] = None


# ── Error ────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    error: str
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None

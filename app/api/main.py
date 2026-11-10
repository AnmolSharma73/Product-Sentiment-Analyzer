"""
FastAPI main application.
All heavy ML models are loaded lazily — /health never blocks.
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.api.schemas.schemas import (
    AnalyzeRequest, AnalyzeResponse,
    BatchAnalyzeRequest,
    HealthResponse, ReadyResponse,
    InsightsResponse,
    RAGQueryRequest, RAGQueryResponse,
    ErrorResponse,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Aspect-Based Sentiment Analysis & RAG-Powered Customer Intelligence API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
origins = [o.strip() for o in settings.cors_origins.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request ID middleware ───────────────────────────────────────────────────
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-ms"] = str(elapsed)
    return response


# ── Global exception handler ───────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(error="Internal server error", details={"type": type(exc).__name__}).model_dump(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH & READINESS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Lightweight health check — always responds quickly without loading models."""
    return HealthResponse(status="ok", version=settings.app_version)


@app.get("/ready", response_model=ReadyResponse, tags=["System"])
async def ready():
    """Readiness check — verifies essential services are available."""
    checks: dict = {}

    # Check vector store
    try:
        from app.services import rag_service
        checks["vector_store"] = "ok" if rag_service._faiss_index is not None else "empty"
    except Exception:
        checks["vector_store"] = "not_loaded"

    # Check core NLP (only VADER — lightweight)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        SentimentIntensityAnalyzer()
        checks["vader"] = "ok"
    except Exception:
        checks["vader"] = "error"

    checks["llm"] = "configured" if settings.llm_enabled else "not_configured"

    status = "ready" if checks.get("vader") == "ok" else "degraded"
    return ReadyResponse(status=status, checks=checks)


# ═══════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/analyze-sentiment", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_sentiment(body: AnalyzeRequest):
    """Analyze a single review for overall and aspect-level sentiment."""
    try:
        from app.services import sentiment_service
        return sentiment_service.analyze_single(body.review, body.use_bert)
    except Exception as e:
        logger.error(f"analyze_sentiment failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/analyze-batch", tags=["Analysis"])
async def analyze_batch(body: BatchAnalyzeRequest):
    """Analyze a batch of reviews and return aggregated results."""
    try:
        from app.services import sentiment_service
        df = sentiment_service.analyze_batch(body.reviews, body.use_bert)
        return {"total": len(df), "results": df.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"analyze_batch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/insights/generate", response_model=InsightsResponse, tags=["Insights"])
async def generate_insights(body: BatchAnalyzeRequest):
    """Analyze reviews and generate structured insights."""
    try:
        from app.services import sentiment_service, insight_service
        df = sentiment_service.analyze_batch(body.reviews, body.use_bert)
        return insight_service.generate_insights(df, body.product_name)
    except Exception as e:
        logger.error(f"generate_insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# RAG
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/rag/index", tags=["RAG"])
async def rag_index(
    reviews: List[str],
    product_name: str = Query(default="Unknown"),
    dataset_id: str = Query(default="default"),
):
    """Index reviews into the FAISS vector store for semantic search."""
    try:
        from app.services import rag_service
        count = rag_service.index_reviews(reviews, product_name, dataset_id)
        return {"indexed": count, "dataset_id": dataset_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/rag/query", response_model=RAGQueryResponse, tags=["RAG"])
async def rag_query(body: RAGQueryRequest):
    """Answer a natural-language question about indexed customer reviews."""
    try:
        from app.services import rag_service
        return rag_service.query(
            question=body.question,
            top_k=body.top_k,
            dataset_id=body.dataset_id,
            product_filter=body.product_filter,
        )
    except Exception as e:
        logger.error(f"rag_query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/rag/dataset/{dataset_id}", tags=["RAG"])
async def delete_dataset(dataset_id: str):
    """Remove all indexed reviews for a given dataset_id."""
    try:
        from app.services import rag_service
        removed = rag_service.clear_dataset(dataset_id)
        return {"removed": removed, "dataset_id": dataset_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# DATASETS — Upload
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/datasets/upload", tags=["Datasets"])
async def upload_dataset(
    file: UploadFile = File(...),
    product_name: str = Query(default="Unknown"),
    dataset_id: str = Query(default=""),
):
    """
    Upload a CSV, TXT, or PDF file of reviews.
    Parses, analyzes, and indexes the reviews automatically.
    """
    import io, uuid as _uuid
    from core.file_parser import FileParser, FileParseError as CoreParseError

    if not dataset_id:
        dataset_id = str(_uuid.uuid4())[:8]

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "txt"
    allowed = {"csv", "txt", "pdf", "xlsx", "json", "jsonl", "docx"}
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: .{ext}")

    contents = await file.read()
    if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large.")

    try:
        parser = FileParser()
        reviews = parser.parse(io.BytesIO(contents), file.filename)
    except CoreParseError as e:
        raise HTTPException(status_code=422, detail=str(e))

    if not reviews:
        raise HTTPException(status_code=422, detail="No review text found in file.")

    # Analyze
    df = sentiment_service.analyze_batch(reviews)
    insights = insight_service.generate_insights(df, product_name)

    # Index for RAG
    indexed = rag_service.index_reviews(reviews, product_name, dataset_id)

    # Save dataset metadata to SQLite store
    from app.data.storage import dataset_store
    dataset_store.save_dataset_meta(
        dataset_id=dataset_id,
        name=file.filename,
        product_name=product_name,
        file_type=ext,
        review_count=len(reviews),
    )

    return {
        "dataset_id": dataset_id,
        "product_name": product_name,
        "review_count": len(reviews),
        "indexed_for_rag": indexed,
        "insights_summary": insights.summary,
    }


@app.get("/api/v1/datasets", tags=["Datasets"])
async def list_datasets():
    """Retrieve metadata of all uploaded datasets."""
    from app.data.storage import dataset_store
    return {"datasets": dataset_store.list_datasets()}


@app.get("/api/v1/datasets/{dataset_id}", tags=["Datasets"])
async def get_dataset(dataset_id: str):
    """Retrieve metadata for a specific dataset."""
    from app.data.storage import dataset_store
    meta = dataset_store.get_dataset(dataset_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Dataset not found.")
    return meta


@app.delete("/api/v1/datasets/{dataset_id}", tags=["Datasets"])
async def delete_dataset_endpoint(dataset_id: str):
    """Delete a dataset and remove its indexed vectors."""
    from app.data.storage import dataset_store
    dataset_store.delete_dataset(dataset_id)
    removed_vectors = rag_service.clear_dataset(dataset_id)
    return {"status": "deleted", "dataset_id": dataset_id, "vectors_removed": removed_vectors}


# ═══════════════════════════════════════════════════════════════════════════
# ASPECT EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class AspectExtractRequest(BaseModel):
    review: str = Field(..., min_length=1, max_length=5000)

@app.post("/api/v1/aspects/extract", tags=["Analysis"])
async def extract_aspects(body: AspectExtractRequest):
    """Extract aspects and syntactic opinion phrases from a review without full sentiment pass."""
    try:
        from core.aspect_extractor import AspectExtractor
        extractor = AspectExtractor(use_bert=False)
        extracted = extractor.extract(body.review)
        return {"review": body.review, "aspects": extracted}
    except Exception as e:
        logger.error(f"Aspect extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# ASYNC BATCH JOBS
# ═══════════════════════════════════════════════════════════════════════════

class JobCreateRequest(BaseModel):
    reviews: List[str] = Field(..., min_length=1)
    product_name: str = Field(default="Unknown Product")
    use_bert: bool = False

@app.post("/api/v1/jobs", tags=["Jobs"])
async def submit_job(body: JobCreateRequest):
    """Submit a batch of reviews for asynchronous background processing."""
    from app.services import job_service
    job_id = job_service.create_job(
        reviews=body.reviews,
        use_bert=body.use_bert,
        product_name=body.product_name,
    )
    return {"job_id": job_id, "status": "pending", "total": len(body.reviews)}


@app.get("/api/v1/jobs/{job_id}", tags=["Jobs"])
async def get_job_status_endpoint(job_id: str):
    """Check the status and progress of an asynchronous job."""
    from app.services import job_service
    status = job_service.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found.")
    return status


@app.get("/api/v1/jobs/{job_id}/results", tags=["Jobs"])
async def get_job_results_endpoint(job_id: str):
    """Fetch results of a completed batch job."""
    from app.services import job_service
    results = job_service.get_job_results(job_id)
    if not results:
        raise HTTPException(status_code=404, detail="Job not found.")
    return results


@app.post("/api/v1/jobs/{job_id}/cancel", tags=["Jobs"])
async def cancel_job_endpoint(job_id: str):
    """Cancel an active or queued batch job."""
    from app.services import job_service
    success = job_service.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job could not be canceled or does not exist.")
    return {"status": "canceled", "job_id": job_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)


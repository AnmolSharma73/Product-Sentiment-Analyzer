"""
Job service for asynchronous batch processing.
Provides in-memory threading queue with status tracking for local execution,
and bridges to Celery when ENABLE_ASYNC_JOBS=true.
"""
from __future__ import annotations

import concurrent.futures
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd

from app.api.schemas.schemas import JobStatus

_jobs_lock = threading.Lock()
_jobs_store: Dict[str, Dict[str, Any]] = {}
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)


def _process_batch_job(job_id: str, reviews: List[str], use_bert: bool, product_name: str):
    try:
        with _jobs_lock:
            if _jobs_store.get(job_id, {}).get("status") == "canceled":
                return
            _jobs_store[job_id]["status"] = "running"
            _jobs_store[job_id]["total"] = len(reviews)

        # Chunk processing to update progress
        chunk_size = 50
        all_results = []
        total = len(reviews)

        for i in range(0, total, chunk_size):
            with _jobs_lock:
                if _jobs_store.get(job_id, {}).get("status") == "canceled":
                    return

            chunk = reviews[i : i + chunk_size]
            from app.services import sentiment_service
            df_chunk = sentiment_service.analyze_batch(chunk, use_bert=use_bert)
            all_results.extend(df_chunk.to_dict(orient="records"))

            with _jobs_lock:
                _jobs_store[job_id]["progress"] = min(i + len(chunk), total)

        with _jobs_lock:
            _jobs_store[job_id]["status"] = "done"
            _jobs_store[job_id]["result_available"] = True
            _jobs_store[job_id]["results"] = all_results

    except Exception as e:
        with _jobs_lock:
            _jobs_store[job_id]["status"] = "failed"
            _jobs_store[job_id]["error"] = str(e)


def create_job(reviews: List[str], use_bert: bool = False, product_name: str = "Unknown") -> str:
    """Submit a batch review job for asynchronous processing."""
    job_id = str(uuid.uuid4())[:12]
    now = datetime.now(timezone.utc)
    with _jobs_lock:
        _jobs_store[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "progress": 0,
            "total": len(reviews),
            "created_at": now,
            "result_available": False,
            "error": None,
            "results": None,
        }

    _executor.submit(_process_batch_job, job_id, reviews, use_bert, product_name)
    return job_id


def get_job_status(job_id: str) -> Optional[JobStatus]:
    """Retrieve the current progress and status of a job."""
    with _jobs_lock:
        data = _jobs_store.get(job_id)
        if not data:
            return None
        return JobStatus(
            job_id=data["job_id"],
            status=data["status"],
            progress=data["progress"],
            total=data["total"],
            created_at=data["created_at"],
            result_available=data["result_available"],
            error=data.get("error"),
        )


def get_job_results(job_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve completed results for a job."""
    with _jobs_lock:
        data = _jobs_store.get(job_id)
        if not data:
            return None
        if data["status"] != "done":
            return {"status": data["status"], "results": None}
        return {"status": "done", "total": data["total"], "results": data.get("results", [])}


def cancel_job(job_id: str) -> bool:
    """Cancel an active or pending job."""
    with _jobs_lock:
        if job_id not in _jobs_store:
            return False
        curr_status = _jobs_store[job_id]["status"]
        if curr_status in ("done", "failed", "canceled"):
            return False
        _jobs_store[job_id]["status"] = "canceled"
        return True

"""
Centralized application configuration using Pydantic Settings.
All environment variables are defined here with sensible defaults.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_env: str = "development"
    app_name: str = "Customer Insights Copilot"
    app_version: str = "2.0.0"
    log_level: str = "INFO"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: str = "*"
    rate_limit: int = 60  # requests per minute per IP

    # Database
    database_url: str = f"sqlite:///{BASE_DIR}/data/copilot.db"

    # Redis (optional — for Celery batch jobs)
    redis_url: str = "redis://localhost:6379/0"
    enable_async_jobs: bool = False

    # ML Models
    sentiment_model: str = "vader"  # vader | deberta
    absa_model_path: str = "yangheng/deberta-v3-base-absa-v1.1"

    # Embeddings & Vector Store
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_path: str = str(BASE_DIR / "data" / "faiss_index")
    retrieval_top_k: int = 10
    similarity_threshold: float = 0.35

    # LLM (optional — rule-based fallback if not set)
    llm_provider: Optional[str] = None   # openai | anthropic | ollama
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4o-mini"
    llm_timeout: int = 30
    llm_max_retries: int = 2

    # Upload limits
    max_upload_size_mb: int = 50

    # Experiment tracking
    enable_experiment_tracking: bool = False
    mlflow_tracking_uri: str = str(BASE_DIR / "data" / "mlruns")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_provider and self.llm_api_key)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

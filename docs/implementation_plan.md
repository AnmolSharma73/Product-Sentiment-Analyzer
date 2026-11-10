# Implementation Plan — Customer Insights Copilot

## Phase 1: Repository Assessment (Complete)

### Existing Capabilities Identified
| Module | Location | Status |
|---|---|---|
| Streamlit UI | `app.py` | Working — preserved as-is |
| File Parser (CSV/PDF/DOCX/TXT/JSON/XML/Parquet) | `core/file_parser.py` | Working |
| Text Preprocessor | `core/preprocessor.py` | Working |
| Aspect Extractor (VADER + spaCy, n-gram matching) | `core/aspect_extractor.py` | Working |
| Sentiment Analyzer (VADER + optional DeBERTa) | `core/sentiment_analyzer.py` | Working |
| Rule-Based Insights Engine | `core/insights_engine.py` | Working |
| Report Generator (PDF/Excel/DOCX/JSON) | `core/report_generator.py` | Working |
| DeBERTa ABSA Wrapper | `models/absa_model.py` | Working (optional, ~500MB) |
| Aspect Keywords / Constants | `utils/constants.py` | Working |
| Logger helper | `utils/helpers.py` | Working |

### Technical Risks
- No FastAPI layer — Streamlit does all backend work (tightly coupled)
- No vector store or RAG capability
- No persistent data storage (SQLite/PostgreSQL)
- No test suite
- No Docker configuration
- No CI/CD
- Utility scripts (fix_indent.py, etc.) committed to root — need cleanup

## Phase 2: Architecture Refactoring

### New modules added
- `app/core/config.py` — Pydantic Settings
- `app/core/logging.py` — Structured logging
- `app/core/exceptions.py` — Custom exceptions
- `app/api/main.py` — FastAPI application
- `app/api/routes/` — Route handlers
- `app/api/schemas/` — Pydantic schemas
- `app/services/sentiment_service.py` — Thin service wrapping core modules
- `app/services/insight_service.py` — Rule-based + LLM insights
- `app/services/rag_service.py` — RAG pipeline

## Phase 3: ABSA Training Pipeline

- `scripts/train_absa.py` — SemEval-2014 / SST fine-tuning
- `scripts/evaluate_models.py` — Benchmark script
- MLflow experiment tracking

## Phase 4: RAG Pipeline

- `app/rag/embeddings.py` — sentence-transformers
- `app/rag/vector_store.py` — FAISS local index
- `app/rag/retrieval.py` — Top-k semantic search
- `app/rag/prompts.py` — Prompt templates

## Phase 5-8: Tests, Docker, CI/CD

- `tests/` — pytest suite
- `Dockerfile` + `docker-compose.yml`
- `.github/workflows/ci.yml`

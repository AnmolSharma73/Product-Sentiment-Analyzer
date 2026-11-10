# Architecture — Customer Insights Copilot

## Current Architecture (Baseline)

```
app.py (Streamlit)
  └── core/file_parser.py
  └── core/sentiment_analyzer.py
        └── core/aspect_extractor.py
        └── core/preprocessor.py
        └── models/absa_model.py (optional)
  └── core/insights_engine.py (rule-based)
  └── core/report_generator.py
```

## Target Architecture

```
                          ┌──────────────────────┐
                          │   Streamlit Frontend  │
                          │   app.py / pages/     │
                          └────────┬─────────────┘
                                   │ HTTP (FastAPI)
                          ┌────────▼─────────────┐
                          │   FastAPI Backend     │
                          │   app/api/main.py     │
                          │   /health /ready      │
                          │   /analyze /insights  │
                          │   /datasets /rag/query│
                          └────────┬─────────────┘
               ┌───────────────────┼──────────────────┐
               │                   │                  │
    ┌──────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
    │ SentimentService│  │InsightService │  │  RAGService   │
    │ (VADER+spaCy)   │  │(Rule+LLM)    │  │(FAISS+LLM)   │
    └──────────┬──────┘  └────────┬──────┘  └────────┬──────┘
               │                   │                  │
    ┌──────────▼──────┐  ┌────────▼──────┐  ┌────────▼──────┐
    │  core/ modules  │  │ Pydantic Val  │  │ Sentence-Trans│
    │  (preserved)    │  │ LLM Provider  │  │ FAISS Index   │
    └─────────────────┘  └───────────────┘  └───────────────┘
```

## Key Design Decisions

1. **FastAPI as backend** — Separates ML logic from UI; enables REST API for external consumers
2. **FAISS as vector store** — Local, no cloud dependency, 100% offline capable
3. **Sentence-transformers for embeddings** — CPU-compatible, no API key needed
4. **Rule-based fallback always present** — App works without LLM API keys
5. **SQLite default, PostgreSQL-compatible** — via SQLAlchemy ORM abstraction
6. **Existing core/ modules preserved** — Not rewritten, only wrapped in service layer

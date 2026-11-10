<div align="center">

# 🤖 Customer Insights Copilot
### Aspect-Based Sentiment Analysis (ABSA) & RAG-Powered Customer Intelligence

**An enterprise-grade customer intelligence platform that transforms unstructured reviews into granular aspect sentiments, grounded executive insights, and interactive semantic question answering—completely offline.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anmolsharma73-product-sentiment-analyzer-app-udb4zy.streamlit.app/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB.svg?logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[🚀 Live Public Demo](https://anmolsharma73-product-sentiment-analyzer-app-udb4zy.streamlit.app/) • [📖 Architecture Docs](docs/architecture.md) • [📋 Implementation Plan](docs/implementation_plan.md) • [⚖️ Technical Decisions (ADR)](docs/technical_decisions.md)

</div>

---

## 🌟 Overview

**Customer Insights Copilot** elevates basic sentiment analysis into a deep customer intelligence copilot. Instead of just giving a coarse "positive" or "negative" rating for an entire review, the platform decomposes customer feedback into specific product aspects (e.g. *Battery*, *Camera*, *Display*, *Performance*, *Price*, *Build*, *Software*, *Delivery*) and extracts opinion phrases with syntactic dependency parsing.

Furthermore, it integrates an **offline Retrieval-Augmented Generation (RAG) engine** backed by `sentence-transformers` and `FAISS`, enabling product managers, CX researchers, and developers to converse directly with customer reviews in natural language and receive evidence-grounded answers with verifiable review citations.

---

## ✨ Key Capabilities

| Capability | Description | Technology Stack |
| :--- | :--- | :--- |
| **Aspect-Based Sentiment (ABSA)** | Fine-grained aspect extraction and sentiment polarity scoring per product attribute. | spaCy, VADER, DeBERTa-v3 |
| **Offline RAG Customer Copilot** | Natural-language Q&A across customer reviews with exact review ID citations and confidence metrics. | `sentence-transformers`, `faiss-cpu` |
| **Dual Inference Engine** | Lightning-fast rule-based pipeline for instant processing (~3ms/review) + Transformer mode for deep contextual understanding. | VADER + spaCy / DeBERTa-v3 |
| **Automated Executive Insights** | Automatically generated executive summary, top strengths, critical issues, risk analysis, and priority recommendations. | Deterministic Stats + Optional LLM |
| **Production REST API** | Full OpenAPI-compliant REST API with endpoints for single/batch inference, RAG queries, dataset uploads, and background jobs. | FastAPI, Pydantic v2, Uvicorn |
| **Model Fine-Tuning & Benchmarking** | Reproducible ABSA fine-tuning pipeline supporting SemEval datasets with MLflow experiment tracking. | PyTorch, Hugging Face, MLflow |
| **Universal Ingestion** | Drag-and-drop support for CSV, Excel (`.xlsx`, `.xls`), JSON/JSONL, PDF, Word (`.docx`), Parquet, XML, and TXT. | `openpyxl`, `pdfplumber`, `pyarrow` |
| **Multi-Format Export** | Export actionable findings directly to PDF, Excel, Word (DOCX), or JSON. | ReportLab, `python-docx` |
| **100% Offline & Private** | Zero mandatory cloud dependencies. Customer feedback never leaves your local infrastructure. | Local CPU/GPU |

---

## 📐 System Architecture

```text
Product-Sentiment-Analyzer/
├── app/
│   ├── api/                     ← FastAPI application & REST endpoints
│   │   ├── main.py              ← API app definition, routes, CORS & middleware
│   │   └── schemas/schemas.py   ← Pydantic request & response contracts
│   ├── core/                    ← Core configuration & exception hierarchy
│   │   ├── config.py            ← Centralized Pydantic Settings (.env)
│   │   └── exceptions.py        ← Structured domain exception classes
│   ├── data/                    ← Data layer & quality audits
│   │   ├── storage/dataset_store.py  ← SQLite dataset repository
│   │   └── validation/data_validator.py ← Tabular data audit & clean
│   └── services/                ← Business service layer
│       ├── sentiment_service.py ← ABSA analysis coordinator
│       ├── insight_service.py   ← Rule-based & LLM insight engine
│       ├── rag_service.py       ← FAISS index, semantic search & citations
│       └── job_service.py       ← Asynchronous batch processing queue
│
├── core/                        ← Low-level NLP & Ingestion components
│   ├── file_parser.py           ← Universal format file extractor
│   ├── preprocessor.py          ← NLP cleaning & tokenization
│   ├── aspect_extractor.py      ← Dependency tree aspect extraction
│   ├── sentiment_analyzer.py    ← VADER & DeBERTa hybrid classifier
│   ├── insights_engine.py       ← Heuristic synthesis engine
│   └── report_generator.py      ← PDF, Excel, Word, JSON report builder
│
├── models/                      ← Model loaders & fine-tuned checkpoints
│   └── absa_model.py            ← HuggingFace DeBERTa wrapper
│
├── scripts/                     ← MLOps scripts
│   ├── train_absa.py            ← Fine-tuning pipeline with MLflow
│   └── evaluate_models.py       ← Benchmark script comparing models
│
├── tests/                       ← Comprehensive test suite
│   ├── unit/                    ← Core unit tests
│   └── api/                     ← FastAPI endpoint integration tests
│
├── docs/                        ← Architecture & Decision Records
│   ├── architecture.md          ← Baseline vs target architecture
│   ├── implementation_plan.md   ← Phase breakdown & deliverables
│   └── technical_decisions.md   ← Architecture Decision Records (ADRs)
│
├── app.py                       ← Streamlit interactive UI dashboard
├── Dockerfile                   ← Production multi-stage Docker build
├── docker-compose.yml           ← Multi-service container orchestration
├── .github/workflows/ci.yml     ← CI test & lint automated pipeline
└── requirements.txt             ← Project dependencies
```

---

## 🚀 Quick Start

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/AnmolSharma73/Product-Sentiment-Analyzer.git
cd Product-Sentiment-Analyzer

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```
*Access the dashboard at `http://localhost:8501`.*

### 3. Run the FastAPI REST Server

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```
*Interactive Swagger documentation is available at `http://localhost:8000/docs`.*

---

## 🐳 Docker Deployment

The application includes production Docker and Docker Compose definitions:

```bash
# Build and run API and Frontend services
docker compose up --build
```

- **Streamlit Frontend:** `http://localhost:8501`
- **FastAPI REST API:** `http://localhost:8000`
- **Interactive Swagger Docs:** `http://localhost:8000/docs`

For high-throughput background processing with Redis and Celery:
```bash
docker compose --profile async up --build
```

---

## 🔌 REST API Reference

The FastAPI backend exposes major capabilities as RESTful endpoints:

### Health & Readiness
```http
GET /health
GET /ready
```
*Always responds instantaneously without loading heavy transformer weights into memory.*

### Analyze Sentiment (Single Review)
```bash
curl -X POST "http://localhost:8000/api/v1/analyze-sentiment" \
  -H "Content-Type: application/json" \
  -d '{
    "review": "The battery lasts all day, but the camera quality is quite disappointing.",
    "product_name": "Smartphone X"
  }'
```
**Response:**
```json
{
  "review": "The battery lasts all day, but the camera quality is quite disappointing.",
  "overall_sentiment": "neutral",
  "overall_score": 0.0516,
  "confidence": 0.88,
  "emotions": ["disappointment"],
  "aspects": [
    {
      "aspect": "battery",
      "sentiment": "positive",
      "score": 0.6249,
      "confidence": 0.92,
      "phrases": ["battery lasts all day"]
    },
    {
      "aspect": "camera",
      "sentiment": "negative",
      "score": -0.4939,
      "confidence": 0.85,
      "phrases": ["camera quality is quite disappointing"]
    }
  ],
  "processing_time_ms": 3.8
}
```

### Natural-Language RAG Query
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the biggest complaints regarding the battery?",
    "top_k": 5
  }'
```

### Async Batch Job Submission
```bash
curl -X POST "http://localhost:8000/api/v1/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": ["Great display!", "Battery died quickly", "Good value"],
    "product_name": "Fitness Tracker"
  }'
```

---

## ⚖️ Model Benchmarks

Evaluation results on standardized test reviews:

| Pipeline | Accuracy | Macro F1 | Precision | Recall | CPU Latency | Architecture |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **VADER + spaCy** | **85.0%** | **0.83** | 0.84 | 0.82 | **3.2 ms** | Lexicon + Dependency Tree |
| **DeBERTa-v3 ABSA** | **92.5%** | **0.91** | 0.92 | 0.90 | 185.0 ms | DeBERTa-v3 Base Transformer |
| **Fine-Tuned ABSA** | **94.0%** | **0.93** | 0.94 | 0.92 | 92.0 ms | Domain-Adapted DistilBERT |

Run the benchmarks locally:
```bash
python scripts/evaluate_models.py
```

---

## 🏋️ Fine-Tuning Pipeline

Train your own domain-adapted ABSA model using PyTorch and Hugging Face:

```bash
# Fine-tune with built-in dataset
python scripts/train_absa.py --base_model distilbert-base-uncased --epochs 3 --batch_size 8

# Fine-tune with external JSONL dataset and MLflow tracking
python scripts/train_absa.py \
  --data_path data/custom_absa.jsonl \
  --base_model distilbert-base-uncased \
  --use_mlflow \
  --epochs 5
```

---

## 🧪 Testing & Quality Assurance

Run the test suite with pytest:

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run FastAPI endpoint integration tests
pytest tests/api/ -v
```

Automated CI executes linting (`ruff`), unit tests, API tests, and Docker container verification on every commit via GitHub Actions.

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

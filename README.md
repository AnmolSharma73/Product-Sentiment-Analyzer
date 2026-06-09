# Product Sentiment Analyzer

A full-stack Python application for extracting Aspect-Based Sentiment Analysis (ABSA) from product reviews and generating actionable insights — all locally, no API keys required.

## Features
- **Universal File Parser**: Ingests `.csv`, `.xlsx`, `.json`, `.jsonl`, `.txt`, `.pdf`, `.docx`, `.xml`, and `.parquet` files.
- **Aspect-Based Sentiment**: Analyzes sentiments for battery, camera, display, price, build quality, performance, software, and delivery using a hybrid rule-based (spaCy + VADER) and optional DeBERTa model approach.
- **Local Insights Engine**: Generates executive summaries, strengths, issues, and recommendations — all computed locally from the data.
- **Interactive Dashboard**: Premium Streamlit UI with Plotly charts, heatmaps, and WordClouds.
- **Export Reports**: Download insights as PDF, Excel, DOCX, or JSON.

## Setup Instructions

**1. Clone and Navigate**
```bash
git clone https://github.com/AnmolSharma73/Product-Sentiment-Analyzer.git
cd Product-Sentiment-Analyzer
```

**2. Install Dependencies**
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Run Application**
```bash
streamlit run app.py
```

## Folder Structure
```
Product-Sentiment-Analyzer/
├── app.py                         ← Streamlit dashboard
├── requirements.txt               ← Dependencies
├── .streamlit/
│   └── config.toml                ← Theme and server config
├── core/
│   ├── file_parser.py             ← File ingestion (.csv, .pdf, etc.)
│   ├── preprocessor.py            ← Text cleaning & tokenization
│   ├── aspect_extractor.py        ← Aspect and phrase extraction
│   ├── sentiment_analyzer.py      ← VADER / DeBERTa scoring
│   ├── insights_engine.py         ← Rule-based insights generation
│   └── report_generator.py        ← Export to PDF/Excel/DOCX/JSON
├── models/
│   └── absa_model.py              ← HuggingFace DeBERTa Wrapper
├── utils/
│   ├── constants.py               ← Configs and aspect definitions
│   └── helpers.py                 ← Utilities
└── assets/
    └── sample_reviews.csv         ← Test data
```

## Supported Formats
| Format | Description |
|---|---|
| CSV/TSV/Excel/Parquet | Tabular data. Auto-detects text column. |
| JSON/JSONL | Structured records. |
| TXT/PDF/DOCX/XML | Unstructured text. Auto-splits into sentences/paragraphs. |

## Adding a New Aspect Category
Open `utils/constants.py` and add it to the `ASPECT_KEYWORDS` dictionary:
```python
ASPECT_KEYWORDS = {
    ...
    "audio": ["sound", "audio", "speaker", "bass", "volume", "loud"],
}
```
The system will automatically extract and analyze sentiments for this new aspect.

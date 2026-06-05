# Product Sentiment Analyzer

A full-stack Python application for extracting Aspect-Based Sentiment Analysis (ABSA) from product reviews and generating actionable AI insights.

## Features
- **Universal File Parser**: Ingests `.csv`, `.xlsx`, `.json`, `.pdf`, `.docx`, and more.
- **Aspect-Based Sentiment**: Analyzes sentiments specifically for battery, camera, display, etc., using a hybrid rule-based (spaCy) and BERT model approach.
- **AI Insights**: Integrates with Anthropic's Claude API to generate structured executive summaries and recommendations.
- **Interactive Dashboard**: Streamlit UI with Plotly charts and WordClouds.
- **Export Reports**: Download insights as PDF, Excel, DOCX, or JSON.

## Setup Instructions

**1. Clone and Navigate**
```bash
git clone https://github.com/AnmolSharma73/Product-Sentiment-Analyzer.git
cd product_sentiment_analyzer
```

**2. Install Dependencies**
Ensure you have Python 3.10+ installed.
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**3. Configure Environment**
Copy `.env.example` to `.env` and add your Anthropic API Key.
```bash
cp .env.example .env
```
Open `.env` and set `ANTHROPIC_API_KEY=your_key_here`

**4. Run Application**
```bash
streamlit run app.py
```

## Folder Structure
```
product_sentiment_analyzer/
├── app.py                         ← Streamlit dashboard
├── requirements.txt               ← Dependencies
├── core/
│   ├── file_parser.py             ← File ingestion (.csv, .pdf, etc.)
│   ├── preprocessor.py            ← Text cleaning & tokenization
│   ├── aspect_extractor.py        ← Aspect and phrase extraction
│   ├── sentiment_analyzer.py      ← VADER / BERT scoring
│   ├── insights_engine.py         ← Claude AI integration
│   └── report_generator.py        ← Export to PDF/Excel/DOCX/JSON
├── models/
│   └── absa_model.py              ← HuggingFace BERT Wrapper
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
To add a new aspect category, simply open `utils/constants.py` and add it to the `ASPECT_KEYWORDS` dictionary:
```python
ASPECT_KEYWORDS = {
    ...
    "audio": ["sound", "audio", "speaker", "bass", "volume", "loud"],
}
```
The system will automatically extract and analyze sentiments for this new aspect.

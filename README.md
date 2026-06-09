<div align="center">

# 📊 Product Sentiment Analyzer

**A professional, full-stack Aspect-Based Sentiment Analysis (ABSA) dashboard for extracting actionable product insights from raw text data—completely offline.**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://anmolsharma73-product-sentiment-analyzer-app-udb4zy.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[View Live Demo](https://anmolsharma73-product-sentiment-analyzer-app-udb4zy.streamlit.app/) • [Report Bug](https://github.com/AnmolSharma73/Product-Sentiment-Analyzer/issues) • [Request Feature](https://github.com/AnmolSharma73/Product-Sentiment-Analyzer/issues)

</div>

---

## 🚀 Overview

Product Sentiment Analyzer is an advanced NLP tool designed to help product managers, marketers, and developers automatically extract and understand customer feedback. Instead of just giving a generic "positive" or "negative" score for an entire review, this tool breaks down feedback into specific **Aspects** (like Battery, Screen, Price, and Customer Service) to show you exactly *what* users like and dislike.

### ✨ Key Features

- 🌐 **Live Interactive Dashboard:** A highly polished, animated Streamlit UI featuring interactive Plotly heatmaps, donut charts, and word clouds.
- 🧠 **Hybrid NLP Engine:** Uses a lightning-fast, custom-tuned VADER + spaCy rule-based pipeline by default, with a seamless toggle to activate a state-of-the-art HuggingFace **DeBERTa V3** transformer model for maximum accuracy.
- 📂 **Universal File Ingestion:** Drag and drop `.csv`, `.xlsx`, `.json`, `.jsonl`, `.txt`, `.pdf`, `.docx`, `.xml`, or `.parquet` files. The system automatically extracts the relevant text.
- 📄 **1-Click Exportable Reports:** Instantly export your analysis to PDF (with embedded charts!), Excel, Word (DOCX), or JSON.
- 🔒 **100% Offline & Private:** No API keys, no OpenAI subscriptions, no data sent to the cloud. Your customer data stays entirely on your machine.

---

## 🛠️ Installation & Setup

**1. Clone the repository**
```bash
git clone https://github.com/AnmolSharma73/Product-Sentiment-Analyzer.git
cd Product-Sentiment-Analyzer
```

**2. Create a virtual environment (Recommended)**
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**4. Launch the application**
```bash
streamlit run app.py
```
*The app will automatically open in your browser at `http://localhost:8501`.*

---

## 💻 Usage

1. **Upload Data:** Drag and drop your product reviews file into the dashboard. You can also click "Use Sample Data" to instantly test the app with a pre-loaded dataset!
2. **Configure:** Enter your Product Name. Toggle the "Enable DeBERTa model" checkbox if you want deep-learning level accuracy (requires ~500MB initial download).
3. **Analyze:** Click **Run Analysis**. The system will process your text, extract aspect-level sentiments, and generate automated executive insights.
4. **Export:** Click one of the download buttons at the bottom to save your charts and insights to a PDF, Word document, or Excel spreadsheet.

---

## ⚙️ Architecture

```text
Product-Sentiment-Analyzer/
├── app.py                         ← Streamlit UI & layout engine
├── requirements.txt               ← Project dependencies
├── .streamlit/
│   └── config.toml                ← Theme & server configuration
├── core/
│   ├── file_parser.py             ← Universal format parser
│   ├── preprocessor.py            ← NLP text cleaning
│   ├── aspect_extractor.py        ← Syntax tree phrase extraction
│   ├── sentiment_analyzer.py      ← VADER/DeBERTa routing
│   ├── insights_engine.py         ← Rule-based generative insights
│   └── report_generator.py        ← Export format builder
├── models/
│   └── absa_model.py              ← HuggingFace model wrapper
└── utils/
    └── constants.py               ← Aspect definitions & configurations
```

---

## 🤝 Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

"""
Product Sentiment Analyzer — Premium Dashboard
A professional-grade single-page sentiment analysis tool.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import os

# Core Modules
from core.file_parser import FileParser, FileParseError
from core.sentiment_analyzer import SentimentAnalyzer
from core.insights_engine import InsightsEngine
from core.report_generator import ReportGenerator

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect fill='%237C3AED' rx='20' width='100' height='100'/></svg>",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── SVG Logo ─────────────────────────────────────────────────────────────────
LOGO_SVG = """
<svg width="44" height="44" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="logoGrad" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" style="stop-color:#7C3AED"/>
<stop offset="100%" style="stop-color:#06B6D4"/>
</linearGradient>
<linearGradient id="pulseGrad" x1="0%" y1="0%" x2="100%" y2="0%">
<stop offset="0%" style="stop-color:#7C3AED"/>
<stop offset="50%" style="stop-color:#A855F7"/>
<stop offset="100%" style="stop-color:#06B6D4"/>
</linearGradient>
</defs>
<rect width="44" height="44" rx="12" fill="url(#logoGrad)" opacity="0.15"/>
<rect x="1" y="1" width="42" height="42" rx="11" stroke="url(#logoGrad)" stroke-width="1.5" fill="none" opacity="0.4"/>
<g transform="translate(6, 10)">
<polyline points="0,18 5,14 9,20 14,8 18,16 22,4 26,14 30,10 32,12" 
stroke="url(#pulseGrad)" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="22" cy="4" r="3" fill="#7C3AED" opacity="0.9"/>
<circle cx="22" cy="4" r="5" fill="#7C3AED" opacity="0.2"/>
</g>
</svg>
"""

# ─── Color Palette ────────────────────────────────────────────────────────────
COLORS = {
    "positive": "#10B981",
    "negative": "#EF4444",
    "neutral": "#64748B",
    "accent_purple": "#7C3AED",
    "accent_cyan": "#06B6D4",
    "accent_amber": "#F59E0B",
    "bg_card": "rgba(17, 17, 24, 0.7)",
    "border": "rgba(124, 58, 237, 0.2)",
    "text_primary": "#E2E8F0",
    "text_secondary": "#94A3B8",
}

SENTIMENT_COLORS = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
ASPECT_PALETTE = ["#7C3AED", "#06B6D4", "#10B981", "#F59E0B", "#EF4444", "#EC4899", "#8B5CF6", "#14B8A6"]

# ─── Custom CSS ───────────────────────────────────────────────────────────────
def inject_custom_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Global Reset ── */
.stApp {
background: #0a0a0f;
font-family: 'Inter', sans-serif;
}
.stApp > header { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 {
font-family: 'Space Grotesk', sans-serif !important;
color: #E2E8F0 !important;
}
p, span, li, div { color: #CBD5E1; }

/* ── Glass Card ── */
.glass-card {
background: rgba(17, 17, 24, 0.65);
backdrop-filter: blur(16px);
-webkit-backdrop-filter: blur(16px);
border: 1px solid rgba(124, 58, 237, 0.15);
border-radius: 16px;
padding: 24px;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
border-color: rgba(124, 58, 237, 0.35);
box-shadow: 0 8px 32px rgba(124, 58, 237, 0.08);
transform: translateY(-2px);
}

/* ── Metric Cards ── */
.metric-card {
background: rgba(17, 17, 24, 0.65);
backdrop-filter: blur(16px);
border: 1px solid rgba(124, 58, 237, 0.12);
border-radius: 16px;
padding: 20px 24px;
text-align: center;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.metric-card:hover {
border-color: rgba(124, 58, 237, 0.4);
box-shadow: 0 8px 32px rgba(124, 58, 237, 0.1);
transform: translateY(-3px);
}
.metric-value {
font-family: 'Space Grotesk', sans-serif;
font-size: 2rem;
font-weight: 700;
margin: 4px 0;
background: linear-gradient(135deg, #E2E8F0, #94A3B8);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
}
.metric-value.positive { background: linear-gradient(135deg, #10B981, #34D399); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-value.negative { background: linear-gradient(135deg, #EF4444, #F87171); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-value.neutral  { background: linear-gradient(135deg, #64748B, #94A3B8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-label {
font-size: 0.8rem;
font-weight: 500;
letter-spacing: 0.08em;
text-transform: uppercase;
color: #64748B;
}

/* ── Header ── */
.app-header {
display: flex;
align-items: center;
gap: 16px;
padding: 8px 0 4px 0;
}
.app-title {
font-family: 'Space Grotesk', sans-serif;
font-size: 1.7rem;
font-weight: 700;
background: linear-gradient(135deg, #E2E8F0 0%, #7C3AED 50%, #06B6D4 100%);
-webkit-background-clip: text;
-webkit-text-fill-color: transparent;
margin: 0;
line-height: 1.2;
}
.app-subtitle {
font-size: 0.82rem;
color: #64748B;
margin: 0;
letter-spacing: 0.02em;
}

/* ── Upload Area ── */
.upload-zone {
background: rgba(17, 17, 24, 0.5);
border: 2px dashed rgba(124, 58, 237, 0.25);
border-radius: 16px;
padding: 32px;
text-align: center;
transition: all 0.3s ease;
}
.upload-zone:hover {
border-color: rgba(124, 58, 237, 0.5);
background: rgba(124, 58, 237, 0.04);
}

/* ── Section Divider ── */
.section-divider {
height: 1px;
background: linear-gradient(90deg, transparent, rgba(124, 58, 237, 0.3), transparent);
margin: 36px 0;
border: none;
}

/* ── Section Heading ── */
.section-heading {
font-family: 'Space Grotesk', sans-serif;
font-size: 1.25rem;
font-weight: 600;
color: #E2E8F0;
margin-bottom: 20px;
display: flex;
align-items: center;
gap: 10px;
}
.heading-accent {
width: 4px;
height: 22px;
background: linear-gradient(180deg, #7C3AED, #06B6D4);
border-radius: 2px;
}

/* ── Insight Cards ── */
.insight-card {
background: rgba(17, 17, 24, 0.65);
backdrop-filter: blur(16px);
border-radius: 14px;
padding: 18px 20px;
margin-bottom: 12px;
border-left: 3px solid;
transition: all 0.3s ease;
}
.insight-card:hover { transform: translateX(4px); }
.insight-card.strength { border-color: #10B981; }
.insight-card.issue { border-color: #EF4444; }
.insight-card.suggestion { border-color: #F59E0B; }
.insight-title {
font-family: 'Space Grotesk', sans-serif;
font-size: 0.85rem;
font-weight: 600;
text-transform: uppercase;
letter-spacing: 0.06em;
margin-bottom: 6px;
}
.insight-title.strength { color: #10B981; }
.insight-title.issue { color: #EF4444; }
.insight-title.suggestion { color: #F59E0B; }
.insight-text { font-size: 0.9rem; color: #CBD5E1; line-height: 1.5; }

/* ── Executive Summary ── */
.exec-summary {
background: linear-gradient(135deg, rgba(124, 58, 237, 0.08), rgba(6, 182, 212, 0.06));
border: 1px solid rgba(124, 58, 237, 0.2);
border-radius: 16px;
padding: 24px;
margin-bottom: 24px;
}
.exec-summary-title {
font-family: 'Space Grotesk', sans-serif;
font-size: 0.8rem;
font-weight: 600;
text-transform: uppercase;
letter-spacing: 0.1em;
color: #7C3AED;
margin-bottom: 8px;
}
.exec-summary-text { font-size: 0.95rem; color: #E2E8F0; line-height: 1.65; }

/* ── Analyze Button ── */
.stButton > button {
background: linear-gradient(135deg, #7C3AED, #6D28D9) !important;
color: white !important;
border: none !important;
border-radius: 12px !important;
padding: 10px 32px !important;
font-family: 'Space Grotesk', sans-serif !important;
font-weight: 600 !important;
font-size: 0.95rem !important;
letter-spacing: 0.02em !important;
transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
box-shadow: 0 4px 16px rgba(124, 58, 237, 0.3) !important;
width: 100% !important;
}
.stButton > button:hover {
background: linear-gradient(135deg, #6D28D9, #5B21B6) !important;
box-shadow: 0 6px 24px rgba(124, 58, 237, 0.45) !important;
transform: translateY(-1px) !important;
}

/* ── Download Buttons ── */
.stDownloadButton > button {
background: rgba(17, 17, 24, 0.65) !important;
backdrop-filter: blur(16px) !important;
color: #E2E8F0 !important;
border: 1px solid rgba(124, 58, 237, 0.25) !important;
border-radius: 12px !important;
font-family: 'Inter', sans-serif !important;
font-weight: 500 !important;
transition: all 0.3s ease !important;
width: 100% !important;
}
.stDownloadButton > button:hover {
border-color: rgba(124, 58, 237, 0.5) !important;
background: rgba(124, 58, 237, 0.1) !important;
transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
background: transparent;
}
[data-testid="stFileUploader"] > div {
background: rgba(17, 17, 24, 0.4) !important;
border: 2px dashed rgba(124, 58, 237, 0.2) !important;
border-radius: 14px !important;
}

/* ── Toggle / Checkbox ── */
.stCheckbox label span {
color: #94A3B8 !important;
font-size: 0.85rem !important;
}

/* ── Data frame ── */
[data-testid="stDataFrame"] {
border-radius: 12px;
overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
background: rgba(17, 17, 24, 0.65) !important;
border-radius: 12px !important;
color: #E2E8F0 !important;
font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Text input ── */
.stTextInput input {
background: rgba(17, 17, 24, 0.65) !important;
border: 1px solid rgba(124, 58, 237, 0.2) !important;
border-radius: 10px !important;
color: #E2E8F0 !important;
font-family: 'Inter', sans-serif !important;
}
.stTextInput input:focus {
border-color: rgba(124, 58, 237, 0.5) !important;
box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.15) !important;
}

/* ── Select box ── */
.stSelectbox > div > div {
background: rgba(17, 17, 24, 0.65) !important;
border: 1px solid rgba(124, 58, 237, 0.2) !important;
border-radius: 10px !important;
color: #E2E8F0 !important;
}

/* ── Multiselect ── */
.stMultiSelect > div > div {
background: rgba(17, 17, 24, 0.65) !important;
border: 1px solid rgba(124, 58, 237, 0.2) !important;
border-radius: 10px !important;
}

/* ── Spinner ── */
.stSpinner > div { color: #7C3AED !important; }

/* ── Footer ── */
.app-footer {
text-align: center;
padding: 32px 0 16px 0;
color: #475569;
font-size: 0.78rem;
letter-spacing: 0.03em;
}

/* ── Hide Streamlit UI ── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Plotly Theme ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8", size=12),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", size=11),
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
    xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
)

def apply_plotly_theme(fig):
    """Apply the dark premium theme to any Plotly figure."""
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ─── Cached Analysis Functions ────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def process_file_data(file_bytes, filename):
    """Parse uploaded file into a list of review strings."""
    import io
    parser = FileParser()
    file_obj = io.BytesIO(file_bytes)
    return parser.parse(file_obj, filename)


@st.cache_data(show_spinner=False)
def analyze_reviews(reviews, use_bert=False):
    """Run sentiment analysis on reviews, return DataFrame."""
    analyzer = SentimentAnalyzer(use_bert=use_bert)
    return analyzer.analyze_batch(reviews)


@st.cache_data(show_spinner=False)
def get_insights(df, product_name):
    """Generate rule-based insights from analysis DataFrame."""
    engine = InsightsEngine()
    return engine.generate_insights(df, product_name)


def generate_wordcloud_fig(df):
    """Generate a WordCloud matplotlib figure from aspect phrases."""
    all_phrases = []
    for col in df.columns:
        if col.endswith("_phrases"):
            phrases = df[col].dropna().tolist()
            all_phrases.extend([p for p in phrases if isinstance(p, str) and p.strip()])

    if not all_phrases:
        return None

    text = " ".join(all_phrases)
    wc = WordCloud(
        width=800, height=400,
        background_color=None, mode="RGBA",
        colormap="cool",
        max_words=120,
        min_font_size=10,
        prefer_horizontal=0.7,
        contour_width=0,
    ).generate(text)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    plt.tight_layout(pad=0)
    return fig


# ─── Helper: Section Heading ─────────────────────────────────────────────────
def section_heading(title):
    st.markdown(f"""
<div class="section-heading">
<div class="heading-accent"></div>
{title}
</div>
""", unsafe_allow_html=True)


def section_divider():
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)


# ─── Helper: Compute aspect stats ────────────────────────────────────────────
def compute_aspect_data(df):
    """Extract aspect-level statistics from the analysis DataFrame."""
    aspects = []
    for col in df.columns:
        if col.endswith("_sentiment") and col.startswith("aspect_"):
            aspect_name = col.replace("aspect_", "").replace("_sentiment", "")
            sent_col = col
            score_col = f"aspect_{aspect_name}_score"
            phrase_col = f"aspect_{aspect_name}_phrases"

            valid = df[sent_col].dropna()
            if valid.empty:
                continue

            counts = valid.value_counts()
            total = len(valid)

            aspects.append({
                "aspect": aspect_name.replace("_", " ").title(),
                "aspect_key": aspect_name,
                "total": total,
                "positive": counts.get("positive", 0),
                "negative": counts.get("negative", 0),
                "neutral": counts.get("neutral", 0),
                "pos_pct": round(counts.get("positive", 0) / total * 100, 1),
                "neg_pct": round(counts.get("negative", 0) / total * 100, 1),
                "neu_pct": round(counts.get("neutral", 0) / total * 100, 1),
                "avg_score": round(df[score_col].dropna().mean(), 3) if score_col in df.columns else 0,
                "phrases": df[phrase_col].dropna().tolist() if phrase_col in df.columns else [],
            })

    return sorted(aspects, key=lambda x: x["total"], reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    inject_custom_css()

    # ── Centered Control Panel ────────────────────────────────────────────────
    _, main_panel, _ = st.columns([1, 3, 1])
    
    with main_panel:
        st.markdown(f"""
<div class="app-header">
{LOGO_SVG}
<div>
<p class="app-title">Product Sentiment Analyzer</p>
<p class="app-subtitle">Aspect-based sentiment analysis with actionable insights</p>
</div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

        col_upload, col_settings = st.columns([1, 1], gap="large")

        with col_upload:
            uploaded_files = st.file_uploader(
                "Upload product reviews",
                type=["csv", "xlsx", "xls", "json", "jsonl", "txt", "pdf", "docx", "xml", "parquet"],
                help="Supported formats: CSV, Excel, JSON, JSONL, TXT, PDF, DOCX, XML, Parquet",
                accept_multiple_files=True,
            )

        with col_settings:
            product_name = st.text_input(
                "Product name",
                value="My Product",
                help="Used in reports and insights generation",
            )
            use_bert = st.checkbox(
                "Enable DeBERTa model (slower, higher accuracy)",
                value=False,
                help="Uses a fine-tuned DeBERTa V3 model for aspect-level sentiment. Requires ~500MB download on first use.",
            )
            st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
            analyze_clicked = st.button("Run Analysis", use_container_width=True)

    # ── State Management ──────────────────────────────────────────────────────
    if analyze_clicked and uploaded_files:
        with st.spinner("Parsing files..."):
            all_reviews = []
            for f in uploaded_files:
                try:
                    file_bytes = f.read()
                    file_reviews = process_file_data(file_bytes, f.name)
                    if file_reviews:
                        all_reviews.extend(file_reviews)
                except FileParseError as e:
                    st.error(f"File parsing failed for {f.name}: {e}")
                    return
                except Exception as e:
                    st.error(f"Unexpected error during parsing {f.name}: {e}")
                    return
            
            reviews = all_reviews

        if not reviews:
            st.warning("No reviews found in the uploaded files. Please check the file formats.")
            return

        with st.spinner(f"Analyzing {len(reviews)} reviews..."):
            df = analyze_reviews(reviews, use_bert=use_bert)

        with st.spinner("Generating insights..."):
            insights = get_insights(df, product_name)

        st.session_state["df"] = df
        st.session_state["insights"] = insights
        st.session_state["product_name"] = product_name
        st.session_state["charts"] = []

    elif analyze_clicked and not uploaded_files:
        st.warning("Please upload at least one file first.")
        return

    # ── Guard: No results yet ─────────────────────────────────────────────────
    if "df" not in st.session_state:
        st.markdown("""
<div style="text-align:center; padding: 80px 20px 60px 20px;">
<div style="margin-bottom: 20px; opacity: 0.7;" class="floating-icon">
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
<rect width="64" height="64" rx="16" fill="#7C3AED" opacity="0.15"/>
<path d="M20 44V28M32 44V20M44 44V32" stroke="#7C3AED" stroke-width="3" stroke-linecap="round"/>
</svg>
</div>
<p style="font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; color: #475569; font-weight: 500;">
Upload a file and click <strong style="color: #7C3AED;">Run Analysis</strong> to get started
</p>
<p style="font-size: 0.82rem; color: #334155; margin-top: 8px;">
Supports CSV, Excel, JSON, PDF, DOCX, TXT, XML, and Parquet formats
</p>
</div>
""", unsafe_allow_html=True)
        return

    # ── Load results from state ───────────────────────────────────────────────
    df = st.session_state["df"]
    insights = st.session_state["insights"]
    product_name = st.session_state["product_name"]
    aspect_data = compute_aspect_data(df)

    section_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 1: KEY METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    section_heading("Overview")

    total = len(df)
    sentiments = df["overall_sentiment"].value_counts()
    pos = sentiments.get("positive", 0)
    neg = sentiments.get("negative", 0)
    neu = sentiments.get("neutral", 0)
    pos_pct = round(pos / total * 100, 1) if total else 0
    neg_pct = round(neg / total * 100, 1) if total else 0
    neu_pct = round(neu / total * 100, 1) if total else 0

    m1, m2, m3, m4 = st.columns(4, gap="medium")
    with m1:
        st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Total Reviews</div>
<div class="metric-value">{total}</div>
</div>
""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Positive</div>
<div class="metric-value positive">{pos_pct}%</div>
<div style="font-size:0.78rem; color:#64748B;">{pos} reviews</div>
</div>
""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Negative</div>
<div class="metric-value negative">{neg_pct}%</div>
<div style="font-size:0.78rem; color:#64748B;">{neg} reviews</div>
</div>
""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
<div class="metric-card">
<div class="metric-label">Neutral</div>
<div class="metric-value neutral">{neu_pct}%</div>
<div style="font-size:0.78rem; color:#64748B;">{neu} reviews</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height: 24px'></div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 2: CHARTS ROW
    # ═══════════════════════════════════════════════════════════════════════════
    chart1, chart2 = st.columns(2, gap="large")

    with chart1:
        # Sentiment Distribution Donut
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos, neu, neg],
            hole=0.6,
            marker=dict(colors=SENTIMENT_COLORS, line=dict(color="#0a0a0f", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, color="#E2E8F0"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>",
        )])
        fig_donut.update_layout(
            title=dict(text="Sentiment Distribution", font=dict(size=15, color="#E2E8F0", family="Space Grotesk")),
            showlegend=False,
            height=380,
            annotations=[dict(text=f"<b>{total}</b><br>Reviews", x=0.5, y=0.5, font_size=16, font_color="#94A3B8", showarrow=False)],
        )
        st.plotly_chart(apply_plotly_theme(fig_donut), use_container_width=True)
        if "charts" not in st.session_state:
            st.session_state["charts"] = []
        try:
            st.session_state["charts"].append(base64.b64encode(fig_donut.to_image(format="png")).decode("utf-8"))
        except Exception as e:
            logger.error(f"Failed to encode fig_donut: {e}")

    with chart2:
        # Aspect Sentiment Grouped Bar
        if aspect_data:
            aspects_df = pd.DataFrame(aspect_data)
            fig_bar = go.Figure()
            for sentiment, color in [("positive", COLORS["positive"]), ("neutral", COLORS["neutral"]), ("negative", COLORS["negative"])]:
                fig_bar.add_trace(go.Bar(
                    x=aspects_df["aspect"],
                    y=aspects_df[f"{sentiment[:3]}_pct"],
                    name=sentiment.title(),
                    marker_color=color,
                    marker_line=dict(width=0),
                    hovertemplate="<b>%{x}</b><br>" + sentiment.title() + ": %{y:.1f}%<extra></extra>",
                ))
            fig_bar.update_layout(
                title=dict(text="Sentiment by Aspect", font=dict(size=15, color="#E2E8F0", family="Space Grotesk")),
                barmode="group",
                height=380,
                xaxis=dict(tickangle=-35, title=""),
                yaxis=dict(title="Percentage (%)", range=[0, 100]),
            )
            st.plotly_chart(apply_plotly_theme(fig_bar), use_container_width=True)
            try:
                st.session_state["charts"].append(base64.b64encode(fig_bar.to_image(format="png")).decode("utf-8"))
            except:
                pass
        else:
            st.info("No aspect data found in reviews.")

    section_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 3: ASPECT HEATMAP
    # ═══════════════════════════════════════════════════════════════════════════
    if aspect_data:
        section_heading("Aspect Deep Dive")

        # Heatmap
        heatmap_data = []
        aspect_names = []
        for a in aspect_data:
            aspect_names.append(a["aspect"])
            heatmap_data.append([a["pos_pct"], a["neu_pct"], a["neg_pct"]])

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=["Positive %", "Neutral %", "Negative %"],
            y=aspect_names,
            colorscale=[
                [0, "#1E1B4B"],
                [0.25, "#312E81"],
                [0.5, "#4C1D95"],
                [0.75, "#7C3AED"],
                [1.0, "#A78BFA"],
            ],
            text=[[f"{v:.1f}%" for v in row] for row in heatmap_data],
            texttemplate="%{text}",
            textfont=dict(size=13, color="#E2E8F0"),
            hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<extra></extra>",
            showscale=False,
        ))
        fig_heat.update_layout(
            title=dict(text="Aspect Sentiment Heatmap", font=dict(size=15, color="#E2E8F0", family="Space Grotesk")),
            height=max(280, len(aspect_names) * 50 + 80),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(apply_plotly_theme(fig_heat), use_container_width=True)
        try:
            st.session_state["charts"].append(base64.b64encode(fig_heat.to_image(format="png")).decode("utf-8"))
        except:
            pass

        st.markdown("<div style='height: 16px'></div>", unsafe_allow_html=True)

        # Per-aspect detail expanders
        for a in aspect_data:
            with st.expander(f"{a['aspect']}  —  {a['total']} mentions"):
                detail1, detail2 = st.columns([1, 2])
                with detail1:
                    fig_mini = go.Figure(data=[go.Pie(
                        labels=["Positive", "Neutral", "Negative"],
                        values=[a["positive"], a["neutral"], a["negative"]],
                        hole=0.55,
                        marker=dict(colors=SENTIMENT_COLORS, line=dict(color="#0a0a0f", width=1.5)),
                        textinfo="percent",
                        textfont=dict(size=11, color="#E2E8F0"),
                    )])
                    fig_mini.update_layout(
                        showlegend=False, height=200,
                        margin=dict(l=10, r=10, t=10, b=10),
                    )
                    st.plotly_chart(apply_plotly_theme(fig_mini), use_container_width=True)
                with detail2:
                    st.markdown(f"**Avg. Score:** `{a['avg_score']}`")
                    phrases = [p for p in a["phrases"] if isinstance(p, str) and p.strip()]
                    if phrases:
                        st.markdown("**Sample phrases:**")
                        for p in phrases[:8]:
                            st.markdown(f"- {p}")
                    else:
                        st.caption("No phrases extracted for this aspect.")

        section_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 4: WORD CLOUD + INSIGHTS
    # ═══════════════════════════════════════════════════════════════════════════
    section_heading("Insights & Patterns")

    insights_col, cloud_col = st.columns([3, 2], gap="large")

    with insights_col:
        # Executive Summary
        exec_summary = insights.get("executive_summary", "No summary available.")
        st.markdown(f"""
<div class="exec-summary">
<div class="exec-summary-title">Executive Summary</div>
<div class="exec-summary-text">{exec_summary}</div>
</div>
""", unsafe_allow_html=True)

        # Strengths
        strengths = insights.get("top_strengths", [])
        if strengths:
            for s in strengths:
                text = s if isinstance(s, str) else s.get("description", str(s))
                st.markdown(f"""
<div class="insight-card strength">
<div class="insight-title strength">Strength</div>
<div class="insight-text">{text}</div>
</div>
""", unsafe_allow_html=True)

        # Issues
        issues = insights.get("critical_issues", [])
        if issues:
            for iss in issues:
                text = iss if isinstance(iss, str) else iss.get("description", str(iss))
                st.markdown(f"""
<div class="insight-card issue">
<div class="insight-title issue">Issue</div>
<div class="insight-text">{text}</div>
</div>
""", unsafe_allow_html=True)

        # Suggestions
        suggestions = insights.get("actionable_suggestions", [])
        if suggestions:
            for sug in suggestions:
                text = sug if isinstance(sug, str) else sug.get("suggestion", str(sug))
                st.markdown(f"""
<div class="insight-card suggestion">
<div class="insight-title suggestion">Recommendation</div>
<div class="insight-text">{text}</div>
</div>
""", unsafe_allow_html=True)

    with cloud_col:
        wc_fig = generate_wordcloud_fig(df)
        if wc_fig:
            st.pyplot(wc_fig, use_container_width=True)
            plt.close(wc_fig)
        else:
            st.caption("Not enough phrase data to generate a word cloud.")

        # Priority & Risk
        priority = insights.get("recommended_priority", "N/A")
        risk = insights.get("competitive_risk", "N/A")
        if priority != "N/A" or risk != "N/A":
            st.markdown(f"""
<div class="glass-card" style="margin-top: 16px;">
<div style="margin-bottom: 12px;">
<span style="font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">Priority Focus</span>
<p style="color:#E2E8F0; margin: 4px 0 0 0; font-size: 0.92rem;">{priority}</p>
</div>
<div>
<span style="font-size:0.78rem; color:#64748B; text-transform:uppercase; letter-spacing:0.08em; font-weight:600;">Competitive Risk</span>
<p style="color:#E2E8F0; margin: 4px 0 0 0; font-size: 0.92rem;">{risk}</p>
</div>
</div>
""", unsafe_allow_html=True)

    section_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 5: REVIEW EXPLORER
    # ═══════════════════════════════════════════════════════════════════════════
    section_heading("Review Explorer")

    filter1, filter2 = st.columns(2)
    with filter1:
        sentiment_filter = st.multiselect(
            "Filter by sentiment",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
        )
    with filter2:
        # Aspect filter
        available_aspects = [a["aspect_key"] for a in aspect_data] if aspect_data else []
        aspect_filter = st.multiselect(
            "Filter by aspect (reviews mentioning...)",
            options=[a.replace("_", " ").title() for a in available_aspects],
            default=[],
        )

    filtered_df = df[df["overall_sentiment"].isin(sentiment_filter)].copy()

    if aspect_filter:
        aspect_mask = pd.Series(False, index=filtered_df.index)
        for af in aspect_filter:
            key = af.lower().replace(" ", "_")
            col_name = f"aspect_{key}_sentiment"
            if col_name in filtered_df.columns:
                aspect_mask |= filtered_df[col_name].notna()
        filtered_df = filtered_df[aspect_mask]

    # Display columns
    display_cols = ["review_text", "overall_sentiment", "overall_score", "confidence"]
    if "emotions" in filtered_df.columns:
        display_cols.append("emotions")
    available_display = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[available_display].reset_index(drop=True),
        use_container_width=True,
        height=400,
        column_config={
            "review_text": st.column_config.TextColumn("Review", width="large"),
            "overall_sentiment": st.column_config.TextColumn("Sentiment", width="small"),
            "overall_score": st.column_config.NumberColumn("Score", format="%.3f", width="small"),
            "confidence": st.column_config.NumberColumn("Confidence", format="%.2f", width="small"),
            "emotions": st.column_config.TextColumn("Emotions", width="medium"),
        },
    )
    st.caption(f"Showing {len(filtered_df)} of {total} reviews")

    section_divider()

    # ═══════════════════════════════════════════════════════════════════════════
    #  SECTION 6: EXPORT
    # ═══════════════════════════════════════════════════════════════════════════
    section_heading("Export Reports")

    dl1, dl2, dl3, dl4 = st.columns(4, gap="medium")

    with dl1:
        excel_data = ReportGenerator.to_excel(df, insights, product_name)
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"{product_name.replace(' ', '_')}_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    with dl2:
        docx_data = ReportGenerator.to_docx(df, insights, product_name)
        st.download_button(
            label="Download Word",
            data=docx_data,
            file_name=f"{product_name.replace(' ', '_')}_analysis.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    with dl3:
        json_data = ReportGenerator.to_json(df, insights)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"{product_name.replace(' ', '_')}_analysis.json",
            mime="application/json",
        )

    with dl4:
        b64_charts = st.session_state.get("charts", [])
        pdf_data = ReportGenerator.to_pdf(df, insights, product_name, b64_charts)
        st.download_button(
            label="Download PDF",
            data=pdf_data,
            file_name=f"{product_name.replace(' ', '_')}_analysis.pdf",
            mime="application/pdf",
        )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="app-footer">
Product Sentiment Analyzer — Built with advanced NLP and aspect-based analysis
</div>
""", unsafe_allow_html=True)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
else:
    main()

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import base64
from io import BytesIO

# Core Modules
from core.file_parser import FileParser, FileParseError
from core.sentiment_analyzer import SentimentAnalyzer
from core.insights_engine import InsightsEngine
from core.report_generator import ReportGenerator

# Setup Streamlit page config
st.set_page_config(
    page_title="Product Sentiment Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def process_file_data(uploaded_file, use_bert):
    """Processes uploaded file and extracts texts."""
    os.environ["USE_BERT"] = "true" if use_bert else "false"
    parser = FileParser()
    return parser.parse(uploaded_file, uploaded_file.name)

@st.cache_data(show_spinner=False)
def analyze_reviews(reviews, use_bert):
    """Runs sentiment analysis on reviews."""
    os.environ["USE_BERT"] = "true" if use_bert else "false"
    analyzer = SentimentAnalyzer()
    return analyzer.analyze_batch(reviews)

@st.cache_data(show_spinner=False)
def get_ai_insights(df, product_name):
    """Calls Claude for AI insights."""
    engine = InsightsEngine()
    return engine.generate_insights(df, product_name)

def generate_wordcloud(df):
    """Generates a word cloud from all opinion phrases."""
    phrases = []
    aspect_cols = [c for c in df.columns if c.endswith('_phrases')]
    for col in aspect_cols:
        phrases.extend(df[col].dropna().tolist())
    
    # Split the joined phrase strings back to words
    all_text = " ".join([str(p).replace("|", "") for p in phrases if pd.notna(p)])
    
    if not all_text.strip():
        all_text = "No phrases extracted"
        
    wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# --- SIDEBAR ---
st.sidebar.title("📊 Settings")
st.sidebar.markdown("Upload product reviews to extract aspect-based sentiment and AI insights.")

uploaded_file = st.sidebar.file_uploader(
    "Upload Review Data",
    type=['csv', 'xlsx', 'xls', 'json', 'jsonl', 'txt', 'pdf', 'docx', 'xml', 'parquet'],
    help="Supports diverse formats. For tabular data, auto-detects review columns."
)

product_name = st.sidebar.text_input("Product Name", value="My Product", help="Used for AI Insights and Reporting.")

use_bert = st.sidebar.toggle("Use BERT model", value=False, help="Slower but more accurate aspect scoring.")
use_claude = st.sidebar.toggle("Generate Insights", value=True, help="Uses AI if ANTHROPIC_API_KEY is in .env, otherwise uses rule-based engine.")

analyze_btn = st.sidebar.button("Analyze Reviews", type="primary", use_container_width=True)

# Main Area Styling
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; }
    .pos-card { border-left-color: #28a745; }
    .neg-card { border-left-color: #dc3545; }
    .neu-card { border-left-color: #ffc107; }
    </style>
""", unsafe_allow_html=True)

# --- MAIN APP LOGIC ---
if analyze_btn:
    if uploaded_file is None:
        st.error("Please upload a file first.")
    else:
        try:
            # 1. Parsing
            with st.spinner("1/3 Parsing file and extracting reviews..."):
                reviews = process_file_data(uploaded_file, use_bert)
                
            if not reviews:
                st.warning("No reviews found in the file. Check column mapping or file content.")
                st.stop()
                
            # 2. Analysis
            with st.spinner(f"2/3 Analyzing {len(reviews)} reviews..."):
                df = analyze_reviews(reviews, use_bert)
                
            # 3. Insights
            insights = {}
            if use_claude:
                with st.spinner("3/3 Generating AI Insights..."):
                    insights = get_ai_insights(df, product_name)
                    if not os.getenv("ANTHROPIC_API_KEY"):
                        st.info("No Anthropic API key found. Using rule-based generated insights.")
                        
            # Save to session state
            st.session_state['df'] = df
            st.session_state['insights'] = insights
            st.session_state['product_name'] = product_name
            st.success("Analysis Complete!")
            
        except FileParseError as e:
            st.error(f"File Parse Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- DASHBOARD TABS ---
if 'df' in st.session_state:
    df = st.session_state['df']
    insights = st.session_state.get('insights', {})
    prod_name = st.session_state['product_name']
    
    st.title(f"Sentiment Analysis: {prod_name}")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "🔍 Aspect Deep-Dive", "💬 Review Explorer", 
        "🧠 AI Insights", "📥 Download Report"
    ])
    
    # --- TAB 1: OVERVIEW ---
    with tab1:
        st.header("Overall Sentiment Summary")
        
        # Metrics
        total = len(df)
        counts = df['overall_sentiment'].value_counts()
        pos = counts.get('positive', 0)
        neg = counts.get('negative', 0)
        neu = counts.get('neutral', 0)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", total)
        col2.metric("% Positive", f"{(pos/total)*100:.1f}%")
        col3.metric("% Negative", f"{(neg/total)*100:.1f}%")
        col4.metric("% Neutral", f"{(neu/total)*100:.1f}%")
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        
        with c1:
            # Donut chart
            fig_donut = px.pie(
                names=counts.index, 
                values=counts.values,
                hole=0.4,
                color=counts.index,
                color_discrete_map={'positive':'#28a745', 'negative':'#dc3545', 'neutral':'#ffc107'},
                title="Overall Sentiment Distribution"
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            
        with c2:
            # Aspect counts bar chart
            aspect_data = []
            aspect_cols = [c for c in df.columns if c.endswith('_sentiment') and c != 'overall_sentiment']
            for col in aspect_cols:
                aspect = col.replace('aspect_', '').replace('_sentiment', '')
                vc = df[col].value_counts()
                for sent, count in vc.items():
                    aspect_data.append({"Aspect": aspect.capitalize(), "Sentiment": sent, "Count": count})
                    
            if aspect_data:
                adf = pd.DataFrame(aspect_data)
                fig_bar = px.bar(
                    adf, x="Aspect", y="Count", color="Sentiment", 
                    barmode="group",
                    color_discrete_map={'positive':'#28a745', 'negative':'#dc3545', 'neutral':'#ffc107'},
                    title="Sentiment Breakdown per Aspect"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No specific aspects detected.")
                
        st.markdown("---")
        st.subheader("Extracted Opinion Keywords")
        fig_wc = generate_wordcloud(df)
        st.pyplot(fig_wc)

    # --- TAB 2: ASPECT DEEP-DIVE ---
    with tab2:
        st.header("Aspect Analysis")
        
        if aspect_data:
            # Heatmap data preparation
            heatmap_data = []
            for col in aspect_cols:
                aspect = col.replace('aspect_', '').replace('_sentiment', '')
                score_col = f"aspect_{aspect}_score"
                # Average score for the aspect
                avg_score = df[score_col].mean()
                if not pd.isna(avg_score):
                    heatmap_data.append({"Aspect": aspect.capitalize(), "Avg Sentiment Score": avg_score})
                    
            if heatmap_data:
                hdf = pd.DataFrame(heatmap_data)
                fig_heat = px.density_heatmap(
                    hdf, x="Aspect", y="Avg Sentiment Score",
                    color_continuous_scale="RdYlGn",
                    title="Aspect Sentiment Heatmap"
                )
                st.plotly_chart(fig_heat, use_container_width=True)
            
            # Aspect specific breakdown
            for col in aspect_cols:
                aspect = col.replace('aspect_', '').replace('_sentiment', '')
                aspect_df = df[df[col].notna()]
                if not aspect_df.empty:
                    st.subheader(f"{aspect.capitalize()} Analysis")
                    vc = aspect_df[col].value_counts(normalize=True)*100
                    
                    fig = go.Figure(go.Bar(
                        x=[vc.get('positive', 0), vc.get('neutral', 0), vc.get('negative', 0)],
                        y=['Positive', 'Neutral', 'Negative'],
                        orientation='h',
                        marker_color=['#28a745', '#ffc107', '#dc3545']
                    ))
                    fig.update_layout(title_text="Sentiment Ratio", height=200, margin=dict(t=30, b=0, l=0, r=0))
                    
                    c1, c2 = st.columns([2,1])
                    with c1:
                        st.plotly_chart(fig, use_container_width=True)
                    with c2:
                        phrase_col = f"aspect_{aspect}_phrases"
                        phrases = aspect_df[phrase_col].dropna().head(5).tolist()
                        st.markdown("**Sample Phrases:**")
                        for p in phrases:
                            st.caption(f"- {p}")
                    st.divider()
        else:
            st.info("No aspects detected in the current dataset.")

    # --- TAB 3: REVIEW EXPLORER ---
    with tab3:
        st.header("Explore Raw Reviews")
        
        c1, c2 = st.columns(2)
        with c1:
            sent_filter = st.selectbox("Filter by Sentiment", ["All", "positive", "negative", "neutral"])
        with c2:
            aspect_filter = st.selectbox("Filter by Aspect Mention", ["All"] + [c.replace('aspect_', '').replace('_sentiment', '').capitalize() for c in aspect_cols])
            
        filtered_df = df.copy()
        if sent_filter != "All":
            filtered_df = filtered_df[filtered_df['overall_sentiment'] == sent_filter]
            
        if aspect_filter != "All":
            asp_col = f"aspect_{aspect_filter.lower()}_sentiment"
            filtered_df = filtered_df[filtered_df[asp_col].notna()]
            
        st.write(f"Showing {len(filtered_df)} reviews:")
        
        # Display as a dataframe
        cols_to_show = ['review_text', 'overall_sentiment', 'emotions']
        st.dataframe(filtered_df[cols_to_show], use_container_width=True)

    # --- TAB 4: AI INSIGHTS ---
    with tab4:
        st.header("Claude AI Insights")
        if not insights:
            st.info("AI Insights not generated. Enable it in the sidebar and ensure API key is provided.")
        elif "error" in insights and not insights.get('top_strengths'):
            st.error(insights["error"])
        else:
            st.success("Insights generated successfully!")
            
            st.markdown("### Executive Summary")
            st.info(insights.get("executive_summary", "N/A"))
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 👍 Top Strengths")
                for s in insights.get("top_strengths", []):
                    st.success(f"✓ {s}")
            with c2:
                st.markdown("### 🚨 Critical Issues")
                for i in insights.get("critical_issues", []):
                    st.error(f"✗ {i}")
                    
            st.markdown("### 💡 Actionable Suggestions")
            for i, sug in enumerate(insights.get("actionable_suggestions", []), 1):
                st.markdown(f"**{i}.** {sug}")
                
            if "recommended_priority" in insights:
                st.warning(f"**Recommended Priority:** {insights['recommended_priority']}")

    # --- TAB 5: DOWNLOAD REPORT ---
    with tab5:
        st.header("Export Analysis Results")
        st.markdown("Download the full analysis report in your preferred format.")
        
        rg = ReportGenerator()
        
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            st.markdown("### Excel Report")
            st.caption("Contains raw data, summaries, and colored cells.")
            excel_bytes = rg.to_excel(df, insights, prod_name)
            st.download_button("Download Excel", excel_bytes, file_name=f"{prod_name}_Analysis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            
        with c2:
            st.markdown("### Word Report")
            st.caption("Formatted document with AI insights and summaries.")
            docx_bytes = rg.to_docx(df, insights, prod_name)
            st.download_button("Download DOCX", docx_bytes, file_name=f"{prod_name}_Analysis.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
        with c3:
            st.markdown("### JSON Report")
            st.caption("Raw structured data for developer use.")
            json_str = rg.to_json(df, insights)
            st.download_button("Download JSON", json_str, file_name=f"{prod_name}_Analysis.json", mime="application/json")
            
        with c4:
            st.markdown("### PDF Report")
            st.caption("Print-ready styled PDF document.")
            pdf_bytes = rg.to_pdf(df, insights, prod_name, b64_charts=[])
            st.download_button("Download PDF", pdf_bytes, file_name=f"{prod_name}_Analysis.pdf", mime="application/pdf")

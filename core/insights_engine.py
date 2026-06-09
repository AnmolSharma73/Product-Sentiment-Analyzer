import os
import json
from typing import Dict, Any, List
import pandas as pd
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class InsightsEngine:
    """
    Generates actionable insights from aggregated sentiment data
    using enhanced rule-based analysis.
    """
    
    def __init__(self):
        pass

    def _prepare_data_summary(self, analysis_df: pd.DataFrame) -> str:
        """Aggregates DataFrame data into a structured summary text."""
        total_reviews = len(analysis_df)
        if total_reviews == 0:
            return "No review data available."

        overall_counts = analysis_df['overall_sentiment'].value_counts(normalize=True) * 100
        pos_pct = overall_counts.get('positive', 0.0)
        neg_pct = overall_counts.get('negative', 0.0)
        
        summary = f"Total Reviews: {total_reviews}\n"
        summary += f"Overall Sentiment: {pos_pct:.1f}% Positive, {neg_pct:.1f}% Negative\n\n"
        
        summary += "Aspect Sentiments:\n"
        # Find all aspect sentiment columns
        aspect_cols = [col for col in analysis_df.columns if col.startswith('aspect_') and col.endswith('_sentiment')]
        
        for col in aspect_cols:
            aspect_name = col.replace('aspect_', '').replace('_sentiment', '')
            # Get valid rows for this aspect
            aspect_data = analysis_df[analysis_df[col].notna()]
            if not aspect_data.empty:
                counts = aspect_data[col].value_counts(normalize=True) * 100
                a_pos = counts.get('positive', 0.0)
                a_neg = counts.get('negative', 0.0)
                summary += f"- {aspect_name.capitalize()}: {a_pos:.1f}% Positive, {a_neg:.1f}% Negative (Mentions: {len(aspect_data)})\n"
                
                # Sample phrases for context
                phrase_col = f"aspect_{aspect_name}_phrases"
                if phrase_col in analysis_df.columns:
                    phrases = analysis_df[phrase_col].dropna().head(3).tolist()
                    summary += f"  Sample phrases: {', '.join([str(p) for p in phrases])}\n"
                    
        return summary

    def _generate_rule_based_insights(self, analysis_df: pd.DataFrame, product_name: str) -> Dict[str, Any]:
        """Generates comprehensive rule-based insights from analysis data."""
        total_reviews = len(analysis_df)
        if total_reviews == 0:
            return {
                "executive_summary": "No reviews available for analysis.",
                "top_strengths": [],
                "critical_issues": [],
                "aspect_deep_dive": {},
                "actionable_suggestions": ["Collect customer reviews to enable sentiment analysis."],
                "competitive_risk": "Unable to assess competitive risk without review data.",
                "sentiment_trend": "No data available for trend analysis.",
                "recommended_priority": "Begin collecting customer feedback."
            }
            
        overall_counts = analysis_df['overall_sentiment'].value_counts(normalize=True) * 100
        pos_pct = overall_counts.get('positive', 0.0)
        neg_pct = overall_counts.get('negative', 0.0)
        neu_pct = overall_counts.get('neutral', 0.0)
        
        # Compute per-aspect statistics
        aspect_scores = {}
        aspect_cols = [col for col in analysis_df.columns if col.startswith('aspect_') and col.endswith('_sentiment')]
        for col in aspect_cols:
            aspect = col.replace('aspect_', '').replace('_sentiment', '')
            valid_data = analysis_df[col].dropna()
            if len(valid_data) == 0:
                continue
            counts = valid_data.value_counts(normalize=True) * 100
            aspect_scores[aspect] = {
                "positive": counts.get('positive', 0.0),
                "negative": counts.get('negative', 0.0),
                "neutral": counts.get('neutral', 0.0),
                "mentions": len(valid_data)
            }
            
        # Sort aspects by positive and negative percentages
        top_positive = sorted(aspect_scores.items(), key=lambda x: x[1]['positive'], reverse=True)
        top_negative = sorted(aspect_scores.items(), key=lambda x: x[1]['negative'], reverse=True)
        
        strengths = [f"{k.capitalize()} ({v['positive']:.1f}% positive across {v['mentions']} mentions)" for k, v in top_positive[:3] if v['positive'] > 0]
        issues = [f"{k.capitalize()} ({v['negative']:.1f}% negative across {v['mentions']} mentions)" for k, v in top_negative[:3] if v['negative'] > 0]
        
        # --- Executive Summary ---
        if pos_pct >= 60:
            overall_tone = "predominantly positive"
        elif neg_pct >= 60:
            overall_tone = "predominantly negative"
        elif pos_pct >= 40:
            overall_tone = "generally positive with notable concerns"
        elif neg_pct >= 40:
            overall_tone = "generally negative with some positives"
        else:
            overall_tone = "mixed"

        exec_summary = (
            f"Analysis of {total_reviews} customer reviews for '{product_name}' reveals "
            f"{overall_tone} sentiment. "
            f"Overall, {pos_pct:.1f}% of reviews are positive, {neg_pct:.1f}% are negative, "
            f"and {neu_pct:.1f}% are neutral. "
        )
        if strengths:
            top_strength_names = [k for k, v in top_positive[:2]]
            exec_summary += f"Customers particularly praised the {' and '.join(top_strength_names)}. "
        if issues:
            top_issue_names = [k for k, v in top_negative[:2] if v['negative'] > 15]
            if top_issue_names:
                exec_summary += f"Key areas of concern include {' and '.join(top_issue_names)}."

        # --- Aspect Deep Dive ---
        aspect_deep_dive = {}
        for aspect, stats in aspect_scores.items():
            mention_count = stats['mentions']
            mention_share = (mention_count / total_reviews) * 100
            deep_text = (
                f"{aspect.capitalize()} was mentioned in {mention_count} reviews "
                f"({mention_share:.1f}% of total). "
                f"Sentiment breakdown: {stats['positive']:.1f}% positive, "
                f"{stats['neutral']:.1f}% neutral, {stats['negative']:.1f}% negative."
            )
            # Add phrase samples if available
            phrase_col = f"aspect_{aspect}_phrases"
            if phrase_col in analysis_df.columns:
                sample_phrases = analysis_df[phrase_col].dropna().head(3).tolist()
                if sample_phrases:
                    deep_text += f" Sample feedback: {'; '.join([str(p) for p in sample_phrases])}."
            aspect_deep_dive[aspect] = deep_text
            
        # --- Actionable Suggestions ---
        suggestions = []
        for k, v in top_negative[:3]:
            if v['negative'] > 20:
                suggestions.append(
                    f"Investigate and improve '{k}' — {v['negative']:.1f}% of {v['mentions']} mentions are negative."
                )
            elif v['negative'] > 10:
                suggestions.append(
                    f"Monitor '{k}' closely — {v['negative']:.1f}% negative sentiment detected."
                )
        for k, v in top_positive[:2]:
            if v['positive'] > 60:
                suggestions.append(
                    f"Leverage strong '{k}' sentiment ({v['positive']:.1f}% positive) in marketing materials."
                )
        if not suggestions:
            suggestions.append("Continue monitoring customer feedback to maintain high satisfaction rates.")
        suggestions.append("Consider collecting more detailed feedback on underperforming aspects.")
            
        # --- Competitive Risk ---
        high_risk_aspects = [k for k, v in top_negative[:3] if v['negative'] > 30]
        if high_risk_aspects:
            competitive_risk = (
                f"High competitive risk in: {', '.join(high_risk_aspects)}. "
                f"Competitors offering better {high_risk_aspects[0]} could attract dissatisfied customers."
            )
        elif issues:
            competitive_risk = (
                f"Moderate competitive risk. Issues with {issues[0].split(' (')[0].lower()} "
                f"could drive some customers to alternatives if not addressed."
            )
        else:
            competitive_risk = "Low immediate competitive risk based on current sentiment data."
        
        # --- Sentiment Trend ---
        sentiment_trend = (
            f"Based on {total_reviews} reviews analyzed: "
            f"the positive-to-negative ratio is {pos_pct:.1f}:{neg_pct:.1f}. "
            f"Temporal trend analysis requires timestamped review data spanning multiple periods."
        )
        
        # --- Recommended Priority ---
        if issues:
            priority_aspect = top_negative[0][0]
            priority_neg = top_negative[0][1]['negative']
            recommended_priority = (
                f"Address '{priority_aspect}' issues first — {priority_neg:.1f}% negative sentiment "
                f"across {top_negative[0][1]['mentions']} mentions makes it the most critical area for improvement."
            )
        else:
            recommended_priority = "Maintain current product quality and continue monitoring feedback for emerging issues."
        
        return {
            "executive_summary": exec_summary,
            "top_strengths": strengths,
            "critical_issues": issues,
            "aspect_deep_dive": aspect_deep_dive,
            "actionable_suggestions": suggestions,
            "competitive_risk": competitive_risk,
            "sentiment_trend": sentiment_trend,
            "recommended_priority": recommended_priority
        }

    def generate_insights(self, analysis_df: pd.DataFrame, product_name: str) -> Dict[str, Any]:
        """
        Generates a structured insights report from sentiment analysis data.
        
        Args:
            analysis_df (pd.DataFrame): Aggregated sentiment data.
            product_name (str): Name of the product analyzed.
            
        Returns:
            Dict: Structured insights with 8 keys: executive_summary, top_strengths,
                  critical_issues, aspect_deep_dive, actionable_suggestions,
                  competitive_risk, sentiment_trend, recommended_priority.
        """
        logger.info(f"Generating rule-based insights for '{product_name}'...")
        return self._generate_rule_based_insights(analysis_df, product_name)

if __name__ == "__main__":
    engine = InsightsEngine()
    # Dummy data
    df = pd.DataFrame({
        "overall_sentiment": ["positive", "negative", "positive"],
        "aspect_camera_sentiment": ["positive", None, "positive"],
        "aspect_battery_sentiment": ["negative", "negative", "negative"]
    })
    
    print("Data Summary Output:\n", engine._prepare_data_summary(df))
    print("\nInsights:\n", json.dumps(engine.generate_insights(df, "Test Product"), indent=2))

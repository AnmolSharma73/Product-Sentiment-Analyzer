import os
import json
from typing import Dict, Any, List
import pandas as pd
from anthropic import Anthropic
from utils.constants import CLAUDE_MODEL
from utils.helpers import setup_logger

logger = setup_logger(__name__)

class InsightsEngine:
    """
    Generates actionable insights from aggregated sentiment data using Claude AI.
    """
    
    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.model = CLAUDE_MODEL
        self.client = Anthropic(api_key=self.api_key) if self.api_key and self.api_key != "your_key_here" else None

    def _prepare_data_summary(self, analysis_df: pd.DataFrame) -> str:
        """Aggregates DataFrame data into a structured prompt text."""
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
        """Generates basic rule-based insights when AI API is unavailable."""
        total_reviews = len(analysis_df)
        if total_reviews == 0:
            return {}
            
        overall_counts = analysis_df['overall_sentiment'].value_counts(normalize=True) * 100
        pos_pct = overall_counts.get('positive', 0.0)
        neg_pct = overall_counts.get('negative', 0.0)
        
        aspect_scores = {}
        aspect_cols = [col for col in analysis_df.columns if col.startswith('aspect_') and col.endswith('_sentiment')]
        for col in aspect_cols:
            aspect = col.replace('aspect_', '').replace('_sentiment', '')
            counts = analysis_df[col].value_counts(normalize=True) * 100
            aspect_scores[aspect] = {
                "positive": counts.get('positive', 0.0),
                "negative": counts.get('negative', 0.0)
            }
            
        # Sort aspects by positive and negative percentages
        top_positive = sorted(aspect_scores.items(), key=lambda x: x[1]['positive'], reverse=True)
        top_negative = sorted(aspect_scores.items(), key=lambda x: x[1]['negative'], reverse=True)
        
        strengths = [f"{k.capitalize()} ({v['positive']:.1f}% positive)" for k, v in top_positive[:3] if v['positive'] > 0]
        issues = [f"{k.capitalize()} ({v['negative']:.1f}% negative)" for k, v in top_negative[:3] if v['negative'] > 0]
        
        exec_summary = f"Analysis of {total_reviews} reviews for {product_name} shows a generally {'positive' if pos_pct >= 50 else 'negative' if neg_pct >= 50 else 'mixed'} sentiment. "
        exec_summary += f"Overall, {pos_pct:.1f}% of reviews are positive and {neg_pct:.1f}% are negative. "
        if strengths:
            exec_summary += f"Customers particularly praised the {', '.join([k for k, v in top_positive[:2]])}."
            
        suggestions = []
        if issues:
            for k, v in top_negative[:3]:
                if v['negative'] > 20:
                    suggestions.append(f"Investigate and improve the {k} as {v['negative']:.1f}% of mentions are negative.")
        if not suggestions:
            suggestions.append("Continue monitoring customer feedback to maintain high satisfaction rates.")
            
        return {
            "executive_summary": exec_summary,
            "top_strengths": strengths,
            "critical_issues": issues,
            "aspect_deep_dive": {k: f"{v['positive']:.1f}% positive, {v['negative']:.1f}% negative" for k, v in aspect_scores.items()},
            "actionable_suggestions": suggestions,
            "competitive_risk": f"Competitors might capitalize on issues with {issues[0].split(' ')[0]}." if issues else "Low immediate competitive risk based on current sentiment.",
            "sentiment_trend": "Sentiment trend analysis requires temporal data spanning several months.",
            "recommended_priority": f"Fix issues related to {issues[0].split(' ')[0]} immediately." if issues else "Maintain current product quality."
        }

    def generate_insights(self, analysis_df: pd.DataFrame, product_name: str) -> Dict[str, Any]:
        """
        Calls Claude API to generate a structured insights report.
        
        Args:
            analysis_df (pd.DataFrame): Aggregated sentiment data.
            product_name (str): Name of the product analyzed.
            
        Returns:
            Dict: Parsed JSON response from Claude containing insights.
        """
        if not self.client:
            logger.info("Anthropic API Key not found. Generating rule-based insights instead.")
            return self._generate_rule_based_insights(analysis_df, product_name)

        data_summary = self._prepare_data_summary(analysis_df)
        
        system_prompt = """You are a product analytics expert helping e-commerce businesses understand 
customer sentiment. Be specific, data-driven, and actionable. 
Always reference actual numbers from the data.
You must output ONLY valid JSON without any markdown formatting block around it."""

        user_prompt = f"""
Analyze the following customer review data for the product: "{product_name}".

Data Summary:
{data_summary}

Return a JSON object with EXACTLY these keys:
- executive_summary: string (3-4 sentences overview)
- top_strengths: list of strings (top 3 things customers love, with % positive)
- critical_issues: list of strings (top 3 pain points, with % negative)
- aspect_deep_dive: object mapping aspect names to a paragraph of analysis
- actionable_suggestions: list of strings (5 specific, implementable recommendations)
- competitive_risk: string (what issues might drive customers to competitors)
- sentiment_trend: string (commentary on review patterns if dates are available)
- recommended_priority: string (which issue to fix first and why)
"""

        try:
            logger.info(f"Calling Claude API for {product_name}...")
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.2,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            # Parse the response text as JSON
            response_text = response.content[0].text.strip()
            
            # Try to strip markdown JSON block if present
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            insights = json.loads(response_text)
            return insights
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude JSON response: {e}")
            logger.debug(f"Raw response: {response_text}")
            return {
                "error": "Failed to parse AI response.",
                "executive_summary": "An error occurred while generating structured insights.",
                "raw_response": response_text
            }
        except Exception as e:
            logger.error(f"Claude API Error: {e}")
            return {
                "error": str(e),
                "executive_summary": f"API Error: {str(e)}"
            }

if __name__ == "__main__":
    engine = InsightsEngine()
    # Dummy data
    df = pd.DataFrame({
        "overall_sentiment": ["positive", "negative", "positive"],
        "aspect_camera_sentiment": ["positive", None, "positive"],
        "aspect_battery_sentiment": ["negative", "negative", "negative"]
    })
    
    print("Data Summary Output:\n", engine._prepare_data_summary(df))

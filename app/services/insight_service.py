"""
Insight service — wraps rule-based InsightsEngine and optionally LLM generation.
Falls back to rule-based if LLM is not configured.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from core.insights_engine import InsightsEngine
from app.api.schemas.schemas import InsightFinding, InsightsResponse, Recommendation
from app.core.config import get_settings

logger = logging.getLogger(__name__)
_engine = InsightsEngine()


def _rule_based_response(df: pd.DataFrame, product_name: str) -> InsightsResponse:
    """Generate structured InsightsResponse using the rule-based engine."""
    raw = _engine.generate_insights(df, product_name)

    # Build key findings from aspect data
    findings: List[InsightFinding] = []
    aspect_cols = [c for c in df.columns if c.startswith("aspect_") and c.endswith("_sentiment")]
    for col in aspect_cols:
        aspect = col.replace("aspect_", "").replace("_sentiment", "")
        valid = df[col].dropna()
        if valid.empty:
            continue
        counts = valid.value_counts()
        total = len(valid)
        dominant = counts.idxmax()
        findings.append(InsightFinding(
            aspect=aspect,
            sentiment=dominant,
            evidence_count=total,
            insight=raw.get("aspect_deep_dive", {}).get(aspect, f"{aspect} mentioned {total} times."),
        ))

    # Build recommendations
    recs: List[Recommendation] = []
    for s in raw.get("actionable_suggestions", []):
        priority = "high" if any(w in s.lower() for w in ["investigate", "critical", "urgent"]) else "medium"
        recs.append(Recommendation(priority=priority, recommendation=s))

    return InsightsResponse(
        summary=raw.get("executive_summary", ""),
        key_findings=findings,
        recommendations=recs,
        source="rule_based",
    )


def _llm_response(df: pd.DataFrame, product_name: str, rule_insights: InsightsResponse) -> InsightsResponse:
    """
    Optionally enhance insights using an LLM.
    Statistics are calculated from actual data — the LLM only writes prose.
    """
    settings = get_settings()
    if not settings.llm_enabled:
        return rule_insights

    # Build a data-grounded prompt
    findings_text = "\n".join(
        f"- {f.aspect}: {f.sentiment} ({f.evidence_count} mentions) — {f.insight}"
        for f in rule_insights.key_findings
    )
    prompt = (
        f"You are a product analyst. Based on the following measured statistics for '{product_name}', "
        f"write a concise executive summary and prioritized recommendations. "
        f"Do NOT invent any numbers. Only use the evidence provided.\n\n"
        f"Findings:\n{findings_text}\n\n"
        f"Existing summary: {rule_insights.summary}\n\n"
        f"Return JSON with keys: summary (str), recommendations (list of str)."
    )

    try:
        if settings.llm_provider == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=settings.llm_api_key, timeout=settings.llm_timeout)
            resp = client.chat.completions.create(
                model=settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            parsed = json.loads(resp.choices[0].message.content)
        elif settings.llm_provider == "anthropic":
            import anthropic
            client = anthropic.Anthropic(api_key=settings.llm_api_key)
            resp = client.messages.create(
                model=settings.llm_model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            parsed = json.loads(resp.content[0].text)
        else:
            logger.warning(f"Unknown LLM provider: {settings.llm_provider}. Falling back to rule-based.")
            return rule_insights

        llm_recs = [
            Recommendation(priority="high", recommendation=r)
            for r in parsed.get("recommendations", [])[:5]
        ]
        return InsightsResponse(
            summary=parsed.get("summary", rule_insights.summary),
            key_findings=rule_insights.key_findings,  # Keep measured data
            recommendations=llm_recs or rule_insights.recommendations,
            source="llm",
        )

    except Exception as e:
        logger.error(f"LLM insight generation failed: {e}. Falling back to rule-based.")
        return rule_insights


def generate_insights(df: pd.DataFrame, product_name: str) -> InsightsResponse:
    """Main entry point. Uses LLM if configured, otherwise rule-based."""
    rule_result = _rule_based_response(df, product_name)
    return _llm_response(df, product_name, rule_result)

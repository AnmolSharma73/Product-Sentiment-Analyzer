"""
Data validation and quality reporting module for ingested review datasets.
Handles encoding detection, column mapping, missing values, duplicates, and anomaly detection.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class ValidationReport:
    total_rows: int = 0
    valid_reviews: int = 0
    empty_reviews: int = 0
    duplicate_reviews: int = 0
    detected_columns: List[str] = field(default_factory=list)
    mapped_text_column: Optional[str] = None
    mapped_product_column: Optional[str] = None
    mapped_rating_column: Optional[str] = None
    mapped_date_column: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True


class DataValidator:
    """Validates raw review tabular data and generates quality audit reports."""

    TEXT_CANDIDATES = [
        "review", "review_text", "text", "comment", "feedback",
        "description", "body", "content", "opinion", "reviews",
    ]
    PRODUCT_CANDIDATES = ["product", "product_name", "item", "title", "asin", "sku"]
    RATING_CANDIDATES = ["rating", "score", "stars", "star_rating"]
    DATE_CANDIDATES = ["date", "timestamp", "created_at", "review_date", "time"]

    @classmethod
    def _find_column(cls, df_columns: List[str], candidates: List[str]) -> Optional[str]:
        cols_lower = {c.lower().strip(): c for c in df_columns}
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        # Partial match
        for cand in candidates:
            for col_lower, orig in cols_lower.items():
                if cand in col_lower:
                    return orig
        return None

    @classmethod
    def validate_dataframe(
        cls,
        df: pd.DataFrame,
        custom_text_col: Optional[str] = None,
        custom_product_col: Optional[str] = None,
    ) -> Tuple[List[str], ValidationReport]:
        """
        Validates DataFrame, resolves column mappings, deduplicates,
        and returns clean list of review strings with validation report.
        """
        report = ValidationReport(total_rows=len(df), detected_columns=list(df.columns))

        if df.empty:
            report.is_valid = False
            report.warnings.append("Dataset contains 0 rows.")
            return [], report

        # Resolve review text column
        text_col = custom_text_col or cls._find_column(list(df.columns), cls.TEXT_CANDIDATES)
        if not text_col or text_col not in df.columns:
            # Fall back to first object/string column
            str_cols = df.select_dtypes(include=["object", "string"]).columns
            if len(str_cols) > 0:
                text_col = str_cols[0]
                report.warnings.append(f"Auto-selected '{text_col}' as text column.")
            else:
                report.is_valid = False
                report.warnings.append("No suitable review text column found.")
                return [], report

        report.mapped_text_column = text_col
        report.mapped_product_column = custom_product_col or cls._find_column(list(df.columns), cls.PRODUCT_CANDIDATES)
        report.mapped_rating_column = cls._find_column(list(df.columns), cls.RATING_CANDIDATES)
        report.mapped_date_column = cls._find_column(list(df.columns), cls.DATE_CANDIDATES)

        # Missing values
        raw_series = df[text_col]
        missing_count = int(raw_series.isna().sum())
        report.empty_reviews = missing_count

        # Extract non-empty strings
        non_empty = raw_series.dropna().astype(str).str.strip()
        non_empty = non_empty[non_empty != ""]

        # Duplicates
        duplicate_count = int(non_empty.duplicated().sum())
        report.duplicate_reviews = duplicate_count
        clean_reviews = non_empty.drop_duplicates().tolist()

        report.valid_reviews = len(clean_reviews)

        if report.valid_reviews == 0:
            report.is_valid = False
            report.warnings.append("All rows in selected column are empty or invalid.")

        if report.duplicate_reviews > 0:
            report.warnings.append(f"Removed {report.duplicate_reviews} duplicate reviews.")

        return clean_reviews, report

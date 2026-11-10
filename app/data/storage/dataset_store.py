"""
SQLite-backed dataset repository for dataset metadata management.
Easily swappable with PostgreSQL via standard SQL queries or SQLAlchemy.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import get_settings


def _get_db_path() -> Path:
    settings = get_settings()
    db_url = settings.database_url
    if db_url.startswith("sqlite:///"):
        raw_path = db_url.replace("sqlite:///", "")
        p = Path(raw_path)
    else:
        p = Path("data/copilot.db")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_db():
    """Initializes tables if they do not exist."""
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                product_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                review_count INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                review_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                review_text TEXT NOT NULL,
                overall_sentiment TEXT,
                overall_score REAL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def save_dataset_meta(
    dataset_id: str,
    name: str,
    product_name: str,
    file_type: str,
    review_count: int,
):
    init_db()
    db_path = _get_db_path()
    now_iso = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO datasets (dataset_id, name, product_name, file_type, review_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (dataset_id, name, product_name, file_type, review_count, now_iso),
        )
        conn.commit()


def list_datasets() -> List[Dict[str, Any]]:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM datasets ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


def get_dataset(dataset_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
        return dict(row) if row else None


def delete_dataset(dataset_id: str) -> bool:
    init_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM reviews WHERE dataset_id = ?", (dataset_id,))
        cur = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        conn.commit()
        return cur.rowcount > 0

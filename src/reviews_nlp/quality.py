"""Contrôle qualité : rapport pandas sur les données brutes + contrôles SQL sur les données nettoyées."""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .config import stars_to_sentiment
from .data import HTML_TAG, MIN_CHARS


def quality_report(df_raw: pd.DataFrame) -> dict:
    """Mesure les défauts de qualité du corpus brut (avant nettoyage)."""
    text = df_raw["text"]
    as_str = text.dropna().astype(str)
    sentiment = stars_to_sentiment(df_raw["stars"])
    counts = sentiment.value_counts()
    lengths = as_str.str.len()

    return {
        "n_rows": int(len(df_raw)),
        "n_missing_text": int(text.isna().sum()),
        "n_too_short": int((as_str.str.strip().str.len() < MIN_CHARS).sum()),
        "n_duplicated_texts": int(as_str.duplicated().sum()),
        "n_html_or_newline_artifacts": int(
            as_str.str.contains(HTML_TAG).sum() + as_str.str.contains("\\n", regex=False).sum()
        ),
        "n_invalid_stars": int((~df_raw["stars"].between(1, 5)).sum()),
        "sentiment_counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 2),
        "length_chars": {
            "mean": round(float(lengths.mean()), 1),
            "median": float(lengths.median()),
            "p95": float(lengths.quantile(0.95)),
        },
    }


def _parse_sql_file(sql_path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    name, buf = None, []
    for line in Path(sql_path).read_text(encoding="utf-8").splitlines():
        match = re.match(r"--\s*name:\s*(\w+)", line)
        if match:
            if name:
                queries[name] = "\n".join(buf).strip().rstrip(";")
            name, buf = match.group(1), []
        elif name and not line.strip().startswith("--"):
            buf.append(line)
    if name:
        queries[name] = "\n".join(buf).strip().rstrip(";")
    return queries


def run_sql_checks(db_path: Path, sql_path: Path) -> dict[str, list[dict]]:
    """Exécute les requêtes de sql/quality_checks.sql et renvoie les résultats."""
    results = {}
    with sqlite3.connect(db_path) as conn:
        for name, query in _parse_sql_file(sql_path).items():
            results[name] = pd.read_sql_query(query, conn).to_dict(orient="records")
    return results

"""Shared utilities for the printing analytics project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
REPORTS_DIR = OUTPUT_DIR / "reports"


def ensure_directories() -> None:
    """Create the project output directories if they do not exist."""
    for directory in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FIGURES_DIR,
        TABLES_DIR,
        REPORTS_DIR,
        PROJECT_ROOT / "notebooks",
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def slugify(value: object, max_length: int = 90) -> str:
    """Return a safe, readable filename slug."""
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return (text[:max_length] or "output")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while returning NaN for zero denominators."""
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def weighted_va_margin(va_amount: pd.Series, sell_price: pd.Series) -> float:
    """Return grouped VA margin as sum(VA Amount) divided by sum(Sell Price)."""
    va_total = pd.to_numeric(va_amount, errors="coerce").sum(min_count=1)
    revenue_total = pd.to_numeric(sell_price, errors="coerce").sum(min_count=1)
    if pd.isna(revenue_total) or revenue_total == 0:
        return np.nan
    return float(va_total / revenue_total)


def save_table(df: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Save a dataframe to CSV and return the written path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path


def save_json(payload: dict, path: Path) -> Path:
    """Save a dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def currency(value: float | int | None) -> str:
    """Format a value as a whole-pound business figure."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"GBP {float(value):,.0f}"


def percent(value: float | int | None, decimals: int = 1) -> str:
    """Format a decimal percentage value."""
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{decimals}f}%"


def dataframe_to_markdown(
    df: pd.DataFrame,
    max_rows: int = 10,
    float_format: str = "{:,.2f}",
) -> str:
    """Render a compact markdown table without requiring tabulate."""
    if df is None or df.empty:
        return "_No rows available._"

    preview = df.head(max_rows).copy()
    preview.columns = [str(col) for col in preview.columns]

    def fmt(value: object) -> str:
        if isinstance(value, (float, np.floating)):
            if pd.isna(value):
                return ""
            return float_format.format(float(value))
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}"
        if pd.isna(value):
            return ""
        return str(value)

    rows = [[fmt(value) for value in row] for row in preview.to_numpy()]
    widths = [
        max(len(str(col)), *(len(row[index]) for row in rows))
        for index, col in enumerate(preview.columns)
    ]
    header = "| " + " | ".join(
        str(col).ljust(widths[index]) for index, col in enumerate(preview.columns)
    ) + " |"
    rule = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [
        "| " + " | ".join(row[index].ljust(widths[index]) for index in range(len(widths))) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def display_format_for_column(column_name: str, max_abs: float | int | None = None) -> str:
    """Choose a Streamlit/Pandas display format from business column names.

    Counts, dates, days, cadence, impressions, and production quantities are
    deliberately kept as plain numbers. The previous formatter treated every
    column containing "order" as currency, which incorrectly displayed fields
    such as Order Count and Reorder Cadence Days with a pound sign.
    """
    label = str(column_name).lower().replace("_", " ")
    compact_label = label.replace(" ", "")

    percent_terms = ("%", "margin", "markup", "mup", "share")
    decimal_day_terms = (
        "average reorder",
        "median reorder",
        "reorder cadence",
        "reorder gap",
        "expected reorder",
        "customer tenure",
    )
    whole_number_terms = (
        "count",
        "distinct order days",
        "days since",
        "lead time",
        "quantity",
        "impressions",
        "jobs",
        "customers",
        "week",
        "year",
        "month",
        "plates",
        "press hours",
        "press hrs",
    )
    whole_number_compact_terms = ("presshours", "presshrs")
    currency_terms = (
        "revenue",
        "va",
        "value",
        "price",
        "purchase",
        "labour",
        "paper",
        "amount",
        "profit",
        "sell",
        "rebate",
        "handling",
    )
    currency_phrases = ("average order", "order value", "order va")

    if any(term in label for term in percent_terms):
        try:
            max_value = float(max_abs) if max_abs is not None else 0.0
        except (TypeError, ValueError):
            max_value = 0.0
        return "{:.1%}" if max_value <= 1.5 else "{:.1f}%"

    if any(term in label for term in decimal_day_terms):
        return "{:,.1f}"

    if any(term in label for term in whole_number_terms) or any(
        term in compact_label for term in whole_number_compact_terms
    ):
        return "{:,.0f}"

    if any(term in label for term in currency_terms) or any(
        phrase in label for phrase in currency_phrases
    ):
        return "\u00a3{:,.0f}"

    return "{:,.0f}"

def flatten_columns(columns: Iterable[object]) -> list[str]:
    """Flatten pandas multi-index columns into readable strings."""
    flattened: list[str] = []
    for column in columns:
        if isinstance(column, tuple):
            parts = [str(part) for part in column if part not in ("", None)]
            flattened.append(" ".join(parts).strip())
        else:
            flattened.append(str(column))
    return flattened





"""Data loading, cleaning, outlier detection, and data quality reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import dataframe_to_markdown, safe_divide, slugify


COLUMN_RENAMES = {
    "Puchases": "Purchases",
    "Rep Name": "Rep",
}

NUMERIC_COLUMNS = [
    "Year",
    "Month",
    "Week No",
    "Quantity",
    "Sell Price",
    "Mup%",
    "VA Amount",
    "VA/24",
    "VA%",
    "VA/K",
    "Rebate",
    "Purchases",
    "Press hrs",
    "Impressions",
    "Handling",
    "Labour",
    "Paper",
    "labmup",
    "manadj",
    "mupnett",
    "Plates",
    "AmtInv",
]

PERCENTAGE_COLUMNS = ["Mup%", "VA%"]
DATE_COLUMNS = ["SalesIn", "SalesOut", "Ship date"]
CATEGORICAL_COLUMNS = [
    "Title",
    "CustomerID",
    "Job Status",
    "Customer Name",
    "Rep",
    "Region",
    "Industry",
    "Work Type",
    "Product Type",
    "Binding Type",
    "Currency",
]


@dataclass
class CleanResult:
    """Container for cleaned data and quality metadata."""

    raw_rows: int
    raw_columns: int
    cleaned_rows: int
    cleaned_columns: int
    duplicate_rows_removed: int
    missing_before: pd.DataFrame
    missing_after: pd.DataFrame
    outlier_report: pd.DataFrame
    cleaning_actions: list[str]


def load_raw_data(path: str | Path, sheet_name: str | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a CSV or Excel workbook containing printing job data."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    metadata: dict[str, Any] = {"source_path": str(path), "source_type": path.suffix.lower()}

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        with pd.ExcelFile(path) as workbook:
            metadata["sheet_names"] = workbook.sheet_names
            selected_sheet = sheet_name or (
                "Master Plain (Anon)"
                if "Master Plain (Anon)" in workbook.sheet_names
                else workbook.sheet_names[0]
            )
            metadata["selected_sheet"] = selected_sheet
            df = pd.read_excel(workbook, sheet_name=selected_sheet)
    elif path.suffix.lower() == ".csv":
        metadata["selected_sheet"] = None
        df = pd.read_csv(path)
    else:
        raise ValueError("Supported input formats are .csv, .xlsx, .xlsm, and .xls")

    metadata["raw_shape"] = df.shape
    return df, metadata


def load_field_definitions(path: str | Path) -> pd.DataFrame:
    """Load the optional field definitions tab from the workbook."""
    path = Path(path)
    if path.suffix.lower() not in {".xlsx", ".xlsm", ".xls"}:
        return pd.DataFrame()
    try:
        with pd.ExcelFile(path) as workbook:
            if "Field Definitions" not in workbook.sheet_names:
                return pd.DataFrame()
            return pd.read_excel(workbook, sheet_name="Field Definitions")
    except ValueError:
        return pd.DataFrame()


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert currency, comma-formatted, and percentage-like strings to numbers.

    Currency dash placeholders such as "-", "GBP-", and "GBP -" are source-system
    blanks, not real negative values, so they are normalised to missing values.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[£$???,]", "", regex=True)
        .str.strip()
        .str.replace(r"^[\s\-–—]+$", "", regex=True)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _normalise_percentages(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Ensure percentage columns use decimal units, for example 0.125 for 12.5%."""
    actions: list[str] = []
    for column in PERCENTAGE_COLUMNS:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        high_values = series.abs().dropna()
        if not high_values.empty and high_values.quantile(0.95) > 1.5:
            df[column] = series / 100
            actions.append(f"Converted {column} from whole percent units to decimals.")
        else:
            df[column] = series
            actions.append(f"Validated {column} as decimal percentage units.")
    return df, actions


def _missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Create a missing-value summary table."""
    summary = pd.DataFrame(
        {
            "column": df.columns,
            "missing_count": df.isna().sum().to_numpy(),
            "missing_pct": (df.isna().mean() * 100).round(2).to_numpy(),
            "dtype": [str(dtype) for dtype in df.dtypes],
        }
    )
    return summary.sort_values(["missing_count", "column"], ascending=[False, True])


def _impute_missing_sell_price_from_formula(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Fill missing or zero Sell Price from Purchases + Rebate + VA Amount.

    Rebate is treated as zero when blank because source systems often leave
    no-rebate jobs empty. Purchases and VA Amount must be present so the imputed
    value remains traceable to the commercial formula.
    """
    if "Sell Price" not in df.columns:
        return df, 0

    df["Sell Price Formula Imputed"] = False
    df["Sell Price Was Imputed"] = False
    required = {"Purchases", "VA Amount"}
    if not required.issubset(df.columns):
        return df, 0

    sell_price = pd.to_numeric(df["Sell Price"], errors="coerce")
    purchases = pd.to_numeric(df["Purchases"], errors="coerce")
    rebate = pd.to_numeric(df.get("Rebate", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    va_amount = pd.to_numeric(df["VA Amount"], errors="coerce")

    # In the source workbook, zero sell price is commonly used as the same
    # business placeholder as a blank price, so both are formula-imputed.
    missing_sell_price = sell_price.isna() | sell_price.eq(0)
    can_impute = missing_sell_price & purchases.notna() & va_amount.notna()
    imputed_sell_price = purchases + rebate + va_amount

    df.loc[can_impute, "Sell Price"] = imputed_sell_price.loc[can_impute]
    df.loc[can_impute, "Sell Price Formula Imputed"] = True
    df.loc[can_impute, "Sell Price Was Imputed"] = True
    return df, int(can_impute.sum())


def detect_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    iqr_multiplier: float = 1.5,
) -> pd.DataFrame:
    """Detect numeric outliers using the interquartile-range method."""
    columns = columns or [col for col in NUMERIC_COLUMNS if col in df.columns]
    rows: list[dict[str, Any]] = []

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (iqr_multiplier * iqr)
        upper = q3 + (iqr_multiplier * iqr)
        mask = (pd.to_numeric(df[column], errors="coerce") < lower) | (
            pd.to_numeric(df[column], errors="coerce") > upper
        )
        rows.append(
            {
                "column": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_count": int(mask.sum()),
                "outlier_pct": round(float(mask.mean() * 100), 2),
                "minimum": float(series.min()),
                "maximum": float(series.max()),
            }
        )

    return pd.DataFrame(rows).sort_values("outlier_count", ascending=False)


def clean_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, CleanResult]:
    """Clean the raw printing dataset while preserving auditable quality metadata."""
    df = raw_df.copy()
    actions: list[str] = []
    raw_rows, raw_columns = df.shape

    df.columns = [str(column).strip() for column in df.columns]
    rename_map = {old: new for old, new in COLUMN_RENAMES.items() if old in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
        actions.append(f"Renamed columns: {rename_map}.")

    missing_before = _missing_summary(df)

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        df = df.drop_duplicates().reset_index(drop=True)
        actions.append(f"Removed {duplicate_rows} duplicate rows.")
    else:
        actions.append("No duplicate rows found.")

    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
            actions.append(f"Converted {column} to datetime.")

    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = _to_numeric(df[column])
    actions.append("Converted numeric, currency, and production metric columns.")

    df, percentage_actions = _normalise_percentages(df)
    actions.extend(percentage_actions)

    for column in NUMERIC_COLUMNS:
        if column in df.columns and df[column].isna().any():
            # Preserve source-system missingness before any business-rule imputation.
            # This keeps anomalies visible in Streamlit even when a safe numeric
            # value is needed for downstream aggregations.
            df[f"{slugify(column)}_source_missing"] = df[column].isna().astype(int)

    df, sell_price_imputed_count = _impute_missing_sell_price_from_formula(df)
    if sell_price_imputed_count:
        actions.append(
            "Imputed missing or zero Sell Price values as Purchases plus Rebate plus VA Amount "
            f"for {sell_price_imputed_count:,} row(s)."
        )

    if {"Purchases", "Sell Price", "VA Amount"}.issubset(df.columns):
        missing = df["Purchases"].isna()
        df.loc[missing, "Purchases"] = (
            df.loc[missing, "Sell Price"] - df.loc[missing, "VA Amount"]
        )
        actions.append("Imputed missing Purchases as Sell Price less VA Amount where possible.")

    if {"VA%", "VA Amount", "Sell Price"}.issubset(df.columns):
        missing = df["VA%"].isna()
        df.loc[missing, "VA%"] = safe_divide(
            df.loc[missing, "VA Amount"],
            df.loc[missing, "Sell Price"],
        )
        actions.append("Recomputed missing VA% from VA Amount divided by Sell Price.")

    for column in NUMERIC_COLUMNS:
        if column not in df.columns:
            continue
        if df[column].isna().any():
            df[f"{slugify(column)}_missing"] = df[column].isna().astype(int)
            fill_value = df[column].median()
            if pd.isna(fill_value):
                fill_value = 0
            df[column] = df[column].fillna(fill_value)
            actions.append(f"Filled missing {column} values with median {fill_value:,.2f}.")

    for column in CATEGORICAL_COLUMNS:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()
            df[column] = df[column].replace({"": pd.NA}).fillna("Unknown")

    outlier_report = detect_outliers_iqr(df)
    missing_after = _missing_summary(df)

    result = CleanResult(
        raw_rows=raw_rows,
        raw_columns=raw_columns,
        cleaned_rows=int(df.shape[0]),
        cleaned_columns=int(df.shape[1]),
        duplicate_rows_removed=duplicate_rows,
        missing_before=missing_before,
        missing_after=missing_after,
        outlier_report=outlier_report,
        cleaning_actions=actions,
    )
    return df, result


def build_data_quality_markdown(result: CleanResult) -> str:
    """Build a markdown data quality report."""
    actions = "\n".join(f"- {action}" for action in result.cleaning_actions)
    missing_before = dataframe_to_markdown(result.missing_before, max_rows=20)
    missing_after = dataframe_to_markdown(result.missing_after, max_rows=20)
    outliers = dataframe_to_markdown(result.outlier_report, max_rows=20)

    return f"""# Data Quality Report

## Dataset Shape

- Raw rows: {result.raw_rows:,}
- Raw columns: {result.raw_columns:,}
- Cleaned rows: {result.cleaned_rows:,}
- Cleaned columns: {result.cleaned_columns:,}
- Duplicate rows removed: {result.duplicate_rows_removed:,}

## Cleaning Actions

{actions}

## Missing Values Before Cleaning

{missing_before}

## Missing Values After Cleaning

{missing_after}

## IQR Outlier Detection

Outliers are flagged for review rather than removed automatically, because unusually large
jobs may be commercially important key accounts.

{outliers}
"""

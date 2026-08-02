"""Independent business calculation validation for printing analytics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_divide


TOLERANCE = 1e-6


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce")


def _comparison(
    df: pd.DataFrame,
    metric: str,
    current: pd.Series,
    expected: pd.Series,
    formula: str,
    tolerance: float = TOLERANCE,
) -> tuple[dict[str, object], pd.DataFrame]:
    diff = (current - expected).replace([np.inf, -np.inf], np.nan)
    comparable = current.notna() & expected.notna()
    failures = comparable & diff.abs().gt(tolerance)
    summary = {
        "Metric": metric,
        "Formula Checked": formula,
        "Rows Compared": int(comparable.sum()),
        "Failures": int(failures.sum()),
        "Pass Rate": 1 - (failures.sum() / comparable.sum()) if comparable.sum() else np.nan,
        "Mean Absolute Difference": float(diff.loc[comparable].abs().mean())
        if comparable.any()
        else np.nan,
        "Max Absolute Difference": float(diff.loc[comparable].abs().max())
        if comparable.any()
        else np.nan,
        "Status": "PASS" if not failures.any() else "REVIEW",
    }
    details = pd.DataFrame(
        {
            "Row": df.index,
            "Metric": metric,
            "Current": current,
            "Expected": expected,
            "Difference": diff,
            "Pass": ~failures,
        }
    )
    details = details.loc[failures].head(200)
    return summary, details


def validate_row_calculations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate derived fields independently and compare to current values."""

    checks: list[tuple[str, pd.Series, pd.Series, str]] = []

    sell_price = _numeric(df, "Sell Price")
    purchases = _numeric(df, "Purchases")
    rebate = _numeric(df, "Rebate")
    va_amount = _numeric(df, "VA Amount")
    impressions = _numeric(df, "Impressions")
    labour = _numeric(df, "Labour")
    labmup = _numeric(df, "labmup")
    manadj = _numeric(df, "manadj")
    mupnett = _numeric(df, "mupnett")

    checks.append(
        (
            "VA Amount",
            va_amount,
            sell_price - purchases - rebate,
            "Sell Price - Purchases - Rebate",
        )
    )
    checks.append(("VA%", _numeric(df, "VA%"), safe_divide(va_amount, sell_price), "VA Amount / Sell Price"))
    checks.append(
        (
            "VA/K",
            _numeric(df, "VA/K"),
            safe_divide(va_amount, impressions / 1000),
            "VA Amount / (Impressions / 1000)",
        )
    )
    checks.append(("mupnett", mupnett, labmup + manadj, "labmup + manadj"))
    checks.append(("Mup%", _numeric(df, "Mup%"), safe_divide(mupnett, labour), "mupnett / Labour"))

    if "Profit Margin" in df.columns:
        checks.append(("Profit Margin", _numeric(df, "Profit Margin"), safe_divide(va_amount, sell_price), "VA Amount / Sell Price"))
    if "Revenue" in df.columns:
        checks.append(("Revenue", _numeric(df, "Revenue"), sell_price, "Sell Price"))
    if "Profit" in df.columns:
        checks.append(("Profit", _numeric(df, "Profit"), va_amount, "VA Amount"))
    if "Direct Cost Estimate" in df.columns:
        expected_cost = (
            _numeric(df, "Purchases")
            + _numeric(df, "Labour")
            + _numeric(df, "Paper")
            + _numeric(df, "Handling")
        )
        checks.append(("Direct Cost Estimate", _numeric(df, "Direct Cost Estimate"), expected_cost, "Purchases + Labour + Paper + Handling"))

    summaries = []
    detail_frames = []
    for metric, current, expected, formula in checks:
        summary, details = _comparison(df, metric, current, expected, formula)
        summaries.append(summary)
        detail_frames.append(details)

    summary_df = pd.DataFrame(summaries)
    details_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    return summary_df, details_df


def validate_monthly_summary(df: pd.DataFrame, monthly_table: pd.DataFrame) -> pd.DataFrame:
    """Compare monthly table output with an independent groupby recalculation."""

    if "Sales Month" not in df.columns or monthly_table is None or monthly_table.empty:
        return pd.DataFrame()
    expected = (
        df.groupby("Sales Month", dropna=False)
        .agg(
            Jobs=("Title", "count"),
            Revenue=("Sell Price", "sum"),
            VA_Amount=("VA Amount", "sum"),
            Average_Markup=("Mup%", "mean"),
            Quantity=("Quantity", "sum"),
            Press_Hours=("Press hrs", "sum"),
        )
        .reset_index()
    )
    expected["VA_Margin"] = safe_divide(expected["VA_Amount"], expected["Revenue"])
    merged = expected.merge(
        monthly_table,
        on="Sales Month",
        how="outer",
        suffixes=("_expected", "_current"),
    )
    rows = []
    for metric in ["Jobs", "Revenue", "VA_Amount", "VA_Margin", "Average_Markup", "Quantity", "Press_Hours"]:
        expected_col = f"{metric}_expected"
        current_col = f"{metric}_current"
        if expected_col not in merged or current_col not in merged:
            continue
        diff = (merged[current_col] - merged[expected_col]).abs()
        rows.append(
            {
                "Metric": f"Monthly {metric}",
                "Rows Compared": int(diff.notna().sum()),
                "Failures": int(diff.gt(TOLERANCE).sum()),
                "Max Absolute Difference": float(diff.max()) if diff.notna().any() else np.nan,
                "Status": "PASS" if not diff.gt(TOLERANCE).any() else "REVIEW",
            }
        )
    return pd.DataFrame(rows)



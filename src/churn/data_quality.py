"""Data-quality audit for customer churn modelling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ChurnConfig
from .features import prepare_churn_transactions


def _record(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    mask: pd.Series,
    category: str,
    severity: str,
    action: str,
    explanation: str,
    scope: str = "row",
) -> None:
    mask = mask.reindex(df.index, fill_value=False).fillna(False).astype(bool)
    count = int(mask.sum())
    pct = float(count / len(df)) if len(df) else 0.0
    sample = df.loc[mask].head(1)
    rows.append(
        {
            "Scope": scope,
            "Severity": severity,
            "Category": category,
            "Record Key": str(sample.index[0]) if not sample.empty else "",
            "CustomerID": sample.get("CustomerID", pd.Series([""])).iloc[0] if not sample.empty else "",
            "Customer Name": sample.get("Customer Name", pd.Series([""])).iloc[0] if not sample.empty else "",
            "Row Count": count,
            "Percentage": pct,
            "Recommended Action": action,
            "Business Explanation": explanation,
        }
    )


def audit_churn_data(df: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    """Create a visible exceptions report for churn modelling.

    Rows are not removed here. The report separates data-quality errors from
    valid but unusual business conditions so users can decide what to review.
    """

    tx = prepare_churn_transactions(df, config)
    rows: list[dict[str, object]] = []
    total = len(tx)
    if total == 0:
        return pd.DataFrame(
            [{
                "Scope": "dataset",
                "Severity": "FAIL",
                "Category": "Empty dataset",
                "Record Key": "",
                "CustomerID": "",
                "Customer Name": "",
                "Row Count": 0,
                "Percentage": 0.0,
                "Recommended Action": "Stop modelling",
                "Business Explanation": "No transactions are available for churn analysis.",
            }]
        )

    _record(
        rows,
        tx,
        tx.duplicated(keep=False),
        "Duplicate transaction rows",
        "WARNING",
        "Flag and review before contractual reporting",
        "Repeated rows can overstate customer frequency and value.",
    )
    _record(
        rows,
        tx,
        tx["CustomerID"].eq("UNKNOWN"),
        "Missing customer identifier",
        "FAIL",
        "Exclude from customer-level modelling until resolved",
        "Churn is defined at customer level, so transactions need a stable CustomerID.",
    )
    _record(
        rows,
        tx,
        tx["SalesIn"].isna(),
        "Invalid or missing order date",
        "FAIL",
        "Exclude from time-window modelling; keep in data-quality report",
        "Snapshot features and churn labels require a valid order date.",
    )
    _record(
        rows,
        tx,
        tx["Is Source Non Positive Sell Price"],
        "Zero or negative revenue",
        "WARNING",
        "Flag; use estimated sell price where available",
        "Zero or negative revenue can represent credits, missing prices, or data-entry issues.",
    )
    _record(
        rows,
        tx,
        tx["Is Source Sell Price Below Purchase"],
        "Sell price below purchase cost",
        "WARNING",
        "Flag; include only when confirmed as real loss-making work",
        "Below-cost jobs may be valid strategic work, credits, or pricing mistakes.",
    )
    _record(
        rows,
        tx,
        tx["Is Negative Margin"],
        "Negative value-added margin",
        "WARNING",
        "Flag and review by account owner",
        "Loss-making jobs can distort value-at-risk and margin trend features.",
    )
    _record(
        rows,
        tx,
        tx["Is Incomplete Status"],
        "Open, held, or incomplete job status",
        "WARNING",
        "Flag; do not use as confirmed repeat purchase without review",
        "Incomplete jobs may not represent settled customer demand.",
    )
    _record(
        rows,
        tx,
        tx.get("Currency", pd.Series("", index=tx.index)).fillna("").astype(str).str.strip().eq(""),
        "Missing currency",
        "WARNING",
        "Flag; confirm whether values are GBP or another currency",
        "Currency differences affect value ranking and revenue-at-risk comparisons.",
    )

    if "Currency" in tx.columns and tx["Currency"].nunique(dropna=True) > 1:
        rows.append(
            {
                "Scope": "dataset",
                "Severity": "WARNING",
                "Category": "Multiple currencies",
                "Record Key": "",
                "CustomerID": "",
                "Customer Name": "",
                "Row Count": int(total),
                "Percentage": 1.0,
                "Recommended Action": "Segment or convert currencies before financial comparison",
                "Business Explanation": "The data contains multiple currencies; the current project treats source values as reported.",
            }
        )

    activity = tx[tx["Is Activity Status"] & tx["SalesIn"].notna()].copy()
    if not activity.empty:
        customer_counts = activity.groupby("CustomerID")["_Order Date"].nunique()
        single_order_customers = customer_counts[customer_counts <= 1]
        rows.append(
            {
                "Scope": "customer",
                "Severity": "WARNING",
                "Category": "Single-order customers",
                "Record Key": ", ".join(single_order_customers.index.astype(str).tolist()[:5]),
                "CustomerID": "",
                "Customer Name": "",
                "Row Count": int(single_order_customers.shape[0]),
                "Percentage": float(single_order_customers.shape[0] / max(customer_counts.shape[0], 1)),
                "Recommended Action": "Handle with cold-start rules instead of the repeat-order churn model",
                "Business Explanation": "One-time buyers do not have enough history to estimate a normal reorder cycle.",
            }
        )
        tenure = activity.groupby("CustomerID")["_Order Date"].agg(["min", "max"])
        insufficient = tenure[(tenure["max"] - tenure["min"]).dt.days < config.min_history_days]
        rows.append(
            {
                "Scope": "customer",
                "Severity": "WARNING",
                "Category": "Insufficient customer history",
                "Record Key": ", ".join(insufficient.index.astype(str).tolist()[:5]),
                "CustomerID": "",
                "Customer Name": "",
                "Row Count": int(insufficient.shape[0]),
                "Percentage": float(insufficient.shape[0] / max(tenure.shape[0], 1)),
                "Recommended Action": "Keep in dashboard but exclude from supervised churn training",
                "Business Explanation": "Short-tenure customers make churn labels unstable and should be monitored separately.",
            }
        )

    va = pd.to_numeric(tx.get("VA Amount", pd.Series(0, index=tx.index)), errors="coerce")
    if va.notna().sum() >= 10:
        q1, q3 = va.quantile([0.25, 0.75])
        iqr = q3 - q1
        outlier_mask = (va < q1 - 3 * iqr) | (va > q3 + 3 * iqr)
        _record(
            rows,
            tx,
            outlier_mask,
            "Outlier VA values",
            "WARNING",
            "Flag for review; keep when commercially valid",
            "Large positive or negative VA values can be valid key accounts or data errors.",
        )

    report = pd.DataFrame(rows)
    if not report.empty:
        report["Percentage"] = report["Percentage"].fillna(0.0)
    return report

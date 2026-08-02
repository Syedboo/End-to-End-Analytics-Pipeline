"""Business rule and anomaly detection framework for printing jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BusinessRule:
    """A reusable data-quality rule with business-facing remediation guidance."""

    name: str
    status: str
    action: str
    explanation: str
    mask: Callable[[pd.DataFrame], pd.Series]


def _false_mask(df: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=df.index)


def _column(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def build_business_rules() -> list[BusinessRule]:
    """Return the anomaly rules used consistently by the app and tests."""

    return [
        BusinessRule(
            name="Missing Sell Price",
            status="FAIL",
            action="Impute before commercial analysis",
            explanation=(
                "Sell Price is required for revenue, VA%, margin, pricing, and "
                "customer value calculations."
            ),
            mask=lambda df: _column(df, "Sell Price").isna(),
        ),
        BusinessRule(
            name="Zero Revenue",
            status="FAIL",
            action="Exclude from revenue KPIs unless confirmed as valid non-charge work",
            explanation=(
                "A zero sell price usually indicates missing pricing or a non-billable "
                "job; including it suppresses revenue and margin."
            ),
            mask=lambda df: _column(df, "Sell Price").fillna(0).eq(0),
        ),
        BusinessRule(
            name="Negative Revenue",
            status="FAIL",
            action="Review as credit note or data error before KPI inclusion",
            explanation="Negative sell price changes commercial totals and may represent a credit.",
            mask=lambda df: _column(df, "Sell Price").lt(0),
        ),
        BusinessRule(
            name="Sell Price < Purchase Cost",
            status="FAIL",
            action="Flag as loss-making and exclude from margin benchmarking by default",
            explanation=(
                "The job appears priced below external purchase cost before labour, "
                "paper, handling, and rebate effects are considered."
            ),
            mask=lambda df: _column(df, "Sell Price").lt(_column(df, "Purchases")),
        ),
        BusinessRule(
            name="Negative Margin",
            status="FAIL",
            action="Review pricing and production records",
            explanation="Negative VA or margin indicates the job destroyed value or has bad inputs.",
            mask=lambda df: (
                _column(df, "VA Amount").lt(0)
                | _column(df, "Profit Margin").lt(0)
                | _column(df, "VA%").lt(0)
            ),
        ),
        BusinessRule(
            name="Impossible Percentage",
            status="WARNING",
            action="Flag and validate against source system",
            explanation=(
                "Percentages below -100% or above 100% can occur in severe loss jobs, "
                "but they should be reviewed before interpretation."
            ),
            mask=lambda df: (
                _column(df, "VA%").abs().gt(1)
                | _column(df, "Mup%").abs().gt(5)
            ),
        ),
        BusinessRule(
            name="Division By Zero Risk",
            status="WARNING",
            action="Keep row but suppress ratio calculations where denominator is zero",
            explanation=(
                "Zero sell price, impressions, press hours, or labour can create undefined "
                "ratios such as VA%, VA/K, VA per press hour, or markup."
            ),
            mask=lambda df: (
                _column(df, "Sell Price").fillna(0).eq(0)
                | _column(df, "Impressions").fillna(0).eq(0)
                | _column(df, "Press hrs").fillna(0).eq(0)
                | _column(df, "Labour").fillna(0).eq(0)
            ),
        ),
        BusinessRule(
            name="Outlier VA Values",
            status="WARNING",
            action="Flag, do not automatically remove",
            explanation=(
                "High-value jobs may be legitimate strategic accounts; review outliers "
                "instead of deleting them."
            ),
            mask=_va_outlier_mask,
        ),
        BusinessRule(
            name="Missing Impressions",
            status="WARNING",
            action="Flag as production data missing",
            explanation="Impressions are required for VA/K and production-efficiency analysis.",
            mask=lambda df: _column(df, "Impressions").isna() | _column(df, "Impressions").eq(0),
        ),
        BusinessRule(
            name="Missing Press Hours",
            status="WARNING",
            action="Flag as production time missing",
            explanation="Press hours are required for VA per press hour and capacity analysis.",
            mask=lambda df: _column(df, "Press hrs").isna() | _column(df, "Press hrs").eq(0),
        ),
        BusinessRule(
            name="Missing Purchase Values",
            status="WARNING",
            action="Impute only when Sell Price and VA Amount reconcile",
            explanation="Purchase cost is needed to validate VA and identify loss-making work.",
            mask=lambda df: (
                _column(df, "Purchases").isna()
                | _column(df, "Purchases").eq(0)
                | _column(df, "purchases_source_missing", 0).eq(1)
                | _column(df, "purchases_missing", 0).eq(1)
            ),
        ),
        BusinessRule(
            name="Negative Financial Values",
            status="WARNING",
            action="Flag and classify as valid credit/loss or source-system error",
            explanation=(
                "Negative financial values can be valid credits or loss-making jobs, but "
                "they should remain visible to business users."
            ),
            mask=lambda df: _negative_financial_mask(df),
        ),
    ]


def _va_outlier_mask(df: pd.DataFrame) -> pd.Series:
    va = _column(df, "VA Amount")
    valid = va.dropna()
    if valid.empty:
        return _false_mask(df)
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return _false_mask(df)
    return va.lt(q1 - 1.5 * iqr) | va.gt(q3 + 1.5 * iqr)


def _negative_financial_mask(df: pd.DataFrame) -> pd.Series:
    columns = [
        col
        for col in [
            "Sell Price",
            "Purchases",
            "VA Amount",
            "Rebate",
            "Handling",
            "Labour",
            "Paper",
            "AmtInv",
        ]
        if col in df.columns
    ]
    if not columns:
        return _false_mask(df)
    return df[columns].apply(pd.to_numeric, errors="coerce").lt(0).any(axis=1)


def apply_business_rules(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all business rules and return row flags plus a report.

    Rows are not removed here. The Streamlit sidebar controls whether flagged rows
    are included in commercial calculations, keeping anomalies visible to users.
    """

    rules = build_business_rules()
    flagged = df.copy()
    reason_lists = pd.Series([[] for _ in range(len(flagged))], index=flagged.index)
    highest_status = pd.Series("PASS", index=flagged.index, dtype="object")
    report_rows: list[dict[str, object]] = []
    status_rank = {"PASS": 0, "WARNING": 1, "FAIL": 2}

    for rule in rules:
        try:
            mask = rule.mask(flagged).fillna(False).astype(bool)
        except Exception:
            mask = _false_mask(flagged)
        column_name = f"Rule: {rule.name}"
        flagged[column_name] = mask
        for idx in flagged.index[mask]:
            reason_lists.at[idx] = [*reason_lists.at[idx], rule.name]
        highest_status = highest_status.mask(
            mask & (status_rank[rule.status] > highest_status.map(status_rank)),
            rule.status,
        )
        count = int(mask.sum())
        report_rows.append(
            {
                "Anomaly": rule.name,
                "Severity": rule.status,
                "Row Count": count,
                "Percentage of Dataset": count / len(flagged) if len(flagged) else 0.0,
                "Recommended Action": rule.action,
                "Business Explanation": rule.explanation,
            }
        )

    flagged["Quality Status"] = highest_status
    flagged["Reason"] = reason_lists.apply(lambda values: "; ".join(values) if values else "PASS")
    flagged["Is Flagged"] = flagged["Quality Status"].ne("PASS")
    report = pd.DataFrame(report_rows).sort_values(
        ["Severity", "Row Count"],
        ascending=[True, False],
    )
    return flagged, report


def filter_flagged_rows(df: pd.DataFrame, include_flagged: bool) -> pd.DataFrame:
    """Return either all rows or only rows that pass the business rule engine."""

    if include_flagged or "Is Flagged" not in df.columns:
        return df.copy()
    return df.loc[~df["Is Flagged"]].copy()

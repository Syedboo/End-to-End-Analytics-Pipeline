"""Configuration objects for leakage-safe churn analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChurnConfig:
    """Business and modelling settings for the churn pipeline.

    Defaults are intentionally conservative for commercial print data: customers
    may order irregularly, so the churn label is only created when the customer
    was expected to reorder during the prediction window.
    """

    observation_window_days: int = 365
    prediction_window_days: int = 60
    gap_days: int = 0
    snapshot_frequency: str = "MS"
    min_history_days: int = 90
    min_distinct_order_days: int = 3
    min_reorder_window_days: int = 30
    recent_value_window_days: int = 180
    random_state: int = 42
    contact_capacity: int = 10
    contact_cost: float = 25.0
    intervention_success_rate: float = 0.20
    retained_value_rate: float = 0.35
    model_version: str = "churn_v1_2026_08_01"
    top_k_rates: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20)
    activity_statuses: tuple[str, ...] = (
        "z-Closed",
        "Checked",
        "Shipped",
        "Digital Finished",
        "x-Inv AutoDirect",
    )
    incomplete_status_terms: tuple[str, ...] = (
        "Scheduled",
        "Hold",
        "Waiting",
        "HELD",
    )
    categorical_features: tuple[str, ...] = (
        "Rep",
        "Region",
        "Industry",
        "Currency",
        "Primary Product Type",
        "Primary Work Type",
        "Customer Value Band",
        "Frequency Segment",
    )
    numeric_features: tuple[str, ...] = (
        "Customer Tenure Days",
        "Days Since Last Order",
        "Distinct Order Days",
        "Order Count Lifetime",
        "Order Count 30d",
        "Order Count 60d",
        "Order Count 90d",
        "Order Count 180d",
        "Order Count 365d",
        "Revenue Lifetime",
        "Revenue 30d",
        "Revenue 60d",
        "Revenue 90d",
        "Revenue 180d",
        "Revenue 365d",
        "VA Lifetime",
        "VA 90d",
        "VA 180d",
        "Average Order Value",
        "Median Order Value",
        "Max Order Value",
        "Average VA Margin",
        "Average Reorder Days",
        "Median Reorder Days",
        "Reorder P75 Days",
        "Reorder P90 Days",
        "Max Reorder Days",
        "Std Reorder Days",
        "Recent Reorder Interval Days",
        "Expected Reorder Window Days",
        "Reorder Alert Threshold Days",
        "High Risk Threshold Days",
        "Likely Churn Threshold Days",
        "Days Beyond Max Reorder Gap",
        "Days Beyond Likely Churn Threshold",
        "Inactivity To Cadence Ratio",
        "Missed Expected Cycles",
        "Days Overdue",
        "Revenue 90d Trend",
        "Order 90d Trend",
        "VA 90d Trend",
        "Margin 90d Change",
        "Product Diversity",
        "Work Type Diversity",
        "Dominant Product Share",
        "Anomalous Job Count",
        "Anomalous Job Pct",
        "Negative Margin Pct",
        "Incomplete Job Pct",
        "Estimated Sell Price Pct",
    )


def load_churn_config(path: str | Path | None = None) -> ChurnConfig:
    """Load config when available; otherwise return conservative defaults.

    The project keeps a YAML file for business readability. To avoid adding a
    runtime dependency, this loader currently returns defaults unless PyYAML is
    installed. Unknown keys are ignored.
    """

    if path is None:
        return ChurnConfig()
    path = Path(path)
    if not path.exists():
        return ChurnConfig()
    try:
        import yaml  # type: ignore
    except Exception:
        return ChurnConfig()
    with path.open("r", encoding="utf-8") as handle:
        loaded: dict[str, Any] = yaml.safe_load(handle) or {}
    valid = {field.name for field in ChurnConfig.__dataclass_fields__.values()}
    values = {key: value for key, value in loaded.items() if key in valid}
    return ChurnConfig(**values)

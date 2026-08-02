"""Leakage-safe customer snapshot feature engineering for churn analysis."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import ChurnConfig


FINANCIAL_COLUMNS = (
    "Sell Price",
    "Estimated Sell Price",
    "VA Amount",
    "Purchases",
    "Labour",
    "Paper",
    "Handling",
    "Quantity",
    "Press hrs",
    "Impressions",
)


def safe_divide(numerator: float | pd.Series, denominator: float | pd.Series) -> float | pd.Series:
    """Divide while returning zero for missing, infinite, or zero denominators."""

    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(np.asarray(numerator, dtype="float64"), dtype="float64"),
        where=np.asarray(denominator, dtype="float64") != 0,
    )
    if np.isscalar(numerator) and np.isscalar(denominator):
        return float(result)
    return pd.Series(result, index=getattr(numerator, "index", None)).replace([np.inf, -np.inf], 0).fillna(0)


def _normalise_id(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _num(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(0.0, index=df.index)
    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def _mode(series: pd.Series) -> str:
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return "Unknown"
    return str(values.mode().iloc[0])


def _sum_between(df: pd.DataFrame, snapshot_date: pd.Timestamp, days: int, column: str) -> float:
    start = snapshot_date - pd.Timedelta(days=days)
    mask = (df["_Order Date"] > start) & (df["_Order Date"] <= snapshot_date)
    return float(_num(df.loc[mask], column).sum())


def _count_between(df: pd.DataFrame, snapshot_date: pd.Timestamp, days: int) -> int:
    start = snapshot_date - pd.Timedelta(days=days)
    mask = (df["_Order Date"] > start) & (df["_Order Date"] <= snapshot_date)
    return int(df.loc[mask, "_Order Date"].nunique())


def _pct_change(current: float, previous: float) -> float:
    if pd.isna(previous) or abs(previous) < 1e-9:
        return 0.0 if abs(current) < 1e-9 else 1.0
    return float((current - previous) / abs(previous))


def _share(df: pd.DataFrame, condition: pd.Series) -> float:
    if df.empty:
        return 0.0
    return float(condition.fillna(False).mean())


def _band_by_quantile(value: float, quantiles: dict[str, float]) -> str:
    if value >= quantiles.get("q90", np.inf):
        return "Strategic"
    if value >= quantiles.get("q70", np.inf):
        return "High"
    if value >= quantiles.get("q35", np.inf):
        return "Medium"
    return "Low"


def prepare_churn_transactions(df: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    """Return transaction data with typed fields and row-level churn flags.

    The function does not delete anomalies. The model uses these flags so the
    business can separate real customer behaviour from questionable input data.
    """

    tx = df.copy()
    if "CustomerID" not in tx.columns:
        tx["CustomerID"] = "UNKNOWN"
    if "Customer Name" not in tx.columns:
        tx["Customer Name"] = tx["CustomerID"]
    tx["CustomerID"] = tx["CustomerID"].map(_normalise_id)
    tx["Customer Name"] = tx["Customer Name"].map(_normalise_id)

    tx["SalesIn"] = pd.to_datetime(tx.get("SalesIn"), errors="coerce")
    tx["_Order Date"] = tx["SalesIn"].dt.normalize()

    for column in FINANCIAL_COLUMNS:
        if column in tx.columns:
            tx[column] = pd.to_numeric(tx[column], errors="coerce")

    if "Estimated Sell Price" in tx.columns:
        tx["Churn Revenue"] = tx["Estimated Sell Price"].where(
            tx["Estimated Sell Price"].notna(), _num(tx, "Sell Price")
        )
    else:
        tx["Churn Revenue"] = _num(tx, "Sell Price")
    tx["Churn Revenue"] = pd.to_numeric(tx["Churn Revenue"], errors="coerce").fillna(0.0)
    source_sell_price = pd.to_numeric(tx.get("Sell Price"), errors="coerce")
    source_purchases = _num(tx, "Purchases")
    tx["Is Source Missing Sell Price"] = source_sell_price.isna()
    tx["Is Source Non Positive Sell Price"] = source_sell_price.fillna(0.0) <= 0
    tx["Is Source Sell Price Below Purchase"] = source_sell_price.fillna(0.0) < source_purchases

    job_status = tx.get("Job Status", pd.Series("", index=tx.index)).astype(str)
    tx["Is Activity Status"] = job_status.isin(config.activity_statuses)
    terms = tuple(term.lower() for term in config.incomplete_status_terms)
    tx["Is Incomplete Status"] = job_status.str.lower().apply(
        lambda value: any(term.lower() in value for term in terms)
    )
    tx["Is Non Positive Revenue"] = tx["Churn Revenue"].fillna(0.0) <= 0
    tx["Is Negative Margin"] = _num(tx, "VA Amount") < 0
    tx["Is Sell Price Below Purchase"] = tx["Is Source Sell Price Below Purchase"] | (tx["Churn Revenue"] < _num(tx, "Purchases"))
    tx["Is Estimated Price"] = tx.get(
        "Sell Price Was Imputed",
        pd.Series(False, index=tx.index),
    ).fillna(False).astype(bool)
    tx["Is Anomalous Job"] = (
        tx["Is Source Missing Sell Price"]
        | tx["Is Source Non Positive Sell Price"]
        | tx["Is Negative Margin"]
        | tx["Is Sell Price Below Purchase"]
        | tx["Is Incomplete Status"]
        | tx["SalesIn"].isna()
    )
    return tx


def generate_snapshot_dates(
    transactions: pd.DataFrame,
    config: ChurnConfig,
    latest_only: bool = False,
    reference_date: pd.Timestamp | str | None = None,
) -> list[pd.Timestamp]:
    """Create monthly snapshot dates where a future label can be observed."""

    valid_dates = transactions["_Order Date"].dropna()
    if valid_dates.empty:
        return []
    if reference_date is not None:
        return [pd.to_datetime(reference_date).normalize()]
    max_date = valid_dates.max().normalize()
    if latest_only:
        return [max_date]
    start = valid_dates.min().normalize() + pd.Timedelta(days=config.observation_window_days)
    end = max_date - pd.Timedelta(days=config.prediction_window_days + config.gap_days)
    if start > end:
        return []
    start = start.to_period("M").to_timestamp()
    dates = pd.date_range(start=start, end=end, freq=config.snapshot_frequency)
    return [date.normalize() for date in dates]


def _build_one_customer_snapshot(
    customer_id: str,
    customer_history: pd.DataFrame,
    observation: pd.DataFrame,
    future: pd.DataFrame,
    snapshot_date: pd.Timestamp,
    config: ChurnConfig,
    global_reorder_days: float,
) -> dict[str, object]:
    """Build one leakage-safe customer feature row for one snapshot date."""

    history = customer_history.sort_values("_Order Date")
    order_dates = history["_Order Date"].dropna().drop_duplicates().sort_values()
    if order_dates.empty:
        return {}

    first_order = order_dates.iloc[0]
    last_order = order_dates.iloc[-1]
    distinct_order_days = int(order_dates.nunique())
    gaps = order_dates.diff().dt.days.dropna()

    average_reorder = float(gaps.mean()) if not gaps.empty else np.nan
    median_reorder = float(gaps.median()) if not gaps.empty else np.nan
    p75_reorder = float(gaps.quantile(0.75)) if not gaps.empty else np.nan
    p90_reorder = float(gaps.quantile(0.90)) if not gaps.empty else np.nan
    max_reorder = float(gaps.max()) if not gaps.empty else np.nan
    std_reorder = float(gaps.std(ddof=0)) if len(gaps) > 1 else 0.0
    recent_interval = float(gaps.iloc[-1]) if not gaps.empty else np.nan
    cadence = median_reorder if pd.notna(median_reorder) else average_reorder
    if pd.isna(cadence):
        cadence = global_reorder_days
    expected_window = max(float(cadence), float(config.min_reorder_window_days))
    reorder_alert_threshold = max(
        p75_reorder if pd.notna(p75_reorder) else expected_window,
        expected_window * 1.25,
    )
    high_risk_threshold = max(
        p90_reorder if pd.notna(p90_reorder) else expected_window,
        expected_window * 2.0,
    )
    max_gap_grace_days = max(7.0, expected_window * 0.5)
    max_gap_threshold = (max_reorder + max_gap_grace_days) if pd.notna(max_reorder) else expected_window * 3.0
    likely_churn_threshold = min(max_gap_threshold, expected_window * 3.0)
    expected_next = last_order + pd.Timedelta(days=expected_window)

    days_since_last = max(0, int((snapshot_date - last_order).days))
    days_overdue = max(0, int((snapshot_date - expected_next).days))
    days_beyond_max_gap = max(0.0, days_since_last - max_reorder) if pd.notna(max_reorder) else 0.0
    days_beyond_likely_churn = max(0.0, days_since_last - likely_churn_threshold)
    inactivity_ratio = float(days_since_last / expected_window) if expected_window else 0.0
    missed_cycles = max(0.0, (days_since_last - expected_window) / expected_window) if expected_window else 0.0

    lifetime_revenue = float(_num(history, "Churn Revenue").sum())
    lifetime_va = float(_num(history, "VA Amount").sum())
    lifetime_orders = int(history["_Order Date"].nunique())

    revenue_30 = _sum_between(history, snapshot_date, 30, "Churn Revenue")
    revenue_60 = _sum_between(history, snapshot_date, 60, "Churn Revenue")
    revenue_90 = _sum_between(history, snapshot_date, 90, "Churn Revenue")
    revenue_180 = _sum_between(history, snapshot_date, 180, "Churn Revenue")
    revenue_365 = _sum_between(history, snapshot_date, 365, "Churn Revenue")
    va_90 = _sum_between(history, snapshot_date, 90, "VA Amount")
    va_180 = _sum_between(history, snapshot_date, 180, "VA Amount")
    va_365 = _sum_between(history, snapshot_date, 365, "VA Amount")

    order_30 = _count_between(history, snapshot_date, 30)
    order_60 = _count_between(history, snapshot_date, 60)
    order_90 = _count_between(history, snapshot_date, 90)
    order_180 = _count_between(history, snapshot_date, 180)
    order_365 = _count_between(history, snapshot_date, 365)

    prev_90_start = snapshot_date - pd.Timedelta(days=180)
    prev_90_end = snapshot_date - pd.Timedelta(days=90)
    prev_90 = history[(history["_Order Date"] > prev_90_start) & (history["_Order Date"] <= prev_90_end)]
    prev_90_revenue = float(_num(prev_90, "Churn Revenue").sum())
    prev_90_va = float(_num(prev_90, "VA Amount").sum())
    prev_90_orders = int(prev_90["_Order Date"].nunique()) if not prev_90.empty else 0
    recent_margin = safe_divide(va_90, revenue_90)
    previous_margin = safe_divide(prev_90_va, prev_90_revenue)

    product_counts = history.get("Product Type", pd.Series(index=history.index, dtype=object)).fillna("Unknown")
    product_diversity = int(product_counts.nunique()) if not product_counts.empty else 0
    dominant_product_share = float(product_counts.value_counts(normalize=True).iloc[0]) if not product_counts.empty else 0.0
    work_values = history.get("Work Type", pd.Series(index=history.index, dtype=object)).fillna("Unknown")
    work_diversity = int(work_values.nunique()) if not work_values.empty else 0

    tenure_days = max(0, int((snapshot_date - first_order).days))
    eligible = (
        tenure_days >= config.min_history_days
        and distinct_order_days >= config.min_distinct_order_days
        and expected_next <= snapshot_date + pd.Timedelta(days=config.prediction_window_days + config.gap_days)
    )
    churn_label = np.nan
    if eligible:
        churn_label = 0 if not future.empty else 0
        churn_label = 1 if future.empty else 0

    cold_start = distinct_order_days < config.min_distinct_order_days
    already_dormant = days_since_last > expected_window * 3

    return {
        "CustomerID": customer_id,
        "Customer Name": _mode(history["Customer Name"]),
        "Snapshot Date": snapshot_date,
        "First Order Date": first_order,
        "Last Order Date": last_order,
        "Expected Next Order Date": expected_next,
        "Prediction Window End": snapshot_date + pd.Timedelta(days=config.prediction_window_days + config.gap_days),
        "Churn Label": churn_label,
        "Eligible For Training": bool(eligible),
        "Cold Start Flag": bool(cold_start),
        "Already Dormant Flag": bool(already_dormant),
        "Customer Tenure Days": tenure_days,
        "Days Since Last Order": days_since_last,
        "Distinct Order Days": distinct_order_days,
        "Order Count Lifetime": lifetime_orders,
        "Order Count 30d": order_30,
        "Order Count 60d": order_60,
        "Order Count 90d": order_90,
        "Order Count 180d": order_180,
        "Order Count 365d": order_365,
        "Revenue Lifetime": lifetime_revenue,
        "Revenue 30d": revenue_30,
        "Revenue 60d": revenue_60,
        "Revenue 90d": revenue_90,
        "Revenue 180d": revenue_180,
        "Revenue 365d": revenue_365,
        "VA Lifetime": lifetime_va,
        "VA 90d": va_90,
        "VA 180d": va_180,
        "VA 365d": va_365,
        "Average Order Value": float(_num(history, "Churn Revenue").mean()),
        "Median Order Value": float(_num(history, "Churn Revenue").median()),
        "Max Order Value": float(_num(history, "Churn Revenue").max()),
        "Min Order Value": float(_num(history, "Churn Revenue").min()),
        "Average VA Margin": float(safe_divide(lifetime_va, lifetime_revenue)),
        "Average Reorder Days": average_reorder,
        "Median Reorder Days": median_reorder,
        "Reorder P75 Days": p75_reorder,
        "Reorder P90 Days": p90_reorder,
        "Max Reorder Days": max_reorder,
        "Std Reorder Days": std_reorder,
        "Recent Reorder Interval Days": recent_interval,
        "Expected Reorder Window Days": expected_window,
        "Reorder Alert Threshold Days": reorder_alert_threshold,
        "High Risk Threshold Days": high_risk_threshold,
        "Likely Churn Threshold Days": likely_churn_threshold,
        "Days Beyond Max Reorder Gap": days_beyond_max_gap,
        "Days Beyond Likely Churn Threshold": days_beyond_likely_churn,
        "Inactivity To Cadence Ratio": inactivity_ratio,
        "Missed Expected Cycles": missed_cycles,
        "Days Overdue": days_overdue,
        "Revenue 90d Trend": _pct_change(revenue_90, prev_90_revenue),
        "Order 90d Trend": _pct_change(float(order_90), float(prev_90_orders)),
        "VA 90d Trend": _pct_change(va_90, prev_90_va),
        "Margin 90d Change": float(recent_margin - previous_margin),
        "Product Diversity": product_diversity,
        "Work Type Diversity": work_diversity,
        "Dominant Product Share": dominant_product_share,
        "Primary Product Type": _mode(history.get("Product Type", pd.Series(index=history.index))),
        "Primary Work Type": _mode(history.get("Work Type", pd.Series(index=history.index))),
        "Rep": _mode(history.get("Rep", pd.Series(index=history.index))),
        "Region": _mode(history.get("Region", pd.Series(index=history.index))),
        "Industry": _mode(history.get("Industry", pd.Series(index=history.index))),
        "Currency": _mode(history.get("Currency", pd.Series(index=history.index))),
        "Anomalous Job Count": int(history.get("Is Anomalous Job", pd.Series(False, index=history.index)).sum()),
        "Anomalous Job Pct": _share(history, history.get("Is Anomalous Job", pd.Series(False, index=history.index))),
        "Negative Margin Pct": _share(history, history.get("Is Negative Margin", pd.Series(False, index=history.index))),
        "Incomplete Job Pct": _share(history, history.get("Is Incomplete Status", pd.Series(False, index=history.index))),
        "Estimated Sell Price Pct": _share(history, history.get("Is Estimated Price", pd.Series(False, index=history.index))),
        "Future Order Count": int(future["_Order Date"].nunique()),
        "Churn Definition": (
            "No activity in prediction window after the customer was expected to reorder"
        ),
    }


def build_customer_snapshots(
    transactions: pd.DataFrame,
    config: ChurnConfig,
    snapshot_dates: Iterable[pd.Timestamp] | None = None,
    latest_only: bool = False,
    reference_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Create one or more leakage-safe customer snapshots.

    Every feature is calculated from rows on or before the snapshot date. The
    label, when requested, uses only the future prediction window.
    """

    tx = prepare_churn_transactions(transactions, config)
    tx = tx[tx["SalesIn"].notna()].copy()
    if tx.empty:
        return pd.DataFrame()

    dates = list(snapshot_dates or generate_snapshot_dates(tx, config, latest_only, reference_date))
    activity = tx[tx["Is Activity Status"]].copy()
    if activity.empty:
        activity = tx.copy()

    all_order_dates = activity[["CustomerID", "_Order Date"]].drop_duplicates().sort_values(["CustomerID", "_Order Date"])
    all_gaps = all_order_dates.groupby("CustomerID")["_Order Date"].diff().dt.days.dropna()
    global_reorder_days = float(all_gaps.median()) if not all_gaps.empty else float(config.min_reorder_window_days)
    global_reorder_days = max(global_reorder_days, float(config.min_reorder_window_days))

    rows: list[dict[str, object]] = []
    for snapshot_date in dates:
        snapshot_date = pd.to_datetime(snapshot_date).normalize()
        historical = activity[activity["_Order Date"] <= snapshot_date]
        if historical.empty:
            continue
        future_start = snapshot_date + pd.Timedelta(days=config.gap_days)
        future_end = future_start + pd.Timedelta(days=config.prediction_window_days)
        future_window = activity[(activity["_Order Date"] > future_start) & (activity["_Order Date"] <= future_end)]
        observation_start = snapshot_date - pd.Timedelta(days=config.observation_window_days)
        observation_window = historical[historical["_Order Date"] > observation_start]

        for customer_id, customer_history in historical.groupby("CustomerID"):
            observation = observation_window[observation_window["CustomerID"] == customer_id]
            if observation.empty:
                continue
            customer_future = future_window[future_window["CustomerID"] == customer_id]
            row = _build_one_customer_snapshot(
                customer_id=customer_id,
                customer_history=customer_history,
                observation=observation,
                future=customer_future,
                snapshot_date=snapshot_date,
                config=config,
                global_reorder_days=global_reorder_days,
            )
            if row:
                rows.append(row)

    snapshots = pd.DataFrame(rows)
    if snapshots.empty:
        return snapshots

    for snapshot_date, index in snapshots.groupby("Snapshot Date").groups.items():
        values = snapshots.loc[index, "Revenue Lifetime"].fillna(0)
        quantiles = {
            "q35": float(values.quantile(0.35)),
            "q70": float(values.quantile(0.70)),
            "q90": float(values.quantile(0.90)),
        }
        snapshots.loc[index, "Customer Value Band"] = values.apply(lambda value: _band_by_quantile(float(value), quantiles))

    cadence = snapshots["Expected Reorder Window Days"].fillna(config.min_reorder_window_days)
    snapshots["Frequency Segment"] = np.select(
        [cadence <= 45, cadence <= 90],
        ["High frequency", "Medium frequency"],
        default="Low frequency",
    )
    snapshots["Churn Label"] = pd.to_numeric(snapshots["Churn Label"], errors="coerce")
    return snapshots


def build_latest_customer_features(
    transactions: pd.DataFrame,
    config: ChurnConfig,
    reference_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Build the current customer snapshot used for operational inference."""

    prepared = prepare_churn_transactions(transactions, config)
    dates = generate_snapshot_dates(prepared, config, latest_only=True, reference_date=reference_date)
    return build_customer_snapshots(prepared, config, snapshot_dates=dates, latest_only=True)

"""End-to-end production churn pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ChurnConfig, load_churn_config
from .data_quality import audit_churn_data
from .features import build_customer_snapshots, build_latest_customer_features, prepare_churn_transactions
from .model import ChurnModelOutputs, fit_churn_models


@dataclass
class ChurnPipelineResult:
    """Outputs produced by the customer churn pipeline."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    threshold_table: pd.DataFrame
    feature_importance: pd.DataFrame
    segment_performance: pd.DataFrame
    lift_gain: pd.DataFrame
    calibration_table: pd.DataFrame
    data_quality_report: pd.DataFrame
    snapshots: pd.DataFrame
    dashboard_tables: dict[str, pd.DataFrame]
    reports: dict[str, str]
    metadata: dict[str, Any]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _save_table(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return str(path)


def _write_text(text: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path)


def _frame_markdown(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "No rows produced.\n"
    view = df.head(max_rows).copy()
    try:
        return view.to_markdown(index=False) + "\n"
    except Exception:
        return "```csv\n" + view.to_csv(index=False) + "```\n"


def _audit_markdown() -> str:
    return """# Current Churn Pipeline Audit

The source data has no explicit churn flag, so churn must be inferred from repeat-order behaviour. The previous implementation was useful as a dashboard heuristic, but it was not a production churn model.

| Category | File/line | Finding | Why it matters | Correction implemented |
|---|---:|---|---|---|
| Business-logic weakness | src/feature_engineering.py:103 | Churn was a single latest-date lifecycle heuristic, not a labelled prediction problem. | It cannot estimate out-of-time model performance or prove that risk scores predict future inactivity. | Added temporal customer snapshots with observation and prediction windows. |
| Data leakage risk | main.py:111 | Customer lifecycle features were created before sell-price imputation and business-rule anomaly flags. | Churn values and anomaly features did not use the corrected analytical dataset. | The new churn pipeline runs after imputation and business rules in main.py. |
| Evaluation weakness | src/modelling.py:94 | Existing VA model uses a random train/test split. | Random splits are unsuitable for churn because future customer behaviour can leak into training. | Churn model uses chronological train, validation, and test snapshots. |
| Business-logic weakness | src/feature_engineering.py:202 | Inactivity was treated uniformly across customers except for basic cadence. | Low-frequency but healthy accounts can be unfairly marked as churned. | Labels require the customer to be expected to reorder inside the future prediction window. |
| Maintainability issue | main.py:138 | A stray markdown heading exists inside Python indentation. | It is harmless after the syntax fix, but makes the pipeline harder to maintain. | The churn audit is now generated in a dedicated report file. |
| Operational risk | src/sell_price_imputation.py:156 and src/modelling.py:128 | Random forests used n_jobs=-1. | On this Windows environment parallel workers can fail with joblib pipe permission errors and can make Streamlit less responsive. | Patched to n_jobs=1 for stable local execution. |
| Nice-to-have improvement | appstreamlit.py customer analytics | Streamlit can show current heuristic tables, but did not expose production churn artefacts. | Business users need traceable risk, value, reasons, and actions. | The pipeline now exports dashboard-ready churn prediction tables. |

## Churn Definition Validated

A customer is treated as a repeat-order account only when it has enough historical order days and tenure to estimate a reorder cycle. A positive churn label is assigned at a snapshot date only when all of the following are true:

- The customer had at least the configured minimum history and distinct order days before the snapshot.
- Its expected next-order date fell inside the prediction horizon.
- It placed no qualifying activity-status order during the future prediction window.

This avoids labelling naturally infrequent customers as churned simply because they have long gaps between orders.
"""


def _data_quality_markdown(report: pd.DataFrame, metadata: dict[str, Any]) -> str:
    return f"""# Churn Data Quality Report

Rows are not silently removed. The exceptions below identify data-quality errors, business-rule exceptions, statistical anomalies, and valid but unusual commercial transactions that should be visible to users.

## Dataset Context

- Observation window: {metadata.get('observation_window_days')} days
- Prediction window: {metadata.get('prediction_window_days')} days
- Gap period: {metadata.get('gap_days')} days
- Churn label: behavioural, inferred from customer inactivity after expected reorder

## Exceptions Summary

{_frame_markdown(report, max_rows=40)}
"""


def _evaluation_markdown(outputs: ChurnModelOutputs) -> str:
    meta = outputs.metadata
    return f"""# Customer Churn Model Evaluation Report

## Executive Summary

The pipeline builds monthly customer snapshots, fits only on historical snapshots, validates on later snapshots, and scores the latest available customer state. The final ranking combines calibrated churn risk with customer value, urgency, and cadence thresholds including P75, P90, and max historical reorder gaps. The visible Priority Score is normalized to 0-100; Priority Score Raw preserves the auditable formula.

## Churn Definition

No explicit churn label exists in the source data. Churn is inferred as no qualifying order in the future prediction window after the customer was expected to reorder based on its own historical cadence.

## Time Windows

- Observation window: {meta.get('observation_window_days')} days
- Gap period: {meta.get('gap_days')} days
- Prediction window: {meta.get('prediction_window_days')} days
- Train dates: {meta.get('train_start')} to {meta.get('train_end')}
- Validation dates: {meta.get('validation_start')} to {meta.get('validation_end')}
- Test dates: {meta.get('test_start')} to {meta.get('test_end')}

## Model Selection

- Selected model: {meta.get('selected_model')}
- Calibration: {meta.get('calibration')}
- Recommended threshold: {meta.get('recommended_threshold')}
- Eligible labelled rows: {meta.get('eligible_rows')}
- Churn rate: {meta.get('churn_rate')}

## Metrics

{_frame_markdown(outputs.metrics, max_rows=30)}

## Threshold Economics

{_frame_markdown(outputs.threshold_table, max_rows=30)}

## Feature Importance

{_frame_markdown(outputs.feature_importance, max_rows=20)}

## Segment Performance

{_frame_markdown(outputs.segment_performance, max_rows=30)}

## Lift and Gain

{_frame_markdown(outputs.lift_gain, max_rows=30)}

## Calibration

{_frame_markdown(outputs.calibration_table, max_rows=15)}

## Limitations

- The churn label is inferred from behaviour, not confirmed lost-account outcomes.
- Source values contain multiple currencies and are treated as reported unless the wider profitability pipeline supplies conversions.
- The prediction target is repeat-order inactivity, not necessarily permanent customer loss.
- Survival analysis was assessed conceptually; the current implementation approximates time-to-next-order through cadence, expected next-order date, and days overdue without adding heavy survival dependencies.
"""


def _dashboard_tables(predictions: pd.DataFrame, outputs: ChurnModelOutputs) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {"churn_dashboard_customers": predictions.copy()}
    if not predictions.empty:
        rep_col = "Assigned sales representative"
        if rep_col in predictions.columns:
            tables["churn_risk_by_rep"] = (
                predictions.groupby(rep_col, dropna=False)
                .agg(
                    Customers=("Customer ID", "nunique"),
                    Average_Risk=("Calibrated Churn Probability", "mean"),
                    Revenue_At_Risk=("Revenue at Risk", "sum"),
                    Priority_Score=("Priority Score", "sum"),
                )
                .reset_index()
                .sort_values("Priority_Score", ascending=False)
            )
        driver_rows: list[dict[str, object]] = []
        for reason_col in ["Churn Reason 1", "Churn Reason 2", "Churn Reason 3"]:
            if reason_col not in predictions.columns:
                continue
            for reason, group in predictions[predictions[reason_col].astype(str).str.strip() != ""].groupby(reason_col):
                driver_rows.append(
                    {
                        "Churn Driver": reason,
                        "Customers": int(group["Customer ID"].nunique()),
                        "Revenue at Risk": float(group["Revenue at Risk"].sum()),
                        "Average Risk": float(group["Calibrated Churn Probability"].mean()),
                    }
                )
        tables["churn_drivers"] = pd.DataFrame(driver_rows).sort_values(
            ["Customers", "Revenue at Risk"], ascending=[False, False]
        ) if driver_rows else pd.DataFrame()
    tables["churn_risk_by_segment"] = outputs.segment_performance.copy()
    tables["churn_threshold_table"] = outputs.threshold_table.copy()
    tables["churn_model_metrics"] = outputs.metrics.copy()
    tables["churn_lift_gain"] = outputs.lift_gain.copy()
    tables["churn_calibration"] = outputs.calibration_table.copy()
    tables["churn_feature_importance"] = outputs.feature_importance.copy()
    return tables


def run_churn_pipeline(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    config: ChurnConfig | None = None,
    save_outputs: bool = True,
) -> ChurnPipelineResult:
    """Run the complete churn pipeline and optionally save artefacts."""

    if config is None:
        config = load_churn_config(_repo_root() / "config" / "churn_config.yaml")
    base_output = Path(output_dir) if output_dir is not None else _repo_root() / "outputs"
    reports_dir = base_output / "reports"
    tables_dir = base_output / "tables"
    predictions_dir = base_output / "predictions"
    exceptions_dir = base_output / "exceptions"

    transactions = prepare_churn_transactions(df, config)
    data_quality = audit_churn_data(transactions, config)
    snapshots = build_customer_snapshots(transactions, config)
    latest_features = build_latest_customer_features(transactions, config)
    model_outputs = fit_churn_models(snapshots, latest_features, config)
    predictions = model_outputs.predictions
    dashboard = _dashboard_tables(predictions, model_outputs)

    audit_md = _audit_markdown()
    evaluation_md = _evaluation_markdown(model_outputs)
    data_quality_md = _data_quality_markdown(data_quality, model_outputs.metadata)

    reports: dict[str, str] = {}
    if save_outputs:
        reports["audit"] = _write_text(audit_md, reports_dir / "churn_pipeline_audit.md")
        reports["evaluation"] = _write_text(evaluation_md, reports_dir / "churn_model_evaluation.md")
        reports["data_quality"] = _write_text(data_quality_md, reports_dir / "churn_data_quality_report.md")
        _save_table(predictions, predictions_dir / "customer_churn_predictions.csv")
        _save_table(data_quality, exceptions_dir / "churn_data_quality_exceptions.csv")
        _save_table(snapshots, tables_dir / "churn_customer_snapshots.csv")
        for name, table in dashboard.items():
            _save_table(table, tables_dir / f"{name}.csv")

    metadata = dict(model_outputs.metadata)
    metadata.update(
        {
            "customer_churn_predictions": str(predictions_dir / "customer_churn_predictions.csv") if save_outputs else None,
            "churn_model_evaluation": reports.get("evaluation"),
            "churn_pipeline_audit": reports.get("audit"),
            "churn_data_quality_report": reports.get("data_quality"),
            "data_quality_exceptions": str(exceptions_dir / "churn_data_quality_exceptions.csv") if save_outputs else None,
        }
    )

    return ChurnPipelineResult(
        predictions=predictions,
        metrics=model_outputs.metrics,
        threshold_table=model_outputs.threshold_table,
        feature_importance=model_outputs.feature_importance,
        segment_performance=model_outputs.segment_performance,
        lift_gain=model_outputs.lift_gain,
        calibration_table=model_outputs.calibration_table,
        data_quality_report=data_quality,
        snapshots=snapshots,
        dashboard_tables=dashboard,
        reports={"audit": audit_md, "evaluation": evaluation_md, "data_quality": data_quality_md, **reports},
        metadata=metadata,
    )

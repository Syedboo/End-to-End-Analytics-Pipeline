"""Leakage-safe churn modelling, calibration, prioritisation, and explanations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ChurnConfig


@dataclass
class ChurnModelOutputs:
    """Container for customer churn model artefacts."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame
    threshold_table: pd.DataFrame
    feature_importance: pd.DataFrame
    segment_performance: pd.DataFrame
    lift_gain: pd.DataFrame
    calibration_table: pd.DataFrame
    selected_model: str
    recommended_threshold: float
    metadata: dict[str, Any]


def _available(columns: tuple[str, ...], df: pd.DataFrame) -> list[str]:
    return [column for column in columns if column in df.columns]


def _finite_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(values.astype(float), nan=0.0, posinf=1.0, neginf=0.0), 0.001, 0.999)


def _rule_probability(df: pd.DataFrame) -> np.ndarray:
    """Interpretable fallback score for small or one-class training sets."""

    ratio = pd.to_numeric(df.get("Inactivity To Cadence Ratio", 0), errors="coerce").fillna(0)
    overdue = pd.to_numeric(df.get("Days Overdue", 0), errors="coerce").fillna(0)
    rev_trend = pd.to_numeric(df.get("Revenue 90d Trend", 0), errors="coerce").fillna(0)
    order_trend = pd.to_numeric(df.get("Order 90d Trend", 0), errors="coerce").fillna(0)
    missed = pd.to_numeric(df.get("Missed Expected Cycles", 0), errors="coerce").fillna(0)
    beyond_max = pd.to_numeric(df.get("Days Beyond Max Reorder Gap", 0), errors="coerce").fillna(0)
    beyond_likely = pd.to_numeric(df.get("Days Beyond Likely Churn Threshold", 0), errors="coerce").fillna(0)

    score = 0.08
    score += np.where(ratio > 1.0, 0.18, 0.0)
    score += np.where(ratio > 1.25, 0.18, 0.0)
    score += np.where(ratio > 2.0, 0.20, 0.0)
    score += np.where(overdue > 0, 0.12, 0.0)
    score += np.where(beyond_max > 0, 0.16, 0.0)
    score += np.where(beyond_likely > 0, 0.12, 0.0)
    score += np.where(missed >= 1, 0.10, 0.0)
    score += np.where(rev_trend <= -0.30, 0.08, 0.0)
    score += np.where(order_trend <= -0.30, 0.06, 0.0)
    return _finite_prob(np.asarray(score, dtype=float))


def _temporal_split(labeled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = sorted(pd.to_datetime(labeled["Snapshot Date"]).dropna().unique())
    if len(dates) < 3:
        return labeled.copy(), labeled.iloc[0:0].copy(), labeled.iloc[0:0].copy()
    train_end = max(1, int(len(dates) * 0.60))
    val_end = max(train_end + 1, int(len(dates) * 0.80))
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]
    train = labeled[labeled["Snapshot Date"].isin(train_dates)].copy()
    validation = labeled[labeled["Snapshot Date"].isin(val_dates)].copy()
    test = labeled[labeled["Snapshot Date"].isin(test_dates)].copy()
    return train, validation, test


def _classification_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    value: np.ndarray,
    model_name: str,
    threshold: float,
    top_rates: tuple[float, ...],
) -> dict[str, float | str]:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    prob = _finite_prob(np.asarray(prob, dtype=float))
    predicted = (prob >= threshold).astype(int)
    labels_present = np.unique(y_true)
    row: dict[str, float | str] = {"Model": model_name}
    if len(labels_present) > 1:
        row["ROC AUC"] = float(roc_auc_score(y_true, prob))
        row["PR AUC"] = float(average_precision_score(y_true, prob))
        row["Log Loss"] = float(log_loss(y_true, prob, labels=[0, 1]))
    else:
        row["ROC AUC"] = np.nan
        row["PR AUC"] = np.nan
        row["Log Loss"] = np.nan
    row["Brier Score"] = float(brier_score_loss(y_true, prob))
    row["Recall"] = float(recall_score(y_true, predicted, zero_division=0))
    row["Precision"] = float(precision_score(y_true, predicted, zero_division=0))
    row["F1"] = float(f1_score(y_true, predicted, zero_division=0))
    row["Balanced Accuracy"] = float(balanced_accuracy_score(y_true, predicted))
    cm = confusion_matrix(y_true, predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    row["Specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    row["Threshold"] = threshold
    row["Sample Size"] = int(len(y_true))
    row["Churn Rate"] = float(y_true.mean()) if len(y_true) else 0.0

    value = np.nan_to_num(np.asarray(value, dtype=float), nan=0.0)
    total_churners = max(float(y_true.sum()), 1.0)
    churn_value_total = max(float(value[y_true == 1].sum()), 1.0)
    order = np.argsort(-prob)
    for rate in top_rates:
        k = max(1, int(np.ceil(len(y_true) * rate))) if len(y_true) else 0
        selected = order[:k]
        row[f"Precision Top {int(rate * 100)}%"] = float(y_true[selected].mean()) if k else 0.0
        row[f"Recall Top {int(rate * 100)}%"] = float(y_true[selected].sum() / total_churners) if k else 0.0
        row[f"Revenue Weighted Recall Top {int(rate * 100)}%"] = (
            float(value[selected][y_true[selected] == 1].sum() / churn_value_total) if k else 0.0
        )
    return row


def _lift_gain_table(y_true: np.ndarray, prob: np.ndarray, value: np.ndarray, model_name: str) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    prob = _finite_prob(np.asarray(prob, dtype=float))
    value = np.nan_to_num(np.asarray(value, dtype=float), nan=0.0)
    if len(y_true) == 0:
        return pd.DataFrame()
    ranked = pd.DataFrame({"y": y_true, "prob": prob, "value": value}).sort_values("prob", ascending=False)
    total_churners = max(float(ranked["y"].sum()), 1.0)
    base_rate = max(float(ranked["y"].mean()), 1e-9)
    total_value = max(float(ranked.loc[ranked["y"] == 1, "value"].sum()), 1.0)
    rows = []
    for decile in range(1, 11):
        end = max(1, int(np.ceil(len(ranked) * decile / 10)))
        selected = ranked.iloc[:end]
        capture = float(selected["y"].sum() / total_churners)
        precision = float(selected["y"].mean())
        rows.append(
            {
                "Model": model_name,
                "Population Share": decile / 10,
                "Customers Selected": int(end),
                "Churn Capture": capture,
                "Precision": precision,
                "Lift": float(precision / base_rate),
                "Revenue Weighted Capture": float(selected.loc[selected["y"] == 1, "value"].sum() / total_value),
            }
        )
    return pd.DataFrame(rows)


def _threshold_table(
    scored: pd.DataFrame,
    config: ChurnConfig,
    threshold_values: list[float] | None = None,
) -> pd.DataFrame:
    if scored.empty:
        return pd.DataFrame()
    if threshold_values is None:
        threshold_values = [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    rows: list[dict[str, object]] = []
    y = pd.to_numeric(scored.get("Churn Label", np.nan), errors="coerce")
    for threshold in threshold_values:
        selected = scored[scored["Calibrated Churn Probability"] >= threshold]
        selected_y = y.loc[selected.index]
        true_churners = float(selected_y.sum()) if selected_y.notna().any() else np.nan
        total_churners = float(y.sum()) if y.notna().any() else np.nan
        precision = float(true_churners / len(selected)) if len(selected) and not pd.isna(true_churners) else np.nan
        recall = float(true_churners / total_churners) if total_churners and total_churners > 0 else np.nan
        expected_retained = float(
            (
                selected["Calibrated Churn Probability"]
                * selected["Expected Customer Value"]
                * config.intervention_success_rate
                * config.retained_value_rate
            ).sum()
        )
        campaign_cost = float(len(selected) * config.contact_cost)
        rows.append(
            {
                "Threshold": threshold,
                "Customers Flagged": int(len(selected)),
                "Precision": precision,
                "Recall": recall,
                "Expected Revenue At Risk": float(selected["Revenue at Risk"].sum()),
                "Expected Retained Value": expected_retained,
                "Estimated Campaign Cost": campaign_cost,
                "Expected Net Value": expected_retained - campaign_cost,
            }
        )

    capacity = max(1, int(config.contact_capacity))
    top_n = scored.sort_values("Priority Score", ascending=False).head(capacity)
    expected_retained = float(
        (
            top_n["Calibrated Churn Probability"]
            * top_n["Expected Customer Value"]
            * config.intervention_success_rate
            * config.retained_value_rate
        ).sum()
    )
    rows.append(
        {
            "Threshold": "Top capacity",
            "Customers Flagged": int(len(top_n)),
            "Precision": np.nan,
            "Recall": np.nan,
            "Expected Revenue At Risk": float(top_n["Revenue at Risk"].sum()),
            "Expected Retained Value": expected_retained,
            "Estimated Campaign Cost": float(len(top_n) * config.contact_cost),
            "Expected Net Value": expected_retained - float(len(top_n) * config.contact_cost),
        }
    )
    return pd.DataFrame(rows)


def _risk_category(row: pd.Series) -> str:
    prob = float(row.get("Calibrated Churn Probability", 0.0))
    band = str(row.get("Customer Value Band", "Low"))
    if prob >= 0.65 and band in {"Strategic", "High"}:
        return "Critical"
    if prob >= 0.65:
        return "Urgent"
    if prob >= 0.35 and band in {"Strategic", "High"}:
        return "Protect"
    if prob >= 0.20:
        return "Monitor"
    return "Low priority"


def _reason_and_action(row: pd.Series) -> tuple[str, str, str, str]:
    reasons: list[str] = []
    days_overdue = float(row.get("Days Overdue", 0) or 0)
    days_since = float(row.get("Days Since Last Order", 0) or 0)
    cadence = float(row.get("Expected Reorder Window Days", 0) or 0)
    ratio = float(row.get("Inactivity To Cadence Ratio", 0) or 0)
    revenue_trend = float(row.get("Revenue 90d Trend", 0) or 0)
    order_trend = float(row.get("Order 90d Trend", 0) or 0)
    margin_change = float(row.get("Margin 90d Change", 0) or 0)
    negative_margin_pct = float(row.get("Negative Margin Pct", 0) or 0)
    incomplete_pct = float(row.get("Incomplete Job Pct", 0) or 0)
    max_gap = float(row.get("Max Reorder Days", 0) or 0)
    p90_gap = float(row.get("Reorder P90 Days", 0) or 0)
    beyond_max = float(row.get("Days Beyond Max Reorder Gap", 0) or 0)
    beyond_likely = float(row.get("Days Beyond Likely Churn Threshold", 0) or 0)

    if beyond_max > 0 and max_gap > 0:
        reasons.append(
            f"No order for {int(days_since)} days; longest historical reorder gap was {int(round(max_gap))} days."
        )
    elif beyond_likely > 0:
        threshold = float(row.get("Likely Churn Threshold Days", 0) or 0)
        reasons.append(
            f"No order for {int(days_since)} days; above likely churn threshold of {int(round(threshold))} days."
        )
    elif p90_gap > 0 and days_since > p90_gap:
        reasons.append(
            f"No order for {int(days_since)} days; above the customer's 90th percentile reorder gap of {int(round(p90_gap))} days."
        )
    elif days_overdue > 0 and cadence > 0:
        reasons.append(
            f"No order for {int(days_since)} days versus a usual reorder cycle of {int(round(cadence))} days."
        )
    if ratio >= 1.25:
        reasons.append(f"Customer is {ratio:.1f} times beyond its normal reorder cadence.")
    if revenue_trend <= -0.30:
        reasons.append(f"Recent revenue is down {abs(revenue_trend):.0%} versus the previous 90 days.")
    if order_trend <= -0.30:
        reasons.append(f"Recent order frequency is down {abs(order_trend):.0%} versus the previous 90 days.")
    if margin_change <= -0.10:
        reasons.append("Recent margin has weakened materially versus the previous 90 days.")
    if incomplete_pct >= 0.10:
        reasons.append("A high share of recent jobs are open, held, or incomplete.")
    if negative_margin_pct >= 0.10:
        reasons.append("A high share of historical jobs were loss-making.")
    if bool(row.get("Cold Start Flag", False)):
        reasons.append("Customer has limited repeat-order history, so risk confidence is lower.")
    if not reasons:
        reasons.append("Risk is driven by a combination of recency, value, and account behaviour.")

    reasons = (reasons + ["", ""])[:3]
    action = "Monitor in the normal account-review cycle."
    first_reason = reasons[0].lower()
    if "no order" in first_reason or "beyond" in first_reason:
        action = "Ask the account owner to contact the customer about upcoming print demand and reorder timing."
    elif "revenue is down" in first_reason or "frequency is down" in first_reason:
        action = "Review recent account activity, competitor risk, and missed reorder opportunities."
    elif "margin" in first_reason or "loss-making" in first_reason:
        action = "Run a pricing and profitability review before the next quote."
    elif "open, held, or incomplete" in first_reason:
        action = "Escalate service or production blockers before a sales intervention."
    elif bool(row.get("Cold Start Flag", False)):
        action = "Use onboarding and next-order prompts rather than treating the customer as proven churn risk."
    return reasons[0], reasons[1], reasons[2], action


def _add_business_scoring(scored: pd.DataFrame, config: ChurnConfig) -> pd.DataFrame:
    result = scored.copy()
    value = np.maximum(
        pd.to_numeric(result.get("VA 180d", 0), errors="coerce").fillna(0),
        pd.to_numeric(result.get("VA Lifetime", 0), errors="coerce").fillna(0) * 0.25,
    )
    revenue_at_risk = np.maximum(
        pd.to_numeric(result.get("Revenue 180d", 0), errors="coerce").fillna(0),
        pd.to_numeric(result.get("Revenue Lifetime", 0), errors="coerce").fillna(0) * 0.25,
    )
    urgency = pd.to_numeric(result.get("Inactivity To Cadence Ratio", 0), errors="coerce").fillna(0)
    urgency = urgency.clip(lower=0.5, upper=2.0)
    margin_factor = 1 + pd.to_numeric(result.get("Average VA Margin", 0), errors="coerce").fillna(0).clip(lower=0, upper=1)
    result["Expected Customer Value"] = value.astype(float)
    result["Revenue at Risk"] = revenue_at_risk.astype(float)
    result["Urgency Factor"] = urgency.astype(float)
    result["Priority Score Raw"] = (
        result["Calibrated Churn Probability"]
        * result["Expected Customer Value"]
        * result["Urgency Factor"]
        * margin_factor
    ).fillna(0.0)
    max_priority = result["Priority Score Raw"].max()
    if pd.notna(max_priority) and max_priority > 0:
        result["Priority Score"] = result["Priority Score Raw"] / max_priority * 100
    else:
        result["Priority Score"] = 0.0
    result["Priority Rank"] = result["Priority Score Raw"].rank(method="dense", ascending=False).astype(int)
    result["Priority Score Explanation"] = (
        "0-100 score from calibrated churn probability, expected customer value, "
        "urgency and margin; rank is relative to the current scored customers."
    )
    result["Churn-risk category"] = result.apply(_risk_category, axis=1)
    reason_action = result.apply(_reason_and_action, axis=1, result_type="expand")
    reason_action.columns = [
        "Churn Reason 1",
        "Churn Reason 2",
        "Churn Reason 3",
        "Recommended retention action",
    ]
    result = pd.concat([result, reason_action], axis=1)
    result["Top three churn reasons"] = result[["Churn Reason 1", "Churn Reason 2", "Churn Reason 3"]].apply(
        lambda values: " | ".join([str(value) for value in values if str(value).strip()]),
        axis=1,
    )
    result["Model version"] = config.model_version
    return result


def _calibrate(raw_val: np.ndarray, y_val: np.ndarray, raw_new: np.ndarray) -> tuple[np.ndarray, str]:
    if len(np.unique(y_val)) < 2 or len(y_val) < 20:
        return _finite_prob(raw_new), "Not calibrated - validation set too small or one-class"
    try:
        from sklearn.linear_model import LogisticRegression

        calibrator = LogisticRegression(max_iter=1000)
        calibrator.fit(raw_val.reshape(-1, 1), y_val)
        return _finite_prob(calibrator.predict_proba(raw_new.reshape(-1, 1))[:, 1]), "Platt scaling"
    except Exception as exc:
        return _finite_prob(raw_new), f"Calibration fallback: {exc}"


def _calibration_table(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    if len(y_true) == 0:
        return pd.DataFrame()
    frame = pd.DataFrame({"Actual": y_true, "Probability": _finite_prob(np.asarray(prob, dtype=float))})
    frame["Bin"] = pd.cut(frame["Probability"], bins=np.linspace(0, 1, bins + 1), include_lowest=True)
    return (
        frame.groupby("Bin", observed=False)
        .agg(
            Customers=("Actual", "size"),
            Average_Predicted_Probability=("Probability", "mean"),
            Actual_Churn_Rate=("Actual", "mean"),
        )
        .reset_index()
    )


def _segment_performance(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or "Churn Label" not in scored:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for segment_column in ["Industry", "Region", "Rep", "Customer Value Band", "Frequency Segment"]:
        if segment_column not in scored.columns:
            continue
        for segment, group in scored.groupby(segment_column, dropna=False):
            y = pd.to_numeric(group["Churn Label"], errors="coerce").dropna()
            if y.empty:
                continue
            rows.append(
                {
                    "Segment Type": segment_column,
                    "Segment": segment,
                    "Customers": int(len(group)),
                    "Observed Churn Rate": float(y.mean()),
                    "Average Predicted Risk": float(group["Calibrated Churn Probability"].mean()),
                    "Average Priority Score": float(group["Priority Score"].mean()),
                    "Revenue at Risk": float(group["Revenue at Risk"].sum()),
                }
            )
    return pd.DataFrame(rows)


def _feature_importance(pipeline: Any, numeric_features: list[str], categorical_features: list[str], model_name: str) -> pd.DataFrame:
    if pipeline is None:
        return pd.DataFrame()
    try:
        model = pipeline.named_steps["model"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = list(numeric_features)
        if categorical_features:
            encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
            feature_names.extend(encoder.get_feature_names_out(categorical_features).tolist())
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])
        else:
            return pd.DataFrame()
        return (
            pd.DataFrame({"Feature": feature_names[: len(importance)], "Importance": importance, "Model": model_name})
            .sort_values("Importance", ascending=False)
            .head(40)
        )
    except Exception:
        return pd.DataFrame()


def fit_churn_models(snapshots: pd.DataFrame, latest_features: pd.DataFrame, config: ChurnConfig) -> ChurnModelOutputs:
    """Fit baselines and supervised churn models, then score latest customers."""

    if snapshots.empty or latest_features.empty:
        empty = pd.DataFrame()
        return ChurnModelOutputs(empty, empty, empty, empty, empty, empty, empty, "No model", 0.5, {})

    labeled = snapshots[snapshots["Eligible For Training"] & snapshots["Churn Label"].notna()].copy()
    labeled["Churn Label"] = labeled["Churn Label"].astype(int)
    train, validation, test = _temporal_split(labeled) if not labeled.empty else (labeled, labeled, labeled)
    eval_frame = test if not test.empty else validation if not validation.empty else train

    numeric_features = _available(config.numeric_features, snapshots)
    categorical_features = _available(config.categorical_features, snapshots)
    features = numeric_features + categorical_features
    metrics_rows: list[dict[str, object]] = []
    lift_tables: list[pd.DataFrame] = []
    selected_model = "Rule-based cadence baseline"
    selected_pipeline: Any = None
    selected_validation_raw: np.ndarray | None = None
    calibration_note = "Rule probability"
    recommended_threshold = 0.5

    if not eval_frame.empty:
        y_eval = eval_frame["Churn Label"].astype(int).to_numpy()
        eval_value = pd.to_numeric(eval_frame.get("Revenue 180d", 0), errors="coerce").fillna(0).to_numpy()
        for name, prob in [
            ("Recency threshold baseline", (pd.to_numeric(eval_frame["Inactivity To Cadence Ratio"], errors="coerce").fillna(0) >= 1.25).astype(float).to_numpy()),
            ("RFM scoring baseline", _rule_probability(eval_frame)),
        ]:
            metrics_rows.append(_classification_metrics(y_eval, prob, eval_value, name, 0.5, config.top_k_rates))
            lift_tables.append(_lift_gain_table(y_eval, prob, eval_value, name))

    can_fit = not train.empty and len(np.unique(train["Churn Label"])) > 1 and bool(features)
    if can_fit:
        try:
            from sklearn.compose import ColumnTransformer
            from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            from sklearn.tree import DecisionTreeClassifier

            try:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
            transformers = []
            if numeric_features:
                transformers.append(
                    (
                        "numeric",
                        Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                        numeric_features,
                    )
                )
            if categorical_features:
                transformers.append(
                    (
                        "categorical",
                        Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]),
                        categorical_features,
                    )
                )
            preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
            models: dict[str, Any] = {
                "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=config.random_state),
                "Decision Tree": DecisionTreeClassifier(max_depth=4, min_samples_leaf=8, class_weight="balanced", random_state=config.random_state),
                "Random Forest": RandomForestClassifier(
                    n_estimators=180,
                    min_samples_leaf=6,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=1,
                ),
                "Gradient Boosting": GradientBoostingClassifier(random_state=config.random_state),
            }
            try:
                from xgboost import XGBClassifier

                models["XGBoost"] = XGBClassifier(
                    n_estimators=120,
                    max_depth=3,
                    learning_rate=0.05,
                    subsample=0.90,
                    colsample_bytree=0.90,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=config.random_state,
                    n_jobs=1,
                )
            except Exception:
                pass

            X_train = train[features]
            y_train = train["Churn Label"].astype(int)
            eval_value = pd.to_numeric(eval_frame.get("Revenue 180d", 0), errors="coerce").fillna(0).to_numpy()
            y_eval = eval_frame["Churn Label"].astype(int).to_numpy() if not eval_frame.empty else np.array([])

            candidate_rows: list[dict[str, object]] = []
            pipelines: dict[str, Any] = {}
            raw_validation: dict[str, np.ndarray] = {}
            for name, model in models.items():
                try:
                    pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
                    pipeline.fit(X_train, y_train)
                    pipelines[name] = pipeline
                    if not validation.empty:
                        raw_validation[name] = _finite_prob(pipeline.predict_proba(validation[features])[:, 1])
                    if not eval_frame.empty:
                        prob_eval = _finite_prob(pipeline.predict_proba(eval_frame[features])[:, 1])
                        row = _classification_metrics(y_eval, prob_eval, eval_value, name, 0.5, config.top_k_rates)
                        metrics_rows.append(row)
                        candidate_rows.append(row)
                        lift_tables.append(_lift_gain_table(y_eval, prob_eval, eval_value, name))
                except Exception as model_exc:
                    metrics_rows.append({"Model": name, "Error": str(model_exc)})

            if candidate_rows:
                candidate_metrics = pd.DataFrame(candidate_rows)
                sort_cols = ["PR AUC", "Brier Score", "Recall Top 20%"]
                for col in sort_cols:
                    if col not in candidate_metrics:
                        candidate_metrics[col] = np.nan
                candidate_metrics = candidate_metrics.sort_values(
                    ["PR AUC", "Recall Top 20%", "Brier Score"],
                    ascending=[False, False, True],
                    na_position="last",
                )
                selected_model = str(candidate_metrics.iloc[0]["Model"])
                selected_pipeline = pipelines[selected_model]
                selected_validation_raw = raw_validation.get(selected_model)
        except Exception as exc:
            metrics_rows.append({"Model": "Supervised model error", "Error": str(exc)})

    latest = latest_features.copy()
    if selected_pipeline is not None and features:
        raw_latest = _finite_prob(selected_pipeline.predict_proba(latest[features])[:, 1])
        if not validation.empty and selected_validation_raw is not None:
            calibrated, calibration_note = _calibrate(
                selected_validation_raw,
                validation["Churn Label"].astype(int).to_numpy(),
                raw_latest,
            )
        else:
            calibrated, calibration_note = raw_latest, "No validation set for calibration"
        latest["Churn probability"] = raw_latest
        latest["Calibrated Churn Probability"] = calibrated
    else:
        latest["Churn probability"] = _rule_probability(latest)
        latest["Calibrated Churn Probability"] = latest["Churn probability"]
        calibration_note = "Rule-based score; no supervised calibration"

    latest = _add_business_scoring(latest, config)
    threshold_tbl = _threshold_table(latest, config)
    numeric_thresholds = threshold_tbl[pd.to_numeric(threshold_tbl["Threshold"], errors="coerce").notna()].copy()
    if not numeric_thresholds.empty:
        recommended_threshold = float(
            numeric_thresholds.sort_values("Expected Net Value", ascending=False).iloc[0]["Threshold"]
        )

    feature_importance = _feature_importance(selected_pipeline, numeric_features, categorical_features, selected_model)
    segment_perf = _segment_performance(latest)
    calibration_tbl = pd.DataFrame()
    if selected_pipeline is not None and not validation.empty and selected_validation_raw is not None:
        calibrated_val, _ = _calibrate(
            selected_validation_raw,
            validation["Churn Label"].astype(int).to_numpy(),
            selected_validation_raw,
        )
        calibration_tbl = _calibration_table(validation["Churn Label"].astype(int).to_numpy(), calibrated_val)
    metrics = pd.DataFrame(metrics_rows)
    lift_gain = pd.concat([table for table in lift_tables if not table.empty], ignore_index=True) if lift_tables else pd.DataFrame()

    prediction_cols = [
        "CustomerID",
        "Customer Name",
        "Snapshot Date",
        "Churn probability",
        "Calibrated Churn Probability",
        "Churn-risk category",
        "Priority Rank",
        "Priority Score",
        "Priority Score Raw",
        "Priority Score Explanation",
        "Customer Value Band",
        "Revenue at Risk",
        "Expected Next Order Date",
        "Days Overdue",
        "Top three churn reasons",
        "Recommended retention action",
        "Rep",
        "Model version",
        "Expected Customer Value",
        "Days Since Last Order",
        "Expected Reorder Window Days",
        "Reorder P75 Days",
        "Reorder P90 Days",
        "Max Reorder Days",
        "Reorder Alert Threshold Days",
        "High Risk Threshold Days",
        "Likely Churn Threshold Days",
        "Days Beyond Max Reorder Gap",
        "Days Beyond Likely Churn Threshold",
        "Inactivity To Cadence Ratio",
        "Churn Reason 1",
        "Churn Reason 2",
        "Churn Reason 3",
        "Industry",
        "Region",
        "Frequency Segment",
    ]
    prediction_cols = [column for column in prediction_cols if column in latest.columns]
    predictions = latest[prediction_cols].copy().rename(
        columns={
            "CustomerID": "Customer ID",
            "Rep": "Assigned sales representative",
        }
    )
    predictions = predictions.sort_values(["Priority Rank", "Priority Score"], ascending=[True, False]).reset_index(drop=True)

    metadata = {
        "selected_model": selected_model,
        "recommended_threshold": recommended_threshold,
        "calibration": calibration_note,
        "model_version": config.model_version,
        "observation_window_days": config.observation_window_days,
        "prediction_window_days": config.prediction_window_days,
        "gap_days": config.gap_days,
        "training_snapshots": int(labeled["Snapshot Date"].nunique()) if not labeled.empty else 0,
        "training_rows": int(len(train)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "eligible_rows": int(len(labeled)),
        "churn_rate": float(labeled["Churn Label"].mean()) if not labeled.empty else np.nan,
        "train_start": str(train["Snapshot Date"].min().date()) if not train.empty else None,
        "train_end": str(train["Snapshot Date"].max().date()) if not train.empty else None,
        "validation_start": str(validation["Snapshot Date"].min().date()) if not validation.empty else None,
        "validation_end": str(validation["Snapshot Date"].max().date()) if not validation.empty else None,
        "test_start": str(test["Snapshot Date"].min().date()) if not test.empty else None,
        "test_end": str(test["Snapshot Date"].max().date()) if not test.empty else None,
    }
    return ChurnModelOutputs(
        predictions=predictions,
        metrics=metrics,
        threshold_table=threshold_tbl,
        feature_importance=feature_importance,
        segment_performance=segment_perf,
        lift_gain=lift_gain,
        calibration_table=calibration_tbl,
        selected_model=selected_model,
        recommended_threshold=recommended_threshold,
        metadata=metadata,
    )

"""Sell Price estimation for missing or commercially implausible rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SellPriceImputationResult:
    """Container for sell-price imputation outputs."""

    data: pd.DataFrame
    model_metrics: pd.DataFrame
    selected_model: str
    feature_importance: pd.DataFrame


NUMERIC_FEATURES = [
    "Purchases",
    "Rebate",
    "Paper",
    "Labour",
    "Handling",
    "Press hrs",
    "Impressions",
    "Plates",
    "Quantity",
    "VA/K",
]

CATEGORICAL_FEATURES = [
    "CustomerID",
    "Rep",
    "Region",
    "Industry",
    "Work Type",
    "Product Type",
    "Binding Type",
    "Currency",
]


def _numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric column aligned to df, using a default when absent."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def rows_needing_sell_price_estimate(df: pd.DataFrame) -> pd.Series:
    """Identify rows where actual Sell Price is missing, zero, negative, or below purchases."""

    sell_price = pd.to_numeric(df.get("Sell Price"), errors="coerce")
    purchases = pd.to_numeric(df.get("Purchases"), errors="coerce")
    return sell_price.isna() | sell_price.le(0) | sell_price.lt(purchases.fillna(0))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denom)))


def _rule_based_estimate(df: pd.DataFrame) -> pd.Series:
    """Estimate sell price from cost base and peer pricing ratios.

    The rule deliberately avoids VA Amount and VA%, because those fields are
    derived from Sell Price and can reproduce bad zero-price rows.
    """

    base = (
        _numeric_column(df, "Purchases")
        + _numeric_column(df, "Rebate")
        + _numeric_column(df, "Labour")
        + _numeric_column(df, "Handling")
    )
    return base.clip(lower=0)


def estimate_sell_price(
    df: pd.DataFrame,
    random_state: int = 42,
    fast_mode: bool = False,
) -> SellPriceImputationResult:
    """Compare several estimation methods and populate Estimated Sell Price.

    Existing Sell Price is never overwritten. The selected model estimates only
    rows whose price is missing, non-positive, or below purchase cost.
    """

    data = df.copy()
    needs_estimate = rows_needing_sell_price_estimate(data)
    formula_imputed = data.get(
        "Sell Price Formula Imputed",
        pd.Series(False, index=data.index),
    ).fillna(False).astype(bool)
    valid_train = ~needs_estimate & pd.to_numeric(data["Sell Price"], errors="coerce").gt(0)
    data["Sell Price Needs Estimate"] = needs_estimate
    data["Sell Price Was Imputed"] = formula_imputed | needs_estimate
    data["Sell Price Source"] = np.select(
        [formula_imputed, needs_estimate],
        ["Formula Imputed", "Estimated"],
        default="Actual",
    )

    if fast_mode or valid_train.sum() < 50:
        estimate = _rule_based_estimate(data)
        data["Estimated Sell Price"] = np.where(needs_estimate, estimate, data["Sell Price"])
        data["Sell Price Confidence"] = np.select(
            [formula_imputed, needs_estimate],
            ["Formula", "Low"],
            default="Actual",
        )
        model_name = "Rule-based cost floor"
        if fast_mode and valid_train.sum() >= 50:
            model_name = "Rule-based cost floor (dashboard fast mode)"
        return SellPriceImputationResult(
            data=data,
            model_metrics=pd.DataFrame(
                [{"Model": model_name, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}]
            ),
            selected_model=model_name,
            feature_importance=pd.DataFrame(),
        )

    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import ElasticNet, LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeRegressor
    except ImportError:
        estimate = _rule_based_estimate(data)
        data["Estimated Sell Price"] = np.where(needs_estimate, estimate, data["Sell Price"])
        data["Sell Price Confidence"] = np.select(
            [formula_imputed, needs_estimate],
            ["Formula", "Low"],
            default="Actual",
        )
        return SellPriceImputationResult(
            data=data,
            model_metrics=pd.DataFrame(
                [{"Model": "Rule-based cost floor", "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}]
            ),
            selected_model="Rule-based cost floor",
            feature_importance=pd.DataFrame(),
        )

    numeric_features = [col for col in NUMERIC_FEATURES if col in data.columns]
    categorical_features = [col for col in CATEGORICAL_FEATURES if col in data.columns]
    features = numeric_features + categorical_features
    x = data.loc[valid_train, features].replace([np.inf, -np.inf], np.nan)
    y = pd.to_numeric(data.loc[valid_train, "Sell Price"], errors="coerce")

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", encoder)]),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "ElasticNet": ElasticNet(alpha=0.05, l1_ratio=0.25, random_state=random_state, max_iter=5000),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=random_state),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            min_samples_leaf=4,
            random_state=random_state,
            n_jobs=1,
        ),
    }
    for package, class_name, model_name in [
        ("xgboost", "XGBRegressor", "XGBoost"),
        ("lightgbm", "LGBMRegressor", "LightGBM"),
        ("catboost", "CatBoostRegressor", "CatBoost"),
    ]:
        try:
            module = __import__(package, fromlist=[class_name])
            estimator = getattr(module, class_name)
            if model_name == "CatBoost":
                models[model_name] = estimator(verbose=False, random_seed=random_state)
            else:
                models[model_name] = estimator(random_state=random_state)
        except Exception:
            continue

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    metrics = [
        _evaluate_rule_model("Rule-based cost floor", x_test, y_test, data.loc[x_test.index])
    ]
    fitted: dict[str, Pipeline] = {}

    for name, estimator in models.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        pred = np.maximum(pipeline.predict(x_test), 0)
        metrics.append(
            {
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                "MAPE": _mape(y_test.to_numpy(), pred),
                "R2": r2_score(y_test, pred),
            }
        )
        fitted[name] = pipeline

    metrics_df = pd.DataFrame(metrics).sort_values(["RMSE", "MAE"], ascending=[True, True])
    selected_model = str(metrics_df.iloc[0]["Model"])

    if selected_model == "Rule-based cost floor":
        estimate = _rule_based_estimate(data)
        feature_importance = pd.DataFrame()
    else:
        best_pipeline = fitted[selected_model]
        estimate = pd.Series(
            np.maximum(best_pipeline.predict(data[features].replace([np.inf, -np.inf], np.nan)), 0),
            index=data.index,
        )
        feature_importance = _feature_importance(best_pipeline, numeric_features, categorical_features, selected_model)

    floor = (
        _numeric_column(data, "Purchases")
        + _numeric_column(data, "Rebate")
    ).clip(lower=0)
    estimate = pd.Series(np.maximum(estimate, floor), index=data.index)
    data["Estimated Sell Price"] = np.where(needs_estimate, estimate, data["Sell Price"])
    best_mape = metrics_df.iloc[0]["MAPE"]
    confidence = "High" if pd.notna(best_mape) and best_mape <= 0.15 else "Medium" if pd.notna(best_mape) and best_mape <= 0.35 else "Low"
    data["Sell Price Confidence"] = np.select(
        [formula_imputed, needs_estimate],
        ["Formula", confidence],
        default="Actual",
    )
    return SellPriceImputationResult(data, metrics_df, selected_model, feature_importance)


def _evaluate_rule_model(name: str, x_test: pd.DataFrame, y_test: pd.Series, original_rows: pd.DataFrame) -> dict[str, float | str]:
    pred = _rule_based_estimate(original_rows).reindex(x_test.index).fillna(0).to_numpy()
    y = y_test.to_numpy()
    return {
        "Model": name,
        "MAE": float(np.mean(np.abs(y - pred))),
        "RMSE": float(np.sqrt(np.mean((y - pred) ** 2))),
        "MAPE": _mape(y, pred),
        "R2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def _feature_importance(pipeline: Any, numeric_features: list[str], categorical_features: list[str], model_name: str) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        try:
            names.extend(encoder.get_feature_names_out(categorical_features).tolist())
        except AttributeError:
            names.extend(encoder.get_feature_names(categorical_features).tolist())
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(np.ravel(model.coef_))
    else:
        return pd.DataFrame()
    return (
        pd.DataFrame({"Model": model_name, "Feature": names, "Importance": values})
        .sort_values("Importance", ascending=False)
        .head(40)
    )

"""Predictive modelling for Value Added (VA Amount)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelResults:
    """Container for modelling outputs."""

    metrics: pd.DataFrame
    feature_importance: pd.DataFrame
    diagnostics: pd.DataFrame
    predictions: pd.DataFrame
    best_model_name: str | None


def train_regression_models(
    df: pd.DataFrame,
    target: str = "VA Amount",
    random_state: int = 42,
) -> ModelResults:
    """Train regression models to predict VA Amount and return evaluation artefacts."""
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
    except ImportError:
        message = "scikit-learn is not installed; install requirements.txt to run modelling."
        return ModelResults(
            metrics=pd.DataFrame({"message": [message]}),
            feature_importance=pd.DataFrame({"message": [message]}),
            diagnostics=pd.DataFrame({"message": [message]}),
            predictions=pd.DataFrame(),
            best_model_name=None,
        )

    model_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[target]).copy()
    numeric_features = [
        col
        for col in [
            "Sell Price",
            "Quantity",
            "Mup%",
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
            "Direct Cost Estimate",
            "Cost to Sales Ratio",
            "Labour Share of Sales",
            "Paper Share of Sales",
            "Purchases Share of Sales",
            "Invoice Lead Time Days",
            "Ship Lead Time Days",
        ]
        if col in model_df.columns
    ]
    categorical_features = [
        col
        for col in [
            "Job Status",
            "Rep",
            "Region",
            "Industry",
            "Work Type",
            "Product Type",
            "Binding Type",
            "Currency",
        ]
        if col in model_df.columns
    ]

    features = numeric_features + categorical_features
    x = model_df[features]
    y = model_df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", encoder),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    models: dict[str, Any] = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=random_state,
            min_samples_leaf=2,
            n_jobs=1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    try:
        from xgboost import XGBRegressor

        models["XGBoost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=random_state,
        )
    except ImportError:
        pass

    metrics_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    prediction_frames: list[pd.DataFrame] = []

    for name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        metrics_rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
                "R2": r2_score(y_test, predictions),
            }
        )
        fitted_models[name] = pipeline
        prediction_frames.append(
            pd.DataFrame(
                {
                    "model": name,
                    "actual_va_amount": y_test.to_numpy(),
                    "predicted_va_amount": predictions,
                    "residual": y_test.to_numpy() - predictions,
                }
            )
        )

    metrics = pd.DataFrame(metrics_rows).sort_values(["R2", "RMSE"], ascending=[False, True])
    best_model_name = metrics.iloc[0]["model"] if not metrics.empty else None
    best_pipeline = fitted_models.get(best_model_name) if best_model_name else None

    feature_importance = (
        _feature_importance(best_pipeline, numeric_features, categorical_features, best_model_name)
        if best_pipeline is not None
        else pd.DataFrame()
    )
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    diagnostics = _regression_diagnostics(
        predictions[predictions["model"] == best_model_name],
        best_model_name,
    )

    return ModelResults(
        metrics=metrics,
        feature_importance=feature_importance,
        diagnostics=diagnostics,
        predictions=predictions,
        best_model_name=best_model_name,
    )


def _feature_names(
    pipeline: Any,
    numeric_features: list[str],
    categorical_features: list[str],
) -> list[str]:
    preprocessor = pipeline.named_steps["preprocessor"]
    names = list(numeric_features)
    if categorical_features:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        try:
            names.extend(encoder.get_feature_names_out(categorical_features).tolist())
        except AttributeError:
            names.extend(encoder.get_feature_names(categorical_features).tolist())
    return names


def _feature_importance(
    pipeline: Any,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str,
) -> pd.DataFrame:
    names = _feature_names(pipeline, numeric_features, categorical_features)
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        table = pd.DataFrame(
            {
                "model": model_name,
                "feature": names,
                "importance": importance,
                "absolute_importance": importance,
            }
        )
    elif hasattr(model, "coef_"):
        coefficients = np.ravel(model.coef_)
        table = pd.DataFrame(
            {
                "model": model_name,
                "feature": names,
                "importance": coefficients,
                "absolute_importance": np.abs(coefficients),
            }
        )
    else:
        return pd.DataFrame()

    return table.sort_values("absolute_importance", ascending=False).head(50)


def _regression_diagnostics(predictions: pd.DataFrame, model_name: str | None) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    residual = predictions["residual"]
    actual = predictions["actual_va_amount"]
    predicted = predictions["predicted_va_amount"]
    return pd.DataFrame(
        [
            {
                "model": model_name,
                "rows": len(predictions),
                "mean_residual": residual.mean(),
                "median_residual": residual.median(),
                "residual_std": residual.std(),
                "mean_actual": actual.mean(),
                "mean_predicted": predicted.mean(),
                "residual_actual_corr": residual.corr(actual),
                "residual_predicted_corr": residual.corr(predicted),
            }
        ]
    )

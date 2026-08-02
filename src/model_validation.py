"""Model validation diagnostics for the analytics pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd


POTENTIAL_LEAKAGE_FEATURES = {
    "Sell Price",
    "Revenue",
    "Profit",
    "Profit Margin",
    "VA%",
    "VA/K",
    "VA per Impression",
    "VA per Press Hour",
    "Customer Lifetime VA",
}


def validate_model_inputs(df: pd.DataFrame, target: str = "VA Amount") -> dict[str, pd.DataFrame]:
    """Create feature diagnostics for target leakage, redundancy, VIF, and variance."""

    numeric = (
        df.select_dtypes(include=[np.number])
        .replace([np.inf, -np.inf], np.nan)
        .drop(columns=[target], errors="ignore")
    )
    target_series = pd.to_numeric(df[target], errors="coerce") if target in df.columns else pd.Series(dtype="float64")

    corr = numeric.join(target_series.rename(target)).corr(numeric_only=True)
    target_corr = (
        corr[target]
        .drop(labels=[target], errors="ignore")
        .dropna()
        .abs()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "Feature", target: "Absolute Correlation With Target"})
    )
    leakage = pd.DataFrame(
        {
            "Feature": sorted(POTENTIAL_LEAKAGE_FEATURES.intersection(df.columns)),
            "Concern": "Derived from or tightly coupled to VA/Sell Price; review before predictive use.",
        }
    )
    high_corr = target_corr[target_corr["Absolute Correlation With Target"].ge(0.95)].copy()
    if not high_corr.empty:
        high_corr["Concern"] = "Correlation >= 0.95 suggests target leakage or formula dependency."
        leakage = pd.concat([leakage, high_corr[["Feature", "Concern"]]], ignore_index=True).drop_duplicates()

    redundant = _redundant_features(numeric)
    vif = _vif_table(numeric)
    variance = (
        numeric.var(numeric_only=True)
        .sort_values()
        .reset_index()
        .rename(columns={"index": "Feature", 0: "Variance"})
    )
    variance["Recommendation"] = np.where(
        variance["Variance"].fillna(0).le(1e-9),
        "Remove zero-variance predictor",
        "Retain unless redundant or leaky",
    )
    return {
        "correlation_matrix": corr.reset_index().rename(columns={"index": "Feature"}),
        "target_correlations": target_corr,
        "target_leakage": leakage,
        "redundant_features": redundant,
        "vif": vif,
        "variance_analysis": variance,
        "feature_recommendations": _feature_recommendations(target_corr, leakage, redundant, variance),
    }


def permutation_importance_report(df: pd.DataFrame, target: str = "VA Amount", random_state: int = 42) -> pd.DataFrame:
    """Compute permutation importance for a compact random-forest validation model."""

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
    except ImportError:
        return pd.DataFrame({"Message": ["scikit-learn not installed; permutation importance unavailable."]})

    features = [
        col
        for col in [
            "Sell Price",
            "Purchases",
            "Press hrs",
            "Impressions",
            "Handling",
            "Labour",
            "Paper",
            "Mup%",
            "mupnett",
            "labmup",
            "manadj",
            "Plates",
            "Quantity",
        ]
        if col in df.columns and col != target
    ]
    model_df = df[features + [target]].replace([np.inf, -np.inf], np.nan).dropna(subset=[target])
    if len(model_df) < 100:
        return pd.DataFrame({"Message": ["Not enough rows for permutation importance."]})
    if len(model_df) > 2500:
        model_df = model_df.sample(2500, random_state=random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        model_df[features],
        model_df[target],
        test_size=0.25,
        random_state=random_state,
    )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    min_samples_leaf=3,
                    random_state=random_state,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(x_train, y_train)
    result = permutation_importance(
        pipeline,
        x_test,
        y_test,
        n_repeats=5,
        random_state=random_state,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "Feature": features,
                "Importance Mean": result.importances_mean,
                "Importance Std": result.importances_std,
            }
        )
        .sort_values("Importance Mean", ascending=False)
        .reset_index(drop=True)
    )


def shap_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return a SHAP availability note.

    SHAP is optional and usually not installed in lightweight Streamlit deployments.
    The app exposes this note rather than failing.
    """

    try:
        __import__("shap")
    except Exception:
        return pd.DataFrame(
            {
                "SHAP Status": ["Unavailable"],
                "Explanation": ["Install shap to compute SHAP values for tree models."],
            }
        )
    return pd.DataFrame(
        {
            "SHAP Status": ["Available"],
            "Explanation": ["SHAP is installed; extend this report for local deep-dive notebooks."],
        }
    )


def _redundant_features(numeric: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    corr = numeric.corr(numeric_only=True).abs()
    rows = []
    cols = list(corr.columns)
    for i, left in enumerate(cols):
        for right in cols[i + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and value >= threshold:
                rows.append(
                    {
                        "Feature A": left,
                        "Feature B": right,
                        "Correlation": value,
                        "Recommendation": "Review one of these predictors for removal.",
                    }
                )
    return pd.DataFrame(rows).sort_values("Correlation", ascending=False) if rows else pd.DataFrame()


def _vif_table(numeric: pd.DataFrame) -> pd.DataFrame:
    sample = numeric.dropna(axis=1, how="all").fillna(numeric.median(numeric_only=True))
    sample = sample.loc[:, sample.std(numeric_only=True).gt(0)]
    if sample.shape[1] < 2:
        return pd.DataFrame()
    if sample.shape[0] > 2500:
        sample = sample.sample(2500, random_state=42)
    rows = []
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        values = sample.to_numpy(dtype=float)
        for index, column in enumerate(sample.columns):
            rows.append({"Feature": column, "VIF": float(variance_inflation_factor(values, index))})
    except Exception:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score

        for column in sample.columns:
            others = sample.drop(columns=[column])
            model = LinearRegression().fit(others, sample[column])
            r2 = r2_score(sample[column], model.predict(others))
            vif = np.inf if r2 >= 0.999999 else 1 / (1 - r2)
            rows.append({"Feature": column, "VIF": float(vif)})
    vif = pd.DataFrame(rows).sort_values("VIF", ascending=False)
    vif["Recommendation"] = np.where(vif["VIF"].gt(10), "High multicollinearity", "Acceptable")
    return vif


def _feature_recommendations(
    target_corr: pd.DataFrame,
    leakage: pd.DataFrame,
    redundant: pd.DataFrame,
    variance: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    leakage_set = set(leakage.get("Feature", []))
    low_variance_set = set(variance.loc[variance["Variance"].fillna(0).le(1e-9), "Feature"]) if not variance.empty else set()
    redundant_set = set(redundant.get("Feature B", [])) if not redundant.empty else set()
    for _, row in target_corr.iterrows():
        feature = row["Feature"]
        reasons = []
        if feature in leakage_set:
            reasons.append("possible target leakage")
        if feature in redundant_set:
            reasons.append("redundant with another predictor")
        if feature in low_variance_set:
            reasons.append("zero or near-zero variance")
        if row["Absolute Correlation With Target"] < 0.02:
            reasons.append("weak standalone relationship")
        recommendation = "Review/remove" if reasons else "Retain for model testing"
        rows.append(
            {
                "Feature": feature,
                "Absolute Correlation With Target": row["Absolute Correlation With Target"],
                "Recommendation": recommendation,
                "Reason": "; ".join(reasons) if reasons else "No immediate concern",
            }
        )
    return pd.DataFrame(rows)

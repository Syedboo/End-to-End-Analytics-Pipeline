"""Exploratory, business, and statistical analysis functions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .utils import safe_divide


RANKING_DIMENSIONS = {
    "customers": ["CustomerID", "Customer Name"],
    "industries": ["Industry"],
    "regions": ["Region"],
    "product_types": ["Product Type"],
    "work_types": ["Work Type"],
    "binding_types": ["Binding Type"],
    "sales_representatives": ["Rep"],
}


def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    summary = numeric.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary["missing_count"] = numeric.isna().sum()
    summary["missing_pct"] = numeric.isna().mean() * 100
    summary["skew"] = numeric.skew(numeric_only=True)
    summary["kurtosis"] = numeric.kurtosis(numeric_only=True)
    return summary.reset_index().rename(columns={"index": "column"})


def categorical_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise categorical columns for coverage and concentration."""
    rows: list[dict[str, Any]] = []
    categorical = df.select_dtypes(include=["object", "string", "category"]).columns
    for column in categorical:
        counts = df[column].value_counts(dropna=False)
        top_value = counts.index[0] if not counts.empty else None
        rows.append(
            {
                "column": column,
                "unique_values": int(df[column].nunique(dropna=True)),
                "top_value": top_value,
                "top_count": int(counts.iloc[0]) if not counts.empty else 0,
                "top_pct": float(counts.iloc[0] / len(df) * 100) if len(df) and not counts.empty else 0,
                "missing_count": int(df[column].isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("unique_values", ascending=False)


def profitability_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate revenue, VA, margin, markup, and production metrics by a dimension."""
    available_group_cols = [col for col in group_cols if col in df.columns]
    if not available_group_cols:
        return pd.DataFrame()

    grouped = (
        df.groupby(available_group_cols, dropna=False)
        .agg(
            Jobs=("Title", "count"),
            Customers=("CustomerID", "nunique"),
            Revenue=("Sell Price", "sum"),
            VA_Amount=("VA Amount", "sum"),
            Average_VA=("VA Amount", "mean"),
            Average_Markup=("Mup%", "mean"),
            Quantity=("Quantity", "sum"),
            Press_Hours=("Press hrs", "sum"),
            Impressions=("Impressions", "sum"),
            Labour=("Labour", "sum"),
            Paper=("Paper", "sum"),
            Purchases=("Purchases", "sum"),
        )
        .reset_index()
    )
    grouped["VA_Margin"] = safe_divide(grouped["VA_Amount"], grouped["Revenue"])
    grouped["VA per Press Hour"] = grouped["VA_Amount"] / grouped["Press_Hours"].replace(0, np.nan)
    grouped["Revenue Share"] = grouped["Revenue"] / grouped["Revenue"].sum()
    grouped["VA Share"] = grouped["VA_Amount"] / grouped["VA_Amount"].sum()
    return grouped.sort_values("VA_Amount", ascending=False)


def build_business_tables(df: pd.DataFrame, top_n: int = 25) -> dict[str, pd.DataFrame]:
    """Create all business aggregation tables requested in the brief."""
    tables: dict[str, pd.DataFrame] = {}

    for name, group_cols in RANKING_DIMENSIONS.items():
        table = profitability_table(df, group_cols)
        tables[f"{name}_profitability"] = table
        tables[f"top_{name}_by_revenue"] = table.sort_values("Revenue", ascending=False).head(top_n)
        tables[f"top_{name}_by_va"] = table.sort_values("VA_Amount", ascending=False).head(top_n)
        tables[f"top_{name}_by_va_margin"] = table.sort_values(
            ["VA_Margin", "Revenue"],
            ascending=[False, False],
        ).head(top_n)
        tables[f"top_{name}_by_average_markup"] = table.sort_values(
            ["Average_Markup", "Revenue"],
            ascending=[False, False],
        ).head(top_n)

    monthly = (
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
        .sort_values("Sales Month")
    )
    monthly["VA_Margin"] = safe_divide(monthly["VA_Amount"], monthly["Revenue"])
    tables["monthly_sales_profitability_trend"] = monthly
    return tables


def relationship_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate Pearson and Spearman correlations for requested relationships."""
    pairs = [
        ("Sell Price", "VA Amount"),
        ("Labour", "VA Amount"),
        ("Paper", "Sell Price"),
        ("Press hrs", "VA Amount"),
        ("Impressions", "VA Amount"),
        ("Mup%", "Profit"),
        ("Quantity", "Profit"),
        ("Purchases", "VA Amount"),
        ("Handling", "VA Amount"),
    ]
    rows: list[dict[str, Any]] = []
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        data = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 3:
            continue
        rows.append(
            {
                "x": x_col,
                "y": y_col,
                "rows": len(data),
                "pearson_corr": data[x_col].corr(data[y_col], method="pearson"),
                "spearman_corr": data[x_col].rank().corr(data[y_col].rank(), method="pearson"),
            }
        )
    return pd.DataFrame(rows).sort_values("pearson_corr", ascending=False)


def run_statistical_analysis(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run correlation, ANOVA, t-test, chi-square, and effect-size analyses."""
    outputs: dict[str, pd.DataFrame] = {}
    numeric_cols = [
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
            "Plates",
            "Profit Margin",
            "VA Amount",
        ]
        if col in df.columns
    ]

    corr = df[numeric_cols].corr(numeric_only=True)
    outputs["correlation_matrix"] = corr.reset_index().rename(columns={"index": "variable"})

    try:
        from scipy import stats
    except ImportError:
        message = "scipy is not installed; install requirements.txt to run inferential tests."
        outputs["anova_tests"] = pd.DataFrame({"message": [message]})
        outputs["t_tests"] = pd.DataFrame({"message": [message]})
        outputs["chi_square_tests"] = pd.DataFrame({"message": [message]})
        return outputs

    outputs["anova_tests"] = _anova_tests(df, stats)
    outputs["t_tests"] = _t_tests(df, stats)
    outputs["chi_square_tests"] = _chi_square_tests(df, stats)
    outputs["target_correlations"] = _target_correlations(df, numeric_cols, stats)
    return outputs


def _anova_tests(df: pd.DataFrame, stats: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in ["Industry", "Work Type", "Product Type", "Binding Type", "Rep", "Region"]:
        if column not in df.columns:
            continue
        groups = [
            group["VA Amount"].dropna()
            for _, group in df.groupby(column)
            if len(group["VA Amount"].dropna()) >= 2
        ]
        if len(groups) < 2:
            continue
        statistic, p_value = stats.f_oneway(*groups)
        grand_mean = df["VA Amount"].mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
        ss_total = ((df["VA Amount"] - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total else np.nan
        rows.append(
            {
                "category": column,
                "groups": len(groups),
                "f_statistic": statistic,
                "p_value": p_value,
                "eta_squared": eta_squared,
                "interpretation": "significant" if p_value < 0.05 else "not significant",
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def _t_tests(df: pd.DataFrame, stats: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "Mup%" in df.columns:
        median_markup = df["Mup%"].median()
        low = df.loc[df["Mup%"] <= median_markup, "VA Amount"].dropna()
        high = df.loc[df["Mup%"] > median_markup, "VA Amount"].dropna()
        if len(low) >= 2 and len(high) >= 2:
            statistic, p_value = stats.ttest_ind(high, low, equal_var=False)
            rows.append(
                {
                    "test": "High markup vs low markup VA Amount",
                    "group_a": "Above median markup",
                    "group_b": "At or below median markup",
                    "mean_a": high.mean(),
                    "mean_b": low.mean(),
                    "t_statistic": statistic,
                    "p_value": p_value,
                    "interpretation": "significant" if p_value < 0.05 else "not significant",
                }
            )

    if "Work Type" in df.columns:
        counts = df["Work Type"].value_counts()
        if len(counts) >= 2:
            group_a_name, group_b_name = counts.index[:2]
            group_a = df.loc[df["Work Type"] == group_a_name, "VA Amount"].dropna()
            group_b = df.loc[df["Work Type"] == group_b_name, "VA Amount"].dropna()
            if len(group_a) >= 2 and len(group_b) >= 2:
                statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
                rows.append(
                    {
                        "test": "Two largest work types VA Amount",
                        "group_a": group_a_name,
                        "group_b": group_b_name,
                        "mean_a": group_a.mean(),
                        "mean_b": group_b.mean(),
                        "t_statistic": statistic,
                        "p_value": p_value,
                        "interpretation": "significant" if p_value < 0.05 else "not significant",
                    }
                )
    return pd.DataFrame(rows)


def _chi_square_tests(df: pd.DataFrame, stats: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "High VA Job" not in df.columns:
        return pd.DataFrame()

    for column in ["Industry", "Work Type", "Product Type", "Binding Type", "Rep", "Region"]:
        if column not in df.columns:
            continue
        crosstab = pd.crosstab(df[column], df["High VA Job"])
        if crosstab.shape[0] < 2 or crosstab.shape[1] < 2:
            continue
        statistic, p_value, dof, _ = stats.chi2_contingency(crosstab)
        rows.append(
            {
                "category": column,
                "chi_square": statistic,
                "degrees_of_freedom": dof,
                "p_value": p_value,
                "interpretation": "associated" if p_value < 0.05 else "not associated",
            }
        )
    return pd.DataFrame(rows).sort_values("p_value")


def _target_correlations(df: pd.DataFrame, numeric_cols: list[str], stats: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    target = "VA Amount"
    for column in numeric_cols:
        if column == target:
            continue
        data = df[[column, target]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(data) < 3:
            continue
        statistic, p_value = stats.pearsonr(data[column], data[target])
        rows.append(
            {
                "variable": column,
                "pearson_corr": statistic,
                "p_value": p_value,
                "abs_corr": abs(statistic),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_corr", ascending=False)



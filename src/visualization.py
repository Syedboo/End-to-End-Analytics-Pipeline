"""Static and interactive visualisations for printing job analytics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import slugify


def _compact_number(value: float) -> str:
    """Format large values using business-friendly K/M/B suffixes."""
    value = float(value)
    abs_value = abs(value)
    sign = "-" if value < 0 else ""

    for threshold, suffix in (
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ):
        if abs_value >= threshold:
            scaled = abs_value / threshold
            label = f"{scaled:.1f}".rstrip("0").rstrip(".")
            return f"{sign}{label}{suffix}"

    return f"{value:,.0f}"


def _currency_label(value: float, _position: int | None = None) -> str:
    """Render numeric axis ticks as compact pound sterling labels."""
    value = float(value)
    sign = "-" if value < 0 else ""
    return f"{sign}£{_compact_number(abs(value))}"


def _percentage_label(value: float, _position: int | None = None) -> str:
    """Render percentages whether stored as decimals or whole percentages."""
    scaled = value * 100 if abs(value) <= 1.5 else value
    return f"{scaled:.1f}%"


def _apply_currency_axis(ax: Any, axis: str = "x") -> None:
    """Apply pound formatting and suppress Matplotlib scientific offsets."""
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(_currency_label)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.offsetText.set_visible(False)
    else:
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_visible(False)


def _apply_percentage_axis(ax: Any, axis: str = "y") -> None:
    """Apply percentage formatting and suppress scientific offsets."""
    from matplotlib.ticker import FuncFormatter

    formatter = FuncFormatter(_percentage_label)
    if axis == "x":
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.offsetText.set_visible(False)
    else:
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.offsetText.set_visible(False)



def _apply_plotly_currency_axis(fig: Any, axis: str) -> None:
    """Use compact GBP labels in Plotly without scientific notation."""
    update = fig.update_xaxes if axis == "x" else fig.update_yaxes
    update(
        tickprefix="£",
        tickformat="~s",
        exponentformat="none",
        showexponent="none",
        separatethousands=True,
    )


def _apply_plotly_dashboard_layout(fig: Any, height: int = 320) -> None:
    """Keep Streamlit dashboard charts compact and readable."""
    fig.update_layout(
        height=height,
        margin=dict(l=45, r=20, t=48, b=38),
        hovermode="closest",
        font=dict(size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
def _plotting_stack() -> tuple[Any, Any]:
    """Import plotting libraries lazily so dependency errors are clear."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise ImportError(
            "matplotlib and seaborn are required for static charts. "
            "Install the project dependencies with `pip install -r requirements.txt`."
        ) from exc

    sns.set_theme(style="whitegrid", context="talk")
    return plt, sns


def generate_all_figures(
    df: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, str]:
    """Generate all requested publication-quality and interactive charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures: dict[str, str] = {}

#    figures.update(_distribution_figures(df, output_dir))
#    figures.update(_missing_heatmap(df, output_dir))
#    figures.update(_correlation_heatmap(df, output_dir))
#    figures.update(_pairplot(df, output_dir))
    figures.update(_ranking_charts(business_tables, output_dir))
    figures.update(_trend_charts(business_tables, output_dir))
#   figures.update(_relationship_charts(df, output_dir))
    figures.update(_interactive_charts(df, business_tables, output_dir))
    return figures


def _distribution_figures(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    figures: dict[str, str] = {}
    for column in ["Profit", "Sell Price", "VA Amount", "Mup%", "Quantity"]:
        if column not in df.columns:
            continue
        plot_data = df[column].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_data.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
        sns.histplot(plot_data, bins=40, kde=True, ax=axes[0], color="#2454a6")
        axes[0].set_title(f"{column} Distribution")
        sns.boxplot(x=plot_data, ax=axes[1], color="#8ab17d")
        axes[1].set_title(f"{column} Boxplot")
        sns.violinplot(x=plot_data, ax=axes[2], color="#e9c46a")
        axes[2].set_title(f"{column} Violin Plot")
        for axis in axes:
            axis.set_xlabel(column)
            if column in {"Profit", "Sell Price", "VA Amount"}:
                _apply_currency_axis(axis, "x")
            elif column in {"Mup%", "VA%"}:
                _apply_percentage_axis(axis, "x")
        fig.tight_layout()
        path = output_dir / f"distribution_{slugify(column)}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures[f"distribution_{slugify(column)}"] = str(path)
    return figures


def _missing_heatmap(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    missing = df.isna()
    if not missing.any().any():
        return {}
    sample = missing.sample(min(len(missing), 1200), random_state=42)
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(sample, cbar=False, ax=ax, cmap=["#f7f7f7", "#d95f02"])
    ax.set_title("Missing Value Heatmap")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Sampled Jobs")
    fig.tight_layout()
    path = output_dir / "missing_value_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"missing_value_heatmap": str(path)}


def _correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    columns = [
        col
        for col in [
            "Sell Price",
            "Quantity",
            "Mup%",
            "VA Amount",
            "Purchases",
            "Press hrs",
            "Impressions",
            "Handling",
            "Labour",
            "Paper",
            "Profit Margin",
            "VA per Press Hour",
        ]
        if col in df.columns
    ]
    corr = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, linewidths=0.4, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {"correlation_heatmap": str(path)}


def _pairplot(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    columns = [
        col
        for col in ["Sell Price", "VA Amount", "Mup%", "Labour", "Paper", "Press hrs"]
        if col in df.columns
    ]
    sample = df[columns].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sample) > 1000:
        sample = sample.sample(1000, random_state=42)
    grid = sns.pairplot(sample, diag_kind="kde", corner=True, plot_kws={"alpha": 0.35, "s": 18})
    grid.fig.suptitle("Selected Pairplot", y=1.02)
    path = output_dir / "pairplot_selected_variables.png"
    grid.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(grid.fig)
    return {"pairplot_selected_variables": str(path)}


def _ranking_charts(business_tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    chart_specs = {
        "top_customers_by_va": "Customer Name",
        "top_industries_by_va": "Industry",
        "top_regions_by_va": "Region",
        "top_product_types_by_va": "Product Type",
        "top_work_types_by_va": "Work Type",
        "top_binding_types_by_va": "Binding Type",
        "top_sales_representatives_by_va": "Rep",
    }
    figures: dict[str, str] = {}
    for table_name, label_col in chart_specs.items():
        table = business_tables.get(table_name)
        if table is None or table.empty or label_col not in table.columns:
            continue
        plot_data = table.head(10).sort_values("VA_Amount", ascending=False)
        fig, ax = plt.subplots(figsize=(8.8, 3.9))
        sns.barplot(data=plot_data, x="VA_Amount", y=label_col, ax=ax, color="#287c8e")
        ax.set_title(table_name.replace("_", " ").title())
        ax.set_xlabel("Total VA Amount (GBP)")
        ax.set_ylabel("")
        _apply_currency_axis(ax, "x")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"{table_name}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures[table_name] = str(path)
    return figures


def _trend_charts(business_tables: dict[str, pd.DataFrame], output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    monthly = business_tables.get("monthly_sales_profitability_trend")
    if monthly is None or monthly.empty:
        return {}
    figures: dict[str, str] = {}
    plot_data = monthly.copy()
    plot_data["Sales Month"] = pd.to_datetime(plot_data["Sales Month"])

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    sns.lineplot(data=plot_data, x="Sales Month", y="Revenue", marker="o", ax=ax, label="Revenue")
    sns.lineplot(data=plot_data, x="Sales Month", y="VA_Amount", marker="o", ax=ax, label="VA Amount")
    ax.set_title("Monthly Sales and Profitability Trend")
    ax.set_xlabel("")
    ax.set_ylabel("Value (£)")
    _apply_currency_axis(ax, "y")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    path = output_dir / "monthly_sales_profitability_trend.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures["monthly_sales_profitability_trend"] = str(path)

    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    sns.lineplot(data=plot_data, x="Sales Month", y="VA_Margin", marker="o", ax=ax, color="#b44d12")
    ax.set_title("Monthly Average VA Margin")
    ax.set_xlabel("")
    ax.set_ylabel("Average VA Margin")
    _apply_percentage_axis(ax, "y")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = output_dir / "monthly_profitability_margin_trend.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    figures["monthly_profitability_margin_trend"] = str(path)
    return figures


def _relationship_charts(df: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    plt, sns = _plotting_stack()
    pairs = [
        ("Sell Price", "VA Amount"),
        ("Labour", "VA Amount"),
        ("Paper", "Sell Price"),
        ("Press hrs", "VA Amount"),
        ("Impressions", "VA Amount"),
        ("Mup%", "Profit"),
        ("Quantity", "Profit"),
    ]
    figures: dict[str, str] = {}
    for x_col, y_col in pairs:
        if x_col not in df.columns or y_col not in df.columns:
            continue
        plot_data = df[[x_col, y_col, "Work Type"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(plot_data) > 2500:
            plot_data = plot_data.sample(2500, random_state=42)
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        sns.scatterplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            hue="Work Type" if "Work Type" in plot_data.columns else None,
            alpha=0.55,
            s=35,
            ax=ax,
        )
        sns.regplot(
            data=plot_data,
            x=x_col,
            y=y_col,
            scatter=False,
            ax=ax,
            color="#111111",
            line_kws={"linewidth": 2},
        )
        ax.set_title(f"{x_col} vs {y_col}")
        if x_col in {"Sell Price", "VA Amount", "Purchases", "Handling", "Labour", "Paper", "Profit"}:
            _apply_currency_axis(ax, "x")
        if y_col in {"Sell Price", "VA Amount", "Purchases", "Handling", "Labour", "Paper", "Profit"}:
            _apply_currency_axis(ax, "y")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        path = output_dir / f"scatter_{slugify(x_col)}_vs_{slugify(y_col)}.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        figures[f"scatter_{slugify(x_col)}_vs_{slugify(y_col)}"] = str(path)
    return figures


def _interactive_charts(
    df: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
    output_dir: Path,
) -> dict[str, str]:
    try:
        import plotly.express as px
    except ImportError:
        return {}

    figures: dict[str, str] = {}
    scatter_data = df.copy()
    if len(scatter_data) > 3000:
        scatter_data = scatter_data.sample(3000, random_state=42)

    fig = px.scatter(
        scatter_data,
        x="Sell Price",
        y="VA Amount",
        color="Work Type",
        size="Quantity",
        hover_data=["Customer Name", "Industry", "Product Type", "Rep"],
        title="Interactive Sell Price vs VA Amount",
    )
    fig.update_xaxes(title_text="Sell Price (£)")
    fig.update_yaxes(title_text="VA Amount (£)")
    _apply_plotly_currency_axis(fig, "x")
    _apply_plotly_currency_axis(fig, "y")
    _apply_plotly_dashboard_layout(fig, height=320)
    fig.update_layout(legend_title_text="Work Type")
    path = output_dir / "interactive_sell_price_vs_va_amount.html"
    fig.write_html(path, include_plotlyjs=True)
    figures["interactive_sell_price_vs_va_amount"] = str(path)

    monthly = business_tables.get("monthly_sales_profitability_trend")
    if monthly is not None and not monthly.empty:
        trend = monthly.melt(
            id_vars=["Sales Month"],
            value_vars=["Revenue", "VA_Amount"],
            var_name="Metric",
            value_name="Value",
        )
        fig = px.line(
            trend,
            x="Sales Month",
            y="Value",
            color="Metric",
            markers=True,
            title="Interactive Monthly Revenue and VA Trend",
        )
        fig.update_yaxes(title_text="Value (£)")
        _apply_plotly_currency_axis(fig, "y")
        _apply_plotly_dashboard_layout(fig, height=300)
        fig.update_layout(legend_title_text="Metric")
        path = output_dir / "interactive_monthly_trend.html"
        fig.write_html(path, include_plotlyjs=True)
        figures["interactive_monthly_trend"] = str(path)

    return figures



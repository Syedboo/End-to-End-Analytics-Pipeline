"""Automated markdown and HTML business reporting."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .data_cleaning import CleanResult
from .utils import currency, dataframe_to_markdown, percent


def generate_business_report(
    clean_result: CleanResult,
    descriptive: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
    relationship_table: pd.DataFrame,
    statistical_tables: dict[str, pd.DataFrame],
    model_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
    figures: dict[str, str],
    output_dir: Path,
) -> dict[str, Path]:
    """Generate board-ready markdown and HTML reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    context = _derive_context(
        business_tables,
        relationship_table,
        model_metrics,
        feature_importance,
        customer_lifecycle,
    )
    markdown = _build_markdown(
        clean_result,
        descriptive,
        business_tables,
        relationship_table,
        statistical_tables,
        model_metrics,
        feature_importance,
        customer_lifecycle,
        figures,
        context,
    )
    markdown_path = output_dir / "business_report.md"
    markdown_path.write_text(markdown, encoding="utf-8")

    html_report = _build_html(
        clean_result,
        business_tables,
        relationship_table,
        statistical_tables,
        model_metrics,
        feature_importance,
        customer_lifecycle,
        figures,
        context,
    )
    html_path = output_dir / "business_report.html"
    html_path.write_text(html_report, encoding="utf-8")
    return {"markdown": markdown_path, "html": html_path}


def _derive_context(
    business_tables: dict[str, pd.DataFrame],
    relationship_table: pd.DataFrame,
    model_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
) -> dict[str, Any]:
    def first_value(table_name: str, column: str, fallback: str = "n/a") -> Any:
        table = business_tables.get(table_name)
        if table is None or table.empty or column not in table.columns:
            return fallback
        return table.iloc[0][column]

    context: dict[str, Any] = {
        "top_customer": first_value("top_customers_by_va", "Customer Name"),
        "top_customer_va": first_value("top_customers_by_va", "VA_Amount", 0),
        "top_industry": first_value("top_industries_by_va", "Industry"),
        "top_product": first_value("top_product_types_by_va", "Product Type"),
        "top_work_type": first_value("top_work_types_by_va", "Work Type"),
        "top_region": first_value("top_regions_by_va", "Region"),
        "top_rep": first_value("top_sales_representatives_by_va", "Rep"),
        "best_model": None,
        "best_r2": None,
        "strongest_relationship": None,
        "strongest_predictor": None,
    }
    if model_metrics is not None and not model_metrics.empty and "R2" in model_metrics.columns:
        best = model_metrics.sort_values("R2", ascending=False).iloc[0]
        context["best_model"] = best["model"]
        context["best_r2"] = best["R2"]
    if context["best_model"] is None:
        context["best_model"] = "Not run"
    if relationship_table is not None and not relationship_table.empty:
        rel = relationship_table.reindex(
            relationship_table["pearson_corr"].abs().sort_values(ascending=False).index
        ).iloc[0]
        context["strongest_relationship"] = f"{rel['x']} vs {rel['y']}"
    if feature_importance is not None and not feature_importance.empty and "feature" in feature_importance:
        context["strongest_predictor"] = feature_importance.iloc[0]["feature"]
    if customer_lifecycle is not None and not customer_lifecycle.empty:
        at_risk = customer_lifecycle[
            customer_lifecycle["Churn Risk"].isin(["Likely churn", "High risk", "Due for reorder"])
        ]
        context["follow_up_count"] = len(at_risk)
        context["follow_up_va"] = at_risk["Customer Lifetime VA"].sum() if not at_risk.empty else 0
    else:
        context["follow_up_count"] = 0
        context["follow_up_va"] = 0
    return context


def _r2_text(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.3f}"


def _build_markdown(
    clean_result: CleanResult,
    descriptive: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
    relationship_table: pd.DataFrame,
    statistical_tables: dict[str, pd.DataFrame],
    model_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
    figures: dict[str, str],
    context: dict[str, Any],
) -> str:
    r2_text = _r2_text(context["best_r2"])
    figure_entries: list[str] = []
    for key, path in figures.items():
        figure_path = Path(path)
        if figure_path.exists() and figure_path.suffix.lower() in {".png", ".html"}:
            figure_entries.append(f"- [{figure_path.name}](../figures/{figure_path.name})")
        else:
            figure_entries.append(f"- {key}: {path}")
    figure_lines = "\n".join(figure_entries) if figure_entries else "- No figures generated."
    model_summary = dataframe_to_markdown(model_metrics, max_rows=10)
    feature_summary = dataframe_to_markdown(feature_importance, max_rows=15)
    relationship_summary = dataframe_to_markdown(relationship_table, max_rows=12)
    customer_follow_up = (
        customer_lifecycle[
            customer_lifecycle["Churn Risk"].isin(["Likely churn", "High risk", "Due for reorder"])
        ]
        .sort_values("Customer Lifetime VA", ascending=False)
        .head(15)
        if customer_lifecycle is not None and not customer_lifecycle.empty
        else pd.DataFrame()
    )

    return f"""# W&G Baird Printing Analytics Report

## Executive Summary

This project analyses historical commercial printing jobs to identify the factors associated
with revenue, Value Added (VA Amount), margin, sales effectiveness, and operational
efficiency. VA Amount is used as the primary profit and value-added proxy because the
field definition describes it as the labour and markup value created by each job.

Headline findings:

- Most valuable customer by VA: **{context['top_customer']}** ({currency(context['top_customer_va'])}).
- Highest-VA industry: **{context['top_industry']}**.
- Highest-VA product type: **{context['top_product']}**.
- Highest-VA work type: **{context['top_work_type']}**.
- Highest-VA region: **{context['top_region']}**.
- Highest-VA sales representative: **{context['top_rep']}**.
- Strongest inspected relationship: **{context['strongest_relationship']}**.
- Best predictive model: **{context['best_model']}** with R2 of **{r2_text}**.
- Follow-up opportunity: **{context['follow_up_count']}** customers are due or high priority,
  representing historical VA of **{currency(context['follow_up_va'])}**.

## Design Science Research Methodology Framing

1. Problem identification: profitability is shaped by customer mix, pricing, product mix,
   production effort, materials, and sales ownership.
2. Objectives: produce a reusable data product that refreshes with same-format data,
   highlights commercial priorities, and predicts VA Amount.
3. Design and development: modular Python pipeline for cleaning, feature engineering,
   visualisation, statistics, modelling, and automated reporting.
4. Demonstration: the supplied W&G Baird sample workbook is processed end to end.
5. Evaluation: outputs include quality checks, inferential tests, model performance,
   feature importance, and residual diagnostics.
6. Communication: this report and the generated charts are suitable for a concise
   board-level presentation.

## Data Cleaning Summary

- Raw rows: {clean_result.raw_rows:,}
- Cleaned rows: {clean_result.cleaned_rows:,}
- Duplicate rows removed: {clean_result.duplicate_rows_removed:,}
- Missing fields were retained in the data quality report and imputed only where needed
  for numeric analysis or modelling.
- Outliers are flagged rather than removed because large jobs may be commercially
  important strategic accounts.

## EDA Findings

Descriptive statistics were produced for all numeric fields, including revenue, quantity,
markup, VA Amount, materials, labour, press hours, and impressions.

{dataframe_to_markdown(descriptive, max_rows=12)}

Generated visualisations:

{figure_lines}

## Business Insights

### Top Customers by VA

{dataframe_to_markdown(business_tables.get('top_customers_by_va', pd.DataFrame()), max_rows=10)}

### Top Industries by VA

{dataframe_to_markdown(business_tables.get('top_industries_by_va', pd.DataFrame()), max_rows=10)}

### Top Regions by VA

{dataframe_to_markdown(business_tables.get('top_regions_by_va', pd.DataFrame()), max_rows=10)}

### Top Product Types by VA

{dataframe_to_markdown(business_tables.get('top_product_types_by_va', pd.DataFrame()), max_rows=10)}

### Top Work Types by VA

{dataframe_to_markdown(business_tables.get('top_work_types_by_va', pd.DataFrame()), max_rows=10)}

### Top Sales Representatives by VA

{dataframe_to_markdown(business_tables.get('top_sales_representatives_by_va', pd.DataFrame()), max_rows=10)}

### Follow-up and Reorder Opportunities

{dataframe_to_markdown(customer_follow_up, max_rows=15)}

## Relationship Analysis

{relationship_summary}

## Statistical Analysis

### ANOVA

{dataframe_to_markdown(statistical_tables.get('anova_tests', pd.DataFrame()), max_rows=10)}

### T-tests

{dataframe_to_markdown(statistical_tables.get('t_tests', pd.DataFrame()), max_rows=10)}

### Chi-square Tests

{dataframe_to_markdown(statistical_tables.get('chi_square_tests', pd.DataFrame()), max_rows=10)}

## Model Performance

{model_summary}

### Feature Importance

{feature_summary}

## Recommendations

- Protect and expand the highest-VA customer and industry segments through account plans,
  reorder prompts, and targeted cross-selling into similar product types.
- Use markup analysis to identify work priced below the value created; review low-margin
  jobs by product, work type, and sales representative before repeating similar quotes.
- Treat labour, paper, purchases, and press-hour intensity as operational levers. Jobs with
  weak VA per press hour or high cost-to-sales ratios should be reviewed for estimating,
  scheduling, supplier pricing, or production process improvements.
- Use the churn and reorder table as a weekly sales action list. Customers with strong
  historical VA, overdue reorder cadence, or days beyond their historical max gap should receive priority follow-up.
- Use model feature importance as a decision-support tool, not an automated pricing rule:
  the model highlights variables associated with VA, while commercial judgement remains
  essential for strategic accounts and unusual jobs.

## Future Work

- Add live database ingestion and scheduled refreshes.
- Add a customer-level forecast for expected next order value and timing.
- Include true gross margin if full cost accounting fields become available.
- Build a Streamlit or Dash interface for sales and production users.
- Add model monitoring to detect drift as pricing, product mix, and customer behaviour change.
"""


def _build_html(
    clean_result: CleanResult,
    business_tables: dict[str, pd.DataFrame],
    relationship_table: pd.DataFrame,
    statistical_tables: dict[str, pd.DataFrame],
    model_metrics: pd.DataFrame,
    feature_importance: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
    figures: dict[str, str],
    context: dict[str, Any],
) -> str:
    r2_text = _r2_text(context["best_r2"])

    def table(df: pd.DataFrame, max_rows: int = 10) -> str:
        if df is None or df.empty:
            return "<p><em>No rows available.</em></p>"
        return df.head(max_rows).to_html(index=False, classes="data-table", border=0)

    def image(name: str) -> str:
        path = figures.get(name)
        if not path:
            return ""
        if not Path(path).exists() or Path(path).suffix.lower() != ".png":
            return ""
        filename = html.escape(Path(path).name)
        title = html.escape(name.replace("_", " ").title())
        return f'<figure><img src="../figures/{filename}" alt="{title}"><figcaption>{title}</figcaption></figure>'

    follow_up = (
        customer_lifecycle[
            customer_lifecycle["Churn Risk"].isin(["Likely churn", "High risk", "Due for reorder"])
        ]
        .sort_values("Customer Lifetime VA", ascending=False)
        .head(15)
        if customer_lifecycle is not None and not customer_lifecycle.empty
        else pd.DataFrame()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>W&G Baird Printing Analytics Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 0; line-height: 1.45; }}
    header {{ background: #12355b; color: white; padding: 36px 48px; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }}
    h1, h2, h3 {{ line-height: 1.15; }}
    h2 {{ border-bottom: 2px solid #d9e2ec; padding-bottom: 6px; margin-top: 36px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 14px; }}
    .metric {{ border-left: 4px solid #2f80ed; background: #f5f7fa; padding: 14px 16px; }}
    .metric strong {{ display: block; font-size: 0.82rem; color: #52616b; text-transform: uppercase; }}
    .metric span {{ font-size: 1.15rem; font-weight: 700; }}
    .data-table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 0.9rem; }}
    .data-table th, .data-table td {{ border-bottom: 1px solid #d9e2ec; padding: 8px 10px; text-align: left; }}
    .data-table th {{ background: #eef2f7; }}
    figure {{ margin: 24px 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #d9e2ec; }}
    figcaption {{ font-size: 0.9rem; color: #52616b; margin-top: 6px; }}
    li {{ margin-bottom: 6px; }}
  </style>
</head>
<body>
<header>
  <h1>W&amp;G Baird Printing Analytics Report</h1>
  <p>Commercial decision support for profitability, pricing, customer focus, and production efficiency.</p>
</header>
<main>
  <section class="summary">
    <div class="metric"><strong>Rows analysed</strong><span>{clean_result.cleaned_rows:,}</span></div>
    <div class="metric"><strong>Top customer by VA</strong><span>{html.escape(str(context['top_customer']))}</span></div>
    <div class="metric"><strong>Top industry</strong><span>{html.escape(str(context['top_industry']))}</span></div>
    <div class="metric"><strong>Best model</strong><span>{html.escape(str(context['best_model']))}</span></div>
  </section>

  <h2>Executive Summary</h2>
  <p>VA Amount is used as the main commercial profit proxy. The analysis identifies the customer,
  industry, product, region, work type, and sales representative segments associated with the
  strongest value creation, then evaluates statistical relationships and predictive models.</p>
  <ul>
    <li>Most valuable customer by VA: <strong>{html.escape(str(context['top_customer']))}</strong>
      ({currency(context['top_customer_va'])}).</li>
    <li>Highest-VA work type: <strong>{html.escape(str(context['top_work_type']))}</strong>.</li>
    <li>Follow-up opportunity: <strong>{context['follow_up_count']}</strong> customers due or high priority,
      representing historical VA of <strong>{currency(context['follow_up_va'])}</strong>.</li>
    <li>Best predictive model: <strong>{html.escape(str(context['best_model']))}</strong>
      with R2 of <strong>{r2_text}</strong>.</li>
  </ul>

  <h2>Key Visualisations</h2>
  {image('monthly_sales_profitability_trend')}
  {image('top_customers_by_va')}
  {image('correlation_heatmap')}
  {image('scatter_sell_price_vs_va_amount')}

  <h2>Business Rankings</h2>
  <h3>Top Customers by VA</h3>{table(business_tables.get('top_customers_by_va'), 10)}
  <h3>Top Industries by VA</h3>{table(business_tables.get('top_industries_by_va'), 10)}
  <h3>Top Product Types by VA</h3>{table(business_tables.get('top_product_types_by_va'), 10)}
  <h3>Top Work Types by VA</h3>{table(business_tables.get('top_work_types_by_va'), 10)}
  <h3>Top Sales Representatives by VA</h3>{table(business_tables.get('top_sales_representatives_by_va'), 10)}

  <h2>Customer Follow-up Opportunities</h2>
  {table(follow_up, 15)}

  <h2>Relationship and Statistical Analysis</h2>
  <h3>Requested Relationships</h3>{table(relationship_table, 12)}
  <h3>ANOVA</h3>{table(statistical_tables.get('anova_tests'), 10)}
  <h3>T-tests</h3>{table(statistical_tables.get('t_tests'), 10)}
  <h3>Chi-square Tests</h3>{table(statistical_tables.get('chi_square_tests'), 10)}

  <h2>Predictive Modelling</h2>
  {table(model_metrics, 10)}
  <h3>Feature Importance</h3>{table(feature_importance, 15)}

  <h2>Recommendations</h2>
  <ul>
    <li>Focus sales effort on high-VA customers and similar sectors where repeatable value is strongest.</li>
    <li>Review pricing for low-margin and negative-markup work before renewing comparable jobs.</li>
    <li>Target operational improvements where labour, paper, purchases, or press hours dilute VA.</li>
    <li>Use the reorder/churn table as a practical weekly follow-up queue for account managers.</li>
    <li>Extend the platform into a scheduled dashboard with refreshed data and customer-level forecasting.</li>
  </ul>
</main>
</body>
</html>"""

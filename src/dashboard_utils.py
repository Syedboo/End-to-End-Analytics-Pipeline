"""Small pure helpers used by the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd


def filter_dashboard_data(
    analysis_df: pd.DataFrame,
    selected_years: tuple[str, ...],
    selected_months: tuple[str, ...],
    include_flagged_rows: bool,
) -> pd.DataFrame:
    """Apply global dashboard filters without mutating the source data."""
    filtered = analysis_df[
        analysis_df["Year"].astype(str).isin(selected_years)
        & analysis_df["Month"].astype(str).isin(selected_months)
    ].copy()

    if not include_flagged_rows and "Quality Status" in filtered.columns:
        filtered = filtered[filtered["Quality Status"].eq("PASS")].copy()

    return filtered


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric series aligned to the input frame."""
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _weighted_margin_from_totals(va_amount: float, revenue: float) -> float:
    """Return grouped VA margin from total VA and total revenue."""
    if pd.isna(revenue) or revenue == 0:
        return float("nan")
    return float(va_amount / revenue)


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    """Return a count-aware noun phrase for executive dashboard cards."""
    noun = singular if count == 1 else (plural or f"{singular}s")
    return f"{count:,} {noun}"


def build_data_confidence_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Summarise row quality in plain board-facing language."""
    total_rows = int(len(df))
    if total_rows == 0:
        summary = pd.DataFrame(
            columns=["Quality Status", "Rows", "Share", "Board Meaning"]
        )
        return summary, {
            "score": 0.0,
            "label": "No data",
            "message": "No rows are available for the current filters.",
        }

    if "Quality Status" in df.columns:
        status = df["Quality Status"].fillna("PASS").astype(str)
    else:
        status = pd.Series("PASS", index=df.index)

    meanings = {
        "PASS": "Reliable records for executive KPIs",
        "WARNING": "Records requiring review before detailed profitability use",
        "FAIL": "Invalid records requiring correction before margin or pricing decisions",
    }
    rows = []
    for quality_status in ["PASS", "WARNING", "FAIL"]:
        count = int(status.eq(quality_status).sum())
        rows.append(
            {
                "Quality Status": quality_status,
                "Rows": count,
                "Share": count / total_rows,
                "Board Meaning": meanings[quality_status],
            }
        )
    summary = pd.DataFrame(rows)
    pass_share = float(summary.loc[summary["Quality Status"].eq("PASS"), "Share"].iloc[0])
    fail_share = float(summary.loc[summary["Quality Status"].eq("FAIL"), "Share"].iloc[0])

    if fail_share > 0.05:
        label = "Needs attention"
        message = "Several records should be reviewed before pricing or margin decisions."
    elif pass_share >= 0.90:
        label = "High confidence"
        message = "Most rows are suitable for executive KPIs."
    elif pass_share >= 0.75:
        label = "Moderate confidence"
        message = "Headline KPIs are usable, with visible exceptions to review."
    else:
        label = "Needs attention"
        message = "A large share of rows need review or missing production fields."

    return summary, {"score": pass_share, "label": label, "message": message}


def build_pricing_review_table(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Aggregate records that need pricing or margin review."""
    if df.empty:
        return pd.DataFrame()

    review = df.copy()
    sell_price = _numeric(review, "Sell Price")
    purchases = _numeric(review, "Purchases")
    va_amount = _numeric(review, "VA Amount")

    reason = review.get("Reason", pd.Series("", index=review.index)).fillna("").astype(str)
    price_below_purchase = sell_price.lt(purchases.fillna(0))
    zero_revenue = sell_price.fillna(0).eq(0)
    negative_margin = va_amount.lt(0) | reason.str.contains("Negative Margin", case=False, regex=False)
    review_mask = price_below_purchase | zero_revenue | negative_margin
    review = review.loc[review_mask].copy()
    if review.empty:
        return pd.DataFrame()

    review["_Below Purchase"] = price_below_purchase.loc[review.index].astype(int)
    review["_Zero Revenue"] = zero_revenue.loc[review.index].astype(int)
    review["_Negative Margin"] = negative_margin.loc[review.index].astype(int)

    group_cols = [
        column for column in ["Customer Name", "Product Type", "Work Type", "Rep"]
        if column in review.columns
    ]
    if not group_cols:
        return pd.DataFrame()

    grouped = (
        review.groupby(group_cols, dropna=False)
        .agg(
            Jobs=("Title", "count") if "Title" in review.columns else (group_cols[0], "size"),
            Revenue=("Sell Price", "sum"),
            VA_Amount=("VA Amount", "sum"),
            Purchase_Cost=("Purchases", "sum"),
            Below_Purchase_Jobs=("_Below Purchase", "sum"),
            Zero_Revenue_Jobs=("_Zero Revenue", "sum"),
            Negative_Margin_Jobs=("_Negative Margin", "sum"),
        )
        .reset_index()
    )
    grouped["VA_Margin"] = grouped.apply(
        lambda row: _weighted_margin_from_totals(row["VA_Amount"], row["Revenue"]),
        axis=1,
    )
    grouped["Value at Review"] = grouped["Purchase_Cost"].abs() + grouped["VA_Amount"].abs()

    def action(row: pd.Series) -> str:
        if row["Below_Purchase_Jobs"] > 0:
            return "Review quote, purchase cost, and approval route before repeating similar work."
        if row["Zero_Revenue_Jobs"] > 0:
            return "Confirm whether jobs are non-charge work, credits, or missing prices."
        return "Review pricing and production assumptions for repeat loss-making work."

    grouped["Recommended Action"] = grouped.apply(action, axis=1)
    grouped["Owner"] = "Commercial / Estimating"
    return grouped.sort_values(
        ["Value at Review", "Negative_Margin_Jobs", "Below_Purchase_Jobs"],
        ascending=[False, False, False],
    ).head(top_n)


def build_executive_focus_cards(
    df: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarise the three commercial themes executives should act on first."""
    follow_up_count = 0
    value_at_risk = 0.0
    if customer_lifecycle is not None and not customer_lifecycle.empty:
        follow_up_risks = {"Likely churn", "High risk", "Due for reorder"}
        follow_up = customer_lifecycle[
            customer_lifecycle.get("Churn Risk", pd.Series(index=customer_lifecycle.index, dtype="object"))
            .isin(follow_up_risks)
        ].copy()
        if not follow_up.empty:
            follow_up_count = int(follow_up.get("Customer Name", follow_up.index.to_series()).nunique())
            value_col = "Value at Risk" if "Value at Risk" in follow_up.columns else "Customer Lifetime VA"
            value_at_risk = pd.to_numeric(follow_up.get(value_col), errors="coerce").fillna(0).sum()

    pricing = build_pricing_review_table(df, top_n=max(len(df), 1))
    pricing_count = int(len(pricing))
    pricing_value = (
        pd.to_numeric(pricing.get("Value at Review", pd.Series(dtype=float)), errors="coerce")
        .abs()
        .fillna(0)
        .sum()
    )

    product_table = business_tables.get("product_types_profitability", pd.DataFrame())
    if not product_table.empty and {"Product Type", "Revenue", "VA_Margin"}.issubset(product_table.columns):
        low_margin_products = product_table[
            pd.to_numeric(product_table["Revenue"], errors="coerce").gt(0)
            & pd.to_numeric(product_table["VA_Margin"], errors="coerce").lt(0.25)
        ]
        product_count = int(low_margin_products["Product Type"].nunique())
        product_revenue = pd.to_numeric(low_margin_products["Revenue"], errors="coerce").fillna(0).sum()
    else:
        product_count = 0
        product_revenue = 0.0

    return pd.DataFrame(
        [
            {
                "Theme": "Customer retention",
                "Headline": f"{_plural(follow_up_count, 'customer')} {'needs' if follow_up_count == 1 else 'need'} immediate follow-up",
                "Value": value_at_risk,
                "Value Label": "potential value",
                "Detail": "Prioritise accounts whose reorder cycle is overdue or risk level is elevated.",
            },
            {
                "Theme": "Pricing review",
                "Headline": f"{_plural(pricing_count, 'customer-product combination')} {'requires' if pricing_count == 1 else 'require'} review",
                "Value": pricing_value,
                "Value Label": "value under review",
                "Detail": "Focus on zero-revenue, below-purchase, and negative-margin work.",
            },
            {
                "Theme": "Product mix",
                "Headline": f"{_plural(product_count, 'low-margin product category', 'low-margin product categories')} identified",
                "Value": product_revenue,
                "Value Label": "affected revenue",
                "Detail": "Review promotion, estimating, and pricing for low-margin product groups.",
            },
        ]
    )


def build_recommended_actions(
    df: pd.DataFrame,
    customer_lifecycle: pd.DataFrame,
    business_tables: dict[str, pd.DataFrame],
    top_n: int = 12,
) -> pd.DataFrame:
    """Create a concise board-facing action list from dashboard outputs."""
    actions: list[dict[str, object]] = []

    if customer_lifecycle is not None and not customer_lifecycle.empty:
        follow_up_risks = {"Likely churn", "High risk", "Due for reorder"}
        customers = customer_lifecycle[
            customer_lifecycle.get("Churn Risk", pd.Series(index=customer_lifecycle.index, dtype="object"))
            .isin(follow_up_risks)
        ].copy()
        if not customers.empty:
            sort_col = "Priority Rank" if "Priority Rank" in customers.columns else "Priority Score"
            ascending = sort_col == "Priority Rank"
            customers = customers.sort_values(sort_col, ascending=ascending).head(5)
            for _, row in customers.iterrows():
                value = row.get("Value at Risk", row.get("Average Annual VA", row.get("Customer Lifetime VA", 0)))
                actions.append(
                    {
                        "Priority": len(actions) + 1,
                        "Area": "Customer retention",
                        "Issue": f"{row.get('Customer Name', 'Customer')} is {str(row.get('Churn Risk', '')).lower()}",
                        "Evidence": row.get("Churn Reason", "Reorder cadence indicates follow-up is due."),
                        "Recommended Action": "Ask the account owner to contact the customer and confirm upcoming print demand.",
                        "Owner": "Sales / Account manager",
                        "Value at Stake": value,
                    }
                )

    pricing = build_pricing_review_table(df, top_n=4)
    for _, row in pricing.iterrows():
        customer = row.get("Customer Name", "Customer group")
        product = row.get("Product Type", "product mix")
        evidence = (
            f"{int(row.get('Jobs', 0))} job(s), "
            f"{int(row.get('Negative_Margin_Jobs', 0))} negative-margin, "
            f"{int(row.get('Below_Purchase_Jobs', 0))} below purchase cost."
        )
        actions.append(
            {
                "Priority": len(actions) + 1,
                "Area": "Pricing review",
                "Issue": f"Review {customer} - {product}",
                "Evidence": evidence,
                "Recommended Action": row.get("Recommended Action"),
                "Owner": row.get("Owner", "Commercial / Estimating"),
                "Value at Stake": row.get("Value at Review", 0),
            }
        )

    product_table = business_tables.get("product_types_profitability", pd.DataFrame())
    if not product_table.empty and {"Product Type", "Revenue", "VA_Margin"}.issubset(product_table.columns):
        low_margin = product_table[
            pd.to_numeric(product_table["Revenue"], errors="coerce").gt(0)
            & pd.to_numeric(product_table["VA_Margin"], errors="coerce").lt(0.25)
        ].sort_values("Revenue", ascending=False).head(3)
        for _, row in low_margin.iterrows():
            actions.append(
                {
                    "Priority": len(actions) + 1,
                    "Area": "Product mix",
                    "Issue": f"Low-margin product: {row.get('Product Type')}",
                    "Evidence": f"Weighted VA margin is {row.get('VA_Margin', 0):.1%} on revenue of GBP {row.get('Revenue', 0):,.0f}.",
                    "Recommended Action": "Review pricing, estimating assumptions, and whether this work should be promoted selectively.",
                    "Owner": "Commercial Director",
                    "Value at Stake": row.get("Revenue", 0),
                }
            )

    if "Quality Status" in df.columns:
        warning_fail = int(df["Quality Status"].isin(["WARNING", "FAIL"]).sum())
        if warning_fail:
            actions.append(
                {
                    "Priority": len(actions) + 1,
                    "Area": "Data confidence",
                    "Issue": "Improve production and pricing data capture",
                    "Evidence": f"{warning_fail:,} filtered row(s) are flagged as warning or fail.",
                    "Recommended Action": "Agree ownership for correcting missing press hours, impressions, zero revenue, and below-cost jobs.",
                    "Owner": "Operations / Finance",
                    "Value at Stake": pd.to_numeric(df.get("Sell Price", pd.Series(dtype=float)), errors="coerce").sum(),
                }
            )

    if not actions:
        return pd.DataFrame(
            columns=["Priority", "Area", "Issue", "Evidence", "Recommended Action", "Owner", "Value at Stake"]
        )

    return pd.DataFrame(actions).head(top_n)

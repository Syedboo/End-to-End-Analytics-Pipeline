"""Feature engineering for commercial printing profitability analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import safe_divide


def add_business_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create reusable financial, production, pricing, and customer lifecycle features."""
    engineered = df.copy()

    engineered["Profit"] = engineered["VA Amount"]
    engineered["Revenue"] = engineered["Sell Price"]
    engineered["Profit Margin"] = safe_divide(engineered["VA Amount"], engineered["Sell Price"])
    engineered["Markup Decimal"] = engineered["Mup%"]

    cost_columns = [col for col in ["Purchases", "Labour", "Paper", "Handling"] if col in engineered]
    engineered["Direct Cost Estimate"] = engineered[cost_columns].sum(axis=1, min_count=1)
    engineered["Contribution Estimate"] = engineered["Sell Price"] - engineered["Direct Cost Estimate"]
    engineered["Cost to Sales Ratio"] = safe_divide(
        engineered["Direct Cost Estimate"],
        engineered["Sell Price"],
    )

    engineered["Revenue per Impression"] = safe_divide(
        engineered["Sell Price"],
        engineered["Impressions"],
    )
    engineered["VA per Impression"] = safe_divide(engineered["VA Amount"], engineered["Impressions"])
    engineered["Revenue per Press Hour"] = safe_divide(
        engineered["Sell Price"],
        engineered["Press hrs"],
    )
    engineered["VA per Press Hour"] = safe_divide(engineered["VA Amount"], engineered["Press hrs"])
    engineered["Labour Share of Sales"] = safe_divide(engineered["Labour"], engineered["Sell Price"])
    engineered["Paper Share of Sales"] = safe_divide(engineered["Paper"], engineered["Sell Price"])
    engineered["Purchases Share of Sales"] = safe_divide(
        engineered["Purchases"],
        engineered["Sell Price"],
    )

    engineered["SalesIn"] = pd.to_datetime(engineered["SalesIn"], errors="coerce")
    engineered["SalesOut"] = pd.to_datetime(engineered["SalesOut"], errors="coerce")
    engineered["Ship date"] = pd.to_datetime(engineered["Ship date"], errors="coerce")

    engineered["Sales Month"] = pd.to_datetime(
        {
            "year": engineered["Year"],
            "month": engineered["Month"],
            "day": 1,
        },
        errors="coerce",
    )
    engineered["Sales Month"] = engineered["Sales Month"].fillna(
        engineered["SalesIn"].dt.to_period("M").dt.to_timestamp()
    )
    engineered["Sales Quarter"] = engineered["SalesIn"].dt.to_period("Q").astype("string")
    engineered["Invoice Lead Time Days"] = (
        engineered["SalesOut"] - engineered["SalesIn"]
    ).dt.days
    engineered["Ship Lead Time Days"] = (
        engineered["Ship date"] - engineered["SalesIn"]
    ).dt.days

    engineered["High VA Job"] = (
        engineered["VA Amount"] >= engineered["VA Amount"].median()
    ).astype(int)
    engineered["Low Margin Job"] = (
        engineered["Profit Margin"] <= engineered["Profit Margin"].quantile(0.25)
    ).astype(int)

    customer_lifecycle = build_customer_lifecycle_features(engineered)
    if not customer_lifecycle.empty:
        lifecycle_columns = [
            'CustomerID',
            'Days Since Last Order',
            'Average Reorder Days',
            'Median Reorder Days',
            'Reorder P75 Days',
            'Reorder P90 Days',
            'Max Reorder Days',
            'Reorder Cadence Days',
            'Reorder Alert Threshold Days',
            'High Risk Threshold Days',
            'Likely Churn Threshold Days',
            'Days Beyond Max Reorder Gap',
            'Days Beyond Likely Churn Threshold',
            'Predicted Next Order Date',
            'Churn Risk',
            'Churn Confidence',
            'Churn Reason',
            'Value at Risk',
            'Active Months',
            'Average Monthly VA',
            'Average Annual VA',
            'Priority Rank',
            'Priority Score',
            'Priority Score Raw',
            'Priority Score Explanation',
            'Customer Lifetime Revenue',
            'Customer Lifetime VA',
        ]
        lifecycle_columns = [
            column for column in lifecycle_columns if column in customer_lifecycle.columns
        ]
        engineered = engineered.merge(
            customer_lifecycle[lifecycle_columns],
            on='CustomerID',
            how='left',
        )

    return engineered, customer_lifecycle


def build_customer_lifecycle_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    # Estimate reorder cadence and follow-up risk by customer. Cadence is
    # based on distinct order dates because one order can create several job
    # rows on the same day. Median cadence is the normal behaviour anchor,
    # while P75, P90 and max gaps describe how late the customer can be before
    # an alert becomes a genuine churn concern.
    required = {'CustomerID', 'Customer Name', 'SalesIn', 'Sell Price', 'VA Amount'}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    orders = df.dropna(subset=['SalesIn']).copy()
    orders['SalesIn'] = pd.to_datetime(orders['SalesIn'], errors='coerce')
    orders = orders.dropna(subset=['SalesIn']).sort_values(['CustomerID', 'SalesIn'])
    if orders.empty:
        return pd.DataFrame()

    if 'Profit Margin' not in orders.columns:
        orders['Profit Margin'] = safe_divide(orders['VA Amount'], orders['Sell Price'])

    orders['_Order Date'] = orders['SalesIn'].dt.normalize()
    orders['_Order Month'] = orders['_Order Date'].dt.to_period('M').dt.to_timestamp()
    order_count_source = 'Title' if 'Title' in orders.columns else 'CustomerID'

    distinct_order_dates = orders[
        ['CustomerID', '_Order Date']
    ].drop_duplicates().sort_values(['CustomerID', '_Order Date'])

    interval_records = []
    for customer_id, values in distinct_order_dates.groupby('CustomerID')['_Order Date']:
        gaps = values.sort_values().diff().dt.days.dropna()
        interval_records.append(
            {
                'CustomerID': customer_id,
                'Average Reorder Days': gaps.mean() if not gaps.empty else np.nan,
                'Median Reorder Days': gaps.median() if not gaps.empty else np.nan,
                'Reorder P75 Days': gaps.quantile(0.75) if not gaps.empty else np.nan,
                'Reorder P90 Days': gaps.quantile(0.90) if not gaps.empty else np.nan,
                'Max Reorder Days': gaps.max() if not gaps.empty else np.nan,
                'Reorder Interval Count': int(len(gaps)),
            }
        )
    intervals = pd.DataFrame.from_records(interval_records)

    customer = (
        orders.groupby(['CustomerID', 'Customer Name'], dropna=False)
        .agg(
            Order_Count=(order_count_source, 'count'),
            Distinct_Order_Days=('_Order Date', 'nunique'),
            First_Order=('SalesIn', 'min'),
            Last_Order=('SalesIn', 'max'),
            Customer_Lifetime_Revenue=('Sell Price', 'sum'),
            Customer_Lifetime_VA=('VA Amount', 'sum'),
            Active_Months=('_Order Month', 'nunique'),
            Average_Order_Value=('Sell Price', 'mean'),
            Average_Order_VA=('VA Amount', 'mean'),
        )
        .reset_index()
    )
    customer.columns = [column.replace('_', ' ') for column in customer.columns]
    customer['Average VA Margin'] = safe_divide(
        customer['Customer Lifetime VA'],
        customer['Customer Lifetime Revenue'],
    )

    customer = customer.merge(intervals, on='CustomerID', how='left')
    customer['Average Monthly VA'] = safe_divide(
        customer['Customer Lifetime VA'],
        customer['Active Months'],
    ).fillna(0)
    customer['Average Annual VA'] = customer['Average Monthly VA'] * 12
    if reference_date is None:
        as_of_date = orders['_Order Date'].max()
    else:
        as_of_date = pd.to_datetime(reference_date, errors='coerce')
        if pd.isna(as_of_date):
            as_of_date = orders['_Order Date'].max()
        as_of_date = as_of_date.normalize()

    customer['Analysis As Of Date'] = as_of_date
    customer['Customer Tenure Days'] = (
        as_of_date - customer['First Order'].dt.normalize()
    ).dt.days.clip(lower=0).astype(float)
    customer['Days Since Last Order'] = (
        as_of_date - customer['Last Order'].dt.normalize()
    ).dt.days.clip(lower=0).astype(float)

    # Median is the normal cadence; the 30-day floor avoids over-alerting
    # customers who sometimes place multiple jobs in the same campaign week.
    customer['Reorder Cadence Days'] = customer['Median Reorder Days'].fillna(
        customer['Average Reorder Days']
    )
    customer['Expected Reorder Window Days'] = customer['Reorder Cadence Days'].apply(
        lambda value: max(float(value), 30.0) if pd.notna(value) else np.nan
    )
    customer['Reorder Alert Threshold Days'] = np.maximum(
        customer['Reorder P75 Days'].fillna(customer['Expected Reorder Window Days']),
        customer['Expected Reorder Window Days'] * 1.25,
    )
    customer['High Risk Threshold Days'] = np.maximum(
        customer['Reorder P90 Days'].fillna(customer['Expected Reorder Window Days']),
        customer['Expected Reorder Window Days'] * 2.0,
    )
    max_gap_grace_days = np.maximum(7.0, customer['Expected Reorder Window Days'] * 0.5)
    max_gap_threshold = customer['Max Reorder Days'] + max_gap_grace_days
    three_cycle_threshold = customer['Expected Reorder Window Days'] * 3.0
    customer['Likely Churn Threshold Days'] = np.minimum(
        max_gap_threshold.fillna(three_cycle_threshold),
        three_cycle_threshold,
    )
    customer['Days Beyond Max Reorder Gap'] = (
        customer['Days Since Last Order'] - customer['Max Reorder Days']
    ).clip(lower=0).fillna(0)
    customer['Days Beyond Likely Churn Threshold'] = (
        customer['Days Since Last Order'] - customer['Likely Churn Threshold Days']
    ).clip(lower=0).fillna(0)
    customer['Predicted Next Order Date'] = pd.NaT
    has_cadence = customer['Reorder Cadence Days'].notna()
    customer.loc[has_cadence, 'Predicted Next Order Date'] = (
        customer.loc[has_cadence, 'Last Order']
        + pd.to_timedelta(customer.loc[has_cadence, 'Reorder Cadence Days'], unit='D')
    )

    def confidence(row: pd.Series) -> str:
        if row['Distinct Order Days'] >= 5 and row['Customer Tenure Days'] >= 90:
            return 'High'
        if row['Distinct Order Days'] >= 3:
            return 'Medium'
        return 'Low'

    customer['Churn Confidence'] = customer.apply(confidence, axis=1)

    def classify(row: pd.Series) -> str:
        distinct_days = row['Distinct Order Days']
        days_since = row['Days Since Last Order']
        expected_window = row['Expected Reorder Window Days']
        if distinct_days < 2 or pd.isna(expected_window):
            return 'Single-order customer'
        if days_since > row['Likely Churn Threshold Days']:
            return 'Likely churn'
        if days_since > row['High Risk Threshold Days']:
            return 'High risk'
        if days_since > row['Reorder Alert Threshold Days']:
            return 'Due for reorder'
        return 'Active cadence'

    customer['Churn Risk'] = customer.apply(classify, axis=1)

    def churn_reason(row: pd.Series) -> str:
        days_since = int(round(row['Days Since Last Order']))
        if row['Churn Risk'] == 'Single-order customer':
            return 'Only one distinct order date; use onboarding or reactivation rules.'
        if row['Churn Risk'] == 'Likely churn':
            max_gap = row['Max Reorder Days']
            if pd.notna(max_gap) and row['Days Beyond Max Reorder Gap'] > 0:
                return f'No order for {days_since} days; longest historical reorder gap was {int(round(max_gap))} days.'
            threshold = row['Likely Churn Threshold Days']
            return f'No order for {days_since} days; above likely churn threshold of {int(round(threshold))} days.'
        if row['Churn Risk'] == 'High risk':
            threshold = row['High Risk Threshold Days']
            return f'No order for {days_since} days; above high-risk threshold of {int(round(threshold))} days.'
        if row['Churn Risk'] == 'Due for reorder':
            threshold = row['Reorder Alert Threshold Days']
            return f'No order for {days_since} days; above reorder alert threshold of {int(round(threshold))} days.'
        expected = row['Expected Reorder Window Days']
        return f'Within expected reorder window of {int(round(expected))} days.'

    customer['Churn Reason'] = customer.apply(churn_reason, axis=1)

    recent_window_start = as_of_date - pd.Timedelta(days=90)
    recent_orders = orders[
        (orders['_Order Date'] >= recent_window_start)
        & (orders['_Order Date'] <= as_of_date)
    ]
    recent_value = (
        recent_orders.groupby('CustomerID')
        .agg(
            Recent_90_Day_Revenue=('Sell Price', 'sum'),
            Recent_90_Day_VA=('VA Amount', 'sum'),
        )
        .reset_index()
    )
    recent_value.columns = [column.replace('_', ' ') for column in recent_value.columns]
    customer = customer.merge(recent_value, on='CustomerID', how='left')
    customer[['Recent 90 Day Revenue', 'Recent 90 Day VA']] = customer[
        ['Recent 90 Day Revenue', 'Recent 90 Day VA']
    ].fillna(0)

    risk_weights = {
        'Likely churn': 4.0,
        'High risk': 3.0,
        'Due for reorder': 2.0,
        'Active cadence': 0.25,
        'Single-order customer': 0.1,
    }
    confidence_weights = {'High': 1.0, 'Medium': 0.65, 'Low': 0.35}
    follow_up_risks = {'Likely churn', 'High risk', 'Due for reorder'}

    customer['Risk Weight'] = customer['Churn Risk'].map(risk_weights).fillna(0.1)
    customer['Confidence Weight'] = customer['Churn Confidence'].map(
        confidence_weights
    ).fillna(0.35)
    margin_multiplier = 1 + customer['Average VA Margin'].clip(lower=0, upper=1).fillna(0)
    annualised_va_exposure = customer['Average Annual VA'].fillna(0)
    customer['Value at Risk'] = np.where(
        customer['Churn Risk'].isin(follow_up_risks),
        annualised_va_exposure,
        0.0,
    )
    customer['Priority Score Raw'] = (
        customer['Value at Risk']
        * customer['Risk Weight']
        * customer['Confidence Weight']
        * margin_multiplier
    )
    max_raw_score = customer['Priority Score Raw'].max()
    if pd.notna(max_raw_score) and max_raw_score > 0:
        customer['Priority Score'] = customer['Priority Score Raw'] / max_raw_score * 100
    else:
        customer['Priority Score'] = 0.0
    customer['Priority Rank'] = (
        customer['Priority Score Raw']
        .rank(method='dense', ascending=False)
        .astype(int)
    )
    customer['Priority Score Explanation'] = (
        'Priority uses annualised VA at risk, churn-risk tier, confidence and margin; '
        'rank is relative to the current filtered customer set.'
    )

    risk_rank = {
        'Likely churn': 0,
        'High risk': 1,
        'Due for reorder': 2,
        'Active cadence': 3,
        'Single-order customer': 4,
    }
    customer['Churn Risk Rank'] = customer['Churn Risk'].map(risk_rank).fillna(9)
    customer = customer.sort_values(
        ['Priority Rank', 'Churn Risk Rank', 'Customer Lifetime VA'],
        ascending=[True, True, False],
    ).drop(columns=['Churn Risk Rank'])
    return customer


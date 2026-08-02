# Commercial Printing Analytics Project

This project analyses historical W&G Baird commercial printing jobs and turns the
raw workbook into a reusable decision-support system for profitability, pricing,
customer prioritisation, production efficiency, reorder timing, and churn follow-up.

The code is written as a refreshable pipeline: replace the input file with another
same-format workbook or CSV, rerun `main.py`, and the processed data, tables,
charts, model outputs, and report are regenerated.

## Business Questions

- Which customers, industries, regions, product types, work types, binding types,
  and sales representatives create the most value?
- How do markup, labour, paper, purchases, press hours, impressions, quantity, and
  handling relate to Value Added (VA Amount)?
- Do work types and other categorical segments differ statistically in profitability?
- Which customers are due for reorder or may need proactive follow-up?
- Which variables best predict VA Amount?

## Project Structure

```text
printing_analytics/
|-- data/
|   |-- raw/
|   |-- processed/
|-- notebooks/
|   |-- analysis.ipynb
|-- src/
|   |-- data_cleaning.py
|   |-- feature_engineering.py
|   |-- eda.py
|   |-- visualization.py
|   |-- modelling.py
|   |-- reporting.py
|   |-- utils.py
|-- outputs/
|   |-- figures/
|   |-- tables/
|   |-- reports/
|-- requirements.txt
|-- README.md
|-- main.py
```

## Setup

Python 3.12 is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

XGBoost is optional. If installed, the pipeline automatically adds it to the
model comparison:

```bash
pip install xgboost
```

## Run

The sample workbook should live at:

```text
data/raw/sample_dataset.xlsx
```

Run the full pipeline:

```bash
python main.py
```

Run against a different CSV or Excel file:

```bash
python main.py --input path/to/new_data.xlsx
```

Skip figure generation when plotting dependencies are unavailable:

```bash
python main.py --skip-figures
```

## Outputs

Key outputs are regenerated under `outputs/`:

- `outputs/reports/business_report.html` - board-ready report.
- `outputs/reports/business_report.md` - markdown report.
- `outputs/reports/data_quality_report.md` - cleaning and missingness summary.
- `outputs/tables/*.csv` - reusable tables for dissertation appendices or slide decks.
- `outputs/figures/*.png` - publication-quality static charts.
- `outputs/figures/*.html` - interactive Plotly charts.
- `data/processed/printing_jobs_cleaned.csv` - cleaned, feature-rich dataset.

## Analytical Approach

The project follows a Design Science Research Methodology framing:

1. Problem identification: commercial profitability depends on customer mix,
   pricing, product mix, and operational efficiency.
2. Objectives: create a repeatable analytics artefact that produces actionable
   insight from same-format data.
3. Design and development: build modular Python components for cleaning,
   feature engineering, EDA, visualisation, statistical testing, modelling, and
   reporting.
4. Demonstration: apply the artefact to the supplied W&G Baird sample data.
5. Evaluation: assess data quality, statistical significance, model performance,
   feature importance, and practical business usefulness.
6. Communication: generate board-level reports and presentation-ready charts.

## Notes on Profitability

The workbook defines `VA Amount` as value added from labour and markups, so the
pipeline treats VA Amount as the primary commercial profit/value proxy. True gross
profit may require extra accounting fields if material, overhead, supplier, and
rebate treatment needs to be reconciled at finance-led precision.

## Modules

- `data_cleaning.py`: loads CSV/Excel, normalises data types, fixes `Puchases` to
  `Purchases`, handles missing values, removes duplicates, and detects outliers.
- `feature_engineering.py`: creates financial, operational, margin, productivity,
  reorder, and churn-risk features.
- `eda.py`: produces descriptive statistics, rankings, trends, correlations, ANOVA,
  t-tests, and chi-square tests.
- `visualization.py`: creates distributions, boxplots, violin plots, heatmaps,
  pairplots, rankings, trend charts, scatter plots, and Plotly HTML charts.
- `modelling.py`: compares Linear Regression, Random Forest, Gradient Boosting,
  and optional XGBoost for VA Amount prediction.
- `reporting.py`: writes the markdown and HTML business report.



## Production Customer Churn Pipeline

The repository now includes a leakage-safe churn pipeline under `src/churn/`.
It is additive to the existing profitability analytics and runs automatically
from `main.py` after sell-price imputation and business-rule anomaly detection.

### Churn Definition

The source workbook has no explicit churn flag. Churn is therefore inferred from
repeat-order behaviour:

- A customer must have sufficient history and enough distinct order days to be a
  repeat-order account.
- A snapshot is created at a monthly reference date using only transactions known
  on or before that date.
- A customer is labelled churned only if its expected next-order date falls inside
  the future prediction window and no qualifying order arrives in that window.

This prevents naturally low-frequency customers from being treated as churned
just because they buy less often.

### Churn Configuration

Business assumptions are configurable in `config/churn_config.yaml`, including:

- `observation_window_days`
- `prediction_window_days`
- `gap_days`
- `min_history_days`
- `min_distinct_order_days`
- `contact_capacity`
- `contact_cost`
- `intervention_success_rate`
- `retained_value_rate`

### Churn Outputs

Running `python main.py --skip-figures` or `python main.py` produces:

- `outputs/predictions/customer_churn_predictions.csv` - ranked customer risk file.
- `outputs/reports/churn_pipeline_audit.md` - audit of the previous churn approach.
- `outputs/reports/churn_model_evaluation.md` - model metrics, thresholds, lift, calibration, and limitations.
- `outputs/reports/churn_data_quality_report.md` - churn-specific exception summary.
- `outputs/exceptions/churn_data_quality_exceptions.csv` - visible anomaly register.
- `outputs/tables/churn_customer_snapshots.csv` - leakage-safe training snapshots.
- `outputs/tables/churn_dashboard_customers.csv` - dashboard-ready customer table.
- `outputs/tables/churn_risk_by_segment.csv` - risk by industry, region, rep, value band, and frequency segment.
- `outputs/tables/churn_drivers.csv` - customer-count and revenue-at-risk by churn driver.
- `outputs/tables/churn_threshold_table.csv` - commercial threshold economics.

### Models Compared

The churn pipeline evaluates simple baselines before stronger models:

- Recency threshold baseline.
- RFM/cadence scoring baseline.
- Logistic Regression.
- Decision Tree.
- Random Forest.
- Gradient Boosting.
- Optional XGBoost when available and stable in the local environment.

The final ranking combines calibrated churn probability with customer value, urgency, and cadence thresholds including P75/P90/max historical reorder gaps. The default priority logic is:

```text
Priority Score Raw = calibrated churn probability * expected customer value * urgency factor * margin factor
Priority Score = 0-100 normalized version of Priority Score Raw
Priority Rank = rank order by Priority Score Raw
```

## Board-Facing Streamlit Dashboard

The Streamlit application in `appstreamlit.py` is designed to open with a
non-technical executive view before exposing analyst detail. Operational controls
such as file upload, report preparation, and data refresh live in sidebar `Data
Management`, so the main page starts with business results.

The dashboard tabs are:

- `Executive Summary`: six headline performance cards, a plain-English business
  headline, three action-theme cards, top priority actions, two focused charts,
  and a small data-confidence note.
- `Customers`: customer retention risk, reorder timing, value at risk, and sales
  follow-up prioritisation.
- `Overview`: pricing exceptions, monthly trends, interactive visual exploration,
  mix charts, heatmaps, and detailed business tables.
- `Operations`: work-type revenue, Value Added, volume, and production summary table.
- `Reports`: downloadable reports, technical processing details, and the full
  data-quality summary away from the executive page.

Global Year, Month, customer-risk date, and data-quality filters are applied
consistently across KPI cards, charts, customer analytics, operations views,
tables, and reports. Grouped Value Added margin is calculated as weighted margin:
`sum(VA Amount) / sum(Sell Price)`, with zero-revenue groups handled safely.


## Executive Guide

This dashboard is a decision-support tool for understanding commercial
performance in the printing business. It is not intended to replace the finance
system or the judgement of account managers. Its job is to bring the main issues
to the surface quickly: where value is being created, where margin may be leaking,
and which customers or products need attention.

For a board or senior management audience, the most useful place to start is the
`Executive Summary` tab. It answers four questions:

- How is the business performing?
- Where is the business making or losing value?
- Which customers, products, or work types need attention?
- What should management do next?

The dashboard deliberately keeps the front page simple. The more detailed tables,
model outputs, anomaly checks, and report downloads are still available, but they
sit behind the customer, pricing, operations, and reports tabs. This keeps the
main view focused on decisions rather than data processing.

A sensible way to use the dashboard in a management meeting is:

1. Start with Revenue, Value Added, Value Added Margin, Jobs, Average Order Value,
   and Customer Value at Risk.
2. Review the priority action cards for customer retention, pricing review, and
   product mix.
3. Check the trend chart to see whether performance is improving or weakening.
4. Use the customer and pricing tabs to decide which accounts or job types need
   follow-up.
5. Check the data-quality note before making final decisions on detailed margin
   numbers.

## Calculation Assumptions

The dashboard uses a consistent set of commercial assumptions so that every chart,
KPI, and table tells the same story.

### Value Added

`VA Amount` is treated as the main measure of commercial value created by a job.
In simple terms, it is the closest field in the dataset to profit contribution or
value added. It is used for customer rankings, product rankings, work-type
comparisons, margin analysis, and customer value-at-risk calculations.

This does not mean `VA Amount` is the same as statutory profit. It may not include
all overheads, finance adjustments, or management accounting treatments. For that
reason, the dashboard should be read as a commercial performance tool rather than
a final audited finance report.

### Imputed Sell Price

Some jobs have missing or placeholder Sell Price values. In the source data these
can appear as blanks, zero values, or currency placeholders such as `-`, `GBP-`,
or `GBP -`.

Where this happens, the dashboard imputes Sell Price using the business formula:

```text
Sell Price = Purchases + Rebate + VA Amount
```

Rows filled this way are not hidden. They are marked with:

```text
Sell Price Formula Imputed = True
Sell Price Source = Formula Imputed
```

This makes the calculation transparent. The row can still be included in high-level
commercial analysis, but users can see that the original Sell Price was not a clean
source value.

Some other rows may have a Sell Price that is present but still commercially
questionable, for example where Sell Price is below purchase cost. Those rows are
flagged for review rather than silently corrected.

### Weighted Value Added Margin

Grouped margin is calculated using totals, not by averaging row-level percentages.
This is important because small jobs and large jobs should not have equal weight
when measuring customer, product, industry, region, or work-type performance.

The grouped Value Added Margin is:

```text
Value Added Margin = sum(VA Amount) / sum(Sell Price)
```

For example, if a product group has `VA Amount` of `GBP 1,156` and Sell Price of
`GBP 68,350`, its margin is:

```text
1,156 / 68,350 = 1.69%
```

This is more reliable than averaging each job's `VA%`, especially when the data
contains zero-price jobs, credits, or unusually small orders.

### Flagged Records

The dashboard does not delete unusual records by default. Instead, it flags them
so the business can decide how to treat them.

Examples of flagged records include:

- missing or imputed Sell Price
- zero revenue
- Sell Price below purchase cost
- negative Value Added or negative margin
- missing impressions or press hours
- missing purchase values
- unusually high or low Value Added values

The sidebar option `Include records requiring review` controls whether these rows
are included in dashboard calculations. Leaving them included gives a full view of
the dataset. Excluding them gives a cleaner view for margin benchmarking.

The data-quality message on the executive page is therefore not a warning that the
analysis is unusable. It is a reminder that some commercial records need review
before the figures are used for detailed pricing or profitability decisions.

## Testing

Run the focused validation tests with:

```bash
python -m unittest tests.test_sell_price_imputation tests.test_board_dashboard tests.test_dashboard_utils tests.test_display_formatting tests.test_customer_lifecycle tests.test_churn_pipeline tests.test_weighted_va_margin -v
```

The tests cover Sell Price formula imputation, date-window logic, no future
leakage, churn-label creation, cold-start customers, customer priority output
columns, dashboard filters, board action helpers, display formatting, and weighted
Value Added margin calculations.
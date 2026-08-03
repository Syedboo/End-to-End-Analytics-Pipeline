# End to End Analytics
Disclaimer: This is an independent demonstration project. It is not an official product, production system, or internal pipeline of any company.
The repository does not contain confidential company information. Any sample data used for demonstration purposes is synthetic, anonymised, or provided specifically for the assessment.

Live dashboard: https://end-to-end-analytics.streamlit.app/

This dashboard helps senior leaders review commercial printing performance from an Excel workbook. It is designed to answer practical business questions: where revenue is coming from, where value is being created or lost, which customers need attention, and which pricing or production records should be reviewed.

## How To Use The Dashboard

1. Open the live dashboard: https://end-to-end-analytics.streamlit.app/ (If the app appears blank, click Manage app in the bottom-right corner, open the three-dot menu, and select Reboot app.)
2. Note: Because the dashboard is hosted on Streamlit Community Cloud, it may take a minute or two to load, especially after a period of inactivity.
3. Click the double-arrow icon (») in the top-left corner to open the sidebar.
4. In the sidebar, upload the latest Excel workbook.
5. Please Wait for the dashboard to process the file.
6. Use the Year and Month filters in the sidebar to select the required reporting period.


The upload should be an Excel file using the same type of commercial printing job structure as the sample dataset. The most important fields are customer, date, revenue, Value Added, purchases, rebate, product type, work type, industry, region, sales representative and production fields such as press hours and impressions.

## What The Dashboard Shows

The Executive Summary gives a board-level view of:

- Revenue
- Value Added
- Value Added Margin
- Jobs completed
- Average order value
- Customer value at risk
- Priority commercial actions
- Top customers by Value Added

The other tabs provide more detailed analysis of customer retention risk, pricing exceptions, product and work-type performance, operations, data quality and downloadable reports.

## Key Calculation Assumptions

### Value Added

`VA Amount` is treated as the main measure of commercial value created by a job. It is used for customer rankings, product rankings, work-type comparisons, margin analysis and customer value-at-risk calculations.

This should be read as a commercial performance measure, not as a final audited profit figure. Full statutory profit may require additional finance adjustments, overhead treatment or management accounting rules.

### Imputed Sell Price

Some records may have missing or placeholder Sell Price values, such as blanks, zero values, `-`, `GBP-` or similar currency placeholders.

Where this happens, the dashboard estimates Sell Price using:

```text
Sell Price = Purchases + Rebate + VA Amount
```

These rows are not hidden. They are flagged so users can see that the original Sell Price was not a clean source value.

### Weighted Value Added Margin

Grouped margin is calculated using totals, not by averaging individual row percentages.

```text
Value Added Margin = sum(VA Amount) / sum(Sell Price)
```

This is important because a large order should carry more weight than a small order. It gives a fairer margin for customers, products, industries, regions and work types.

### Flagged Records

The dashboard keeps unusual records visible rather than silently deleting them. Records may be flagged for reasons such as:

- missing or imputed Sell Price
- zero revenue
- Sell Price below purchase cost
- negative Value Added or negative margin
- missing impressions or press hours
- missing purchase values
- unusually high or low Value Added

A record being flagged does not automatically mean it is wrong. It means the record needs context before it is used for detailed pricing or margin decisions.

If the dashboard says, for example, that 65% of records are reliable, this does not mean the remaining 35% are unusable. It means 65% passed every validation rule without warning. The remaining records are still retained in the dashboard, but each has at least one issue that should be reviewed.

## Recommended Use In A Board Meeting

Start with the Executive Summary and focus on four questions:

1. How is the business performing?
2. Where are we making or losing value?
3. Which customers or products require attention?
4. What decisions should we take next?

Use the Reports tab when someone asks how the figures were prepared or why records were flagged.

## Important Limitations

This dashboard is a decision-support tool. It supports commercial review, customer prioritisation and pricing discussion. It should not replace the finance system, audited accounts or account-manager judgement.

For sensitive business data, check who can access the Streamlit app before uploading files.

## For Project Maintainers

Run locally with:

```bash
streamlit run streamlit_app.py
```

Run validation tests with:

```bash
python -m unittest discover -v
```

Streamlit Community Cloud entrypoint:

```text
streamlit_app.py
```

Main dashboard code:

```text
appstreamlit.py
```

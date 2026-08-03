# Commercial Printing Analytics Dashboard

This is an independent demonstration project created for the W&G Baird recruitment task. It is not an official company system. The app analyses pseudo-anonymised commercial printing job data and turns it into board-level insights on revenue, Value Added, customer retention risk, pricing exceptions, product mix and data quality.

Live dashboard: https://wg-baird-end-to-end-analytics.streamlit.app/

Created by [Syed Abuthagir S](https://www.linkedin.com/in/syed-abuthagir-s-59710b1bb/).

## For Interviewers: How To View The Dashboard

### Option 1: Use The Live Dashboard

1. Open https://wg-baird-end-to-end-analytics.streamlit.app/
2. If the app is asleep, allow one or two minutes for it to wake up.
3. Open the sidebar using the top-left sidebar control.
4. Upload an Excel workbook in the same format as the assessment dataset.
5. Use the Year and Month filters to review the required period.

If the live dashboard appears blank or takes too long to respond, please use the local fallback below. Streamlit Community Cloud can occasionally sleep, reboot or run slowly during demonstrations.

## Option 2: Run Locally From The GitHub ZIP File

No Git commands are required for this route.

1. Download the repository ZIP file:

   https://github.com/Syedboo/End-to-End-Analytics-Pipeline/archive/refs/heads/main.zip

2. Unzip the file to a normal folder, for example:

   ```text
   Downloads\End-to-End-Analytics-Pipeline-main
   ```

3. Open PowerShell in the unzipped folder.

4. Create and activate a Python environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   If `py -3.12` is not available, install Python 3.12 from https://www.python.org/downloads/ and try again.

5. Install the dashboard requirements:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   streamlit run streamlit_app.py
   ```

7. A browser window should open automatically. If it does not, copy the local URL shown in PowerShell, usually:

   ```text
   http://localhost:8501
   ```

8. Upload the Excel workbook through the sidebar and use the dashboard normally.

## What The Dashboard Shows

The Executive Summary is designed for a non-technical board audience. It focuses on:

- Revenue
- Value Added
- Weighted Value Added Margin
- Jobs completed
- Average order value
- Customer Value at Risk
- Priority commercial actions
- Top customers by Value Added

The supporting pages provide more detail on customer retention risk, pricing exceptions, product and work-type performance, operations, data quality and report outputs.

## Key Calculation Assumptions

### Value Added

`VA Amount` is treated as the main measure of commercial value created by a job. It is used for customer rankings, product rankings, work-type comparisons, margin analysis and customer value-at-risk calculations.

This should be read as a commercial performance measure, not as a final audited profit figure. Full statutory profit may require additional finance adjustments, overhead treatment or management accounting rules.

### Imputed Sell Price

Some records may contain missing or placeholder Sell Price values, such as blanks, zero values, `-`, `GBP-` or similar currency placeholders.

Where this happens, the dashboard estimates Sell Price using:

```text
Sell Price = Purchases + Rebate + VA Amount
```

These rows are not hidden. They are flagged so users can see that the original Sell Price was not a clean source value.

### Weighted Value Added Margin

Grouped margin is calculated using totals, not by averaging individual row percentages.

```text
Weighted Value Added Margin = sum(VA Amount) / sum(Sell Price)
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

A flagged record is not automatically unusable. It means the record needs context before it is used for detailed pricing or margin decisions.

## Notes For The Demonstration

The dashboard is dynamic rather than a static report. When a new Excel file with the same structure is uploaded, the data is cleaned, checked, recalculated and refreshed through the dashboard.

The main business questions it supports are:

1. How is the business performing?
2. Where is value being created or lost?
3. Which customers or products require attention?
4. What decisions should be taken next?

## Troubleshooting

If the live dashboard does not load, run it locally using the ZIP instructions above.

If local installation fails, check that:

- Python 3.12 is installed.
- PowerShell is opened inside the unzipped project folder.
- The virtual environment is activated before installing requirements.
- The command is `streamlit run streamlit_app.py`.

Streamlit entrypoint: `streamlit_app.py`

Main dashboard file: `appstreamlit.py`
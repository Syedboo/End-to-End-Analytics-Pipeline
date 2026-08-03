"""Run the end-to-end commercial printing analytics project."""

from __future__ import annotations

import argparse
import errno
import tempfile
from pathlib import Path

import pandas as pd

from src.business_rules import apply_business_rules
from src.churn.pipeline import run_churn_pipeline
from src.calculation_validation import validate_monthly_summary, validate_row_calculations
from src.data_cleaning import (
    build_data_quality_markdown,
    clean_data,
    load_field_definitions,
    load_raw_data,
)
from src.formula_discovery import discover_formulas
from src.model_validation import permutation_importance_report, shap_report, validate_model_inputs
from src.sell_price_imputation import estimate_sell_price
from src.eda import (
    build_business_tables,
    categorical_summary,
    descriptive_statistics,
    relationship_analysis,
    run_statistical_analysis,
)
from src.feature_engineering import add_business_features
from src.modelling import train_regression_models
from src.reporting import generate_business_report
from src.utils import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    OUTPUT_DIR,
    save_json,
    save_table,
)
from src.visualization import generate_all_figures


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Commercial printing analytics pipeline for W&G Baird sample data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR / "sample_dataset.xlsx",
        help="Path to a CSV or Excel workbook in the expected format.",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Excel sheet name. Defaults to 'Master Plain (Anon)' when available.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip static and interactive chart generation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where reports, tables, and figures are written.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Directory where cleaned datasets are written.",
    )
    return parser.parse_args()


def _resolve_input_path(input_path: Path) -> Path:
    """Resolve the default sample file, with a local Downloads fallback for this task."""
    if input_path.exists():
        return input_path

    fallback = Path.home() / "Downloads" / "SampleDataSet_0724261220455600546.xlsx"
    if input_path.name == "sample_dataset.xlsx" and fallback.exists():
        return fallback
    return input_path


def _prepare_runtime_paths(
    output_dir: Path,
    processed_dir: Path,
    save_outputs: bool,
) -> tuple[Path, Path]:
    """Return writable output paths for local and Snowflake Streamlit runtimes.

    Streamlit in Snowflake mounts the app source as read-only. The dashboard
    usually runs without static exports, but when report downloads are requested
    we still need a writable location. Snowflake exposes /tmp for temporary app
    files, so fall back there if the normal project folders cannot be created.
    """
    output_dir = Path(output_dir)
    processed_dir = Path(processed_dir)

    if not save_outputs:
        runtime_root = Path(tempfile.gettempdir()) / "printing_analytics_runtime"
        return runtime_root / "outputs", runtime_root / "data" / "processed"

    def required_dirs(base_output: Path, base_processed: Path) -> list[Path]:
        return [
            base_processed,
            base_output / "figures",
            base_output / "tables",
            base_output / "reports",
        ]

    try:
        for directory in required_dirs(output_dir, processed_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return output_dir, processed_dir
    except OSError as exc:
        if exc.errno not in {errno.EROFS, errno.EACCES, errno.EPERM}:
            raise

    runtime_root = Path(tempfile.gettempdir()) / "printing_analytics_runtime"
    fallback_output = runtime_root / "outputs"
    fallback_processed = runtime_root / "data" / "processed"
    for directory in required_dirs(fallback_output, fallback_processed):
        directory.mkdir(parents=True, exist_ok=True)
    return fallback_output, fallback_processed


def run_pipeline(
    input_path: Path,
    sheet_name: str | None = None,
    skip_figures: bool = False,
    output_dir: Path = OUTPUT_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    save_outputs: bool = True,
) -> dict:
    """Execute the complete analytics workflow and return paths to key outputs."""
    input_path = _resolve_input_path(input_path)
    output_dir, processed_dir = _prepare_runtime_paths(
        output_dir,
        processed_dir,
        save_outputs,
    )
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    reports_dir = output_dir / "reports"

    raw_df, metadata = load_raw_data(input_path, sheet_name=sheet_name)
    field_definitions = load_field_definitions(input_path)

    cleaned_df, clean_result = clean_data(raw_df)
    analysis_df, customer_lifecycle = add_business_features(cleaned_df)

    # Sell Price is commercially central. We preserve the source value and add
    # an estimate/confidence for missing, zero, negative, or below-purchase rows.
    sell_price_imputation = estimate_sell_price(analysis_df, fast_mode=not save_outputs)
    analysis_df = sell_price_imputation.data

    # Apply row-level business rules without deleting rows. Streamlit controls
    # whether flagged rows are included in calculations.
    analysis_df, anomaly_report = apply_business_rules(analysis_df)

    if not save_outputs:
        run_metadata = {
            "input_metadata": metadata,
            "raw_shape": metadata.get("raw_shape"),
            "processed_path": None,
            "data_quality_report": None,
            "business_report_markdown": None,
            "business_report_html": None,
            "best_model": None,
            "sell_price_imputation_model": sell_price_imputation.selected_model,
            "dashboard_fast_mode": True,
            "anomaly_report": None,
            "calculation_validation": None,
            "formula_discovery_report": None,
            "customer_churn_predictions": None,
            "churn_model_evaluation": None,
            "churn_pipeline_audit": None,
            "churn_data_quality_report": None,
            "churn_selected_model": None,
            "churn_recommended_threshold": None,
        }
        empty = pd.DataFrame()
        return {
            "metadata": run_metadata,
            "analysis_df": analysis_df,
            "cleaned_df": cleaned_df,
            "customer_lifecycle": customer_lifecycle,
            "business_tables": {},
            "descriptive": empty,
            "categorical": empty,
            "relationship_table": empty,
            "statistical_tables": {},
            "model_results": None,
            "figures": {},
            "clean_result": clean_result,
            "anomaly_report": anomaly_report,
            "sell_price_imputation": sell_price_imputation,
            "calculation_validation": empty,
            "calculation_details": empty,
            "monthly_validation": empty,
            "formula_report": empty,
            "model_validation_tables": {},
            "churn_results": None,
        }

    # Production churn runs after imputation and anomaly rules so customer
    # risk, value-at-risk, and explanations use the same data as the dashboard.
    churn_results = run_churn_pipeline(
        analysis_df,
        output_dir=output_dir,
        save_outputs=save_outputs,
    )
    processed_path = None

    if save_outputs:
        processed_path = save_table(
            analysis_df,
            processed_dir / "printing_jobs_cleaned.csv"
            )
    if not field_definitions.empty:
        if save_outputs:
            save_table(field_definitions, tables_dir / "field_definitions.csv")

    quality_markdown = build_data_quality_markdown(clean_result)
    quality_markdown += "\n## Business Rule Anomaly Framework\n\n"
    quality_markdown += "```csv\n" + anomaly_report.to_csv(index=False) + "```\n"


    quality_path = reports_dir / "data_quality_report.md"
    if save_outputs:
        quality_path.write_text(quality_markdown, encoding="utf-8")
        save_table(clean_result.missing_before, tables_dir / "missing_values_before_cleaning.csv")
        save_table(clean_result.missing_after, tables_dir / "missing_values_after_cleaning.csv")
        save_table(clean_result.outlier_report, tables_dir / "outlier_report.csv")
        save_table(anomaly_report, tables_dir / "anomaly_report.csv")

    descriptive = descriptive_statistics(analysis_df)
    categorical = categorical_summary(analysis_df)
    relationship_table = relationship_analysis(analysis_df)
    business_tables = build_business_tables(analysis_df)
    statistical_tables = run_statistical_analysis(analysis_df)
    model_results = train_regression_models(analysis_df)
    calculation_validation, calculation_details = validate_row_calculations(analysis_df)
    monthly_validation = validate_monthly_summary(
        analysis_df,
        business_tables.get("monthly_sales_profitability_trend"),
    )
    formula_report = discover_formulas(analysis_df)
    model_validation_tables = validate_model_inputs(analysis_df)
    model_validation_tables["permutation_importance"] = permutation_importance_report(analysis_df)
    model_validation_tables["shap_status"] = shap_report(analysis_df)

    if save_outputs:
        save_table(descriptive, tables_dir / "descriptive_statistics.csv")
        save_table(categorical, tables_dir / "categorical_summary.csv")
        save_table(relationship_table, tables_dir / "relationship_analysis.csv")
        save_table(customer_lifecycle, tables_dir / "customer_reorder_churn_opportunities.csv")

    if save_outputs:
        for name, table in business_tables.items():
            save_table(table, tables_dir / f"{name}.csv")
    if save_outputs:
        for name, table in statistical_tables.items():
            save_table(table, tables_dir / f"{name}.csv")

    if save_outputs:
        save_table(model_results.metrics, tables_dir / "model_performance.csv")
        save_table(model_results.feature_importance, tables_dir / "feature_importance.csv")
        save_table(model_results.diagnostics, tables_dir / "regression_diagnostics.csv")
        if not model_results.predictions.empty:
            save_table(model_results.predictions, tables_dir / "model_predictions.csv")
        save_table(sell_price_imputation.model_metrics, tables_dir / "sell_price_imputation_model_metrics.csv")
        save_table(sell_price_imputation.feature_importance, tables_dir / "sell_price_imputation_feature_importance.csv")
        save_table(calculation_validation, tables_dir / "calculation_validation_summary.csv")
        save_table(calculation_details, tables_dir / "calculation_validation_details.csv")
        save_table(monthly_validation, tables_dir / "monthly_calculation_validation.csv")
        save_table(formula_report, tables_dir / "formula_discovery_report.csv")
        for name, table in model_validation_tables.items():
            save_table(table, tables_dir / f"model_validation_{name}.csv")

    figures = {}

    if not skip_figures:
        try:

            figure_output = figures_dir if save_outputs else None

            figures = generate_all_figures(
                analysis_df,
                business_tables,
                figure_output,
            )

        except ImportError as exc:
            figures = {
                "figure_generation_warning": str(exc)
            }


    report_paths = {}

    if save_outputs:

        report_paths = generate_business_report(
            clean_result=clean_result,
            descriptive=descriptive,
            business_tables=business_tables,
            relationship_table=relationship_table,
            statistical_tables=statistical_tables,
            model_metrics=model_results.metrics,
            feature_importance=model_results.feature_importance,
            customer_lifecycle=customer_lifecycle,
            figures=figures,
            output_dir=reports_dir,
        )



    run_metadata = {
    "input_metadata": metadata,
    "processed_path": str(processed_path) if processed_path else None,
    "data_quality_report": str(quality_path) if save_outputs else None,
    "business_report_markdown": (
        str(report_paths["markdown"])
        if save_outputs else None
    ),
    "business_report_html": (
        str(report_paths["html"])
        if save_outputs else None
    ),
    "best_model": model_results.best_model_name,
    "sell_price_imputation_model": sell_price_imputation.selected_model,
    "anomaly_report": str(tables_dir / "anomaly_report.csv") if save_outputs else None,
    "calculation_validation": str(tables_dir / "calculation_validation_summary.csv") if save_outputs else None,
    "formula_discovery_report": str(tables_dir / "formula_discovery_report.csv") if save_outputs else None,
    "customer_churn_predictions": churn_results.metadata.get("customer_churn_predictions"),
    "churn_model_evaluation": churn_results.metadata.get("churn_model_evaluation"),
    "churn_pipeline_audit": churn_results.metadata.get("churn_pipeline_audit"),
    "churn_data_quality_report": churn_results.metadata.get("churn_data_quality_report"),
    "churn_selected_model": churn_results.metadata.get("selected_model"),
    "churn_recommended_threshold": churn_results.metadata.get("recommended_threshold"),
    }


    if save_outputs:
        save_json(
            run_metadata,
            reports_dir / "run_metadata.json",
        )

    return {

    # metadata
    "metadata": run_metadata,

    # primary dataframe
    "analysis_df": analysis_df,

    # cleaned dataframe
    "cleaned_df": cleaned_df,

    # lifecycle analysis
    "customer_lifecycle": customer_lifecycle,

    # business outputs
    "business_tables": business_tables,

    "descriptive": descriptive,

    "categorical": categorical,

    "relationship_table": relationship_table,

    "statistical_tables": statistical_tables,

    # model outputs
    "model_results": model_results,

    # figures
    "figures": figures,

    # data quality and validation
    "clean_result": clean_result,
    "anomaly_report": anomaly_report,
    "sell_price_imputation": sell_price_imputation,
    "calculation_validation": calculation_validation,
    "calculation_details": calculation_details,
    "monthly_validation": monthly_validation,
    "formula_report": formula_report,
    "model_validation_tables": model_validation_tables,

    # production churn outputs
    "churn_results": churn_results,
    "churn_predictions": churn_results.predictions,

    # field definitions
    "field_definitions": field_definitions,
    }


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    outputs = run_pipeline(
        args.input,
        sheet_name=args.sheet,
        skip_figures=args.skip_figures,
        output_dir=args.output_dir,
        processed_dir=args.processed_dir,
    )
    print("Analytics pipeline completed successfully.")
    print(f"Processed data: "f"{outputs['metadata']['processed_path']}")
    print(f"Business report: "f"{outputs['metadata']['business_report_html']}")
    print(f"Data quality report: "f"{outputs['metadata']['data_quality_report']}")


if __name__ == "__main__":
    main()

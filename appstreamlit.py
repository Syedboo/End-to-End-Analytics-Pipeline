"""
Streamlit Application
Commercial Printing Analytics Platform

Run using:

streamlit run app.py
"""

from src.dashboard_utils import (
    build_data_confidence_summary,
    build_executive_focus_cards,
    build_pricing_review_table,
    build_recommended_actions,
    filter_dashboard_data,
)
from src.eda import build_business_tables
from src.feature_engineering import build_customer_lifecycle_features

from pathlib import Path
from html import escape
import gc
import tempfile
import time

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

from main import run_pipeline
from src.utils import display_format_for_column, safe_divide

# ----------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------

# Prefer a repo-relative path so the logo works locally and on Streamlit Cloud.
# The absolute path is only a local fallback for this machine.
LOGO_CANDIDATES = [
    Path(__file__).parent / "assets" / "logo.png",
    Path(r"C:\Users\e16013172\WG BAIRD KTP Assignment\Printing Analytics with Streamlit\assets\logo.png"),
]
LOGO_PATH = next((p for p in LOGO_CANDIDATES if p.exists()), None)

st.set_page_config(
    page_title="Commercial Printing Analytics",
    page_icon=str(LOGO_PATH) if LOGO_PATH else ":bar_chart:",
    layout="wide",
    initial_sidebar_state="collapsed",
)




def _safe_unlink_temp_file(path: Path, attempts: int = 5, delay: float = 0.2) -> None:
    """Best-effort cleanup for uploaded temp files on Windows.

    Pandas/openpyxl can briefly leave workbook handles locked after reading.
    A locked temp file should not crash the dashboard, especially during demos.
    """
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            gc.collect()
            if attempt == attempts - 1:
                return
            time.sleep(delay)

@st.cache_data(
    show_spinner=False,
    max_entries=3,
)
def run_cached_pipeline(
    file_bytes: bytes,
    file_name: str,
    generate_export_outputs: bool,
) -> dict:
    """Run the expensive pipeline once and cache only dashboard-needed outputs."""
    suffix = Path(file_name).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_path = Path(tmp.name)

    try:
        outputs = run_pipeline(
            input_path=temp_path,
            skip_figures=not generate_export_outputs,
            save_outputs=generate_export_outputs,
        )
        return {
            "analysis_df": outputs["analysis_df"],
            "figures": outputs.get("figures", {}),
            "metadata": outputs["metadata"],
        }
    finally:
        _safe_unlink_temp_file(temp_path)




def filter_cached_analysis_data(
    analysis_data: pd.DataFrame,
    selected_years: tuple[str, ...],
    selected_months: tuple[str, ...],
    include_flagged_rows: bool,
) -> pd.DataFrame:
    """Cache filtered data so chart tabs do not repeat filter work."""
    return filter_dashboard_data(
        analysis_data,
        selected_years,
        selected_months,
        include_flagged_rows,
    )




def build_cached_dashboard_outputs(
    filtered_data: pd.DataFrame,
    reference_date: str,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Cache filtered business tables and lifecycle features for snappy reruns."""
    return (
        build_business_tables(filtered_data),
        build_customer_lifecycle_features(
            filtered_data,
            reference_date=pd.Timestamp(reference_date),
        ),
    )


# ----------------------------------------------------
# HEADER
# ----------------------------------------------------

header_logo, header_text = st.columns([0.7, 6], vertical_alignment="center")

with header_logo:
    if LOGO_PATH:
        st.image(str(LOGO_PATH))
    else:
        st.markdown("W&G Baird")

with header_text:
    st.title("Commercial Performance Dashboard")
    st.caption("Commercial performance, customer risk and priority actions")

# ----------------------------------------------------
# DATA MANAGEMENT
# ----------------------------------------------------

with st.sidebar.expander("Data Management", expanded=False):
    uploaded_file = st.file_uploader(
        "Upload Excel dataset",
        type=["xlsx", "xls", "csv"],
    )
    generate_export_outputs = st.checkbox(
        "Prepare report downloads",
        value=False,
        help=(
            "Leave this off while reviewing the dashboard. Turn it on only when "
            "you need downloadable report files or static export figures."
        ),
    )
    if st.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

if uploaded_file is None:
    st.info("Open Data Management in the sidebar to upload the latest Excel workbook.")
    st.stop()

# ----------------------------------------------------
# PREPARE DATA
# ----------------------------------------------------

uploaded_bytes = uploaded_file.getvalue()
loading_placeholder = st.empty()
loading_placeholder.markdown(
    """
    <style>
    @keyframes dashboard-spin {
        to { transform: rotate(360deg); }
    }
    .dashboard-loading {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        width: 100%;
        margin: 0.75rem 0 1rem 0;
        padding: 0.85rem 1rem;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        background: #f8fafc;
        color: #111827;
        font-size: 0.98rem;
        font-weight: 600;
    }
    .dashboard-loading-spinner {
        width: 16px;
        height: 16px;
        border: 2px solid #cbd5e1;
        border-top-color: #2563eb;
        border-radius: 999px;
        animation: dashboard-spin 0.8s linear infinite;
        flex: 0 0 auto;
    }
    </style>
    <div class="dashboard-loading">
        <span class="dashboard-loading-spinner"></span>
        <span>Preparing dashboard data...</span>
    </div>
    """,
    unsafe_allow_html=True,
)
results = run_cached_pipeline(
    uploaded_bytes,
    uploaded_file.name,
    generate_export_outputs,
)
loading_placeholder.empty()

if generate_export_outputs:
    st.sidebar.success("Report downloads prepared.")
else:
    st.sidebar.success("Data updated successfully.")

analysis_df = results["analysis_df"]
figures = results["figures"]
metadata = results["metadata"]

POUND = "\u00a3"
DASHBOARD_CHART_HEIGHT = 320
DASHBOARD_TREND_HEIGHT = 430
DASHBOARD_IMAGE_WIDTH = 920
PLOTLY_COLOR_SEQUENCE = [
    "#2563eb",
    "#16a34a",
    "#f97316",
    "#dc2626",
    "#7c3aed",
    "#0891b2",
    "#be123c",
]
DIVERGING_SCALE = "RdYlGn"
SEQUENTIAL_SCALE = "Viridis"

analysis_df["Year"] = pd.to_numeric(analysis_df["Year"], errors="coerce").astype("Int64")
analysis_df["Month"] = pd.to_numeric(analysis_df["Month"], errors="coerce").astype("Int64")

def multiselect_filter(label, series, key):
    values = sorted(series.dropna().astype(str).unique())
    selected = st.sidebar.multiselect(label, ["All"] + values, default=["All"], key=key)

    if not selected or "All" in selected:
        return values

    return selected


def previous_year_comparison_data(
    analysis_data: pd.DataFrame,
    selected_years: list[str],
    selected_months: list[str],
    include_review_records: bool,
) -> pd.DataFrame:
    """Return same-month previous-year data when the current filter is a single year."""
    try:
        year_values = [int(year) for year in selected_years]
    except (TypeError, ValueError):
        return pd.DataFrame()
    if len(year_values) != 1:
        return pd.DataFrame()

    previous_year = str(year_values[0] - 1)
    available_years = set(analysis_data["Year"].dropna().astype(str))
    if previous_year not in available_years:
        return pd.DataFrame()

    return filter_dashboard_data(
        analysis_data,
        (previous_year,),
        tuple(selected_months),
        include_review_records,
    )

st.sidebar.header("Filters")

selected_years = multiselect_filter("Year", analysis_df["Year"], "selected_years")
selected_months = multiselect_filter("Month", analysis_df["Month"], "selected_months")

include_flagged_rows = st.sidebar.checkbox("Include records requiring review", value=True)
selected_years_tuple = tuple(selected_years)
selected_months_tuple = tuple(selected_months)

filtered_df = filter_dashboard_data(
    analysis_df,
    selected_years_tuple,
    selected_months_tuple,
    include_flagged_rows,
)
comparison_df = previous_year_comparison_data(
    analysis_df,
    selected_years,
    selected_months,
    include_flagged_rows,
)

if filtered_df.empty:
    st.warning('No rows match the current filters.')
    st.stop()

valid_sales_dates = pd.to_datetime(filtered_df['SalesIn'], errors='coerce').dropna()
default_churn_as_of = (
    valid_sales_dates.max().normalize()
    if not valid_sales_dates.empty
    else pd.Timestamp.today().normalize()
)
filter_signature = (
    tuple(selected_years),
    tuple(selected_months),
    bool(include_flagged_rows),
)
if st.session_state.get('_churn_filter_signature') != filter_signature:
    st.session_state['_churn_filter_signature'] = filter_signature
    st.session_state['churn_as_of_date'] = default_churn_as_of.date()

as_of_date = st.sidebar.date_input(
    'Customer risk date',
    value=st.session_state['churn_as_of_date'],
    key='churn_as_of_date',
)

PAGE_OPTIONS = [
    "Executive Summary",
    "Overview",
    "Churn Analytics",
    "Operations",
    "Reports",
]
active_page = st.radio(
    "Dashboard section",
    PAGE_OPTIONS,
    horizontal=True,
    label_visibility="collapsed",
    key="active_dashboard_section",
)

OVERVIEW_OPTIONS = [
    "Top Performers",
    "Product Mix & Margin",
    "Pricing Review",
    "Monthly Trends",
    "Supporting Tables",
]
if active_page == "Overview":
    if st.session_state.get("overview_section") not in OVERVIEW_OPTIONS:
        st.session_state["overview_section"] = OVERVIEW_OPTIONS[0]
    overview_section = st.radio(
        "Commercial view",
        OVERVIEW_OPTIONS,
        horizontal=True,
        key="overview_section",
    )
else:
    overview_section = st.session_state.get("overview_section", OVERVIEW_OPTIONS[0])

needs_business_tables = (
    active_page == "Executive Summary"
    or (active_page == "Overview" and overview_section in {"Top Performers", "Monthly Trends", "Supporting Tables"})
)
needs_customer_lifecycle = active_page in {"Executive Summary", "Churn Analytics"}

business_tables = {}
customer_lifecycle = pd.DataFrame()
if needs_business_tables or needs_customer_lifecycle:
    with st.spinner("Updating visible section..."):
        if needs_business_tables:
            business_tables = build_business_tables(filtered_df)
        if needs_customer_lifecycle:
            customer_lifecycle = build_customer_lifecycle_features(
                filtered_df,
                reference_date=pd.Timestamp(as_of_date),
            )
filtered_sales_dates = pd.to_datetime(filtered_df.get("SalesIn"), errors="coerce").dropna()
data_updated_label = (
    filtered_sales_dates.max().strftime("%d %B %Y")
    if not filtered_sales_dates.empty
    else "date unavailable"
)
year_label = "FY" + ", ".join(selected_years) if len(selected_years) <= 2 else "All financial years"
if "Region" in filtered_df.columns:
    regions = sorted(filtered_df["Region"].dropna().astype(str).unique())
    region_label = regions[0] if len(regions) == 1 else "All regions"
else:
    region_label = "All regions"
st.caption(f"Updated {data_updated_label} | {year_label} | {region_label}")

st.sidebar.caption(
    f"{len(filtered_df):,} jobs | "
    f"Revenue {POUND}{filtered_df['Sell Price'].sum():,.0f} | "
    f"Value Added {POUND}{filtered_df['VA Amount'].sum():,.0f}"
)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def format_currency_df(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Apply business-friendly numeric formatting without changing data."""
    fmt = {}
    for col in df.columns:
        if df[col].dtype.kind not in "fi":
            continue
        fmt[col] = display_format_for_column(col, df[col].abs().max())
    return df.style.format(fmt)


def _axis_max_abs(fig, coordinate: str) -> float:
    """Infer the largest numeric value plotted on a Plotly axis."""
    values = []
    for trace in fig.data:
        raw_values = getattr(trace, coordinate, None)
        if raw_values is None:
            continue
        try:
            numeric = pd.to_numeric(pd.Series(list(raw_values)), errors="coerce")
        except TypeError:
            numeric = pd.to_numeric(pd.Series([raw_values]), errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan).dropna()
        if not numeric.empty:
            values.append(numeric.abs().max())

    return float(max(values)) if values else 0.0


def _format_plotly_business_axes(fig):
    """Apply business-friendly GBP, percentage, and count ticks to Plotly figures."""
    for axis_name, updater in (("xaxis", fig.update_xaxes), ("yaxis", fig.update_yaxes)):
        coordinate = "x" if axis_name == "xaxis" else "y"
        axis = getattr(fig.layout, axis_name, None)
        title = getattr(getattr(axis, "title", None), "text", "") or ""
        display_format = display_format_for_column(title, _axis_max_abs(fig, coordinate))

        if display_format.startswith(POUND):
            updater(
                tickprefix=POUND,
                tickformat="~s",
                exponentformat="none",
                showexponent="none",
                separatethousands=True,
            )
        elif display_format == "{:.1%}":
            updater(tickformat=".1%")
        elif display_format == "{:.1f}%":
            updater(tickformat=".1f", ticksuffix="%")
        else:
            updater(separatethousands=True, exponentformat="none", showexponent="none")

TREND_FIGURE_NAMES = {
    "monthly_sales_profitability_trend",
    "monthly_profitability_margin_trend",
    "interactive_monthly_trend",
}


def _apply_interactive_month_axis(fig):
    """Give monthly trend charts a navigable date axis and precise hover line."""
    fig.update_xaxes(
        title_text="Month",
        tickformat="%b %Y",
        nticks=8,
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(label="All", step="all"),
            ]
        ),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        automargin=True,
    )


def render_interactive_monthly_trends(tables: dict[str, pd.DataFrame]) -> None:
    """Render filtered monthly sales and margin trends as interactive Plotly charts."""
    monthly = tables.get("monthly_sales_profitability_trend")
    if monthly is None or monthly.empty:
        st.info("No monthly trend data is available for the current filters.")
        return

    monthly = monthly.copy()
    monthly["Sales Month"] = pd.to_datetime(monthly["Sales Month"], errors="coerce")
    monthly = monthly.dropna(subset=["Sales Month"]).sort_values("Sales Month")
    if monthly.empty:
        st.info("No dated monthly trend data is available for the current filters.")
        return

    trend = monthly.melt(
        id_vars=["Sales Month"],
        value_vars=["Revenue", "VA_Amount"],
        var_name="Metric",
        value_name="Amount",
    )
    trend["Metric"] = trend["Metric"].replace({"VA_Amount": "Value Added"})

    sales_fig = px.line(
        trend,
        x="Sales Month",
        y="Amount",
        color="Metric",
        markers=True,
        title="Monthly Sales Profitability Trend",
    )
    sales_fig.update_traces(
        hovertemplate=f"%{{fullData.name}}<br>%{{x|%b %Y}}<br>{POUND}%{{y:,.0f}}<extra></extra>"
    )
    sales_fig.update_yaxes(
        title_text="Amount",
        tickprefix=POUND,
        tickformat="~s",
        exponentformat="none",
        showexponent="none",
        separatethousands=True,
        zeroline=True,
    )
    _apply_interactive_month_axis(sales_fig)
    sales_fig.update_layout(
        height=DASHBOARD_TREND_HEIGHT,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=45, r=25, t=55, b=45),
        legend_title_text="Metric",
    )
    st.plotly_chart(
        sales_fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

    margin_fig = px.line(
        monthly,
        x="Sales Month",
        y="VA_Margin",
        markers=True,
        title="Monthly Profitability Margin Trend",
    )
    margin_fig.update_traces(
        line=dict(color="#b44d12"),
        marker=dict(color="#b44d12"),
        hovertemplate="VA Margin<br>%{x|%b %Y}<br>%{y:.1%}<extra></extra>",
    )
    margin_fig.update_yaxes(
        title_text="Average VA Margin",
        tickformat=".1%",
        zeroline=True,
    )
    _apply_interactive_month_axis(margin_fig)
    margin_fig.update_layout(
        height=DASHBOARD_TREND_HEIGHT,
        hovermode="x unified",
        dragmode="pan",
        margin=dict(l=45, r=25, t=55, b=45),
        showlegend=False,
    )
    st.plotly_chart(
        margin_fig,
        use_container_width=True,
        config={"scrollZoom": True, "displaylogo": False},
    )

def _style_interactive_figure(fig, height: int = 360, legend_title: str | None = None):
    """Apply a consistent compact, colorful dashboard style to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=45, r=35, t=58, b=45),
        font=dict(size=12),
        colorway=PLOTLY_COLOR_SEQUENCE,
        hoverlabel=dict(font_size=12),
    )
    if legend_title is not None:
        fig.update_layout(legend_title_text=legend_title)
    return fig


def _safe_hover_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return hover columns that exist in a frame."""
    return [column for column in columns if column in df.columns]


def _formatted_display_series(series: pd.Series, column_name: str) -> pd.Series:
    """Create short labels for bar text without changing the underlying values."""
    numeric = pd.to_numeric(series, errors="coerce")
    max_abs = numeric.abs().max() if not numeric.empty else 0
    display_format = display_format_for_column(column_name, max_abs)

    if display_format.startswith(POUND):
        return numeric.map(lambda value: "" if pd.isna(value) else f"{POUND}{value:,.0f}")
    if display_format == "{:.1%}":
        return numeric.map(lambda value: "" if pd.isna(value) else f"{value:.1%}")
    if display_format == "{:.1f}%":
        return numeric.map(lambda value: "" if pd.isna(value) else f"{value:.1f}%")
    if ".1f" in display_format:
        return numeric.map(lambda value: "" if pd.isna(value) else f"{value:,.1f}")
    return numeric.map(lambda value: "" if pd.isna(value) else f"{value:,.0f}")


def render_interactive_rankings(tables: dict[str, pd.DataFrame]) -> None:
    """Render one executive-friendly ranking chart for a selected business area."""
    ranking_specs = [
        ("Customers", "customers_profitability", "Customer Name"),
        ("Products", "product_types_profitability", "Product Type"),
        ("Industries", "industries_profitability", "Industry"),
        ("Work Types", "work_types_profitability", "Work Type"),
        ("Regions", "regions_profitability", "Region"),
        ("Sales Reps", "sales_representatives_profitability", "Rep"),
        ("Binding", "binding_types_profitability", "Binding Type"),
    ]
    available_specs = [
        spec for spec in ranking_specs
        if tables.get(spec[1]) is not None and not tables[spec[1]].empty and spec[2] in tables[spec[1]].columns
    ]
    if not available_specs:
        st.info("No ranking data is available for the current filters.")
        return

    dimension_options = [label for label, _, _ in available_specs]
    if st.session_state.get("interactive_ranking_dimension") not in dimension_options:
        st.session_state["interactive_ranking_dimension"] = dimension_options[0]
    selected_dimension = st.selectbox(
        "Business area",
        dimension_options,
        key="interactive_ranking_dimension",
    )
    label, table_name, label_col = next(
        spec for spec in available_specs if spec[0] == selected_dimension
    )
    table = tables[table_name].replace([np.inf, -np.inf], np.nan).copy()

    metric_options = {
        "Value Added": "VA_Amount",
        "Revenue": "Revenue",
        "VA Margin": "VA_Margin",
        "Average Markup": "Average_Markup",
    }
    metric_options = {
        label: column for label, column in metric_options.items()
        if column in table.columns
    }
    if not metric_options:
        st.info(f"No performance measures are available for {selected_dimension.lower()}.")
        return

    metric_labels = list(metric_options.keys())
    if st.session_state.get("interactive_ranking_metric") not in metric_labels:
        st.session_state["interactive_ranking_metric"] = metric_labels[0]
    metric_label = st.selectbox(
        "Performance measure",
        metric_labels,
        key="interactive_ranking_metric",
    )
    metric_col = metric_options[metric_label]

    plot_data = table.dropna(subset=[metric_col]).sort_values(
        [metric_col, "Revenue"] if "Revenue" in table.columns else [metric_col],
        ascending=False,
    ).head(10)
    if plot_data.empty:
        st.info(f"No {selected_dimension.lower()} ranking data is available.")
        return

    plot_data[label_col] = plot_data[label_col].fillna("Unknown").astype(str)
    plot_data["_Display Value"] = _formatted_display_series(plot_data[metric_col], metric_col)
    color_col = "VA_Margin" if "VA_Margin" in plot_data.columns else metric_col
    is_rate_color = "%" in display_format_for_column(color_col, plot_data[color_col].abs().max())

    chart_args = dict(
        data_frame=plot_data,
        x=metric_col,
        y=label_col,
        orientation="h",
        color=color_col,
        text="_Display Value",
        hover_data=_safe_hover_columns(
            plot_data,
            [
                "CustomerID",
                "Jobs",
                "Revenue",
                "VA_Amount",
                "VA_Margin",
                "Average_Markup",
                "Quantity",
                "Press_Hours",
            ],
        ),
        color_continuous_scale=DIVERGING_SCALE if is_rate_color else SEQUENTIAL_SCALE,
        title=f"Top 10 {selected_dimension} by {metric_label}",
    )
    if is_rate_color:
        chart_args["color_continuous_midpoint"] = 0

    fig = px.bar(**chart_args)
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="", autorange="reversed")
    fig.update_xaxes(title_text=metric_col.replace("_", " "))
    if is_rate_color:
        fig.update_coloraxes(colorbar_title_text=color_col.replace("_", " "), colorbar_tickformat=".0%")
    _format_plotly_business_axes(fig)
    _style_interactive_figure(fig, height=360)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

def render_relationship_explorer(df: pd.DataFrame) -> None:
    """Render an interactive scatter explorer for profitability drivers."""
    relationship_options = {
        "Sell Price vs Value Added": ("Sell Price", "VA Amount"),
        "Labour vs Value Added": ("Labour", "VA Amount"),
        "Paper vs Sell Price": ("Paper", "Sell Price"),
        "Press Hours vs Value Added": ("Press hrs", "VA Amount"),
        "Impressions vs Value Added": ("Impressions", "VA Amount"),
        "Markup vs Profit": ("Mup%", "Profit"),
        "Quantity vs Profit": ("Quantity", "Profit"),
    }
    relationship_options = {
        label: pair for label, pair in relationship_options.items()
        if pair[0] in df.columns and pair[1] in df.columns
    }
    if not relationship_options:
        st.info("No relationship chart data is available for the current filters.")
        return

    selected = st.selectbox(
        "Explore relationship",
        list(relationship_options.keys()),
        key="relationship_explorer_pair",
    )
    x_col, y_col = relationship_options[selected]
    plot_columns = [x_col, y_col] + _safe_hover_columns(
        df,
        ["Work Type", "Customer Name", "Industry", "Product Type", "Rep", "Quantity"],
    )
    plot_data = df[plot_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col]).copy()
    if len(plot_data) > 3000:
        plot_data = plot_data.sample(3000, random_state=42)

    if plot_data.empty:
        st.info("No rows are available for the selected relationship under the current filters.")
        return

    color_col = "Work Type" if "Work Type" in plot_data.columns else None
    size_col = None
    if "Quantity" in plot_data.columns and x_col != "Quantity":
        plot_data["_Quantity Size"] = pd.to_numeric(plot_data["Quantity"], errors="coerce").clip(lower=0)
        if plot_data["_Quantity Size"].max() > 0:
            size_col = "_Quantity Size"

    fig = px.scatter(
        plot_data,
        x=x_col,
        y=y_col,
        color=color_col,
        size=size_col,
        size_max=18,
        color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
        hover_data=_safe_hover_columns(
            plot_data,
            ["Customer Name", "Industry", "Product Type", "Rep", "Quantity"],
        ),
        title=selected,
    )
    fig.update_traces(marker=dict(opacity=0.72, line=dict(width=0.4, color="rgba(31,41,55,0.35)")))
    fig.update_xaxes(title_text=x_col)
    fig.update_yaxes(title_text=y_col)
    if pd.to_numeric(plot_data[x_col], errors="coerce").min() < 0 < pd.to_numeric(plot_data[x_col], errors="coerce").max():
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(75,85,99,0.7)")
    if pd.to_numeric(plot_data[y_col], errors="coerce").min() < 0 < pd.to_numeric(plot_data[y_col], errors="coerce").max():
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(75,85,99,0.7)")
    _format_plotly_business_axes(fig)
    _style_interactive_figure(fig, height=390, legend_title=color_col)
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True, "displaylogo": False})


def render_mix_and_heatmap(df: pd.DataFrame) -> None:
    """Render value mix and margin heatmap charts."""
    mix_view = st.radio(
        "Mix view",
        ["Revenue Mix", "Margin Heatmap"],
        horizontal=True,
        key="mix_margin_view",
    )

    if mix_view == "Revenue Mix":
        required = ["Work Type", "Product Type", "Sell Price", "VA Amount", "Title"]
        if not all(column in df.columns for column in required):
            st.info("Revenue mix needs work type, product type, revenue, and VA amount fields.")
        else:
            mix = (
                df.groupby(["Work Type", "Product Type"], dropna=False)
                .agg(
                    Jobs=("Title", "count"),
                    Revenue=("Sell Price", "sum"),
                    VA_Amount=("VA Amount", "sum"),
                )
                .reset_index()
            )
            mix["VA_Margin"] = safe_divide(mix["VA_Amount"], mix["Revenue"])
            mix["Work Type"] = mix["Work Type"].fillna("Unknown").astype(str)
            mix["Product Type"] = mix["Product Type"].fillna("Unknown").astype(str)
            mix["_Revenue Size"] = mix["Revenue"].clip(lower=0)
            if mix["_Revenue Size"].sum() <= 0:
                mix["_Revenue Size"] = mix["Jobs"]

            fig = px.treemap(
                mix,
                path=["Work Type", "Product Type"],
                values="_Revenue Size",
                color="VA_Margin",
                color_continuous_scale=DIVERGING_SCALE,
                color_continuous_midpoint=0,
                custom_data=["Revenue", "VA_Amount", "Jobs", "VA_Margin"],
                title="Revenue Mix by Work Type and Product Type",
            )
            fig.update_traces(
                hovertemplate=(
                    "<b>%{label}</b><br>Revenue: " + POUND + "%{customdata[0]:,.0f}"
                    "<br>Value Added: " + POUND + "%{customdata[1]:,.0f}"
                    "<br>Jobs: %{customdata[2]:,.0f}"
                    "<br>VA Margin: %{customdata[3]:.1%}<extra></extra>"
                )
            )
            fig.update_coloraxes(colorbar_title_text="VA Margin")
            _style_interactive_figure(fig, height=410)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    else:
        if "Work Type" not in df.columns or not {"Sell Price", "VA Amount"}.issubset(df.columns):
            st.info("Margin heatmap needs work type, revenue, and VA amount fields.")
        else:
            date_source = "Sales Month" if "Sales Month" in df.columns else "SalesIn"
            heat_data = df[["Work Type", "Sell Price", "VA Amount", date_source]].copy()
            heat_data["_Sales Month"] = pd.to_datetime(heat_data[date_source], errors="coerce").dt.to_period("M").dt.to_timestamp()
            heat_data = heat_data.dropna(subset=["_Sales Month"])
            if heat_data.empty:
                st.info("No monthly margin data is available for the current filters.")
            else:
                heat = (
                    heat_data.groupby(["Work Type", "_Sales Month"], dropna=False)
                    .agg(
                        Revenue=("Sell Price", "sum"),
                        VA_Amount=("VA Amount", "sum"),
                    )
                    .reset_index()
                )
                heat["VA_Margin"] = safe_divide(heat["VA_Amount"], heat["Revenue"])
                heat["Work Type"] = heat["Work Type"].fillna("Unknown").astype(str)
                matrix = heat.pivot(index="Work Type", columns="_Sales Month", values="VA_Margin")
                fig = px.imshow(
                    matrix,
                    aspect="auto",
                    color_continuous_scale=DIVERGING_SCALE,
                    labels=dict(x="Month", y="Work Type", color="VA Margin"),
                    title="VA Margin Heatmap by Work Type and Month",
                )
                fig.update_traces(
                    hovertemplate="Work Type: %{y}<br>Month: %{x|%b %Y}<br>VA Margin: %{z:.1%}<extra></extra>"
                )
                fig.update_xaxes(tickformat="%b %Y", nticks=8)
                fig.update_coloraxes(colorbar_title_text="Value Added Margin", colorbar_tickformat=".0%")
                _style_interactive_figure(fig, height=380)
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_distribution_explorer(df: pd.DataFrame) -> None:
    """Render interactive distributions with marginal summaries."""
    metrics = [
        column for column in ["Sell Price", "VA Amount", "Profit", "Mup%", "Profit Margin", "Quantity", "Press hrs"]
        if column in df.columns
    ]
    if not metrics:
        st.info("No numeric distribution metrics are available for the current filters.")
        return

    metric = st.selectbox("Distribution metric", metrics, key="distribution_metric")
    plot_columns = [metric] + _safe_hover_columns(df, ["Work Type"])
    plot_data = df[plot_columns].replace([np.inf, -np.inf], np.nan).dropna(subset=[metric]).copy()
    if len(plot_data) > 5000:
        plot_data = plot_data.sample(5000, random_state=42)
    if plot_data.empty:
        st.info("No values are available for the selected distribution metric.")
        return

    color_col = "Work Type" if "Work Type" in plot_data.columns else None
    fig = px.histogram(
        plot_data,
        x=metric,
        color=color_col,
        marginal="box",
        nbins=45,
        opacity=0.72,
        color_discrete_sequence=PLOTLY_COLOR_SEQUENCE,
        title=f"{metric} Distribution",
    )
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title_text=metric)
    fig.update_yaxes(title_text="Jobs")
    if pd.to_numeric(plot_data[metric], errors="coerce").min() < 0 < pd.to_numeric(plot_data[metric], errors="coerce").max():
        fig.add_vline(x=0, line_width=1, line_dash="dot", line_color="rgba(75,85,99,0.7)")
    _format_plotly_business_axes(fig)
    _style_interactive_figure(fig, height=390, legend_title=color_col)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_static_export_gallery(figures: dict) -> None:
    """Keep generated report figures available without stacking every PNG on screen."""
    if not figures:
        st.info("No generated export figures are available.")
        return

    figure_names = [name for name in figures.keys() if name not in TREND_FIGURE_NAMES]
    if not figure_names:
        st.info("No additional export figures are available.")
        return

    st.caption("These are the generated report/export figures. The interactive tabs are designed for dashboard exploration.")
    selected_figure = st.selectbox(
        "Select generated figure",
        figure_names,
        format_func=lambda value: value.replace("_", " ").title(),
        key="static_export_figure",
    )
    render_figure(selected_figure, figures[selected_figure])


def render_interactive_visual_gallery(df: pd.DataFrame, tables: dict[str, pd.DataFrame], figures: dict) -> None:
    """Render the two board-friendly visual exploration areas."""
    visual_section = st.radio(
        "Commercial chart",
        ["Top Performers", "Product Mix & Margin"],
        horizontal=True,
        key="visual_explorer_section",
    )

    if visual_section == "Top Performers":
        render_interactive_rankings(tables)
    else:
        render_mix_and_heatmap(df)


def _first_value(table: pd.DataFrame | None, label_col: str, value_col: str) -> tuple[str, float]:
    """Return the leading label/value pair for a board summary metric."""
    if table is None or table.empty or label_col not in table.columns or value_col not in table.columns:
        return "n/a", np.nan
    first = table.sort_values(value_col, ascending=False).iloc[0]
    return str(first[label_col]), float(first[value_col])



def _format_compact_currency(value: float | int | None) -> str:
    """Return compact executive currency labels such as GBP 1.5M."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if not np.isfinite(number):
        number = 0.0

    sign = "-" if number < 0 else ""
    absolute = abs(number)
    if absolute >= 1_000_000_000:
        return f"{sign}{POUND}{absolute / 1_000_000_000:.1f}B"
    if absolute >= 1_000_000:
        return f"{sign}{POUND}{absolute / 1_000_000:.2f}M"
    if absolute >= 1_000:
        return f"{sign}{POUND}{absolute / 1_000:.1f}K"
    return f"{sign}{POUND}{absolute:,.0f}"


def _metric_delta(current: float, previous: float | None, *, currency: bool = False, percent: bool = False) -> str | None:
    """Create a board-friendly previous-year comparison when available."""
    if previous is None or pd.isna(previous):
        return None
    try:
        previous_value = float(previous)
        current_value = float(current)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(previous_value) or previous_value == 0:
        return None

    change = current_value - previous_value
    change_pct = change / abs(previous_value)
    if percent:
        return f"{change:+.1f} pts vs prior year"
    if currency:
        return f"{_format_compact_currency(change)} ({change_pct:+.1%}) vs prior year"
    return f"{change:+,.0f} ({change_pct:+.1%}) vs prior year"


def _executive_styles() -> None:
    """Inject compact dashboard card styling for the executive page."""
    st.markdown(
        """
        <style>
        .focus-card, .priority-card {
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            background: #ffffff;
            min-height: 142px;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
        }
        .focus-card-retention { border-top: 4px solid #2563eb; }
        .focus-card-pricing { border-top: 4px solid #dc2626; }
        .focus-card-product { border-top: 4px solid #16a34a; }
        .focus-label {
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
            color: #475569;
            margin-bottom: 0.35rem;
        }
        .focus-headline { font-size: 0.98rem; font-weight: 700; color: #111827; }
        .focus-value { font-size: 1.45rem; font-weight: 800; color: #111827; margin-top: 0.35rem; }
        .focus-detail { font-size: 0.84rem; color: #475569; margin-top: 0.35rem; line-height: 1.3; }
        .priority-card { min-height: 0; margin-bottom: 0.55rem; border-left: 4px solid #f97316; }
        .priority-title { font-size: 0.96rem; font-weight: 750; color: #111827; }
        .priority-action { color: #1f2937; margin-top: 0.2rem; }
        .priority-evidence { color: #64748b; font-size: 0.82rem; margin-top: 0.25rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _quality_status_label(status: str) -> str:
    """Translate validation statuses into board-facing wording."""
    return {
        "PASS": "Reliable records",
        "WARNING": "Records requiring review",
        "FAIL": "Invalid records",
    }.get(str(status), str(status))


def _quality_issue_summary(df: pd.DataFrame) -> str:
    """Return the main business data-quality theme for the filtered records."""
    if "Reason" not in df.columns:
        return "no material issue identified"

    reason = df["Reason"].fillna("").astype(str)
    issue_map = [
        (
            "pricing and cost anomalies",
            reason.str.contains("Sell Price < Purchase Cost", case=False, regex=False)
            | reason.str.contains("Zero Revenue", case=False, regex=False)
            | reason.str.contains("Negative Margin", case=False, regex=False)
            | reason.str.contains("Missing Purchase", case=False, regex=False),
        ),
        (
            "missing production details",
            reason.str.contains("Missing impressions", case=False, regex=False)
            | reason.str.contains("Missing press hours", case=False, regex=False)
            | reason.str.contains("Production Data Missing", case=False, regex=False),
        ),
        (
            "percentage or value-added anomalies",
            reason.str.contains("Impossible", case=False, regex=False)
            | reason.str.contains("VA", case=False, regex=False),
        ),
    ]
    counts = [(label, int(mask.sum())) for label, mask in issue_map]
    counts = [item for item in counts if item[1] > 0]
    return max(counts, key=lambda item: item[1])[0] if counts else "no material issue identified"


def _render_focus_cards(cards: pd.DataFrame) -> None:
    """Render the three executive action-theme cards."""
    cols = st.columns(3)
    class_names = ["retention", "pricing", "product"]
    for index, (_, row) in enumerate(cards.iterrows()):
        with cols[index]:
            st.markdown(
                f"""
                <div class="focus-card focus-card-{class_names[index]}">
                    <div class="focus-label">{escape(str(row['Theme']))}</div>
                    <div class="focus-headline">{escape(str(row['Headline']))}</div>
                    <div class="focus-value">{_format_compact_currency(row['Value'])}</div>
                    <div class="focus-detail">{escape(str(row['Value Label']).title())}<br>{escape(str(row['Detail']))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_priority_actions(actions: pd.DataFrame, max_actions: int = 5) -> None:
    """Render the highest-priority commercial actions as compact cards."""
    st.markdown("**Priority Actions**")
    if actions.empty:
        st.success("No immediate commercial actions were generated for the current filters.")
        return

    if "Area" in actions.columns:
        balanced = actions.groupby("Area", sort=False, group_keys=False).head(1)
        remaining = actions.loc[~actions["Priority"].isin(balanced.get("Priority", pd.Series(dtype=int)))]
        display_actions = pd.concat([balanced, remaining], ignore_index=True).head(max_actions)
    else:
        display_actions = actions.head(max_actions)

    for display_rank, (_, row) in enumerate(display_actions.iterrows(), start=1):
        value = _format_compact_currency(row.get("Value at Stake", 0))
        st.markdown(
            f"""
            <div class="priority-card">
                <div class="priority-title">{display_rank}. {escape(str(row.get('Issue', 'Action required')))}</div>
                <div class="priority-action">{escape(str(row.get('Recommended Action', 'Review with the responsible owner.')))}</div>
                <div class="priority-evidence">{escape(str(row.get('Evidence', '')))} | Value at stake: {value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_executive_trend(tables: dict[str, pd.DataFrame]) -> None:
    """Show revenue and value-added trend using comparable monthly values."""
    monthly = tables.get("monthly_sales_profitability_trend", pd.DataFrame()).copy()
    required = {"Sales Month", "Revenue", "VA_Amount"}
    if monthly.empty or not required.issubset(monthly.columns):
        st.info("Monthly trend is not available for the current filters.")
        return

    monthly["Sales Month"] = pd.to_datetime(monthly["Sales Month"], errors="coerce")
    monthly = monthly.dropna(subset=["Sales Month"]).sort_values("Sales Month")
    plot_data = monthly.melt(
        id_vars="Sales Month",
        value_vars=["Revenue", "VA_Amount"],
        var_name="Metric",
        value_name="Amount",
    )
    plot_data["Metric"] = plot_data["Metric"].map({"Revenue": "Revenue", "VA_Amount": "Value Added"})
    fig = px.line(
        plot_data,
        x="Sales Month",
        y="Amount",
        color="Metric",
        markers=True,
        color_discrete_sequence=["#2563eb", "#16a34a"],
        title="Revenue and Value-Added Trend",
    )
    fig.update_yaxes(title_text="Amount")
    _format_plotly_business_axes(fig)
    _apply_interactive_month_axis(fig)
    _style_interactive_figure(fig, height=320, legend_title="Metric")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def _render_top_customer_chart(tables: dict[str, pd.DataFrame]) -> None:
    """Show the top ten customers by value added within one comparable dimension."""
    customers = tables.get("top_customers_by_va", pd.DataFrame()).copy()
    if customers.empty or not {"Customer Name", "VA_Amount"}.issubset(customers.columns):
        st.info("Customer value-added ranking is not available for the current filters.")
        return

    top_customers = customers.sort_values("VA_Amount", ascending=False).head(10).copy()
    top_customers["_Display VA"] = top_customers["VA_Amount"].map(_format_compact_currency)
    fig = px.bar(
        top_customers,
        x="VA_Amount",
        y="Customer Name",
        color="VA_Margin" if "VA_Margin" in top_customers.columns else None,
        orientation="h",
        color_continuous_scale=SEQUENTIAL_SCALE,
        hover_data=[
            column for column in ["Revenue", "VA_Margin", "Jobs", "Average_Markup"]
            if column in top_customers.columns
        ],
        text="_Display VA",
        title="Top 10 Customers by Value Added",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="", autorange="reversed")
    values = pd.to_numeric(top_customers["VA_Amount"], errors="coerce")
    max_value = values.max()
    min_value = values.min()
    if pd.notna(max_value) and max_value > 0:
        fig.update_xaxes(
            title_text="Value Added",
            range=[min(0, min_value if pd.notna(min_value) else 0), max_value * 1.18],
        )
    else:
        fig.update_xaxes(title_text="Value Added")
    fig.update_coloraxes(colorbar_title_text="Value Added Margin", colorbar_tickformat=".0%")
    _format_plotly_business_axes(fig)
    _style_interactive_figure(fig, height=430)
    fig.update_layout(margin=dict(l=45, r=100, t=58, b=45))
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_board_summary(
    df: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    lifecycle: pd.DataFrame,
    comparison_df: pd.DataFrame | None = None,
) -> None:
    """Render a concise executive summary focused on performance and action."""
    _executive_styles()

    confidence_table, confidence = build_data_confidence_summary(df)
    actions = build_recommended_actions(df, lifecycle, tables, top_n=10)
    focus_cards = build_executive_focus_cards(df, lifecycle, tables)

    total_revenue = pd.to_numeric(df.get("Sell Price", pd.Series(dtype=float)), errors="coerce").sum()
    total_value_added = pd.to_numeric(df.get("VA Amount", pd.Series(dtype=float)), errors="coerce").sum()
    total_jobs = int(len(df))
    customers = int(df["CustomerID"].nunique()) if "CustomerID" in df.columns else 0
    average_order = total_revenue / total_jobs if total_jobs else np.nan
    value_added_margin = (total_value_added / total_revenue * 100) if total_revenue else np.nan
    value_at_risk = float(focus_cards.loc[focus_cards["Theme"].eq("Customer retention"), "Value"].iloc[0])

    comparison = comparison_df if comparison_df is not None and not comparison_df.empty else None
    if comparison is not None:
        previous_revenue = pd.to_numeric(comparison.get("Sell Price", pd.Series(dtype=float)), errors="coerce").sum()
        previous_value_added = pd.to_numeric(comparison.get("VA Amount", pd.Series(dtype=float)), errors="coerce").sum()
        previous_jobs = int(len(comparison))
        previous_average_order = previous_revenue / previous_jobs if previous_jobs else np.nan
        previous_margin = (previous_value_added / previous_revenue * 100) if previous_revenue else np.nan
    else:
        previous_revenue = previous_value_added = previous_jobs = previous_average_order = previous_margin = None

    st.markdown("### Executive Summary")
    st.caption("Comparisons show the same selected months in the previous year when that data is available.")

    k1, k2, k3 = st.columns(3)
    k1.metric("Revenue", f"{POUND}{total_revenue:,.0f}", _metric_delta(total_revenue, previous_revenue, currency=True))
    k2.metric("Value Added", f"{POUND}{total_value_added:,.0f}", _metric_delta(total_value_added, previous_value_added, currency=True))
    k3.metric(
        "Value Added Margin",
        "n/a" if pd.isna(value_added_margin) else f"{value_added_margin:.1f}%",
        _metric_delta(value_added_margin, previous_margin, percent=True),
    )

    k4, k5, k6 = st.columns(3)
    k4.metric("Jobs", f"{total_jobs:,}", _metric_delta(total_jobs, previous_jobs))
    k4.caption(f"{customers:,} active customers")
    k5.metric("Average Order Value", f"{POUND}{average_order:,.0f}" if pd.notna(average_order) else "n/a", _metric_delta(average_order, previous_average_order, currency=True))
    k6.metric("Customer Value at Risk", _format_compact_currency(value_at_risk))

    invalid_records = 0
    if not confidence_table.empty:
        invalid_records = int(confidence_table.loc[confidence_table["Quality Status"].eq("FAIL"), "Rows"].sum())
    main_issue = _quality_issue_summary(df)
    margin_text = "n/a" if pd.isna(value_added_margin) else f"{value_added_margin:.1f}%"
    headline = (
        f"Revenue is {_format_compact_currency(total_revenue)} with {_format_compact_currency(total_value_added)} Value Added "
        f"({margin_text} weighted margin). "
        f"{focus_cards.loc[0, 'Headline'].capitalize()} with {_format_compact_currency(value_at_risk)} potential value, "
        f"and {invalid_records:,} records require correction before detailed profitability decisions."
    )
    st.info(headline)

    _render_focus_cards(focus_cards)
    st.write("")
    _render_priority_actions(actions, max_actions=3)

    top_customer, top_customer_va = _first_value(tables.get("top_customers_by_va"), "Customer Name", "VA_Amount")
    top_product, top_product_va = _first_value(tables.get("top_product_types_by_va"), "Product Type", "VA_Amount")
    top_work_type, top_work_type_va = _first_value(tables.get("top_work_types_by_va"), "Work Type", "VA_Amount")
    st.caption(
        "Top value sources: "
        f"Customer {top_customer} ({_format_compact_currency(top_customer_va)}), "
        f"Product {top_product} ({_format_compact_currency(top_product_va)}), "
        f"Work type {top_work_type} ({_format_compact_currency(top_work_type_va)})."
    )

    _render_top_customer_chart(tables)


def render_pricing_review_dashboard(df: pd.DataFrame) -> None:
    """Render a board-facing view of pricing and margin exceptions."""
    st.subheader("Pricing Review")

    sell_price = pd.to_numeric(df.get("Sell Price"), errors="coerce")
    purchases = pd.to_numeric(df.get("Purchases"), errors="coerce")
    va_amount = pd.to_numeric(df.get("VA Amount"), errors="coerce")
    reason = df.get("Reason", pd.Series("", index=df.index)).fillna("").astype(str)

    below_purchase = int(sell_price.lt(purchases.fillna(0)).sum())
    zero_revenue = int(sell_price.fillna(0).eq(0).sum())
    negative_margin = int((va_amount.lt(0) | reason.str.contains("Negative Margin", case=False, regex=False)).sum())
    pricing_review = build_pricing_review_table(df, top_n=20)
    value_under_review = pd.to_numeric(pricing_review.get("Value at Review", pd.Series(dtype=float)), errors="coerce").sum()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Below Purchase", f"{below_purchase:,}")
    k2.metric("Zero Revenue", f"{zero_revenue:,}")
    k3.metric("Negative Margin", f"{negative_margin:,}")
    k4.metric("Value Under Review", f"{POUND}{value_under_review:,.0f}")

    if pricing_review.empty:
        st.success("No pricing exceptions are present under the current filters.")
        return

    chart_data = pricing_review.head(12).copy()
    chart_data["Group"] = chart_data.apply(
        lambda row: f"{row.get('Customer Name', 'Customer')} | {row.get('Product Type', 'Product')}",
        axis=1,
    )
    fig = px.bar(
        chart_data,
        x="Value at Review",
        y="Group",
        color="VA_Margin",
        orientation="h",
        color_continuous_scale=DIVERGING_SCALE,
        color_continuous_midpoint=0,
        hover_data=[
            column for column in [
                "Jobs",
                "Revenue",
                "VA_Amount",
                "Purchase_Cost",
                "Below_Purchase_Jobs",
                "Zero_Revenue_Jobs",
                "Negative_Margin_Jobs",
                "Recommended Action",
            ]
            if column in chart_data.columns
        ],
        title="Pricing and Margin Exceptions by Customer/Product",
    )
    fig.update_yaxes(title_text="", autorange="reversed")
    fig.update_xaxes(title_text="Value Under Review")
    fig.update_coloraxes(colorbar_title_text="Value Added Margin", colorbar_tickformat=".0%")
    _format_plotly_business_axes(fig)
    _style_interactive_figure(fig, height=420)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.markdown("**Pricing Exception Worklist**")
    display = pricing_review.rename(
        columns={
            "VA_Amount": "Value Added",
            "Purchase_Cost": "Purchase Cost",
            "VA_Margin": "VA Margin",
            "Below_Purchase_Jobs": "Below Purchase Jobs",
            "Zero_Revenue_Jobs": "Zero Revenue Jobs",
            "Negative_Margin_Jobs": "Negative Margin Jobs",
        }
    )
    st.dataframe(format_currency_df(display), use_container_width=True)


def render_data_quality_report(df: pd.DataFrame) -> None:
    """Render detailed data-quality counts away from the executive summary."""
    confidence_table, confidence = build_data_confidence_summary(df)
    st.subheader("Data Quality Summary")
    reliable_share = float(confidence.get("score", 0.0) or 0.0)
    review_records = 0
    if not confidence_table.empty:
        review_records = int(
            confidence_table.loc[
                confidence_table["Quality Status"].isin(["WARNING", "FAIL"]),
                "Rows",
            ].sum()
        )
    review_share = max(0.0, 1.0 - reliable_share)
    main_issue = _quality_issue_summary(df)

    st.info(
        f"{reliable_share:.1%} reliable records does not mean "
        f"{review_share:.1%} of the data is unusable. It means these records "
        "passed every validation rule without warning. "
        f"The remaining {review_records:,} records are still retained in the "
        "dashboard, but each has at least one issue to review before detailed "
        "pricing or margin decisions."
    )
    st.markdown(
        f"Most review items relate to **{main_issue}**. Typical examples include "
        "missing production fields such as impressions or press hours, and "
        "pricing/cost anomalies such as zero Sell Price, imputed Sell Price, "
        "or Sell Price below purchase cost. These records may still be valid "
        "business data, but they need context."
    )

    if confidence_table.empty:
        st.info("No data-quality results are available for the current filters.")
        return

    display = confidence_table.copy()
    display["Record Category"] = display["Quality Status"].map(_quality_status_label)
    display = display[["Record Category", "Rows", "Share", "Board Meaning"]]

    fig = px.bar(
        display,
        x="Record Category",
        y="Rows",
        color="Record Category",
        color_discrete_map={
            "Reliable records": "#16a34a",
            "Records requiring review": "#f97316",
            "Invalid records": "#dc2626",
        },
        text="Rows",
        title="Records by Data Quality Category",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_yaxes(title_text="Records")
    _style_interactive_figure(fig, height=320, legend_title="Record Category")
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.dataframe(format_currency_df(display), use_container_width=True)


def render_figure(name: str, fig):
    """Render a figure regardless of whether the pipeline handed back a
    Plotly figure object, a matplotlib PNG path, or an HTML file path."""
    st.markdown(f"**{name.replace('_', ' ').title()}**")

    # Case 1: a real Plotly figure object
    if hasattr(fig, "to_plotly_json") or hasattr(fig, "update_layout"):
        _format_plotly_business_axes(fig)
        fig.update_layout(height=DASHBOARD_CHART_HEIGHT, margin=dict(l=35, r=20, t=45, b=35))
        st.plotly_chart(fig, use_container_width=True)
        return

    # Case 2: a filesystem path (string or Path) to an image / html export
    if isinstance(fig, (str, Path)):
        path = Path(fig)
        if not path.exists():
            st.warning(f"Figure file not found: {path.name}")
            return
        if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            st.image(str(path), width=DASHBOARD_IMAGE_WIDTH)
        elif path.suffix.lower() == ".html":
            html = path.read_text(encoding="utf-8")
            components.html(html, height=DASHBOARD_CHART_HEIGHT + 35, scrolling=True)
        else:
            st.info(f"Unsupported figure type: {path.name}")
        return

    # Fallback - unknown object type
    st.info("Could not render this figure.")


FOLLOW_UP_RISKS = ['Likely churn', 'High risk', 'Due for reorder']
RISK_DISPLAY_ORDER = [
    'Likely churn',
    'High risk',
    'Due for reorder',
    'Active cadence',
    'Single-order customer',
]


def sort_lifecycle_for_display(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the customer table ordered by commercial actionability first, then
    # value. This makes the tab usable as a sales follow-up worklist.
    if df is None or df.empty or 'Churn Risk' not in df.columns:
        return pd.DataFrame()

    ranked = df.copy()
    risk_rank = {risk: index for index, risk in enumerate(RISK_DISPLAY_ORDER)}
    ranked['_Risk Rank'] = ranked['Churn Risk'].map(risk_rank).fillna(99)
    sort_columns = []
    ascending = []
    if 'Priority Rank' in ranked.columns:
        sort_columns.append('Priority Rank')
        ascending.append(True)
    else:
        sort_columns.append('_Risk Rank')
        ascending.append(True)
        if 'Priority Score' in ranked.columns:
            sort_columns.append('Priority Score')
            ascending.append(False)
    if '_Risk Rank' not in sort_columns:
        sort_columns.append('_Risk Rank')
        ascending.append(True)
    if 'Customer Lifetime VA' in ranked.columns:
        sort_columns.append('Customer Lifetime VA')
        ascending.append(False)
    return ranked.sort_values(sort_columns, ascending=ascending).drop(columns=['_Risk Rank'])

# ----------------------------------------------------
# DASHBOARD SECTIONS
# ----------------------------------------------------

if active_page == "Executive Summary":
    render_board_summary(filtered_df, business_tables, customer_lifecycle, comparison_df)

elif active_page == "Overview":
    if overview_section == "Top Performers":
        st.subheader("Top Performers")
        render_interactive_rankings(business_tables)
    elif overview_section == "Product Mix & Margin":
        st.subheader("Product Mix & Margin")
        render_mix_and_heatmap(filtered_df)
    elif overview_section == "Pricing Review":
        render_pricing_review_dashboard(filtered_df)
    elif overview_section == "Monthly Trends":
        st.subheader("Monthly Trends")
        render_interactive_monthly_trends(business_tables)
    else:
        st.subheader("Supporting Tables")
        if business_tables:
            table_name = st.selectbox("Select business table", list(business_tables.keys()))
            st.dataframe(format_currency_df(business_tables[table_name]), use_container_width=True)
        else:
            st.info("No business tables are available for the current filters.")

elif active_page == "Churn Analytics":
    st.subheader('Customer Retention Risk')
    st.caption(
        f'Customer risk uses {pd.Timestamp(as_of_date):%d %b %Y} as the review date. '
        'Risk reflects distinct order dates, reorder cadence, order value, and overdue follow-up windows.'
    )

    enriched = sort_lifecycle_for_display(customer_lifecycle)

    if enriched.empty:
        st.info('No customer lifecycle data is available for the current filters.')
    else:
        available_risks = [
            risk for risk in RISK_DISPLAY_ORDER
            if risk in set(enriched['Churn Risk'].dropna())
        ]
        extra_risks = sorted(
            set(enriched['Churn Risk'].dropna()) - set(available_risks)
        )
        risk_options = available_risks + extra_risks
        selected_risks = st.multiselect(
            'Filter by customer risk',
            options=risk_options,
            default=risk_options,
        )
        enriched = enriched[enriched['Churn Risk'].isin(selected_risks)]

        if enriched.empty:
            st.info('No customers match the selected customer-risk filters.')
        else:
            follow_up = enriched[enriched['Churn Risk'].isin(FOLLOW_UP_RISKS)]
            likely_churn = int((enriched['Churn Risk'] == 'Likely churn').sum())
            high_risk = int((enriched['Churn Risk'] == 'High risk').sum())
            due_for_reorder = int((enriched['Churn Risk'] == 'Due for reorder').sum())
            value_column = 'Value at Risk' if 'Value at Risk' in follow_up.columns else 'Customer Lifetime VA'
            value_at_risk = follow_up[value_column].sum() if not follow_up.empty else 0
            historical_va = follow_up['Customer Lifetime VA'].sum() if not follow_up.empty else 0

            s1, s2, s3, s4, s5 = st.columns(5)
            s1.metric('Likely Lost', likely_churn)
            s2.metric('High Risk', high_risk)
            s3.metric('Due For Reorder', due_for_reorder)
            if not follow_up.empty:
                st.warning(
                    f'{len(follow_up)} customer(s) need follow-up, representing '
                    f'{POUND}{historical_va:,.0f} of historical VA.'
                )

                if 'Priority Score' in follow_up.columns:
                    sort_by = 'Priority Rank' if 'Priority Rank' in follow_up.columns else 'Priority Score'
                    top_priority = follow_up.sort_values(
                        sort_by, ascending=(sort_by == 'Priority Rank')
                    ).head(10)
                    hover_columns = [
                        column for column in [
                            'CustomerID',
                            'Customer Lifetime VA',
                            'Value at Risk',
                            'Priority Rank',
                            'Priority Score',
                            'Churn Confidence',
                            'Churn Reason',
                            'Days Since Last Order',
                            'Reorder Cadence Days',
                            'Reorder P90 Days',
                            'Max Reorder Days',
                            'Days Beyond Max Reorder Gap',
                        ]
                        if column in top_priority.columns
                    ]
                    fig = px.bar(
                        top_priority,
                        x='Value at Risk',
                        y='Customer Name',
                        color='Churn Risk',
                        orientation='h',
                        hover_data=hover_columns,
                        title='Top Customer Follow-Up Priorities by Value at Risk',
                    )
                    fig.update_yaxes(autorange='reversed')
                    fig.update_xaxes(
                        title_text='Estimated Value at Risk',
                        tickprefix=POUND,
                        tickformat='~s',
                        exponentformat='none',
                        showexponent='none',
                    )
                    fig.update_layout(
                        height=DASHBOARD_CHART_HEIGHT,
                        margin=dict(l=35, r=20, t=45, b=35),
                        legend_title_text='Risk Level',
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.success('No customers are currently due, high risk, or likely lost under the selected filters.')

            display_columns = [
                'Priority Rank',
                'Customer Name',
                'Churn Risk',
                'Churn Reason',
                'Last Order',
                'Days Since Last Order',
                'Predicted Next Order Date',
                'Customer Lifetime Revenue',
                'Average Monthly VA',
                'Average Annual VA',
                'Recent 90 Day VA',
                'Priority Score Explanation',
            ]
            display_labels = {
                'Priority Rank': 'Priority',
                'Customer Name': 'Customer',
                'Churn Risk': 'Risk Level',
                'Churn Reason': 'Main Risk Driver',
                'Last Order': 'Last Order Date',
                'Predicted Next Order Date': 'Expected Next Order',
                'Customer Lifetime Revenue': 'Lifetime Revenue',
                'Average Monthly VA': 'Average Monthly Value Added',
                'Average Annual VA': 'Average Annual Value Added',
                'Recent 90 Day VA': 'Recent 90-Day Value Added',
                'Priority Score Explanation': 'Recommended Attention',
            }
            display_columns = [column for column in display_columns if column in enriched.columns]
            display_df = enriched[display_columns].rename(columns=display_labels)
            st.dataframe(format_currency_df(display_df), use_container_width=True)

elif active_page == "Operations":
    st.subheader("Operations Summary")

    production_summary = (
        filtered_df
        .groupby("Work Type", as_index=False)
        .agg(
            Revenue=("Sell Price", "sum"),
            Jobs=("Title", "count"),
            VA=("VA Amount", "sum"),
            PressHours=("Press hrs", "sum"),
            Impressions=("Impressions", "sum"),
            Labour=("Labour", "sum"),
            Paper=("Paper", "sum"),
            Purchases=("Purchases", "sum"),
        )
        .rename(columns={"VA": "Value Added"})
    )

    st.dataframe(format_currency_df(production_summary), use_container_width=True)

elif active_page == "Reports":
    st.subheader("Reports")
    with st.expander("Data quality summary", expanded=True):
        render_data_quality_report(filtered_df)

    if not generate_export_outputs:
        st.info(
            "Report downloads are not prepared in this view. Turn on 'Prepare report downloads' "
            "in Data Management when you need report downloads or static export figures."
        )

    raw_shape = metadata.get("raw_shape")
    best_model = metadata.get("best_model")

    if raw_shape:
        st.caption(
            f"Processed {raw_shape[0]:,} rows across {raw_shape[1]} fields."
            + (f" Best-performing model: **{best_model}**." if best_model else "")
        )

    with st.expander("Technical processing details"):
        st.json(metadata)

    st.write("")

    col1, col2, col3 = st.columns(3)

    report_path = metadata.get("business_report_markdown")
    if report_path and Path(report_path).exists():
        with open(report_path, "rb") as f:
            col1.download_button(
                "Download Markdown Report", data=f, file_name="business_report.md"
            )

    html_path = metadata.get("business_report_html")
    if html_path and Path(html_path).exists():
        with open(html_path, "rb") as f:
            col2.download_button(
                "Download HTML Report", data=f, file_name="business_report.html"
            )

    dq_path = metadata.get("data_quality_report")
    if dq_path and Path(dq_path).exists():
        with open(dq_path, "rb") as f:
            col3.download_button(
                "Download Data Quality Report", data=f, file_name="data_quality_report.md"
            )

# ----------------------------------------------------
# FOOTER
# ----------------------------------------------------

st.divider()
st.caption(
    "Commercial Printing Analytics Dashboard | Created by "
    "[Syed Abuthagir S](https://www.linkedin.com/in/syed-abuthagir-s-59710b1bb/)"
)

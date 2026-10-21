"""Sales Analytics Dashboard - Streamlit app.

Upload a CSV / Excel / JSON sales file, validate it, clean it, filter it and explore
KPIs and interactive charts. Run locally with:  streamlit run app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
import data_utils as du

st.set_page_config(page_title="Sales Analytics Dashboard", page_icon="📊", layout="wide")


# --------------------------------------------------------------------------- #
# Cached data steps
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Reading file...")
def read_upload(file_bytes: bytes, name: str, sheet: str | None) -> pd.DataFrame:
    return du.standardize_columns(du.load_file(file_bytes, name, sheet))


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    return du.make_demo_data()


@st.cache_data(show_spinner="Cleaning data...")
def run_cleaning(raw: pd.DataFrame, strategy: str, sales_cap: float, qty_cap: int, drop_dups: bool):
    cfg = du.CleaningConfig(strategy, sales_cap, qty_cap, drop_dups)
    return du.clean_data(raw, cfg)


# --------------------------------------------------------------------------- #
# UI building blocks
# --------------------------------------------------------------------------- #
def sidebar_source() -> pd.DataFrame | None:
    """File upload (or demo data). Returns the raw DataFrame or None."""
    st.sidebar.header("1. Data")
    uploaded = st.sidebar.file_uploader("Upload CSV, Excel or JSON", type=["csv", "xlsx", "json"])
    use_demo = st.sidebar.toggle("Use demo data instead", value=False,
                                 help="A synthetic messy dataset so you can try the app without a file.")
    if uploaded is not None:
        file_bytes, sheet = uploaded.getvalue(), None
        if uploaded.name.lower().endswith(".xlsx"):
            try:
                sheets = du.excel_sheets(file_bytes)
                if len(sheets) > 1:
                    sheet = st.sidebar.selectbox("Excel sheet", sheets)
            except Exception as exc:                                  # corrupt workbook
                st.error(f"Could not open the Excel file: {exc}")
                return None
        try:
            return read_upload(file_bytes, uploaded.name, sheet)
        except Exception as exc:
            st.error(f"Could not read **{uploaded.name}**: {exc}")
            return None
    return demo_data() if use_demo else None


def sidebar_cleaning_options() -> dict:
    st.sidebar.header("2. Cleaning")
    with st.sidebar.expander("Cleaning options", expanded=False):
        strategy = st.radio("Missing numeric values", ["impute", "drop"], horizontal=True,
                            format_func=lambda s: "Impute" if s == "impute" else "Drop rows")
        sales_cap = st.number_input("Max valid Sales per line ($)", 1_000.0, 1e9, 100_000.0, step=10_000.0)
        qty_cap = st.number_input("Max valid Quantity per line", 10, 100_000, 1_000, step=100)
        drop_dups = st.checkbox("Remove duplicate rows", value=True)
    return {"strategy": strategy, "sales_cap": float(sales_cap), "qty_cap": int(qty_cap), "drop_dups": drop_dups}


def sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Interactive filters; returns the filtered DataFrame."""
    st.sidebar.header("3. Filters")
    date_range = None
    if "Order Date" in df:
        lo, hi = df["Order Date"].min().date(), df["Order Date"].max().date()
        date_range = st.sidebar.date_input("Order date range", (lo, hi), min_value=lo, max_value=hi)
        if len(date_range) != 2:           # user is mid-selection
            date_range = (lo, hi)

    selections = {}
    for col in ["Region", "Category", "Segment", "Ship Mode"]:
        if col in df:
            options = sorted(df[col].dropna().unique())
            selections[col] = st.sidebar.multiselect(col, options, default=options)
    if "State" in df:
        allowed = df[df["Region"].isin(selections["Region"])] if "Region" in selections else df
        options = sorted(allowed["State"].dropna().unique())
        selections["State"] = st.sidebar.multiselect("State", options, default=options)

    discount_range = None
    if "Discount" in df:
        discount_range = st.sidebar.slider("Discount range", 0.0, 1.0, (0.0, float(df["Discount"].max())),
                                           step=0.05, format="%.2f")
    if st.sidebar.button("Reset filters"):
        st.rerun()
    return du.apply_filters(df, date_range, selections, discount_range)


def render_kpis(df: pd.DataFrame):
    k = du.compute_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"${k['total_sales']:,.0f}")
    c2.metric("Total Profit", f"${k['total_profit']:,.0f}")
    c3.metric("Total Orders", f"{k['total_orders']:,}", help="Unique Order IDs")
    c4.metric("Quantity Sold", f"{k['total_quantity']:,}")
    c5.metric("Profit Margin", f"{k['profit_margin']:.2f}%")


def tab_dashboard(df: pd.DataFrame):
    left, right = st.columns(2)
    with left:
        freq = st.radio("Trend granularity", list(charts.FREQUENCIES), index=1, horizontal=True)
        st.plotly_chart(charts.sales_profit_trend(df, freq))
    with right:
        dims = [c for c in ["Category", "Sub-Category", "Region", "Segment", "Ship Mode", "State"] if c in df]
        c_dim, c_met = st.columns(2)
        dim = c_dim.selectbox("Group by", dims)
        metric = c_met.selectbox("Metric", ["Sales", "Profit", "Quantity"])
        st.plotly_chart(charts.category_bar(df, dim, metric))

    left, right = st.columns(2)
    if {"Region", "Category"} <= set(df.columns):
        left.plotly_chart(charts.region_treemap(df))
        right.plotly_chart(charts.margin_heatmap(df))
    if "Discount" in df:
        left.plotly_chart(charts.discount_profit_scatter(df))
    if {"Segment", "Ship Mode"} <= set(df.columns):
        right.plotly_chart(charts.segment_shipmode_bar(df))


def tab_quality(raw: pd.DataFrame, clean: pd.DataFrame, log: pd.DataFrame, report: dict):
    st.subheader("Raw data validation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{report['rows']:,}")
    c2.metric("Columns", report["columns"])
    c3.metric("Missing cells", f"{report['missing_cells']:,}")
    c4.metric("Duplicate rows", report["duplicate_rows"])
    st.caption("Column names: " + ", ".join(report["column_names"]))
    left, right = st.columns(2)
    left.markdown("**Columns and data types**")
    left.dataframe(report["schema"], hide_index=True)
    right.markdown("**Missing values** (empty cells + placeholders such as 'N/A', 'unknown')")
    right.dataframe(report["missing"], hide_index=True)
    if report["duplicate_rows"]:
        with st.expander(f"Duplicate rows ({report['duplicate_rows']})"):
            st.dataframe(report["duplicate_preview"])
    with st.expander("Raw data preview"):
        st.dataframe(raw.head(200))

    st.subheader("Cleaning log")
    st.dataframe(log, hide_index=True)
    after = du.validate_data(clean)
    a, b, c = st.columns(3)
    a.metric("Rows after cleaning", f"{after['rows']:,}", f"{after['rows'] - report['rows']:+,}")
    b.metric("Missing cells after", after["missing_cells"], f"{after['missing_cells'] - report['missing_cells']:+,}")
    c.metric("Duplicates after", after["duplicate_rows"], f"{after['duplicate_rows'] - report['duplicate_rows']:+,}")


def tab_statistics(df: pd.DataFrame):
    numeric = [c for c in ["Sales", "Quantity", "Discount", "Profit", "Ship Days"] if c in df]
    st.markdown("**Numeric summary**")
    st.dataframe(du.numeric_summary(df, numeric))
    left, right = st.columns(2)
    with left:
        st.markdown("**Outliers (IQR rule)**")
        st.dataframe(du.outlier_summary_iqr(df, numeric), hide_index=True)
        cat_cols = [c for c in du.CATEGORICAL_COLUMNS if c in df]
        st.markdown("**Categorical summary**")
        st.dataframe(du.categorical_summary(df, cat_cols), hide_index=True)
    with right:
        st.plotly_chart(charts.correlation_heatmap(df, numeric))
    st.markdown("**Performance by group**")
    group_cols = [c for c in ["Category", "Sub-Category", "Region", "State", "Segment", "Ship Mode"] if c in df]
    by = st.selectbox("Group by", group_cols, key="stats_group")
    st.dataframe(du.group_summary(df, by), hide_index=True)
    col = st.selectbox("Distribution of", numeric, key="box_col")
    st.plotly_chart(charts.distribution_box(df, col, "Category" if "Category" in df else None))


def tab_data(df: pd.DataFrame, log: pd.DataFrame):
    st.dataframe(df)
    d1, d2 = st.columns(2)
    d1.download_button("Download filtered data (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "filtered_sales_data.csv", "text/csv")
    d2.download_button("Download cleaning log (CSV)", log.to_csv(index=False).encode("utf-8"),
                       "cleaning_log.csv", "text/csv")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    st.title("📊 Sales Analytics Dashboard")
    raw = sidebar_source()
    if raw is None:
        st.info("Upload a **.csv**, **.xlsx** or **.json** sales file in the sidebar "
                "(or switch on *Use demo data*) to begin.")
        st.markdown("Expected columns: " + ", ".join(f"`{c}`" for c in du.CANONICAL_COLUMNS) +
                    ". Required: " + ", ".join(f"`{c}`" for c in du.REQUIRED_COLUMNS) + ".")
        return

    report = du.validate_data(raw)
    if report["missing_required"]:
        st.error("Missing required column(s): " + ", ".join(report["missing_required"]))
        st.dataframe(report["schema"], hide_index=True)
        return

    opts = sidebar_cleaning_options()
    try:
        clean, log = run_cleaning(raw, opts["strategy"], opts["sales_cap"], opts["qty_cap"], opts["drop_dups"])
    except Exception as exc:
        st.error(f"Cleaning failed: {exc}")
        return
    if clean.empty:
        st.error("No valid rows remain after cleaning. Check the cleaning options in the sidebar.")
        return

    filtered = sidebar_filters(clean)
    st.caption(f"Showing {len(filtered):,} of {len(clean):,} cleaned rows")
    if filtered.empty:
        st.warning("No rows match the current filters. Widen the filters in the sidebar.")
        return

    render_kpis(filtered)
    t1, t2, t3, t4 = st.tabs(["📈 Dashboard", "🧹 Data quality", "📐 Statistics", "🗂 Data"])
    with t1:
        tab_dashboard(filtered)
    with t2:
        tab_quality(raw, clean, log, report)
    with t3:
        tab_statistics(filtered)
    with t4:
        tab_data(filtered, log)


main()

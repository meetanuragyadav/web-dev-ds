"""Reusable data functions: loading, validation, cleaning, statistics, KPIs and filters.

Everything here is plain pandas/numpy (no Streamlit), so each function can be
unit-tested on its own and re-used in notebooks or scripts.
"""
from __future__ import annotations

import difflib
import io
import json
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
CANONICAL_COLUMNS = [
    "Order ID", "Order Date", "Ship Date", "Ship Mode", "Customer ID", "Segment",
    "Country", "City", "State", "Region", "Product ID", "Category", "Sub-Category",
    "Product Name", "Sales", "Quantity", "Discount", "Profit",
]
REQUIRED_COLUMNS = ["Order ID", "Order Date", "Sales", "Quantity", "Profit"]
NUMERIC_COLUMNS = ["Sales", "Quantity", "Discount", "Profit"]
DATE_COLUMNS = ["Order Date", "Ship Date"]
ID_COLUMNS = ["Order ID", "Customer ID", "Product ID"]
CATEGORICAL_COLUMNS = ["Ship Mode", "Segment", "Country", "Region", "State", "City",
                       "Category", "Sub-Category"]
SUPPORTED_EXTENSIONS = (".csv", ".xlsx", ".json")

MISSING_TOKENS = {"", "na", "n/a", "n/a?", "nan", "none", "null", "unknown", "not available",
                  "-", "--", "?", "missing", "undefined", "not-a-date"}
TEST_ROW_PATTERN = re.compile(
    r"\btest\b|testville|mystery|gadgets|express ultra|unknown segment|cust-999|prod-9999",
    re.IGNORECASE)
DATE_FORMATS = ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d",
                "%d/%m/%Y", "%d-%m-%Y", "%m-%d-%Y", "%m/%d/%Y", "%d-%b-%Y", "%d %b %Y",
                "%b %d, %Y", "%d-%B-%Y", "%B %d, %Y"]
WORD_NUMBERS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORD_NUMBERS.update({"thirty": 30, "forty": 40, "fifty": 50, "hundred": 100})


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def excel_sheets(file_bytes: bytes) -> list[str]:
    """Return the sheet names of an .xlsx workbook."""
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def load_file(file_bytes: bytes, filename: str, sheet_name: str | None = None) -> pd.DataFrame:
    """Read a CSV / XLSX / JSON file (given as bytes) into a DataFrame."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Please upload .csv, .xlsx or .json.")
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    if ext == ".csv":
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(file_bytes), encoding="latin-1")
    elif ext == ".xlsx":
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name or 0)
    else:
        data = json.loads(file_bytes.decode("utf-8-sig"))
        if isinstance(data, dict):
            list_values = [v for v in data.values() if isinstance(v, list)]
            if list_values:                      # {"data": [ {...}, {...} ]}
                data = max(list_values, key=len)
            else:                                # {"col": [..], "col2": [..]} or a single record
                data = [data] if all(not isinstance(v, (list, dict)) for v in data.values()) else data
        df = pd.DataFrame(data) if isinstance(data, dict) else pd.json_normalize(data)

    if df.empty:
        raise ValueError("The file was read successfully but contains no rows.")
    return df


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to canonical names ('sub_category ' -> 'Sub-Category')."""
    lookup = {_key(c): c for c in CANONICAL_COLUMNS}
    return df.rename(columns={c: lookup.get(_key(c), str(c).strip()) for c in df.columns})


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def _is_placeholder(series: pd.Series) -> pd.Series:
    """True where a cell is a text placeholder such as 'N/A' or 'unknown'."""
    as_text = series.astype(str).str.strip().str.lower()
    return series.notna() & as_text.isin(MISSING_TOKENS)


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Missing values per column: true NaN plus placeholder text like 'N/A'."""
    rows = []
    for col in df.columns:
        nan_count = int(df[col].isna().sum())
        placeholder = int(_is_placeholder(df[col]).sum())
        total = nan_count + placeholder
        rows.append({"Column": col, "Empty (NaN)": nan_count, "Placeholder text": placeholder,
                     "Total missing": total, "% missing": round(100 * total / max(len(df), 1), 2)})
    return pd.DataFrame(rows)


def validate_data(df: pd.DataFrame) -> dict:
    """Dimensions, column names, dtypes, missing values and duplicates of a DataFrame."""
    schema = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(t) for t in df.dtypes],
        "Non-null": df.notna().sum().values,
        "Unique values": df.nunique(dropna=True).values,
    })
    notes = []
    for col, dtype in zip(df.columns, df.dtypes):
        if col in NUMERIC_COLUMNS and not pd.api.types.is_numeric_dtype(dtype):
            notes.append("Expected numeric but contains text")
        elif col in DATE_COLUMNS and not pd.api.types.is_datetime64_any_dtype(dtype):
            notes.append("Dates stored as text (mixed formats possible)")
        else:
            notes.append("")
    schema["Notes"] = notes
    missing = missing_report(df)
    dup_mask = df.duplicated(keep=False)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "schema": schema,
        "missing": missing,
        "missing_cells": int(missing["Total missing"].sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_preview": df[dup_mask].sort_values(list(df.columns)[:2]).head(50),
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in df.columns],
    }


# --------------------------------------------------------------------------- #
# Cleaning helpers
# --------------------------------------------------------------------------- #
def to_number(value) -> float:
    """Convert messy numeric input ('1,234', 'five', '10 percent', 'N/A', 'inf') to float/NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        x = float(value)
    else:
        s = str(value).strip().lower().replace(",", "").replace("$", "")
        if s in MISSING_TOKENS:
            return np.nan
        pct = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*(?:%|percent)", s)
        word_pct = re.fullmatch(r"([a-z]+)\s*(?:%|percent)", s)
        if pct:
            x = float(pct.group(1)) / 100
        elif word_pct and word_pct.group(1) in WORD_NUMBERS:
            x = WORD_NUMBERS[word_pct.group(1)] / 100
        elif s in WORD_NUMBERS:
            x = float(WORD_NUMBERS[s])
        else:
            try:
                x = float(s)
            except ValueError:
                return np.nan
    return x if np.isfinite(x) else np.nan


def parse_date_value(value, not_before: pd.Timestamp | None = None):
    """Parse one date in any of several formats.

    If a string fits several formats (e.g. 03-12-2025), pick the candidate closest
    after ``not_before`` (used so a ship date is never earlier than its order date).
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return pd.NaT
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value)
    text = str(value).strip()
    candidates = []
    for fmt in DATE_FORMATS:
        try:
            candidates.append(pd.Timestamp(datetime.strptime(text, fmt)))
        except ValueError:
            continue
    if not candidates:
        return pd.NaT
    if not_before is not None and pd.notna(not_before):
        later = [c for c in candidates if c >= not_before]
        if later:
            return min(later)
    return candidates[0]


def parse_dates(series: pd.Series, not_before: pd.Series | None = None) -> pd.Series:
    """Vectorised wrapper around :func:`parse_date_value`."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series)
    bounds = not_before if not_before is not None else [None] * len(series)
    parsed = [parse_date_value(v, b) for v, b in zip(series, bounds)]
    return pd.Series(pd.to_datetime(parsed), index=series.index)


def canonicalize_labels(series: pd.Series, cutoff: float = 0.84) -> pd.Series:
    """Unify spelling / case variants ('consumer', 'Consumer', 'SameDay' vs 'Same Day',
    'Office supply' vs 'Office Supplies') to the most frequent spelling."""
    valid = series.dropna()
    if valid.empty:
        return series
    keys = valid.map(lambda v: _key(v))
    canon, mapping = [], {}
    for k in keys.value_counts().index:             # most frequent first
        match = difflib.get_close_matches(k, canon, n=1, cutoff=cutoff)
        mapping[k] = match[0] if match else k
        if not match:
            canon.append(k)
    groups = keys.map(mapping)
    labels = {}
    for g, raw in valid.groupby(groups):
        label = str(raw.mode().iat[0])
        labels[g] = label.title() if (label.islower() or label.isupper()) else label
    out = series.astype(object).copy()
    out.loc[valid.index] = groups.map(labels)
    return out


def _clean_text(series: pd.Series) -> pd.Series:
    """Trim/collapse whitespace and convert placeholder tokens to NaN."""
    def fix(v):
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            return np.nan
        s = re.sub(r"\s+", " ", str(v)).strip()
        return np.nan if s.lower() in MISSING_TOKENS else s
    return series.astype(object).map(fix)


def _mode_map(df: pd.DataFrame, key: str, value: str) -> pd.Series:
    sub = df.dropna(subset=[key, value])
    if sub.empty:
        return pd.Series(dtype=object)
    return sub.groupby(key)[value].agg(lambda x: x.mode().iat[0])


def fix_geography(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Repair City/State/Region conflicts using the most common mapping in the data."""
    df, changes = df.copy(), 0
    if {"City", "State"} <= set(df.columns):
        state_city = _mode_map(df, "State", "City")
        miss = df["City"].isna() & df["State"].notna()
        df.loc[miss, "City"] = df.loc[miss, "State"].map(state_city)
        changes += int(miss.sum())
        city_state = _mode_map(df, "City", "State")
        fixed = df["City"].map(city_state)
        diff = fixed.notna() & (fixed != df["State"])
        df.loc[diff, "State"] = fixed[diff]
        changes += int(diff.sum())
    if {"State", "Region"} <= set(df.columns):
        state_region = _mode_map(df, "State", "Region")
        fixed = df["State"].map(state_region)
        diff = fixed.notna() & (fixed != df["Region"])
        df.loc[diff, "Region"] = fixed[diff]
        changes += int(diff.sum())
    return df, changes


# --------------------------------------------------------------------------- #
# Main cleaning pipeline
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CleaningConfig:
    missing_strategy: str = "impute"      # "impute" or "drop"
    sales_cap: float = 100_000            # sales above this are treated as data errors
    quantity_cap: int = 1_000             # quantity above this is treated as a data error
    drop_duplicates: bool = True


def clean_data(raw: pd.DataFrame, config: CleaningConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean a raw sales table. Returns (clean_df, cleaning_log_df)."""
    cfg = config or CleaningConfig()
    log: list[dict] = []

    def note(step: str, detail: str, n: int | None = None):
        log.append({"Step": step, "Detail": detail, "Affected": n})

    df = standardize_columns(raw).copy().reset_index(drop=True)
    cols = set(df.columns)
    note("Start", f"{len(df)} rows x {df.shape[1]} columns loaded", None)

    # 1. Text columns: whitespace + placeholders -> NaN, IDs upper-cased
    text_cols = [c for c in df.columns if c not in NUMERIC_COLUMNS + DATE_COLUMNS]
    before_na = int(df[text_cols].isna().sum().sum()) if text_cols else 0
    for c in text_cols:
        df[c] = _clean_text(df[c])
        if c in ID_COLUMNS:
            df[c] = df[c].str.upper()
    after_na = int(df[text_cols].isna().sum().sum()) if text_cols else 0
    note("Placeholders to missing", "Trimmed whitespace; 'N/A', 'unknown', '' etc. became missing", after_na - before_na)

    # 2. Remove obvious test / junk rows
    joined = df.apply(lambda r: " ".join(map(str, r.values)), axis=1)
    junk = joined.str.contains(TEST_ROW_PATTERN)
    if junk.any():
        df = df[~junk].reset_index(drop=True)
    note("Remove test/junk rows", "Rows with markers such as 'Testville', 'PROD-9999', 'CUST-999'", int(junk.sum()))

    # 3. Inconsistent labels (case, spacing, typos)
    changed = 0
    for c in [c for c in CATEGORICAL_COLUMNS if c in cols]:
        before = df[c].copy()
        df[c] = canonicalize_labels(df[c])
        changed += int(((before != df[c]) & before.notna()).sum())
    note("Standardise labels", "Unified case/spelling in Ship Mode, Segment, Region, Category, etc.", changed)

    # 4. Geography consistency
    df, geo_changes = fix_geography(df)
    note("Fix geography", "Repaired City/State/Region conflicts using the dominant mapping", geo_changes)

    # 5. Dates
    if "Order Date" in cols:
        raw_od = df["Order Date"]
        df["Order Date"] = parse_dates(raw_od)
        note("Parse Order Date", "Mixed formats (ISO, dd/mm/yyyy, dd-Mon-yyyy...) converted to datetime",
             int((raw_od.notna() & df["Order Date"].notna() & (raw_od.astype(str) != df["Order Date"].dt.strftime("%Y-%m-%d"))).sum()))
        bad = df["Order Date"].isna()
        df = df[~bad].reset_index(drop=True)
        note("Drop invalid Order Date", "Rows whose order date is missing or unparseable", int(bad.sum()))
    if "Ship Date" in cols:
        raw_sd = df["Ship Date"]
        df["Ship Date"] = parse_dates(raw_sd, df["Order Date"] if "Order Date" in df else None)
        before_bad = int(df["Ship Date"].isna().sum())
        early = (df["Ship Date"] < df["Order Date"]) if "Order Date" in df else pd.Series(False, index=df.index)
        df.loc[early, "Ship Date"] = pd.NaT
        note("Fix Ship Date", "Ship date before order date or unparseable set to missing", int(early.sum()) + before_bad)

    # 6. Numeric columns
    for c in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        before = df[c].copy()
        df[c] = df[c].map(to_number)
        coerced = int((before.notna() & pd.to_numeric(before, errors="coerce").isna()).sum())
        note(f"Convert {c}", "Non-numeric text converted ('five' -> 5, 'ten percent' -> 0.1) or set to missing ('N/A', 'inf')", coerced)
    invalid = 0
    if "Sales" in df:
        bad = (df["Sales"] <= 0) | (df["Sales"] > cfg.sales_cap)
        df.loc[bad, "Sales"] = np.nan; invalid += int(bad.sum())
    if "Quantity" in df:
        df["Quantity"] = df["Quantity"].round()
        bad = (df["Quantity"] < 1) | (df["Quantity"] > cfg.quantity_cap)
        df.loc[bad, "Quantity"] = np.nan; invalid += int(bad.sum())
    if "Discount" in df:
        bad = (df["Discount"] < 0) | (df["Discount"] > 1)
        df.loc[bad, "Discount"] = np.nan; invalid += int(bad.sum())
    if "Profit" in df and "Sales" in df:
        bad = df["Profit"] > df["Sales"]
        df.loc[bad, "Profit"] = np.nan; invalid += int(bad.sum())
    note("Invalid values", f"Negative/zero sales, sales > {cfg.sales_cap:,.0f}, quantity outside 1-{cfg.quantity_cap}, "
         "discount outside 0-1, profit > sales set to missing", invalid)

    # 7. Duplicates (full-row; the same Order ID may legitimately repeat for different line items)
    if cfg.drop_duplicates:
        dups = df.duplicated()
        df = df[~dups].reset_index(drop=True)
        note("Remove duplicates", "Exact duplicate rows removed (first kept)", int(dups.sum()))

    # 8. Missing values
    df, miss_log = handle_missing(df, cfg.missing_strategy)
    log.extend(miss_log)

    # 9. Derived column + final types
    if {"Ship Date", "Order Date"} <= set(df.columns):
        df["Ship Days"] = (df["Ship Date"] - df["Order Date"]).dt.days
    if "Quantity" in df:
        df["Quantity"] = df["Quantity"].round().fillna(0).astype(int)
    df = df.sort_values("Order Date").reset_index(drop=True)
    note("Done", f"{len(df)} clean rows x {df.shape[1]} columns", None)
    return df, pd.DataFrame(log)


def handle_missing(df: pd.DataFrame, strategy: str = "impute") -> tuple[pd.DataFrame, list[dict]]:
    """Impute (group medians / modes) or drop rows with missing values."""
    df, log = df.copy(), []
    by = "Sub-Category" if "Sub-Category" in df else None

    # Categorical / identifier columns
    if {"Category", "Sub-Category"} <= set(df.columns):
        n = int(df["Category"].isna().sum())
        df["Category"] = df["Category"].fillna(df["Sub-Category"].map(_mode_map(df, "Sub-Category", "Category")))
        log.append({"Step": "Fill Category", "Detail": "Filled from the Sub-Category's most common Category",
                    "Affected": n - int(df["Category"].isna().sum())})
    if {"Sub-Category", "Product Name"} <= set(df.columns):
        n = int(df["Product Name"].isna().sum())
        df["Product Name"] = df["Product Name"].fillna(df["Sub-Category"].map(_mode_map(df, "Sub-Category", "Product Name")))
        log.append({"Step": "Fill Product Name", "Detail": "Filled from the Sub-Category's most common product",
                    "Affected": n - int(df["Product Name"].isna().sum())})
    filled = 0
    for c in [c for c in CATEGORICAL_COLUMNS + ["Customer ID", "Product ID", "Product Name"] if c in df.columns]:
        n = int(df[c].isna().sum())
        df[c] = df[c].fillna("Unknown"); filled += n
    log.append({"Step": "Fill labels", "Detail": "Remaining missing text fields set to 'Unknown'", "Affected": filled})

    # Ship date
    if {"Ship Date", "Order Date"} <= set(df.columns):
        lead = (df["Ship Date"] - df["Order Date"]).dt.days
        median_lead = lead.groupby(df["Ship Mode"]).transform("median") if "Ship Mode" in df else lead
        median_lead = median_lead.fillna(lead.median() if lead.notna().any() else 0)
        n = int(df["Ship Date"].isna().sum())
        df["Ship Date"] = df["Ship Date"].fillna(df["Order Date"] + pd.to_timedelta(median_lead.round(), unit="D"))
        log.append({"Step": "Fill Ship Date", "Detail": "Order Date + median lead time of the Ship Mode", "Affected": n})

    present = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if strategy == "drop":
        n = int(df[present].isna().any(axis=1).sum())
        df = df.dropna(subset=present).reset_index(drop=True)
        log.append({"Step": "Drop missing numerics", "Detail": "Rows with missing Sales/Quantity/Discount/Profit dropped", "Affected": n})
        return df, log

    def group_median(col):
        if by:
            df[col] = df[col].fillna(df.groupby(by)[col].transform("median"))
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    for col in [c for c in ("Sales", "Quantity") if c in df.columns]:
        n = int(df[col].isna().sum()); group_median(col)
        log.append({"Step": f"Impute {col}", "Detail": "Median of the Sub-Category (global median fallback)", "Affected": n})
    if "Discount" in df:
        n = int(df["Discount"].isna().sum())
        mode = df["Discount"].mode()
        df["Discount"] = df["Discount"].fillna(mode.iat[0] if not mode.empty else 0.0)
        log.append({"Step": "Impute Discount", "Detail": "Most common discount", "Affected": n})
    if {"Profit", "Sales"} <= set(df.columns):
        n = int(df["Profit"].isna().sum())
        margin = df["Profit"] / df["Sales"]
        med = margin.groupby(df[by]).transform("median") if by else margin
        df["Profit"] = df["Profit"].fillna(df["Sales"] * med.fillna(margin.median() if margin.notna().any() else 0))
        log.append({"Step": "Impute Profit", "Detail": "Sales x median profit margin of the Sub-Category", "Affected": n})
    return df, log


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def numeric_summary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    """count, mean, median, std, variance, IQR, skewness, CV, min/quartiles/max."""
    cols = cols or [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        return pd.DataFrame()
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    out = pd.DataFrame({
        "count": desc["count"], "mean": desc["mean"], "median": desc["50%"], "std": desc["std"],
        "variance": df[cols].var(), "min": desc["min"], "Q1": desc["25%"], "Q3": desc["75%"],
        "max": desc["max"], "IQR": desc["75%"] - desc["25%"], "skewness": df[cols].skew(),
        "CV (%)": 100 * desc["std"] / desc["mean"].replace(0, np.nan),
    })
    return out.round(3)


def categorical_summary(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        vc = df[c].value_counts()
        rows.append({"Column": c, "Unique": int(df[c].nunique()), "Top value": vc.index[0] if len(vc) else None,
                     "Top count": int(vc.iloc[0]) if len(vc) else 0,
                     "Top share (%)": round(100 * vc.iloc[0] / len(df), 2) if len(vc) else 0})
    return pd.DataFrame(rows)


def outlier_summary_iqr(df: pd.DataFrame, cols: list[str], k: float = 1.5) -> pd.DataFrame:
    """Count of values outside [Q1 - k*IQR, Q3 + k*IQR] for each numeric column."""
    rows = []
    for c in cols:
        q1, q3 = df[c].quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        rows.append({"Column": c, "Lower bound": round(lo, 2), "Upper bound": round(hi, 2),
                     "Outliers": int(((df[c] < lo) | (df[c] > hi)).sum())})
    return pd.DataFrame(rows)


def group_summary(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Sales, profit, orders, quantity and margin per group."""
    g = df.groupby(by).agg(Sales=("Sales", "sum"), Profit=("Profit", "sum"),
                           Orders=("Order ID", "nunique"), Quantity=("Quantity", "sum"))
    g["Profit Margin (%)"] = (100 * g["Profit"] / g["Sales"].replace(0, np.nan)).round(2)
    return g.round(2).sort_values("Sales", ascending=False).reset_index()


def compute_kpis(df: pd.DataFrame) -> dict:
    """Total Sales, Total Profit, Total Orders (unique Order IDs), Quantity, Profit Margin %."""
    sales, profit = float(df["Sales"].sum()), float(df["Profit"].sum())
    return {
        "total_sales": sales,
        "total_profit": profit,
        "total_orders": int(df["Order ID"].nunique()),
        "total_quantity": int(df["Quantity"].sum()),
        "profit_margin": (profit / sales * 100) if sales else 0.0,
    }


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
def apply_filters(df: pd.DataFrame, date_range=None, selections: dict | None = None,
                  discount_range: tuple[float, float] | None = None) -> pd.DataFrame:
    """Filter by order-date range, categorical multi-selections and a discount range."""
    mask = pd.Series(True, index=df.index)
    if date_range and len(date_range) == 2 and "Order Date" in df:
        start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
        mask &= df["Order Date"].between(start, end)
    for col, chosen in (selections or {}).items():
        if col in df and chosen is not None:
            mask &= df[col].isin(chosen)
    if discount_range and "Discount" in df:
        mask &= df["Discount"].between(*discount_range)
    return df[mask]


# --------------------------------------------------------------------------- #
# Demo data (messy on purpose, mirrors the problems in the lab dataset)
# --------------------------------------------------------------------------- #
def make_demo_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic raw dataset with realistic data-quality problems."""
    rng = np.random.default_rng(seed)
    geo = {"Boston": ("Massachusetts", "East"), "New York": ("New York", "East"),
           "Atlanta": ("Georgia", "South"), "Miami": ("Florida", "South"),
           "Chicago": ("Illinois", "Central"), "Houston": ("Texas", "Central"),
           "Seattle": ("Washington", "West"), "Phoenix": ("Arizona", "West")}
    products = {"Phones": ("Technology", ["Smartphone", "Cordless Phone"]),
                "Binders": ("Office Supplies", ["Ring Binder", "Presentation Binder"]),
                "Chairs": ("Furniture", ["Conference Chair", "Ergonomic Office Chair"]),
                "Tables": ("Furniture", ["Meeting Table", "Coffee Table"]),
                "Storage": ("Office Supplies", ["File Organizer", "Storage Box"]),
                "Accessories": ("Technology", ["Keyboard", "Wireless Mouse"])}
    cities = rng.choice(list(geo), n)
    subs = rng.choice(list(products), n)
    order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 700, n), unit="D")
    ship_dates = order_dates + pd.to_timedelta(rng.integers(1, 9, n), unit="D")
    sales = np.round(rng.lognormal(6, 1.1, n), 2)
    df = pd.DataFrame({
        "Order ID": [f"ORD-{10000 + i}" for i in rng.integers(0, 90, n)],
        "Order Date": order_dates.strftime("%Y-%m-%d"), "Ship Date": ship_dates.strftime("%Y-%m-%d"),
        "Ship Mode": rng.choice(["Standard Class", "Second Class", "First Class", "Same Day"], n),
        "Customer ID": [f"CUST-{100 + i}" for i in rng.integers(0, 60, n)],
        "Segment": rng.choice(["Consumer", "Corporate", "Home Office"], n),
        "Country": "United States", "City": cities,
        "State": [geo[c][0] for c in cities], "Region": [geo[c][1] for c in cities],
        "Product ID": [f"PROD-{1000 + i}" for i in rng.integers(0, 99, n)],
        "Category": [products[s][0] for s in subs], "Sub-Category": subs,
        "Product Name": [rng.choice(products[s][1]) for s in subs],
        "Sales": sales, "Quantity": rng.integers(1, 11, n),
        "Discount": rng.choice([0, 0.05, 0.1, 0.15, 0.2, 0.3], n),
        "Profit": np.round(sales * rng.normal(0.08, 0.22, n), 2),
    }).astype(object)

    def pick(k):
        return rng.choice(n, k, replace=False)
    df.loc[pick(12), "Sales"] = "unknown"; df.loc[pick(12), "Profit"] = "N/A"
    df.loc[pick(6), "Quantity"] = "five"; df.loc[pick(5), "Discount"] = "ten percent"
    df.loc[pick(2), "Sales"] = 9999999.99; df.loc[pick(2), "Discount"] = 1.75
    df.loc[pick(2), "Quantity"] = -3; df.loc[pick(1), "Quantity"] = 10000
    for col in ["Ship Mode", "Customer ID", "Category", "City", "Product Name"]:
        df.loc[pick(8), col] = np.nan
    idx = pick(25)
    df.loc[idx, "Order Date"] = pd.to_datetime(df.loc[idx, "Order Date"]).dt.strftime("%d/%m/%Y")
    idx = pick(25)
    df.loc[idx, "Ship Date"] = pd.to_datetime(df.loc[idx, "Ship Date"]).dt.strftime("%d-%b-%Y")
    df.loc[pick(15), "Segment"] = "consumer"
    idx = pick(10)
    df.loc[idx, "Region"] = df.loc[idx, "Region"].str.upper()
    df.loc[pick(10), "Ship Mode"] = "SameDay"; df.loc[pick(10), "Category"] = "Office supply"
    df.loc[pick(3), "Ship Date"] = "2023-01-02"
    df = pd.concat([df, df.sample(15, random_state=seed)], ignore_index=True)       # duplicates
    junk = pd.DataFrame([{"Order ID": "ORD-10005", "Order Date": "not-a-date", "Ship Mode": "Express Ultra",
                          "Customer ID": "CUST-999", "Segment": "Unknown Segment"},
                         {"City": "Testville", "State": "Test State", "Region": "Northeast",
                          "Product ID": "PROD-9999", "Category": "Gadgets"}])
    return pd.concat([df, junk], ignore_index=True)

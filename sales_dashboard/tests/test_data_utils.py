import io, json, sys, pathlib
import numpy as np, pandas as pd, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import data_utils as du


@pytest.fixture(scope="module")
def cleaned():
    raw = du.make_demo_data()
    return raw, *du.clean_data(raw)


def test_to_number():
    assert du.to_number("five") == 5 and du.to_number("ten percent") == 0.1
    assert np.isnan(du.to_number("N/A")) and np.isnan(du.to_number("inf")) and np.isnan(du.to_number("unknown"))
    assert du.to_number("1,234.5") == 1234.5


def test_parse_date_ambiguous_uses_order_date():
    od = pd.Timestamp("2025-03-08")
    assert du.parse_date_value("03-12-2025", od) == pd.Timestamp("2025-03-12")
    assert du.parse_date_value("26/12/2024") == pd.Timestamp("2024-12-26")
    assert du.parse_date_value("21-Mar-2025") == pd.Timestamp("2025-03-21")
    assert pd.isna(du.parse_date_value("not-a-date"))


def test_canonicalize_labels():
    s = pd.Series(["Consumer"] * 5 + ["consumer", "Home Office", "Home office", "Same Day", "SameDay", "Same Day"])
    out = du.canonicalize_labels(s)
    assert set(out) == {"Consumer", "Home Office", "Same Day"}


def test_validate_reports_problems():
    r = du.validate_data(du.standardize_columns(du.make_demo_data()))
    assert r["duplicate_rows"] >= 15 and r["missing_cells"] > 0 and r["columns"] == 18


def test_clean_output_is_valid(cleaned):
    raw, df, log = cleaned
    assert not df.duplicated().any()
    assert df[["Sales", "Quantity", "Discount", "Profit"]].notna().all().all()
    assert (df["Sales"] > 0).all() and (df["Sales"] <= 100_000).all()
    assert df["Quantity"].between(1, 1000).all() and df["Discount"].between(0, 1).all()
    assert (df["Ship Date"] >= df["Order Date"]).all()
    assert pd.api.types.is_datetime64_any_dtype(df["Order Date"])
    assert set(df["Segment"]) <= {"Consumer", "Corporate", "Home Office", "Unknown"}
    assert set(df["Region"]) <= {"East", "West", "South", "Central", "Unknown"}
    assert set(df["Category"]) <= {"Technology", "Furniture", "Office Supplies", "Unknown"}
    assert "Testville" not in set(df["City"]) and len(log) > 5


def test_drop_strategy():
    df, _ = du.clean_data(du.make_demo_data(), du.CleaningConfig(missing_strategy="drop"))
    assert df[["Sales", "Profit"]].notna().all().all()


def test_kpis_and_filters(cleaned):
    _, df, _ = cleaned
    k = du.compute_kpis(df)
    assert k["total_orders"] == df["Order ID"].nunique()
    assert k["profit_margin"] == pytest.approx(df["Profit"].sum() / df["Sales"].sum() * 100)
    f = du.apply_filters(df, selections={"Region": ["East"]}, discount_range=(0, 0.1))
    assert set(f["Region"]) == {"East"} and f["Discount"].max() <= 0.1
    assert du.compute_kpis(f)["total_sales"] < k["total_sales"]


def test_summary_functions(cleaned):
    _, df, _ = cleaned
    s = du.numeric_summary(df, ["Sales", "Profit"])
    assert {"mean", "median", "std", "IQR", "skewness"} <= set(s.columns)
    assert du.group_summary(df, "Region")["Sales"].sum() == pytest.approx(round(df["Sales"].sum(), 2), rel=1e-3)


def test_loaders_roundtrip(cleaned):
    _, df, _ = cleaned
    small = df.head(20).copy(); small["Order Date"] = small["Order Date"].dt.strftime("%Y-%m-%d")
    small["Ship Date"] = small["Ship Date"].dt.strftime("%Y-%m-%d")
    csv = small.to_csv(index=False).encode()
    assert du.load_file(csv, "a.csv").shape == small.shape
    js = small.to_json(orient="records").encode()
    assert du.load_file(js, "a.json").shape == small.shape
    assert du.load_file(json.dumps({"data": json.loads(js)}).encode(), "b.json").shape == small.shape
    buf = io.BytesIO(); small.to_excel(buf, index=False)
    assert du.load_file(buf.getvalue(), "a.xlsx").shape == small.shape


def test_loader_errors():
    with pytest.raises(ValueError): du.load_file(b"x", "a.txt")
    with pytest.raises(ValueError): du.load_file(b"", "a.csv")

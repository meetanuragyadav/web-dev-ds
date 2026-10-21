# Sales Analytics Dashboard (Streamlit)

Upload a CSV / Excel / JSON sales file, validate it, clean it, filter it, and explore KPIs and interactive charts.

**Live app:** _paste your Streamlit Community Cloud URL here_
**Repository:** _paste your GitHub URL here_

## Features
- **Upload & validation** (`.csv`, `.xlsx`, `.json`): dimensions, column names, data types, missing values (incl. placeholders like `N/A`/`unknown`), duplicate rows.
- **Cleaning**: placeholders to missing, mixed date formats, label inconsistencies (`consumer`/`Consumer`, `SameDay`/`Same Day`, `Office supply`), City/State/Region repair, text numbers (`five`, `ten percent`), invalid values (negative sales, discount > 1, outliers), junk/test rows, duplicates, median/mode imputation or row dropping. Every step is listed in a cleaning log.
- **Filters**: order date, Region, State, Category, Segment, Ship Mode, discount range.
- **KPIs**: Total Sales, Total Profit, Total Orders (unique Order IDs), Quantity Sold, Profit Margin (%).
- **Charts** (Plotly): sales/profit trend, metric by dimension, region treemap, margin heatmap, discount vs margin, segment x ship mode, correlation, box plots.
- **Statistics**: reusable functions in `data_utils.py` (`numeric_summary`, `categorical_summary`, `outlier_summary_iqr`, `group_summary`, `compute_kpis`).

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Try it with `sample_data/sample_messy.csv` or the "Use demo data" toggle.

## Test
```bash
pip install pytest
python -m pytest -q
```

## Deploy (Streamlit Community Cloud)
1. Push this folder to a **public** GitHub repo (`app.py`, `data_utils.py`, `charts.py`, `requirements.txt` at the repo root).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repo, branch `main`, main file `app.py`, then **Deploy**.
4. Open the URL in a private window, upload a file, and check that KPIs and charts update.

## Structure
```
app.py            # Streamlit UI (layout, sidebar, tabs)
data_utils.py     # loading, validation, cleaning, statistics, KPIs, filters
charts.py         # Plotly figure builders
tests/            # pytest unit tests + Streamlit AppTest
sample_data/      # messy sample CSV
```

# ================================================================
# 🛒 AI E‑Commerce Dashboard — Minimal, Clean ("Second Best" style)
# Author: Rafay + ChatGPT
# Notes:
# - Local CSV only: looks for Data/sample.csv or Data/data.csv
# - Minimalistic, card KPIs + clean charts (Plotly)
# - Optional Prophet forecast tab (auto-disables if prophet not installed)
# - Dark/Light toggle + tidy CSS matching the example style
# ================================================================

import os
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# Prophet optional (won't crash if missing)
HAS_PROPHET = True
try:
    from prophet import Prophet
    from prophet.plot import plot_plotly
except Exception:
    HAS_PROPHET = False

# -----------------------------
# Page & base styles
# -----------------------------
st.set_page_config(page_title="AI E‑Commerce Dashboard", page_icon="🛒", layout="wide")

# Theme toggle
is_dark = st.sidebar.toggle("🌗 Dark Mode", value=True)
PLOTLY_TEMPLATE = "plotly_dark" if is_dark else "plotly"

# Minimal CSS inspired by the referenced clean style
if is_dark:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem;}
          .kpi {background:#0f1623;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:14px 16px;}
          .kpi .label{opacity:.7;font-size:.85rem;margin-bottom:.25rem}
          .kpi .value{font-weight:800;font-size:1.7rem;line-height:1}
          .kpi .sub{opacity:.6;font-size:.85rem;margin-top:.25rem}
          .pill{display:inline-block;padding:.25rem .6rem;border-radius:999px;background:#0f1623;border:1px solid rgba(255,255,255,.08);font-size:.8rem}
          .panel{background:#0f1623;border:1px solid rgba(255,255,255,.08);border-radius:16px;padding:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
          .block-container {padding-top: 0.8rem; padding-bottom: 1.2rem;}
          .kpi {background:#ffffff;border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:14px 16px;}
          .kpi .label{opacity:.7;font-size:.85rem;margin-bottom:.25rem}
          .kpi .value{font-weight:800;font-size:1.7rem;line-height:1}
          .kpi .sub{opacity:.65;font-size:.85rem;margin-top:.25rem}
          .pill{display:inline-block;padding:.25rem .6rem;border-radius:999px;background:#f5f5f5;border:1px solid rgba(0,0,0,.06);font-size:.8rem}
          .panel{background:#ffffff;border:1px solid rgba(0,0,0,.06);border-radius:16px;padding:12px}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Data loader (local only)
# -----------------------------

def smart_read_csv(path: str) -> pd.DataFrame:
    tried = []
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        for sep in (",", ";", "\t", "|"):
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                tried.append(f"{enc}/{sep}: {e}")
    raise ValueError("Failed to read CSV. Tried:\n" + "\n".join(tried))

candidates = [
    "Data/sample.csv",
    "Data/data.csv",
    "data/sample.csv",
    "data/data.csv",
]
CSV_PATH = next((p for p in candidates if os.path.exists(p)), None)
if CSV_PATH is None:
    st.error("Please add a CSV at `Data/sample.csv` or `Data/data.csv` in your repo.")
    st.stop()

@st.cache_data(show_spinner=False)
def load_clean(path: str) -> pd.DataFrame:
    df = smart_read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    def find_col(df, names):
        low = [c.lower() for c in df.columns]
        for n in names:
            if n.lower() in low:
                return df.columns[low.index(n.lower())]
        return None

    c_date = find_col(df, ["InvoiceDate", "Invoice Date", "OrderDate", "date", "datetime"])  \
        or "InvoiceDate"
    c_qty = find_col(df, ["Quantity", "Qty", "units"]) or "Quantity"
    c_price = find_col(df, ["UnitPrice", "Unit Price", "Price", "amount"]) or "UnitPrice"
    c_desc = find_col(df, ["Description", "Product", "Item", "ProductName"]) or "Description"
    c_ctry = find_col(df, ["Country", "Market", "Region"]) or "Country"
    c_inv = find_col(df, ["InvoiceNo", "Invoice No", "Order No", "order_id"]) or "InvoiceNo"

    rename = {c_date:"InvoiceDate", c_qty:"Quantity", c_price:"UnitPrice", c_desc:"Description", c_ctry:"Country", c_inv:"InvoiceNo"}
    df = df.rename(columns=rename)

    # Required columns
    if not set(["InvoiceDate","Quantity","UnitPrice"]).issubset(df.columns):
        missing = [c for c in ["InvoiceDate","Quantity","UnitPrice"] if c not in df.columns]
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    # Types & cleaning
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity","UnitPrice"]).copy()
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # Defaults
    if "Description" not in df.columns:
        df["Description"] = "Unknown"
    if "Country" not in df.columns:
        df["Country"] = "Unknown"

    df["Description"] = df["Description"].astype(str).fillna("Unknown")
    df["Country"] = df["Country"].astype(str).fillna("Unknown")

    df["Sales"] = df["Quantity"] * df["UnitPrice"]
    return df

try:
    df = load_clean(CSV_PATH)
    src_label = os.path.basename(CSV_PATH)
except Exception as e:
    st.error(f"Data load error: {e}")
    st.stop()

# -----------------------------
# Sidebar (filters)
# -----------------------------
st.sidebar.subheader("Filters")
st.sidebar.markdown(f"<span class='pill'>Source: {src_label}</span>", unsafe_allow_html=True)

min_d, max_d = df["InvoiceDate"].min(), df["InvoiceDate"].max()
def_start = max(min_d, max_d - pd.Timedelta(days=180)) if pd.notna(min_d) and pd.notna(max_d) else min_d

start, end = st.sidebar.date_input(
    "Date range",
    value=(def_start.date() if pd.notna(def_start) else datetime.today().date(),
           max_d.date() if pd.notna(max_d) else datetime.today().date()),
    min_value=min_d.date() if pd.notna(min_d) else None,
    max_value=max_d.date() if pd.notna(max_d) else None,
)

countries = st.sidebar.multiselect(
    "Country",
    options=sorted(df["Country"].dropna().astype(str).unique().tolist()),
    default=["United Kingdom"] if "United Kingdom" in df["Country"].unique() else [],
)

show_raw = st.sidebar.checkbox("Show raw (first 200 rows)")

mask = (df["InvoiceDate"] >= pd.to_datetime(start)) & (df["InvoiceDate"] <= pd.to_datetime(end))
if countries:
    mask &= df["Country"].isin(countries)
fdf = df.loc[mask].copy()

# -----------------------------
# Header
# -----------------------------
st.title("AI‑Powered E‑Commerce Sales Dashboard")
st.caption("Minimal clean layout · Streamlit + Plotly" + (" · Prophet" if HAS_PROPHET else ""))

# -----------------------------
# KPI cards row (clean/minimal like the example)
# -----------------------------

total_sales = float(fdf["Sales"].sum())
orders = fdf["InvoiceNo"].nunique() if "InvoiceNo" in fdf.columns else int(
    fdf.set_index("InvoiceDate").resample("D")["Sales"].count().shape[0]
)
products = int(fdf["Description"].nunique())
ctry_sum = fdf.groupby("Country")["Sales"].sum().sort_values(ascending=False)
best_ctry = ctry_sum.index[0] if len(ctry_sum) else "N/A"
best_ctry_val = float(ctry_sum.iloc[0]) if len(ctry_sum) else 0.0
avg_order = (total_sales / orders) if orders else 0.0

c1, c2, c3, c4, c5 = st.columns(5, gap="large")
for col, label, value, sub in [
    (c1, "Total Sales", f"£ {total_sales:,.0f}", "Selected period"),
    (c2, "Total Orders", f"{orders:,}", "Unique invoices"),
    (c3, "Total Products", f"{products:,}", "Distinct items"),
    (c4, "Top Country", best_ctry, f"£ {best_ctry_val:,.0f}"),
    (c5, "Average Order Value", f"£ {avg_order:,.2f}", "Sales / Order"),
]:
    with col:
        st.markdown("<div class='kpi'>" \
                    f"<div class='label'>{label}</div>" \
                    f"<div class='value'>{value}</div>" \
                    f"<div class='sub'>{sub}</div>" \
                    "</div>", unsafe_allow_html=True)

st.markdown("\n")

# -----------------------------
# Main panels: Trends + Breakdowns (mirroring the example vibe)
# -----------------------------

colA, colB = st.columns((2, 1), gap="large")

# Daily & Monthly trends
series_daily = (
    fdf.set_index("InvoiceDate").resample("D")["Sales"].sum().fillna(0.0)
)
fig_daily = px.line(series_daily, labels={"value":"Sales (£)", "InvoiceDate":"Date"}, template=PLOTLY_TEMPLATE)
fig_daily.update_layout(title_text="Daily Sales", margin=dict(l=8, r=8, t=36, b=8))
colA.plotly_chart(fig_daily, use_container_width=True)

series_month = series_daily.resample("MS").sum().fillna(0.0)
fig_month = px.area(series_month, labels={"value":"Sales (£)", "InvoiceDate":"Month"}, template=PLOTLY_TEMPLATE)
fig_month.update_traces(hovertemplate="£ %{y:,.0f}")
fig_month.update_layout(title_text="Monthly Sales", margin=dict(l=8, r=8, t=36, b=8))
colB.plotly_chart(fig_month, use_container_width=True)

st.markdown("\n")

colC, colD = st.columns(2, gap="large")

# Top products (by quantity)
prod_qty = fdf.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
fig_prod = px.bar(prod_qty.sort_values(), orientation="h", template=PLOTLY_TEMPLATE,
                  labels={"value":"Units", "index":"Product"})
fig_prod.update_layout(title_text="Top 10 Products (by Quantity)", margin=dict(l=8, r=8, t=36, b=8))
colC.plotly_chart(fig_prod, use_container_width=True)

# Top countries by sales
country_sales = fdf.groupby("Country")["Sales"].sum().sort_values(ascending=False).head(10)
fig_ctry = px.bar(country_sales, template=PLOTLY_TEMPLATE, labels={"value":"Sales (£)", "Country":"Country"})
fig_ctry.update_traces(hovertemplate="£ %{y:,.0f}")
fig_ctry.update_layout(title_text="Top Countries (by Sales)", margin=dict(l=8, r=8, t=36, b=8))
colD.plotly_chart(fig_ctry, use_container_width=True)

# -----------------------------
# Tabs: Forecast (optional) + Raw
# -----------------------------

tab1, tab2 = st.tabs(["🤖 Forecast", "📄 Raw Data"])

with tab1:
    if not HAS_PROPHET:
        st.info("Prophet not installed — add `prophet` to requirements.txt to enable forecasting.")
    else:
        monthly_df = (
            fdf.set_index("InvoiceDate").resample("MS")["Sales"].sum().reset_index()
            .rename(columns={"InvoiceDate":"ds", "Sales":"y"})
        )
        ok = (len(monthly_df) >= 6) and (monthly_df["y"].gt(0).sum() >= 3)
        if not ok:
            st.info("Not enough monthly history for a stable 6‑month forecast. Expand date range.")
        else:
            with st.spinner("Training Prophet model…"):
                m = Prophet()
                m.fit(monthly_df)
                future = m.make_future_dataframe(periods=6, freq="MS")
                fc = m.predict(future)
            fig_fc = plot_plotly(m, fc)
            fig_fc.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=8, r=8, t=36, b=8))
            st.plotly_chart(fig_fc, use_container_width=True)

with tab2:
    st.dataframe(fdf.head(200), use_container_width=True)

# Footer
st.caption("Made with ❤️ by Rafay · Streamlit + Plotly" + (" · Prophet" if HAS_PROPHET else ""))

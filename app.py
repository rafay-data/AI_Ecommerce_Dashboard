# ================================================================
# 🛒 AI E-Commerce Dashboard (Local CSV only - Dark/Light)
# Author: Rafay (updated)
# Notes:
# - Upload feature removed (app loads Data/sample.csv or Data/data.csv)
# - Robust checks for required columns
# - Dark/Light theme toggle retained
# - Prophet forecasting (6 months), download forecast CSV
# ================================================================

import os
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly

# -----------------------------
# Reduce noisy logs
# -----------------------------
os.environ["STREAMLIT_LOG_LEVEL"] = "error"
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="AI E-Commerce Dashboard", page_icon="🛒", layout="wide")

# -----------------------------
# Theme toggle (dark / light)
# -----------------------------
theme = st.sidebar.toggle("🌗 Dark Mode", value=True)
if theme:
    PLOTLY_TEMPLATE = "plotly_dark"
    DARK_BG = "#0e1117"
    DARK_CARD = "#151a23"
    ACCENT = "#4a90e2"
    st.markdown(
        f"""
        <style>
          .stApp {{ background: linear-gradient(180deg, {DARK_BG} 0%, #0b0e14 100%) !important; }}
          .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; }}
          .glass {{
            background: radial-gradient(100% 100% at 0% 0%, {DARK_CARD} 0%, #111827 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 18px;
            padding: 1.0rem 1.1rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.35);
          }}
          .kpi-number {{ font-size: 1.6rem; font-weight: 800; line-height: 1.1; }}
          .kpi-sub {{ opacity: .7; font-size: .9rem; }}
          .pill {{
            display:inline-block; padding:.25rem .6rem; border-radius:999px;
            background:#111827; border:1px solid rgba(255,255,255,.08); font-size:.8rem;
          }}
          .footer {{ opacity:.6; font-size:.9rem; text-align:center; margin-top:1.2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    PLOTLY_TEMPLATE = "plotly"
    st.markdown(
        """
        <style>
          .glass {
            background: #ffffff;
            border: 1px solid rgba(0,0,0,0.06);
            border-radius: 18px;
            padding: 1.0rem 1.1rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.06);
          }
          .kpi-number { font-size: 1.6rem; font-weight: 800; line-height: 1.1; }
          .kpi-sub { opacity: .7; font-size: .9rem; }
          .pill {
            display:inline-block; padding:.25rem .6rem; border-radius:999px;
            background:#f5f5f5; border:1px solid rgba(0,0,0,.06); font-size:.8rem;
          }
          .footer { opacity:.7; font-size:.9rem; text-align:center; margin-top:1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# Smart CSV Reader (local)
# -----------------------------
def read_csv_smart_local(path):
    """Try multiple encodings & separators for a local CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    tried = []
    for enc in ["utf-8", "utf-8-sig", "latin1"]:
        for sep in [",", ";", "\t", "|"]:
            try:
                return pd.read_csv(path, encoding=enc, sep=sep)
            except Exception as e:
                tried.append(f"{enc}/{sep}: {e}")
    raise ValueError("CSV read failed. Tried:\n" + "\n".join(tried))

# -----------------------------
# Choose local file path
# -----------------------------
# check Data/sample.csv first, else Data/data.csv
local_paths = ["Data/sample.csv", "Data/data.csv", "data/sample.csv", "data/data.csv"]
csv_path = None
for p in local_paths:
    if os.path.exists(p):
        csv_path = p
        break

if csv_path is None:
    st.error(
        "Local dataset not found. Please put your CSV as `Data/sample.csv` or `Data/data.csv` inside the project folder."
    )
    st.stop()

# -----------------------------
# Load & preprocess data (robust)
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_clean(path):
    df = read_csv_smart_local(path)

    # Normalize column names (strip)
    df.columns = [c.strip() for c in df.columns.astype(str)]

    # REQUIRED columns for this app
    # InvoiceDate, Quantity, UnitPrice (or Price), Description, Country, InvoiceNo (optional)
    # Try common alternatives
    col_map = {}

    # helper to find a column by candidate names
    def find_col(candidates):
        cols = [c.lower() for c in df.columns]
        for cand in candidates:
            if cand.lower() in cols:
                return df.columns[cols.index(cand.lower())]
        return None

    col_map["InvoiceDate"] = find_col(["InvoiceDate", "Invoice Date", "date", "OrderDate", "invoicedate", "datetime"])
    col_map["Quantity"] = find_col(["Quantity", "Qty", "quantity", "units"])
    col_map["UnitPrice"] = find_col(["UnitPrice", "Unit Price", "Price", "price", "unit_price", "amount"])
    col_map["Description"] = find_col(["Description", "Product", "Item", "ProductName", "product_name"])
    col_map["Country"] = find_col(["Country", "country", "Market", "Region"])
    col_map["InvoiceNo"] = find_col(["InvoiceNo", "Invoice No", "Order No", "order_id", "invoiceno"])

    # If essential columns missing, raise friendly exception
    essential = ["InvoiceDate", "Quantity", "UnitPrice"]
    missing = [k for k in essential if col_map.get(k) is None]
    if missing:
        raise KeyError(f"Required columns missing in CSV: {', '.join(missing)}. Please ensure your CSV has these columns.")

    # Rename to standard names
    rename_dict = {col_map[k]: k for k in col_map if col_map.get(k) is not None}
    df = df.rename(columns=rename_dict).copy()

    # Ensure Description & Country exist (create defaults if not present)
    if "Description" not in df.columns:
        df["Description"] = "Unknown"
    else:
        df["Description"] = df["Description"].fillna("Unknown").astype(str)

    if "Country" not in df.columns:
        df["Country"] = "Unknown"
    else:
        df["Country"] = df["Country"].fillna("Unknown").astype(str)

    # Convert types safely
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df.dropna(subset=["InvoiceDate"]).copy()

    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce")
    df = df.dropna(subset=["Quantity", "UnitPrice"]).copy()

    # Remove non-positive values
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    # Sales column
    df["Sales"] = df["Quantity"] * df["UnitPrice"]

    return df

# Try loading and handle errors
try:
    df = load_and_clean(csv_path)
    source_label = os.path.basename(csv_path)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
except KeyError as e:
    st.error(f"CSV format problem: {e}")
    st.stop()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.markdown("### ⚙️ Filters")
st.sidebar.markdown(f'<span class="pill">Local file: {source_label}</span>', unsafe_allow_html=True)

min_d, max_d = df["InvoiceDate"].min(), df["InvoiceDate"].max()
default_start = max(min_d, max_d - pd.Timedelta(days=180)) if pd.notna(min_d) and pd.notna(max_d) else min_d

date_range = st.sidebar.date_input(
    "Date range",
    value=(default_start.date() if pd.notna(default_start) else datetime.today().date(),
           max_d.date() if pd.notna(max_d) else datetime.today().date()),
    min_value=min_d.date() if pd.notna(min_d) else None,
    max_value=max_d.date() if pd.notna(max_d) else None
)

selected_countries = st.sidebar.multiselect(
    "Country",
    options=sorted(df["Country"].dropna().astype(str).unique().tolist()),
    default=["United Kingdom"] if "United Kingdom" in df["Country"].unique() else []
)

show_raw = st.sidebar.checkbox("Show raw preview (first 200 rows)")

start_date = pd.to_datetime(date_range[0]) if isinstance(date_range, (list, tuple)) else min_d
end_date = pd.to_datetime(date_range[1]) if isinstance(date_range, (list, tuple)) else max_d

mask = (df["InvoiceDate"] >= start_date) & (df["InvoiceDate"] <= end_date)
if selected_countries:
    mask &= df["Country"].astype(str).isin(selected_countries)
fdf = df.loc[mask].copy()

# -----------------------------
# Tabs for layout
# -----------------------------
tab_overview, tab_trends, tab_top, tab_forecast = st.tabs(["🏠 Overview", "📈 Trends", "🏆 Top", "🤖 Forecast"])

# -----------------------------
# Overview: KPIs
# -----------------------------
with tab_overview:
    st.title("🛒 AI-Powered E-Commerce Sales Dashboard")
    total_sales = float(fdf["Sales"].sum())
    total_orders = fdf["InvoiceNo"].nunique() if "InvoiceNo" in fdf.columns else int(fdf.groupby(pd.Grouper(key="InvoiceDate", freq="D"))["Sales"].count().shape[0])
    total_products = int(fdf["Description"].nunique())
    top_country = fdf.groupby("Country")["Sales"].sum().sort_values(ascending=False).head(1)
    top_country_name = top_country.index[0] if len(top_country) else "N/A"
    top_country_sales = float(top_country.iloc[0]) if len(top_country) else 0.0
    aov = total_sales / total_orders if total_orders else 0.0  # Average Order Value

    c1, c2, c3, c4, c5 = st.columns(5)
    boxes = [
        (c1, "Total Sales", f"£ {total_sales:,.0f}", "Selected period"),
        (c2, "Total Orders", f"{total_orders:,}", "Unique invoices"),
        (c3, "Total Products", f"{total_products:,}", "Distinct items sold"),
        (c4, "Top Country", f"{top_country_name}", f"£ {top_country_sales:,.0f}"),
        (c5, "Average Order Value", f"£ {aov:,.2f}", "Sales / Order"),
    ]
    for box, title, value, sub in boxes:
        with box:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.caption(title)
            st.markdown(f"<div class='kpi-number'>{value}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='kpi-sub'>{sub}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Trends: Daily & Monthly
# -----------------------------
with tab_trends:
    st.markdown("#### Sales Trends")
    cA, cB = st.columns((2, 1), gap="large")

    daily = (
        fdf.set_index("InvoiceDate")
           .resample("D")["Sales"]
           .sum()
           .fillna(0.0)
    )

    fig_daily = px.line(daily, title="Daily Sales", labels={"value": "Sales (£)", "InvoiceDate": "Date"}, template=PLOTLY_TEMPLATE)
    fig_daily.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    cA.plotly_chart(fig_daily, use_container_width=True)

    monthly = daily.resample("MS").sum().fillna(0.0)
    fig_monthly = px.area(monthly, title="Monthly Sales", labels={"value": "Sales (£)", "InvoiceDate": "Month"}, template=PLOTLY_TEMPLATE)
    fig_monthly.update_traces(hovertemplate="£ %{y:,.0f}")
    fig_monthly.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    cB.plotly_chart(fig_monthly, use_container_width=True)

# -----------------------------
# Top performers
# -----------------------------
with tab_top:
    st.markdown("#### Top Performers")
    cC, cD = st.columns(2, gap="large")

    top_products = fdf.groupby("Description")["Quantity"].sum().sort_values(ascending=False).head(10)
    fig_top_prod = px.bar(top_products.sort_values(), orientation="h", title="Top 10 Products (by Quantity)", labels={"value": "Units", "index": "Product"}, template=PLOTLY_TEMPLATE)
    fig_top_prod.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    cC.plotly_chart(fig_top_prod, use_container_width=True)

    country_sales = fdf.groupby("Country")["Sales"].sum().sort_values(ascending=False).head(10)
    fig_top_country = px.bar(country_sales, title="Top Countries (by Sales)", labels={"value": "Sales (£)", "Country": "Country"}, template=PLOTLY_TEMPLATE)
    fig_top_country.update_traces(hovertemplate="£ %{y:,.0f}")
    fig_top_country.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    cD.plotly_chart(fig_top_country, use_container_width=True)

# -----------------------------
# Forecast: Prophet (6 months)
# -----------------------------
with tab_forecast:
    st.markdown("#### AI Sales Forecast (Next 6 Months)")
    monthly_sales = fdf.set_index("InvoiceDate").resample("MS")["Sales"].sum().reset_index().rename(columns={"InvoiceDate": "ds", "Sales": "y"})

    if len(monthly_sales) >= 6 and monthly_sales["y"].gt(0).sum() >= 3:
        with st.spinner("Training Prophet model..."):
            m = Prophet()
            m.fit(monthly_sales)
            future = m.make_future_dataframe(periods=6, freq="MS")
            forecast = m.predict(future)

        fig_fc = plot_plotly(m, forecast)
        fig_fc.update_layout(template=PLOTLY_TEMPLATE, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_fc, use_container_width=True)

        out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out.rename(columns={"ds": "Month", "yhat": "Forecast", "yhat_lower": "Low", "yhat_upper": "High"}, inplace=True)
        st.download_button("⬇️ Download Forecast CSV", data=out.to_csv(index=False).encode("utf-8"), file_name="sales_forecast_6m.csv", mime="text/csv", use_container_width=True)
    else:
        st.info("Forecast ke liye kaafi monthly data nahi mila. Thora wide date range select karein ya zyada months ka dataset rakhein.")

# -----------------------------
# Raw Data (optional)
# -----------------------------
if show_raw:
    st.markdown("### 📄 Raw Data Preview")
    st.dataframe(fdf.head(200), use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='footer'>Made with ❤️ by Rafay · Streamlit + Prophet + Plotly</div>", unsafe_allow_html=True)

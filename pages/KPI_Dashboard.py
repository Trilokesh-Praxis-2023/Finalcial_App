import streamlit as st
import pandas as pd
import os
from utils.github_storage import read_csv
from utils.kpi_dashboard import render_kpis, get_income
from utils.kpi_drilldown import render_kpi_suite

# -----------------------------------------------------------
# LOAD GLOBAL CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📊 KPI Intelligence Dashboard")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# SIDEBAR FILTERS (same as main)
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Smart Filters")

f_year  = st.sidebar.multiselect("Year", sorted(df.year.unique()))
f_month = st.sidebar.multiselect("Month", sorted(df.year_month.unique()))
f_cat   = st.sidebar.multiselect("Category", sorted(df.category.unique()))
f_acc   = st.sidebar.multiselect("Account", sorted(df.accounts.unique()))

filtered = df.copy()
if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:
    filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

# -----------------------------------------------------------
# KPI RENDER
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)
render_kpi_suite(filtered, get_income)

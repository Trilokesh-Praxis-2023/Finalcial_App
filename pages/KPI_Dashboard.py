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
# SIDEBAR FILTERS (Compact + Exclude Category)
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Smart Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc  = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    exclude_cat = st.multiselect(
        "",
        sorted(df.category.unique()),
        placeholder="Exclude category..."
    )

# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]

if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]

if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

# 👉 Exclude category logic
if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]

# -----------------------------------------------------------
# ADVANCED KPI STRIP
# -----------------------------------------------------------
st.markdown("### 🧠 Financial Health Indicators")

total_spend = filtered["amount"].sum()
txn_count = len(filtered)
avg_txn = total_spend / txn_count if txn_count else 0

# Most expensive category
top_cat = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

top_cat_name = top_cat.index[0] if not top_cat.empty else "-"
top_cat_value = top_cat.iloc[0] if not top_cat.empty else 0

# Most active account
top_acc = (
    filtered.groupby("accounts")["amount"]
    .sum()
    .sort_values(ascending=False)
)

top_acc_name = top_acc.index[0] if not top_acc.empty else "-"
top_acc_value = top_acc.iloc[0] if not top_acc.empty else 0

# Monthly burn rate
monthly = filtered.groupby("year_month")["amount"].sum()
burn_rate = monthly.mean() if not monthly.empty else 0

# Spending volatility
volatility = monthly.std() if len(monthly) > 1 else 0

# Best & Worst month
best_month = monthly.idxmin() if not monthly.empty else "-"
worst_month = monthly.idxmax() if not monthly.empty else "-"

k1, k2, k3, k4 = st.columns(4)
k1.metric("💳 Avg Transaction", f"₹{avg_txn:,.0f}")
k2.metric("🔥 Monthly Burn Rate", f"₹{burn_rate:,.0f}")
k3.metric("📂 Top Category", top_cat_name, f"₹{top_cat_value:,.0f}")
k4.metric("🏦 Top Account", top_acc_name, f"₹{top_acc_value:,.0f}")

k5, k6, k7 = st.columns(3)
k5.metric("🧾 Total Transactions", txn_count)
k6.metric("🟢 Best Month", best_month)
k7.metric("🔴 Worst Month", worst_month)

st.caption(f"Spending Volatility (std dev): ₹{volatility:,.0f}")

# -----------------------------------------------------------
# KPI RENDER (your original KPIs)
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)
render_kpi_suite(filtered, get_income)

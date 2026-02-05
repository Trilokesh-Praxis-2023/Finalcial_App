import streamlit as st
import pandas as pd
import os
from utils.github_storage import read_csv
from utils.forecasting_ml import forecasting_ui

# -----------------------------------------------------------
# LOAD CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("🔮 AI Forecasting Intelligence")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# COMPACT SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc  = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    f_cat   = st.multiselect("Category", sorted(df.category.unique()))

# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:
    filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

if filtered.empty:
    st.warning("⚠ No data available for forecasting.")
    st.stop()

# -----------------------------------------------------------
# CONTEXT KPIs (what model sees)
# -----------------------------------------------------------
st.markdown("### 🧠 Dataset Snapshot (Model Input View)")

total = filtered["amount"].sum()
months = filtered["year_month"].nunique()
avg_month = total / months if months else 0
txns = len(filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric("💰 Total Spend", f"₹{total:,.0f}")
k2.metric("🗓 Months Covered", months)
k3.metric("📊 Avg / Month", f"₹{avg_month:,.0f}")
k4.metric("🧾 Transactions", txns)

# -----------------------------------------------------------
# RECENT TREND (last 6 months)
# -----------------------------------------------------------
st.markdown("### 📈 Recent Spending Trend")

monthly = (
    filtered.groupby("year_month")["amount"]
    .sum()
    .sort_index()
)

st.line_chart(monthly.tail(6))

# -----------------------------------------------------------
# CATEGORY CONTRIBUTION
# -----------------------------------------------------------
st.markdown("### 🧩 Category Contribution")

cat_share = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(cat_share)

# -----------------------------------------------------------
# FORECAST SECTION
# -----------------------------------------------------------
st.markdown("### 🔮 Forecast Prediction")

forecasting_ui(filtered)

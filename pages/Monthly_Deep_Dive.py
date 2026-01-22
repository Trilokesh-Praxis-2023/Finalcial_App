import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
import os

# Load global CSS
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(layout="wide")
st.title("📅 Monthly Deep Dive")

df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df["year_month"] = df["period"].dt.to_period("M").astype(str)

# ---------------- Monthly Spend ----------------
st.subheader("📊 Monthly Total Spend")
monthly = df.groupby("year_month")["amount"].sum()
st.line_chart(monthly)

# ---------------- Best / Worst Month ----------------
best_month = monthly.idxmin()
worst_month = monthly.idxmax()

st.success(f"🟢 Best Month (lowest spend): **{best_month}** — ₹{monthly.min():.2f}")
st.error(f"🔴 Worst Month (highest spend): **{worst_month}** — ₹{monthly.max():.2f}")

# ---------------- Category breakup per month ----------------
st.subheader("📂 Category Breakdown by Month")
pivot = df.pivot_table(
    values="amount",
    index="year_month",
    columns="category",
    aggfunc="sum",
    fill_value=0
)
st.dataframe(pivot)

# ---------------- Running Total Curve ----------------
st.subheader("📈 Running Total Trend")
df = df.sort_values("period")
df["running_total"] = df["amount"].cumsum()
st.line_chart(df.set_index("period")["running_total"])

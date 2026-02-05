import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
import os

# -----------------------------------------------------------
# LOAD CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📅 Monthly Deep Dive — Category Analytics")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df["year_month"] = df["period"].dt.to_period("M").astype(str)

# -----------------------------------------------------------
# SIDEBAR EXCLUDE CATEGORY
# -----------------------------------------------------------
exclude_cat = st.sidebar.multiselect(
    "",
    sorted(df["category"].unique()),
    placeholder="Filter out category..."
)

if exclude_cat:
    df = df[~df["category"].isin(exclude_cat)]

if df.empty:
    st.warning("No data available after applying filter.")
    st.stop()

st.header("📊 Category-wise Monthly Deep Dive")

# -----------------------------------------------------------
# LOOP PER CATEGORY
# -----------------------------------------------------------
for cat in sorted(df["category"].unique()):
    st.divider()
    st.subheader(f"📂 {cat}")

    dcat = df[df["category"] == cat].copy()
    monthly = dcat.groupby("year_month")["amount"].sum()

    if len(monthly) < 2:
        st.info("Not enough data for meaningful monthly analysis.")
        continue

    # ---------------- KPI ROW ----------------
    best_month = monthly.idxmin()
    worst_month = monthly.idxmax()
    score = max(0, 100 - (monthly.std() / monthly.mean()) * 100)

    k1, k2, k3 = st.columns(3)
    k1.metric("🟢 Best Month", f"{best_month}", f"₹{monthly.min():.0f}")
    k2.metric("🔴 Worst Month", f"{worst_month}", f"₹{monthly.max():.0f}")
    k3.metric("🎯 Consistency", f"{score:.1f} / 100")

    # ---------------- CHART GRID ----------------
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### 📈 Monthly Spend")
        st.line_chart(monthly)

        st.markdown("#### 📈 MoM Growth %")
        mom = (monthly.pct_change() * 100).round(2)
        st.line_chart(mom)

    with c2:
        st.markdown("#### 📆 Avg Spend per Day")
        days = dcat.groupby("year_month")["period"].nunique()
        avg_per_day = (monthly / days).round(2)
        st.bar_chart(avg_per_day)

        st.markdown("#### 📈 Cumulative by Month")
        st.line_chart(monthly.cumsum())

    # ---------------- RUNNING TOTAL FULL WIDTH ----------------
    st.markdown("#### 📈 Running Total Trend")
    dcat_sorted = dcat.sort_values("period")
    dcat_sorted["running_total"] = dcat_sorted["amount"].cumsum()
    st.line_chart(dcat_sorted.set_index("period")["running_total"])

import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
import os

# -----------------------------------------------------------
# LOAD GLOBAL CSS
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
# SIDEBAR EXCLUDE CATEGORY (same logic)
# -----------------------------------------------------------
all_categories = sorted(df["category"].unique())

exclude_cat = st.sidebar.multiselect(
    label="",
    options=all_categories,
    placeholder="Filter out category..."
)

if exclude_cat:
    df = df[~df["category"].isin(exclude_cat)]

if df.empty:
    st.warning("No data available after applying filter.")
    st.stop()

categories = sorted(df["category"].unique())

st.header("📊 Category-wise Monthly Deep Dive")

# -----------------------------------------------------------
# LOOP PER CATEGORY
# -----------------------------------------------------------
for cat in categories:
    st.divider()
    st.subheader(f"📂 Category: {cat}")

    dcat = df[df["category"] == cat].copy()

    monthly = dcat.groupby("year_month")["amount"].sum()

    if len(monthly) < 2:
        st.info("Not enough data for meaningful monthly analysis.")
        continue

    # -------------------------------------------------------
    # Monthly Trend
    # -------------------------------------------------------
    st.markdown("### 📈 Monthly Spend Trend")
    st.line_chart(monthly)

    # -------------------------------------------------------
    # Best / Worst Month
    # -------------------------------------------------------
    best_month = monthly.idxmin()
    worst_month = monthly.idxmax()

    st.success(f"🟢 Best Month: **{best_month}** — ₹{monthly.min():.2f}")
    st.error(f"🔴 Worst Month: **{worst_month}** — ₹{monthly.max():.2f}")

    # -------------------------------------------------------
    # Avg per day
    # -------------------------------------------------------
    st.markdown("### 📆 Average Daily Spend")
    days = dcat.groupby("year_month")["period"].nunique()
    avg_per_day = (monthly / days).round(2)
    st.bar_chart(avg_per_day)

    # -------------------------------------------------------
    # MoM Growth
    # -------------------------------------------------------
    st.markdown("### 📈 Month-over-Month Growth %")
    mom = (monthly.pct_change() * 100).round(2)
    st.line_chart(mom)

    # -------------------------------------------------------
    # Consistency Score
    # -------------------------------------------------------
    score = max(0, 100 - (monthly.std() / monthly.mean()) * 100)
    st.metric("🎯 Consistency Score", f"{score:.1f} / 100")

    # -------------------------------------------------------
    # Running Total Trend
    # -------------------------------------------------------
    st.markdown("### 📈 Running Total")
    dcat_sorted = dcat.sort_values("period")
    dcat_sorted["running_total"] = dcat_sorted["amount"].cumsum()
    st.line_chart(dcat_sorted.set_index("period")["running_total"])

    # -------------------------------------------------------
    # Cumulative by Month
    # -------------------------------------------------------
    st.markdown("### 📈 Cumulative Spend by Month")
    cumulative_month = monthly.cumsum()
    st.line_chart(cumulative_month)

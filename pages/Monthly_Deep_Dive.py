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
# SIDEBAR FILTERS (compact)
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

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

if exclude_cat:
    filtered = filtered[~filtered.category.isin(exclude_cat)]

if filtered.empty:
    st.warning("No data available after applying filters.")
    st.stop()

st.header("📊 Category-wise Monthly Deep Dive")

# -----------------------------------------------------------
# LOOP PER CATEGORY
# -----------------------------------------------------------
for cat in sorted(filtered["category"].unique()):
    st.divider()
    st.subheader(f"📂 {cat}")

    dcat = filtered[filtered["category"] == cat].copy()
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

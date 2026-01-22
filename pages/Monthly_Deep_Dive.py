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
st.title("📅 Monthly Deep Dive")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df["year_month"] = df["period"].dt.to_period("M").astype(str)

# -----------------------------------------------------------
# MONTHLY TOTAL SPEND
# -----------------------------------------------------------
st.subheader("📊 Monthly Total Spend")
monthly = df.groupby("year_month")["amount"].sum()
st.line_chart(monthly)

# -----------------------------------------------------------
# BEST / WORST MONTH
# -----------------------------------------------------------
best_month = monthly.idxmin()
worst_month = monthly.idxmax()

st.success(f"🟢 Best Month (lowest spend): **{best_month}** — ₹{monthly.min():.2f}")
st.error(f"🔴 Worst Month (highest spend): **{worst_month}** — ₹{monthly.max():.2f}")

# -----------------------------------------------------------
# AVERAGE SPEND PER DAY IN MONTH
# -----------------------------------------------------------
st.subheader("📆 Average Daily Spend per Month")

days_in_month = df.groupby("year_month")["period"].nunique()
avg_per_day = (monthly / days_in_month).round(2)

st.bar_chart(avg_per_day)

# -----------------------------------------------------------
# MONTH OVER MONTH GROWTH %
# -----------------------------------------------------------
st.subheader("📈 Month-over-Month Growth %")

mom = monthly.pct_change() * 100
st.line_chart(mom)

# -----------------------------------------------------------
# CATEGORY BREAKDOWN PER MONTH
# -----------------------------------------------------------
st.subheader("📂 Category Breakdown by Month")

pivot = df.pivot_table(
    values="amount",
    index="year_month",
    columns="category",
    aggfunc="sum",
    fill_value=0
)
st.dataframe(pivot)

# -----------------------------------------------------------
# DOMINANT CATEGORY EACH MONTH
# -----------------------------------------------------------
st.subheader("🏆 Dominant Category Each Month")

idx = pivot.idxmax(axis=1)
dom_cat = pd.DataFrame({
    "year_month": idx.index,
    "dominant_category": idx.values
})
st.dataframe(dom_cat)

# -----------------------------------------------------------
# BUDGET VS ACTUAL (Assume 20k)
# -----------------------------------------------------------
st.subheader("💰 Budget vs Actual Spend")

BUDGET = 20000
budget_compare = pd.DataFrame({
    "Actual Spend": monthly,
    "Budget": BUDGET
})
st.line_chart(budget_compare)

# -----------------------------------------------------------
# SAVINGS POTENTIAL INSIGHT
# -----------------------------------------------------------
st.subheader("💡 Savings Potential Insight")

over_budget_months = monthly[monthly > BUDGET]
if not over_budget_months.empty:
    excess = (over_budget_months - BUDGET).sum()
    st.warning(f"You overspent ₹{excess:,.0f} across {len(over_budget_months)} months.")
else:
    st.success("Great discipline! You stayed within budget every month.")

# -----------------------------------------------------------
# MONTHLY CONSISTENCY SCORE
# -----------------------------------------------------------
st.subheader("🎯 Monthly Consistency Score")

score = max(0, 100 - (monthly.std() / monthly.mean()) * 100)
st.metric("Consistency Score", f"{score:.1f} / 100")

# -----------------------------------------------------------
# RUNNING TOTAL CURVE
# -----------------------------------------------------------
st.subheader("📈 Running Total Trend")

df_sorted = df.sort_values("period")
df_sorted["running_total"] = df_sorted["amount"].cumsum()
st.line_chart(df_sorted.set_index("period")["running_total"])

# -----------------------------------------------------------
# MONTHLY CUMULATIVE CURVE
# -----------------------------------------------------------
st.subheader("📈 Cumulative Spend by Month")

cumulative_month = monthly.cumsum()
st.line_chart(cumulative_month)

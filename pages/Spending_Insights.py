import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
import os

# -----------------------------------------------------------
# LOAD GLOBAL CSS (important for multipage)
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("📊 Spending Insights")

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])

# -----------------------------------------------------------
# TOP CATEGORIES
# -----------------------------------------------------------
st.subheader("🏆 Top Categories by Total Spend")
cat_spend = (
    df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)
st.bar_chart(cat_spend)

# -----------------------------------------------------------
# DAILY TREND
# -----------------------------------------------------------
st.subheader("📈 Daily Spending Trend")
daily = df.groupby("period")["amount"].sum()
st.line_chart(daily)

# -----------------------------------------------------------
# AVG DAILY SPEND
# -----------------------------------------------------------
st.metric("💰 Average Daily Spend", f"₹{daily.mean():.2f}")

# -----------------------------------------------------------
# MOST EXPENSIVE DAY
# -----------------------------------------------------------
max_day = daily.idxmax()
st.write(f"💸 Most expensive day: **{max_day.date()}** — ₹{daily.max():.2f}")

# -----------------------------------------------------------
# WEEKDAY VS WEEKEND
# -----------------------------------------------------------
df["dow"] = df["period"].dt.dayofweek
df["type"] = df["dow"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

st.subheader("📅 Weekday vs Weekend Spend (Average)")
ww = df.groupby("type")["amount"].mean()
st.bar_chart(ww)

# -----------------------------------------------------------
# CATEGORY CONTRIBUTION %
# -----------------------------------------------------------
st.subheader("📂 Category Contribution (%)")

total_spend = df["amount"].sum()
cat_pct = (
    df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False) / total_spend * 100
)

st.dataframe(cat_pct.round(2).astype(str) + " %")

# -----------------------------------------------------------
# TOP 5 HIGHEST TRANSACTIONS
# -----------------------------------------------------------
st.subheader("💎 Top 5 Highest Transactions")

top_txn = df.sort_values("amount", ascending=False).head(5)[
    ["period", "category", "accounts", "amount"]
]
st.dataframe(top_txn)

# -----------------------------------------------------------
# HEATMAP DAY vs MONTH
# -----------------------------------------------------------
st.subheader("🗓 Spending Pattern (Day vs Month)")

df["day"] = df["period"].dt.day
df["month_name"] = df["period"].dt.month_name()

heat = df.pivot_table(
    values="amount",
    index="day",
    columns="month_name",
    aggfunc="sum",
    fill_value=0
)

st.dataframe(heat)

# -----------------------------------------------------------
# MONTHLY TREND + MOVING AVERAGE
# -----------------------------------------------------------
st.subheader("📈 Monthly Spend Trend (with smoothing)")

df["year_month"] = df["period"].dt.to_period("M").astype(str)
monthly = df.groupby("year_month")["amount"].sum()
ma = monthly.rolling(3).mean()

st.line_chart(pd.DataFrame({
    "Monthly Spend": monthly,
    "3-Month Moving Avg": ma
}))

# -----------------------------------------------------------
# SPEND CONSISTENCY SCORE
# -----------------------------------------------------------
st.subheader("🎯 Spend Consistency Score")

std = daily.std()
mean = daily.mean()
score = max(0, 100 - (std / mean) * 100)

st.metric("Consistency Score", f"{score:.1f} / 100")
st.caption("Higher score = more predictable spending behavior")

# -----------------------------------------------------------
# MOST EXPENSIVE CATEGORY PER MONTH
# -----------------------------------------------------------
st.subheader("🏅 Most Expensive Category Each Month")

monthly_cat = (
    df.groupby(["year_month", "category"])["amount"]
    .sum()
    .reset_index()
)

idx = monthly_cat.groupby("year_month")["amount"].idxmax()
best_cat_month = monthly_cat.loc[idx]

st.dataframe(best_cat_month.sort_values("year_month"))

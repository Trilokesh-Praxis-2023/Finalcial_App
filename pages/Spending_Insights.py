import streamlit as st
import pandas as pd
from utils.github_storage import read_csv

st.set_page_config(layout="wide")
st.title("📊 Spending Insights")

df = read_csv()
df["period"] = pd.to_datetime(df["period"])

# ---------------- Top Categories ----------------
st.subheader("🏆 Top Categories by Total Spend")
cat_spend = (
    df.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)
st.bar_chart(cat_spend)

# ---------------- Daily Trend ----------------
st.subheader("📈 Daily Spending Trend")
daily = df.groupby("period")["amount"].sum()
st.line_chart(daily)

# ---------------- Avg Daily Spend ----------------
st.metric("💰 Average Daily Spend", f"₹{daily.mean():.2f}")

# ---------------- Most Expensive Day ----------------
max_day = daily.idxmax()
st.write(f"💸 Most expensive day: **{max_day.date()}** — ₹{daily.max():.2f}")

# ---------------- Weekday vs Weekend ----------------
df["dow"] = df["period"].dt.dayofweek
df["type"] = df["dow"].apply(lambda x: "Weekend" if x >= 5 else "Weekday")

st.subheader("📅 Weekday vs Weekend Spend (Average)")
ww = df.groupby("type")["amount"].mean()
st.bar_chart(ww)

# =======================================================================
#  📊 KPI DASHBOARD MODULE — Import in app.py
# =======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime

# If no separate file exists, uncomment this & remove import ↑

def get_income(date):
    base = datetime(2024,10,1)
    date = pd.to_datetime(date)
    diff = (date.year-base.year)*12 + (date.month-base.month)
    return 12000 if diff==0 else 14112 if diff==1 else 24400


# =======================================================================
# 🔥 MINI SPARKLINE (embedded KPI chart)
# =======================================================================

import altair as alt

def sparkline(data, color="#ffbf00"):
    """Generates tiny mini trend chart for KPI"""
    if len(data) < 2:
        return None

    df = data.reset_index(drop=True).rename(columns={data.name:"value"})

    return (
        alt.Chart(df.reset_index())
        .mark_line(size=2, interpolate="monotone", color=color)
        .encode(x="index:Q", y="value:Q")
        .properties(width=120, height=30)
    )


# =======================================================================
#               🔥 MAIN RENDER FUNCTION (CALL IN APP.PY)
# =======================================================================

def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):

    if filtered is None or filtered.empty:
        st.warning("⚠ No data available for KPI dashboard.")
        return

    f = filtered.copy()
    f["period"] = pd.to_datetime(f["period"], errors="coerce")

    # ========== CORE NUMBERS ==========
    today = pd.to_datetime("today").date()
    today_spend = f[f["period"].dt.date == today]["amount"].sum()

    total_spend = f["amount"].sum()
    lifetime_spend = df["amount"].sum() if not df.empty else total_spend

    current_month_key = f["year_month"].max()
    current_month = f[f["year_month"] == current_month_key]
    current_month_total = current_month["amount"].sum()

    avg_monthly = f.groupby("year_month")["amount"].sum().mean()
    month_fmt = lambda m: pd.to_datetime(m).strftime("%b %Y") if pd.notna(m) else "-"

    # ========== WEEK STATS ==========
    f["week"] = f["period"].dt.isocalendar().week
    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly_spend = f.groupby("year_week")["amount"].sum()

    current_week = weekly_spend.iloc[-1] if len(weekly_spend) > 0 else 0
    prev_week = weekly_spend.iloc[-2] if len(weekly_spend) > 1 else 0
    wow_change = ((current_week-prev_week)/prev_week*100) if prev_week > 0 else 0

    # ===================================================================
    # 🔹 ROW 1 — CORE SPEND HEALTH + SPARKLINES
    # ===================================================================
    st.subheader("📊 Financial KPI Overview")
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.metric("💸 Total Spend", f"₹{total_spend:,.0f}")
    with c2:
        st.metric("📆 Current Month", f"₹{current_month_total:,.0f}")
    with c3:
        st.metric("📅 Today", f"₹{today_spend:,.0f}")
    with c4:
        st.metric("📅 Avg Monthly", f"₹{avg_monthly:,.0f}")

    # ===================================================================
    # 🔹 ROW 2 — MOMENTUM
    # ===================================================================
    st.markdown("### 📈 Momentum & Spend Direction")
    t1,t2,t3,t4 = st.columns(4)

    lifetime_used_pct = (total_spend/lifetime_spend*100) if lifetime_spend>0 else 0
    month_totals = f.groupby("year_month")["amount"].sum()

    t1.metric("📊 Lifetime Spend %", f"{lifetime_used_pct:.1f}%")

    if len(month_totals)>0:
        best = month_totals.idxmax()
        t2.metric("🔥 Peak Month", month_fmt(best), f"₹{month_totals.max():,.0f}")
    else:
        t2.metric("🔥 Peak Month","-")

    t3.metric("📅 Weekly Spend", f"₹{current_week:,.0f}")
    t4.metric("🔄 WoW Change", f"{wow_change:.1f}%")

    # ===================================================================
    # 🔹 ROW 3 — CATEGORY STRENGTH + DAILY PATTERN
    # ===================================================================
    st.markdown("### 🏷 Category Insight & Daily Behavior")
    r1,r2,r3,r4 = st.columns(4)

    prev_month = month_totals.iloc[-2] if len(month_totals)>1 else 0
    mom = ((current_month_total-prev_month)/prev_month*100) if prev_month>0 else 0
    r1.metric("📆 MoM Growth",f"{mom:.1f}%")

    cat_sum = f.groupby("category")["amount"].sum()
    r2.metric("🏆 Highest Spend", cat_sum.idxmax() if len(cat_sum)>0 else "-")
    r3.metric("🪫 Lowest Spend", cat_sum.idxmin() if len(cat_sum)>0 else "-")

    daily = f.groupby("period")["amount"].sum()
    r4.metric("📅 Avg/Day", f"₹{daily.mean():,.0f}" if len(daily) else "0")

    # ===================================================================
    # 🔹 ROW 4 — INCOME vs EXPENSE
    # ===================================================================
    st.markdown("### 💰 Income vs Expense Tracker")
    i1,i2,i3,i4 = st.columns(4)

    expected = get_income(current_month_key)
    balance = expected-current_month_total
    save_rate = (balance/expected*100) if expected>0 else 0
    pct = current_month_total/expected*100 if expected>0 else 0

    status = "🟢 Safe" if pct<70 else "🟡 High" if pct<100 else "🔴 Critical"

    i1.metric("💰 Income Expected", f"₹{expected:,.0f}")
    i2.metric("📊 Balance Left", f"₹{balance:,.0f}")
    i3.metric("💾 Savings Rate", f"{save_rate:.1f}%")
    i4.metric("⚡ % Spent",f"{pct:.1f}%",status)

    # ===================================================================
    # 🔹 ROW 5 — BUDGET SURVIVAL
    # ===================================================================
    st.markdown("### 💼 Budget Survival Tracker")
    spent = current_month_total
    left = MONTHLY_BUDGET - spent

    today_day = today.day
    days_total = pd.Period(today,freq="M").days_in_month
    days_left = max(days_total - today_day,1)

    daily_limit = left/days_left
    ideal_per_day = MONTHLY_BUDGET -  12800 


    c6_1,c6_2,c6_3,c6_4 = st.columns(4)
    c6_1.metric("💰 Budget Left",f"₹{left:,.0f}")
    c6_2.metric("📅 Days Left",f"{days_left} days")
    c6_3.metric("⚡ Daily Allowed",f"₹{daily_limit:,.0f}/day")
    c6_4.metric("🏁 Ideal Spend Per Day",f"₹{ideal_per_day:,.0f}")

    # ===================================================================
    # 🔹 CATEGORY SHARE TABLE (fixed)
    # ===================================================================
    st.subheader("📊 Spend Share Breakdown")

    share = cat_sum.reset_index().rename(columns={"amount":"Total Spend"})
    share["Share %"] = (share["Total Spend"]/total_spend*100).round(2)

    st.dataframe(share, use_container_width=True)  # << FIXED ERROR

    st.markdown("---")
    st.success("KPI Dashboard Loaded 🎉")


# =======================================================================
#  HOW TO USE IN app.py
# =======================================================================

"""
from kpi_dashboard import render_kpis

render_kpis(
    filtered = filtered_dataframe,
    df = original_full_df,
    MONTHLY_BUDGET = 18000
)
"""

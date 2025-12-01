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

    current_month_key = f["year_month"].unique().max()
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
    # 🔹 ROW 1 — CORE Spend Summary
    # ===================================================================
    st.subheader("📊 Financial KPI Overview")
    c1,c2,c3,c4 = st.columns(4)

    with c1: st.metric("💸 Total Spend", f"₹{total_spend:,.0f}")
    with c2: st.metric("📆 Current Month", f"₹{current_month_total:,.0f}")
    with c3: st.metric("📅 Today", f"₹{today_spend:,.0f}")
    with c4: st.metric("📅 Avg Monthly", f"₹{avg_monthly:,.0f}")


    # ===================================================================
    # 🔹 ROW 2 — (Moved from bottom) BUDGET SURVIVAL TRACKER
    # ===================================================================
    st.markdown("### 💼 Monthly Budget Survival Tracker")

    today = pd.Timestamp.today()
    month_now = today.strftime("%Y-%m")

    current_month_spend = filtered[filtered.year_month == month_now]["amount"].sum()

    MONTHLY_BUDGET = 18000
    FIXED_RENT     = 13000

    days_total = pd.Period(today, freq="M").days_in_month
    days_left = max(days_total - today.day, 1)

    daily_budget = (MONTHLY_BUDGET - FIXED_RENT) / days_total

    spent_today = filtered[filtered.period.dt.date == today.date()]["amount"].sum()
    save_today  = daily_budget - spent_today
    budget_left = MONTHLY_BUDGET - current_month_spend

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Budget Left", f"₹{budget_left:,.0f}")
    c2.metric("📅 Days Left", f"{days_left}")
    c3.metric("⚡ Daily Budget", f"₹{daily_budget:,.0f}")
    c4.metric("🛒 Spent Today", f"₹{spent_today:,.0f}")
    c5.metric("💾 Save Today", f"₹{save_today:,.0f}")


    # ===================================================================
    # 🔹 ROW 3 — CATEGORY STRENGTH
    # ===================================================================
    st.markdown("### 🏷 Category Insight & Daily Behavior")
    r1,r2,r3,r4 = st.columns(4)

    month_totals = f.groupby("year_month")["amount"].sum()
    prev_month = month_totals.iloc[-2] if len(month_totals)>1 else 0
    mom = ((current_month_total-prev_month)/prev_month*100) if prev_month>0 else 0

    cat_sum = f.groupby("category")["amount"].sum()
    daily = f.groupby("period")["amount"].sum()

    r1.metric("📆 MoM Growth",f"{mom:.1f}%")
    r2.metric("🏆 Highest Spend", cat_sum.idxmax() if len(cat_sum)>0 else "-")
    r3.metric("🪫 Lowest Spend", cat_sum.idxmin() if len(cat_sum)>0 else "-")
    r4.metric("📅 Avg/Day", f"₹{daily.mean():,.0f}" if len(daily) else "0")


    # ===================================================================
    # 🔹 ROW 4 — Income vs Expense
    # ===================================================================
    st.markdown("### 💰 Income vs Expense Balance")
    i1,i2,i3,i4 = st.columns(4)

    expected = get_income(current_month_key)
    balance = expected-current_month_total
    save_rate = (balance/expected*100) if expected>0 else 0
    pct = current_month_total/expected*100 if expected>0 else 0

    status = "🟢 Safe" if pct<70 else "🟡 High" if pct<100 else "🔴 Critical"

    i1.metric("💰 Expected Income", f"₹{expected:,.0f}")
    i2.metric("📊 Balance Left", f"₹{balance:,.0f}")
    i3.metric("💾 Savings Rate", f"{save_rate:.1f}%")
    i4.metric("⚡ % Spent",f"{pct:.1f}%",status)


    # ===================================================================
    # 🔹 ROW 5 — (Moved from top) MOMENTUM + TRENDS
    # ===================================================================
    st.markdown("### 📈 Momentum & Weekly Direction")
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
    # 🔹 SPEND SHARE BREAKDOWN
    # ===================================================================
    st.subheader("📊 Spend Share Breakdown")

    share = cat_sum.reset_index().rename(columns={"amount":"Total Spend"})
    share["Share %"] = (share["Total Spend"]/total_spend*100).round(2)
    st.dataframe(share, use_container_width=True)

    st.markdown("---")
    st.success("KPI Dashboard Updated & Reordered 🔄✨")


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

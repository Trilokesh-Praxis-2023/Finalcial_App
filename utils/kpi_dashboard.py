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

def calc_income(year_month: str) -> float:
    """
    Income logic based on month:
        Oct 2024 → 12000
        Nov 2024 → 14112
        Dec 2024 onward → 24200
    """
    try:
        ym = pd.to_datetime(year_month, format="%Y-%m")
    except:
        return 0.0

    if ym == pd.Timestamp(2024, 10, 1):
        return 12000
    elif ym == pd.Timestamp(2024, 11, 1):
        return 14112
    elif ym >= pd.Timestamp(2024, 12, 1):
        return 24200
    return 0.0



def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):

    # --- safety checks
    if filtered is None or filtered.empty:
        st.warning("⚠ No data available for KPI dashboard.")
        return

    # work on a copy
    f = filtered.copy()

    # ensure period is datetime
    f["period"] = pd.to_datetime(f.get("period", f.get("date", None)), errors="coerce")

    # ensure year_month column exists (yyyy-mm)
    if "year_month" not in f.columns:
        f["year_month"] = f["period"].dt.to_period("M").astype(str)

    # ensure amount numeric
    f["amount"] = pd.to_numeric(f["amount"], errors="coerce").fillna(0.0)

    # ---------- CORE NUMBERS ----------
    today = pd.to_datetime("today").date()
    today_spend = f[f["period"].dt.date == today]["amount"].sum()

    total_spend = f["amount"].sum()
    lifetime_spend = df["amount"].sum() if (df is not None and not df.empty and "amount" in df.columns) else total_spend

    # current month key (safely)
    month_keys = f["year_month"].dropna().unique()
    current_month_key = month_keys.max() if len(month_keys) > 0 else pd.to_datetime(today).strftime("%Y-%m")
    current_month = f[f["year_month"] == current_month_key]
    current_month_total = current_month["amount"].sum()

    avg_monthly = f.groupby("year_month")["amount"].sum().mean() if len(f["year_month"].unique()) > 0 else 0.0
    month_fmt = lambda m: pd.to_datetime(m).strftime("%b %Y") if pd.notna(m) and m != "" else "-"

    # ---------- INCOME CALCULATION ----------
    unique_months = sorted(f["year_month"].unique())
    total_income = sum(calc_income(m) for m in unique_months)
    current_month_income = calc_income(current_month_key)

    # ---------- WEEK STATS ----------
    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly_spend = f.groupby("year_week")["amount"].sum().sort_index()

    current_week = weekly_spend.iloc[-1] if len(weekly_spend) > 0 else 0.0
    prev_week = weekly_spend.iloc[-2] if len(weekly_spend) > 1 else 0.0
    wow_change = ((current_week - prev_week) / prev_week * 100) if prev_week > 0 else 0.0

    # ========== ROW 1 — CORE KPIs (UPDATED WITH INCOME) ==========
    st.subheader("📊 Financial KPI Overview")
    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric("💰 Total Income", f"₹{total_income:,.0f}")

    with c2:
        st.metric("💸 Total Spend", f"₹{total_spend:,.0f}")

    with c3:
        st.metric("📆 Current Month Spend", f"₹{current_month_total:,.0f}")

    with c4:
        st.metric("💰 Current Month Income", f"₹{current_month_income:,.0f}")

    with c5:
        st.metric("📅 Today", f"₹{today_spend:,.0f}")


    # ========== ROW 2 — BUDGET SURVIVAL TRACKER ==========
    st.markdown("### 💼 Monthly Budget Survival Tracker")

    now_ts = pd.Timestamp.now()
    month_now = now_ts.strftime("%Y-%m")

    current_month_spend = f[f["year_month"] == month_now]["amount"].sum()

    FIXED_RENT = 11000 + 400 + 588 + 470

    days_total = pd.Period(now_ts, freq="M").days_in_month
    days_left = max(days_total - now_ts.day, 1)

    daily_budget = (MONTHLY_BUDGET - FIXED_RENT) / days_total if days_total > 0 else 0.0
    spent_today = f[f["period"].dt.date == today]["amount"].sum()
    save_today = daily_budget - spent_today
    budget_left = MONTHLY_BUDGET - current_month_spend

    daily_allowed_left = budget_left / days_left if days_left > 0 else 0.0

    b1, b2, b3, b4, b5, b6 = st.columns(6)
    b1.metric("💰 Budget Left", f"₹{budget_left:,.0f}")
    b2.metric("📅 Days Left", f"{days_left}")
    b3.metric("⚡ Daily Budget", f"₹{daily_budget:,.0f}")
    b4.metric("🛒 Spent Today", f"₹{spent_today:,.0f}")
    b5.metric("💾 Save Today", f"₹{save_today:,.0f}")
    b6.metric("📊 Daily Allowed (Remaining)", f"₹{daily_allowed_left:,.0f}")


    # ========== ROW 3 — CATEGORY STRENGTH ==========
    st.markdown("### 🏷 Category Insight & Daily Behavior")
    r1, r2, r3, r4 = st.columns(4)

    month_totals = f.groupby("year_month")["amount"].sum().sort_index()
    prev_month_total = month_totals.iloc[-2] if len(month_totals) > 1 else 0.0
    mom = ((current_month_total - prev_month_total) / prev_month_total * 100) if prev_month_total > 0 else 0.0

    cat_sum = f.groupby("category")["amount"].sum().sort_values(ascending=False)
    daily = f.groupby("period")["amount"].sum().sort_index()

    r1.metric("📆 MoM Growth", f"{mom:.1f}%")
    r2.metric("🏆 Highest Spend", cat_sum.idxmax() if len(cat_sum) > 0 else "-")
    r3.metric("🪫 Lowest Spend", cat_sum.idxmin() if len(cat_sum) > 0 else "-")
    r4.metric("📅 Avg/Day", f"₹{daily.mean():,.0f}" if len(daily) > 0 else "₹0")


    # ========== ROW 4 — INCOME vs EXPENSE ==========
    st.markdown("### 💰 Income vs Expense Tracker")
    i1, i2, i3, i4 = st.columns(4)

    expected = current_month_income
    balance = expected - current_month_total
    save_rate = (balance / expected * 100) if expected > 0 else 0.0
    pct = (current_month_total / expected * 100) if expected > 0 else 0.0

    status = "🟢 Safe" if pct < 70 else "🟡 High" if pct < 100 else "🔴 Critical"

    i1.metric("💰 Income Expected", f"₹{expected:,.0f}")
    i2.metric("📊 Balance Left", f"₹{balance:,.0f}")
    i3.metric("💾 Savings Rate", f"{save_rate:.1f}%")
    i4.metric("⚡ % Spent", f"{pct:.1f}%", status)


    # ========== ROW 5 — MOMENTUM & SPEND DIRECTION ==========
    st.markdown("### 📈 Momentum & Spend Direction")
    t1, t2, t3, t4 = st.columns(4)

    lifetime_used_pct = (total_spend / lifetime_spend * 100) if lifetime_spend > 0 else 0.0
    month_totals_for_peak = f.groupby("year_month")["amount"].sum()

    t1.metric("📊 Lifetime Spend %", f"{lifetime_used_pct:.1f}%")

    if len(month_totals_for_peak) > 0:
        best = month_totals_for_peak.idxmax()
        t2.metric("🔥 Peak Month", month_fmt(best), f"₹{month_totals_for_peak.max():,.0f}")
    else:
        t2.metric("🔥 Peak Month", "-")

    t3.metric("📅 Weekly Spend", f"₹{current_week:,.0f}")
    t4.metric("🔄 WoW Change", f"{wow_change:.1f}%")

    # ========== SPEND SHARE ==========
    st.subheader("📊 Spend Share Breakdown")
    share = cat_sum.reset_index().rename(columns={"amount": "Total Spend"})
    share["Share %"] = (share["Total Spend"] / total_spend * 100).round(2) if total_spend > 0 else 0.0
    st.dataframe(share, use_container_width=True)

    st.markdown("---")
    st.success("KPI Dashboard Loaded ✅")


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

# =======================================================================
#  📊 KPI DASHBOARD MODULE — Import in app.py
# =======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt

# If no separate file exists, uncomment this & remove import ↑


# =======================================================================
# 🔥 MINI SPARKLINE (embedded KPI chart)
# =======================================================================


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

def get_income(month):
    return calc_income(month)

def fmt_k(n):
    """Format number into K/M style for metrics."""
    try:
        n = float(n)
    except:
        return n

    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    return f"{n:,.0f}"

def rup(n):
    return f"₹{fmt_k(n)}"


def calc_income(year_month: str) -> float:
    """Income logic for each month."""
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

    if filtered is None or filtered.empty:
        st.warning("⚠ No data available for KPI dashboard.")
        return

    # =====================================================
    # 🔧 PREP
    # =====================================================
    f = filtered.copy()
    f["period"] = pd.to_datetime(f["period"], errors="coerce")
    f["year_month"] = f["period"].dt.to_period("M").astype(str)
    f["amount"] = pd.to_numeric(f["amount"], errors="coerce").fillna(0.0)

    today = pd.to_datetime("today").date()

    month_totals = f.groupby("year_month")["amount"].sum().sort_index()
    cat_sum = f.groupby("category")["amount"].sum().sort_values(ascending=False)

    today_spend = f[f["period"].dt.date == today]["amount"].sum()
    total_spend = f["amount"].sum()

    # =====================================================
    # 📆 CURRENT MONTH
    # =====================================================
    month_keys = sorted(f["year_month"].unique())
    current_month_key = month_keys[-1]
    current_month_spend = month_totals.get(current_month_key, 0.0)

    # =====================================================
    # 💰 INCOME (for reporting only)
    # =====================================================
    total_income = sum(calc_income(m) for m in month_keys)
    current_month_income = calc_income(current_month_key)

    pct_spent = (
        current_month_spend / current_month_income * 100
        if current_month_income > 0 else 0
    )

    # =====================================================
    # 💼 BUDGET LOGIC — ₹20,000 FIXED
    # =====================================================
    TOTAL_MONTHLY_BUDGET = 20000

    VARIABLE_BUDGET = TOTAL_MONTHLY_BUDGET 

    now_ts = pd.Timestamp.now()
    days_total = pd.Period(now_ts, freq="M").days_in_month
    days_left = max(days_total - now_ts.day, 1)

    budget_left = VARIABLE_BUDGET - current_month_spend
    daily_allowed_left = max(budget_left / days_left, 0)

    spend_velocity = (
        current_month_spend / now_ts.day if now_ts.day > 0 else 0
    )

    # =====================================================
    # 📈 MoM
    # =====================================================
    if len(month_totals) > 1:
        prev_month = month_totals.iloc[-2]
        mom = (
            (current_month_spend - prev_month) / prev_month * 100
            if prev_month > 0 else 0
        )
    else:
        mom = 0

    # =====================================================
    # 📊 WoW
    # =====================================================
    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly = f.groupby("year_week")["amount"].sum().sort_index()

    current_week = weekly.iloc[-1] if len(weekly) else 0
    prev_week = weekly.iloc[-2] if len(weekly) > 1 else 0
    wow = (
        (current_week - prev_week) / prev_week * 100
        if prev_week > 0 else 0
    )

    # =====================================================
    # ========== ROW 1 — CORE KPIs ==========
    # =====================================================
    st.subheader("📊 Financial KPI Overview")
    a1, a2, a3, a4 = st.columns(4)

    a1.metric("💰 Total Income", rup(total_income))
    a2.metric("💸 Total Spend", rup(total_spend))
    a3.metric("🛒 Today Spend", rup(today_spend))
    a4.metric("⚡ % Spent (Income)", f"{pct_spent:.1f}%")

    # =====================================================
    # ========== ROW 2 — BUDGET HEALTH ==========
    # =====================================================
    st.markdown("### 💼 Monthly Budget Health")

    b1, b2, b3, b4, b5 = st.columns(5)

    b1.metric("💰 Variable Budget Left", rup(budget_left))
    b2.metric("📅 Days Left", days_left)
    b3.metric("⚡ Daily Allowed", rup(daily_allowed_left))
    b4.metric("📆 Month Spend", rup(current_month_spend))
    b5.metric("🚀 Spend Velocity", rup(spend_velocity))

    # =====================================================
    # ========== ROW 3 — TRENDS ==========
    # =====================================================
    st.markdown("### 📈 Trends & Growth")
    t1, t2 = st.columns(2)

    t1.metric("📆 MoM Growth", f"{mom:.1f}%")
    t2.metric("🔄 WoW Change", f"{wow:.1f}%")

    # =====================================================
    # ========== ROW 4 — CATEGORY INSIGHTS ==========
    # =====================================================
    st.markdown("### 🏷 Category Insights")

    if len(cat_sum):
        st.metric("🏆 Highest Spend Category", cat_sum.idxmax())
    else:
        st.metric("🏆 Highest Spend Category", "-")

    st.subheader("📊 Spend Share Breakdown")

    share = cat_sum.reset_index().rename(columns={"amount": "Total Spend"})
    share["Share %"] = (
        (share["Total Spend"] / total_spend * 100).round(2)
        if total_spend > 0 else 0
    )
    share["Total Spend"] = share["Total Spend"].apply(rup)

    st.dataframe(share, use_container_width=True)

    st.success("KPI Dashboard Loaded ✅")

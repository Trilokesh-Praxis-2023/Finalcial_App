# =======================================================================
#  📊 KPI DASHBOARD MODULE — With LIVE Pulse Animated KPIs
# =======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime
import altair as alt

# =======================================================================
#  Income Model (used in Income vs Expense section)
# =======================================================================

def get_income(date):
    base = datetime(2024,10,1)
    date = pd.to_datetime(date)
    diff = (date.year-base.year)*12 + (date.month-base.month)

    # You can expand this later if income changes monthly
    return 12000 if diff==0 else 14112 if diff==1 else 24400   # <— FIXED


# ===========================================================
# 🔥 Sparkline generator (unchanged)
# ===========================================================
def sparkline(data, color="#ffbf00"):
    if len(data) < 2:
        return None
    df = data.reset_index(drop=True).rename(columns={data.name:"value"})
    return (
        alt.Chart(df.reset_index())
        .mark_line(size=2, interpolate="monotone", color=color)
        .encode(x="index:Q", y="value:Q")
        .properties(width=120, height=30)
    )


# ===========================================================
# ⛽ Auto KPI Pulse Decision
# -----------------------------------------------------------
# prev — previous period / comparison baseline
# value — current KPI
# Returns CSS class to attach for pulse
# ===========================================================
def pulse(prev, value):
    if prev is None: return "kpi-pulse"                  # default soft pulse
    if value > prev: return "kpi-grow"                   # green growth
    if value < prev: return "kpi-drop"                   # red warning drop
    return "kpi-pulse"                                   # neutral gold beat


# ===========================================================
#  MAIN RENDER FUNCTION (Call inside app.py)
# ===========================================================
def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):

    if filtered is None or filtered.empty:
        st.warning("⚠ No data available for KPI dashboard.")
        return

    f = filtered.copy()
    f["period"] = pd.to_datetime(f["period"], errors="coerce")

    # ===================== CORE NUMBERS =====================
    today = pd.to_datetime("today").date()
    today_spend = f[f["period"].dt.date == today]["amount"].sum()

    total_spend = f["amount"].sum()
    lifetime_spend = df["amount"].sum() if not df.empty else total_spend

    current_month_key = f["year_month"].max()
    current_month = f[f["year_month"] == current_month_key]
    current_month_total = current_month["amount"].sum()

    avg_monthly = f.groupby("year_month")["amount"].sum().mean()

    # ===================== WEEKLY METRICS =====================
    f["week"] = f["period"].dt.isocalendar().week
    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly_spend = f.groupby("year_week")["amount"].sum()

    current_week = weekly_spend.iloc[-1] if len(weekly_spend)>0 else 0
    prev_week    = weekly_spend.iloc[-2] if len(weekly_spend)>1 else None

    wow_change = round(((current_week-prev_week)/prev_week*100),1) if prev_week else 0

    # ===========================================================
    # 🔥 ROW 1 — PULSING KPI CARD VALUES
    # ===========================================================
    st.subheader("📊 Financial KPI Overview")
    c1,c2,c3,c4 = st.columns(4)

    c1.markdown(f"<h2 class='{pulse(None,total_spend)}'>₹{total_spend:,.0f}</h2><p>Total Spend</p>",unsafe_allow_html=True)
    c2.markdown(f"<h2 class='{pulse(avg_monthly,current_month_total)}'>₹{current_month_total:,.0f}</h2><p>Current Month</p>",unsafe_allow_html=True)
    c3.markdown(f"<h2 class='{pulse(None,today_spend)}'>₹{today_spend:,.0f}</h2><p>Today</p>",unsafe_allow_html=True)
    c4.markdown(f"<h2 class='{pulse(None,avg_monthly)}'>₹{avg_monthly:,.0f}</h2><p>Avg Monthly</p>",unsafe_allow_html=True)

    # ===========================================================
    # 🔥 ROW 2 — MOMENTUM
    # ===========================================================
    st.markdown("### 📈 Momentum & Spend Direction")
    t1,t2,t3,t4 = st.columns(4)

    lifetime_pct = (total_spend/lifetime_spend*100) if lifetime_spend>0 else 0
    month_totals = f.groupby("year_month")["amount"].sum()

    t1.markdown(f"<h3 class='{pulse(None,lifetime_pct)}'>{lifetime_pct:.1f}%</h3><p>Lifetime Spend %</p>",unsafe_allow_html=True)

    if len(month_totals):
        best = month_totals.idxmax()
        t2.markdown(f"<h3 class='kpi-pulse'>{best}</h3><p>Peak Month</p>",unsafe_allow_html=True)

    t3.markdown(f"<h3 class='{pulse(prev_week,current_week)}'>₹{current_week:,.0f}</h3><p>Weekly Spend</p>",unsafe_allow_html=True)
    t4.markdown(f"<h3 class='{pulse(None,wow_change)}'>{wow_change:.1f}%</h3><p>WoW Change</p>",unsafe_allow_html=True)

    # ===========================================================
    # 🔥 ROW 3 — CATEGORY & BEHAVIOR
    # ===========================================================
    st.markdown("### 🏷 Category Insight & Daily Behavior")
    r1,r2,r3,r4 = st.columns(4)

    prev_month = month_totals.iloc[-2] if len(month_totals)>1 else None
    mom = round(((current_month_total-prev_month)/prev_month*100),1) if prev_month else 0

    r1.markdown(f"<h3 class='{pulse(prev_month,current_month_total)}'>{mom:.1f}%</h3><p>MoM Growth</p>",unsafe_allow_html=True)

    cat_sum = f.groupby("category")["amount"].sum()
    r2.metric("🏆 Highest Spend", cat_sum.idxmax() if len(cat_sum)>0 else "-")
    r3.metric("🪫 Lowest Spend", cat_sum.idxmin() if len(cat_sum)>0 else "-")

    daily = f.groupby("period")["amount"].sum()
    r4.markdown(f"<h3 class='kpi-pulse'>₹{daily.mean():,.0f}</h3><p>Avg/day</p>",unsafe_allow_html=True)

    # ===========================================================
    # 🔥 ROW 4 — INCOME VS EXPENSE — PULSE ATTACHED HERE
    # ===========================================================
    st.markdown("### 💰 Income vs Expense Balance")
    i1,i2,i3,i4 = st.columns(4)

    expected = get_income(current_month_key)
    balance = expected-current_month_total
    save_rate = (balance/expected*100) if expected else 0
    pct = (current_month_total/expected*100) if expected else 0

    i1.markdown(f"<h3 class='kpi-pulse'>₹{expected:,.0f}</h3><p>Income Expected</p>",unsafe_allow_html=True)
    i2.markdown(f"<h3 class='{pulse(None,balance)}'>₹{balance:,.0f}</h3><p>Balance Left</p>",unsafe_allow_html=True)
    i3.markdown(f"<h3 class='{pulse(None,save_rate)}'>{save_rate:.1f}%</h3><p>Savings Rate</p>",unsafe_allow_html=True)
    i4.markdown(f"<h3 class='{pulse(None,pct)}'>{pct:.1f}%</h3><p>Percent Spent</p>",unsafe_allow_html=True)

    # ===========================================================
    # 🔥 ROW 5 — BUDGET SURVIVAL — Pulse On Required Savings
    # ===========================================================
    st.markdown("### 💼 Budget Survival Tracker")

    today = pd.Timestamp.today()
    current_month_total = filtered[filtered.year_month==today.strftime("%Y-%m")]["amount"].sum()

    spent = current_month_total
    left  = MONTHLY_BUDGET - spent

    days_total = pd.Period(today,freq="M").days_in_month
    days_left  = max(days_total - today.day, 1)

    daily_limit   = left/days_left
    ideal_per_day = (18000 - 12800)/days_total
    save_per_day  = ideal_per_day - daily_limit   # ← pulsed feature

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(f"<h3 class='{pulse(None,left)}'>₹{left:,.0f}</h3><p>Budget Left</p>",unsafe_allow_html=True)
    c2.markdown(f"<h3 class='kpi-pulse'>{days_left}</h3><p>Days Left</p>",unsafe_allow_html=True)
    c3.markdown(f"<h3 class='{pulse(None,daily_limit)}'>₹{daily_limit:,.0f}</h3><p>Allowed/day</p>",unsafe_allow_html=True)
    c4.markdown(f"<h3 class='kpi-pulse'>₹{ideal_per_day:,.0f}</h3><p>Ideal/day</p>",unsafe_allow_html=True)
    c5.markdown(f"<h3 class='{pulse(None,save_per_day)}'>₹{save_per_day:,.0f}</h3><p>Save/day Required</p>",unsafe_allow_html=True)

    # ===========================================================
    # 📊 CATEGORY SHARE TABLE
    # ===========================================================
    st.subheader("📊 Spend Share Breakdown")
    share = cat_sum.reset_index().rename(columns={"amount":"Total Spend"})
    share["Share %"] = (share["Total Spend"]/total_spend*100).round(2)
    st.dataframe(share,use_container_width=True)

    st.markdown("---")
    st.success("🚀 KPI Dashboard Loaded With Pulse Animation")

# =======================================================================
#  📊 KPI DASHBOARD MODULE — Import in app.py
# =======================================================================

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import altair as alt
import os, joblib


# =======================================================================
# 🔥 MINI SPARKLINE (embedded KPI chart)
# =======================================================================

def sparkline(data, color="#ffbf00"):
    """Generates tiny mini trend chart for KPI"""
    if len(data) < 2:
        return None

    df = data.reset_index(drop=True).rename(columns={data.name: "value"})

    return (
        alt.Chart(df.reset_index())
        .mark_line(size=2, interpolate="monotone", color=color)
        .encode(x="index:Q", y="value:Q")
        .properties(width=120, height=30)
    )


# =======================================================================
# 💰 FORMATTING HELPERS
# =======================================================================

def fmt_k(n):
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


# =======================================================================
# 💼 INCOME LOGIC
# =======================================================================

def calc_income(year_month: str) -> float:
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


# =======================================================================
# 🔮 CURRENT MONTH FORECAST (FROM DAILY ML MODEL)
# =======================================================================

def get_current_month_forecast(
    filtered,
    DAILY_MODEL_PATH="models/daily_forecast_model.pkl"
):
    if not os.path.exists(DAILY_MODEL_PATH):
        return None

    model = joblib.load(DAILY_MODEL_PATH)

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    today = pd.Timestamp.today().normalize()
    end_of_month = today + pd.offsets.MonthEnd(0)

    future_dates = pd.date_range(today, end_of_month)

    if future_dates.empty:
        return None

    future = pd.DataFrame({"period": future_dates})
    future["day"]   = future["period"].dt.day
    future["dow"]   = future["period"].dt.dayofweek
    future["month"] = future["period"].dt.month
    future["t"]     = range(len(daily), len(daily) + len(future))

    remaining_forecast = model.predict(
        future[["day","dow","month","t"]]
    ).clip(0).sum()

    spent_so_far = df[
        df["period"].dt.to_period("M") == today.to_period("M")
    ]["amount"].sum()

    return {
        "spent_so_far": spent_so_far,
        "remaining_forecast": remaining_forecast,
        "forecast_total": spent_so_far + remaining_forecast
    }


# =======================================================================
#               🔥 MAIN KPI RENDER FUNCTION
# =======================================================================

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
    # 💰 INCOME
    # =====================================================
    total_income = sum(calc_income(m) for m in month_keys)
    current_month_income = calc_income(current_month_key)

    pct_spent = (
        current_month_spend / current_month_income * 100
        if current_month_income > 0 else 0
    )

    # =====================================================
    # 💼 BUDGET LOGIC
    # =====================================================
    TOTAL_MONTHLY_BUDGET = MONTHLY_BUDGET

    now_ts = pd.Timestamp.now()
    days_total = pd.Period(now_ts, freq="M").days_in_month
    days_left = max(days_total - now_ts.day, 1)

    budget_left = TOTAL_MONTHLY_BUDGET - current_month_spend
    daily_allowed_left = max(budget_left / days_left, 0)

    spend_velocity = (
        current_month_spend / now_ts.day if now_ts.day > 0 else 0
    )

    # =====================================================
    # 📈 MoM & WoW
    # =====================================================
    mom = (
        ((month_totals.iloc[-1] - month_totals.iloc[-2]) / month_totals.iloc[-2] * 100)
        if len(month_totals) > 1 and month_totals.iloc[-2] > 0 else 0
    )

    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly = f.groupby("year_week")["amount"].sum().sort_index()

    wow = (
        ((weekly.iloc[-1] - weekly.iloc[-2]) / weekly.iloc[-2] * 100)
        if len(weekly) > 1 and weekly.iloc[-2] > 0 else 0
    )

    # =====================================================
    # 🔮 FORECAST
    # =====================================================
    forecast = get_current_month_forecast(filtered)

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

    b1.metric("💰 Budget Left", rup(budget_left))
    b2.metric("📅 Days Left", days_left)
    b3.metric("⚡ Daily Allowed", rup(daily_allowed_left))
    b4.metric("📆 Month Spend", rup(current_month_spend))
    b5.metric("🚀 Spend Velocity", rup(spend_velocity))

    # =====================================================
    # ========== ROW 3 — FORECAST KPIs ==========
    # =====================================================
    st.markdown("### 🔮 Forecast Outlook (AI)")

    f1, f2, f3 = st.columns(3)

    if forecast:
        f1.metric("🤖 Forecasted Month Spend", rup(forecast["forecast_total"]))
        f2.metric(
            "📊 Forecast vs Budget",
            rup(forecast["forecast_total"] - TOTAL_MONTHLY_BUDGET),
            delta="Over Budget" if forecast["forecast_total"] > TOTAL_MONTHLY_BUDGET else "Under Budget"
        )
        f3.metric(
            "⚡ Forecast Daily Avg (Remaining)",
            rup(forecast["remaining_forecast"] / days_left if days_left > 0 else 0)
        )
    else:
        f1.metric("🤖 Forecasted Month Spend", "—")
        f2.metric("📊 Forecast vs Budget", "—")
        f3.metric("⚡ Forecast Daily Avg", "—")

    # =====================================================
    # ========== ROW 4 — TRENDS ==========
    # =====================================================
    st.markdown("### 📈 Trends & Growth")

    t1, t2 = st.columns(2)
    t1.metric("📆 MoM Growth", f"{mom:.1f}%")
    t2.metric("🔄 WoW Change", f"{wow:.1f}%")

    # =====================================================
    # ========== ROW 5 — CATEGORY INSIGHTS ==========
    # =====================================================
    st.markdown("### 🏷 Category Insights")

    st.metric(
        "🏆 Highest Spend Category",
        cat_sum.idxmax() if len(cat_sum) else "-"
    )

    share = cat_sum.reset_index().rename(columns={"amount": "Total Spend"})
    share["Share %"] = (
        (share["Total Spend"] / total_spend * 100).round(2)
        if total_spend > 0 else 0
    )
    share["Total Spend"] = share["Total Spend"].apply(rup)

    st.dataframe(share, use_container_width=True)

    st.success("KPI Dashboard Loaded ✅")

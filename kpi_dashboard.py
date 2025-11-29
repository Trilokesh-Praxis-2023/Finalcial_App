# ================================================================
#  📊 ADVANCED KPI DASHBOARD MODULE (IMPORT & USE IN app.py)
# ================================================================

import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime


# ================================================================
# 🔥 Income Function (Your original logic — reused everywhere)
# ================================================================
def get_income(date):
    base = datetime(2024, 10, 1)
    date = pd.to_datetime(date)
    diff = (date.year - base.year) * 12 + (date.month - base.month)
    return 12000 if diff == 0 else 14112 if diff == 1 else 24400






# ===================================================================
# 🚀 MAIN FUNCTION — CALL THIS IN app.py TO DISPLAY KPI PANEL
# ===================================================================
def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):

    if filtered is None or filtered.empty:
        st.info("No data available for KPI dashboard. Adjust filters or add entries.")
        return

    # Make a working copy so we don't mutate the original df used outside
    f = filtered.copy()

    # ===== PRE CALCULATIONS =====
    today = pd.to_datetime("today").date()
    f["period"] = pd.to_datetime(f["period"], errors="coerce")

    today_spend = f[f["period"].dt.date == today]["amount"].sum()

    total_spend = f["amount"].sum()
    lifetime_spend = df["amount"].sum() if not df.empty else total_spend

    current_month_key = f["year_month"].max()
    current_month = f[f["year_month"] == current_month_key]
    current_month_total = current_month["amount"].sum()

    avg_monthly = f.groupby("year_month")["amount"].sum().mean()
    month_fmt = lambda m: pd.to_datetime(m).strftime("%b %Y") if pd.notna(m) else "-"

    # ===== WEEKLY =====
    f["week"] = f["period"].dt.isocalendar().week
    f["year_week"] = f["period"].dt.strftime("%Y-W%U")
    weekly_spend = f.groupby("year_week")["amount"].sum()

    current_week_total = weekly_spend.iloc[-1] if len(weekly_spend) > 0 else 0
    previous_week = weekly_spend.iloc[-2] if len(weekly_spend) > 1 else 0
    wow_change = (
        (current_week_total - previous_week) / previous_week * 100
        if previous_week > 0
        else 0
    )

    # =========================================================
    # 🔹 ROW 1 — CORE SPEND HEALTH + SPARKLINES
    # =========================================================
    st.subheader("📊 Financial KPI Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("💸 Total Spend (Filtered)", f"₹{total_spend:,.0f}")

    with c2:
        st.metric("📆 Current Month Spend", f"₹{current_month_total:,.0f}")

    with c3:
        st.metric("📅 Today", f"₹{today_spend:,.0f}")

    with c4:
        st.metric("📅 Avg Monthly Spend", f"₹{avg_monthly:,.0f}")
        month_series = f.groupby("year_month")["amount"].sum().reset_index()

    # =========================================================
    # 🔹 ROW 2 — MOMENTUM & TREND DIRECTION
    # =========================================================
    st.markdown("### 📈 Momentum & Spend Direction")
    t1, t2, t3, t4 = st.columns(4)

    lifetime_used_pct = (total_spend / lifetime_spend * 100) if lifetime_spend > 0 else 0
    t1.metric("📊 Lifetime Spend % Used", f"{lifetime_used_pct:.1f}%")

    month_totals = f.groupby("year_month")["amount"].sum()
    if len(month_totals) > 0:
        best_month = month_totals.idxmax()
        best_month_amt = month_totals.max()
        t2.metric("🔥 Peak Month", month_fmt(best_month), f"₹{best_month_amt:,.0f}")
    else:
        t2.metric("🔥 Peak Month", "-")

    t3.metric("📅 Current Week Spend", f"₹{current_week_total:,.0f}")
    t4.metric("🔄 WoW Change", f"{wow_change:.1f}%", delta_color="inverse")

    # =========================================================
    # 🔹 ROW 3 — CATEGORY STRENGTH & DAILY PATTERN
    # =========================================================
    st.markdown("### 🏷 Category Strength + Daily Pattern")
    r1, r2, r3, r4 = st.columns(4)

    prev_month = month_totals.iloc[-2] if len(month_totals) > 1 else 0
    mom_change = (
        (current_month_total - prev_month) / prev_month * 100
        if prev_month > 0
        else 0
    )
    r1.metric("📆 MoM Spend Change", f"{mom_change:.1f}%")

    cat_sum = f.groupby("category")["amount"].sum()
    r2.metric("🏆 Top Category", cat_sum.idxmax() if len(cat_sum) > 0 else "-")
    r3.metric("🪫 Lowest Category", cat_sum.idxmin() if len(cat_sum) > 0 else "-")

    daily_series = f.groupby("period")["amount"].sum()
    daily_avg = daily_series.mean() if len(daily_series) > 0 else 0
    r4.metric("📅 Avg/Day Spend", f"₹{daily_avg:,.0f}")

    # =========================================================
    # 🔹 ROW 4 — INCOME vs EXPENSE IMPACT
    # =========================================================
    st.markdown("### 💰 Income vs Expense Impact")
    i1, i2, i3, i4 = st.columns(4)

    expected_income = get_income(current_month_key)
    balance_left = expected_income - current_month_total
    save_rate = (balance_left / expected_income * 100) if expected_income > 0 else 0

    expense_pct = current_month_total / expected_income * 100 if expected_income > 0 else 0
    status = (
        "🟢 Safe" if expense_pct < 70 else
        "🟡 High" if expense_pct < 100 else
        "🔴 Critical"
    )

    i1.metric("💰 Income Expected", f"₹{expected_income:,.0f}")
    i2.metric("📊 Balance Left", f"₹{balance_left:,.0f}")
    i3.metric("💾 Savings Rate", f"{save_rate:.1f}%")
    i4.metric("⚡ % Income Spent", f"{expense_pct:.1f}%", status)

    # Extra baseline KPIs (safe daily target / ratio)
    b1, b2 = st.columns(2)
    month_days_passed = max(1, datetime.now().day)
    ideal_daily = expected_income / 30 if expected_income > 0 else 0
    ratio = expense_pct  # reuse
    b1.metric("📆 Ideal Daily Spend Target", f"₹{ideal_daily:,.0f}/day")
    b2.metric("📉 Spend vs Income Ratio", f"{ratio:.1f}%")

    # =========================================================
    # 🔹 ROW 5 — ACTIVITY / LIFETIME HEALTH
    # =========================================================
    st.markdown("### 📅 Lifetime Activity")
    a1, a2, a3, a4 = st.columns(4)

    active_days = f["period"].nunique()
    a1.metric("📆 Active Days Logged", f"{active_days} days")

    monthly_full = df.groupby("year_month")["amount"].sum()
    income_hist = [get_income(m) for m in monthly_full.index] if len(monthly_full) > 0 else []
    total_income = sum(income_hist)
    lifetime_savings = total_income - lifetime_spend
    burn_pct = (lifetime_spend / total_income * 100) if total_income > 0 else 0

    a2.metric("💰 Total Income Est.", f"₹{total_income:,.0f}")
    a3.metric(
        "🏦 Lifetime Savings",
        f"₹{lifetime_savings:,.0f}",
        "🟢" if lifetime_savings > 0 else "🔴",
    )
    a4.metric(
        "🔥 Lifetime Income Burn %",
        f"{burn_pct:.1f}%",
        "🟢 Good" if burn_pct < 75 else "🟡 High" if burn_pct < 100 else "🔴 Critical",
    )

    # =========================================================
    # 🔹 ROW 6 — MONTHLY BUDGET LEFT + DAILY LIMIT
    # =========================================================
    st.markdown("### 💼 Budget Survival Tracking")

    spent = current_month_total
    budget_left = MONTHLY_BUDGET - spent

    today_day = today.day
    days_month = pd.Period(today, freq="M").days_in_month
    days_left = max(days_month - today_day, 1)

    daily_limit = budget_left / days_left if days_left > 0 else 0

    c6_1, c6_2, c6_3 = st.columns(3)

    c6_1.metric(
        "💰 Budget Remaining",
        f"₹{budget_left:,.0f}",
        "🟢 Good" if budget_left > 6000 else "🟡 Low" if budget_left > 0 else "🔴 Over",
    )
    c6_2.metric("📆 Days Left", f"{days_left} days")
    c6_3.metric(
        "⚡ Daily Spend Allowed",
        f"₹{daily_limit:,.0f}/day",
        "🟢 Comfortable" if daily_limit > 450 else "🟡 Tight"
        if daily_limit > 150
        else "🔴 Risk",
    )

    st.markdown("---")

    # =================================================
    # 🧠 SMART SPEND REDUCTION ADVISOR
    # =================================================
    st.subheader("🧠 Smart Spend Reduction Suggestions")

    suggestions = []

    # Category grouping
    cat_group = cat_sum  # already computed above

    if len(cat_group) > 0:
        max_cat = cat_group.idxmax()
        max_cat_val = cat_group.max()
    else:
        max_cat, max_cat_val = "None", 0

    # Savings & ratio re-use
    ratio = expense_pct
    save_rate = (expected_income - current_month_total) / expected_income * 100 if expected_income > 0 else 0

    # 1️⃣ Income vs Expense
    if ratio > 120:
        suggestions.append("🔴 Danger — Spending >120% of income. Immediate cut necessary.")
    elif ratio > 100:
        suggestions.append("🟥 Overspending — You exceeded your income this month.")
    elif ratio > 80:
        suggestions.append("🟡 You are nearing income cap — reduce optional bills.")
    else:
        suggestions.append("🟢 Expenses under control — good month management!")

    # 2️⃣ Savings Condition
    if save_rate < 10:
        suggestions.append("🚨 Savings under 10% — extremely risky month.")
    elif save_rate < 25:
        suggestions.append("⚠ Improve savings to 25% for future stability.")
    else:
        suggestions.append("🟢 Good savings health!")

    # 3️⃣ Category Reduction Plan
    if max_cat_val > 0:
        suggestions.append(
            f"💡 Reduce **{max_cat}** by ~15% → Save ~₹{max_cat_val*0.15:,.0f}"
        )

    # 4️⃣ Daily Spend Health
    if daily_avg > ideal_daily > 0:
        suggestions.append(
            f"⚡ Daily spending too high → Maintain < ₹{ideal_daily:,.0f}/day"
        )
    else:
        suggestions.append("👍 Daily spend is stable and healthy.")

    # 5️⃣ Spike Detection (Safe Mode)
    if len(cat_group) > 0:
        mean_sp = cat_group.mean()
        for c, v in cat_group.items():
            if v > mean_sp * 1.4:
                suggestions.append(
                    f"⚡ {c} spending jumped unusually — track & reduce habit."
                )

    if suggestions:
        for s in suggestions:
            st.write(s)
    else:
        st.info("No suggestions — filtered data too small to analyze.")

    # ======================================================
    # 📊 CATEGORY PERFORMANCE & GROWTH ANALYTICS (SMART)
    # ======================================================
    st.subheader("📊 Category Performance & Growth Analytics (Smart Signals)")

    cat_summary = cat_sum.sort_values(ascending=False)
    cat_month = f.groupby(["year_month", "category"])["amount"].sum().reset_index()

    c1_, c2_, c3_ = st.columns(3)

    # 1️⃣ Top Category Contribution
    if len(cat_summary) > 0:
        top_cat = cat_summary.idxmax()
        top_cat_val = cat_summary.max()
        share_top = (top_cat_val / total_spend * 100) if total_spend > 0 else 0
        c1_.metric("🥇 Top Category by Spend", top_cat, f"{share_top:.2f}% share")
    else:
        c1_.metric("🥇 Top Category by Spend", "-", "0%")

    # 2️⃣ & 3️⃣ Trend Growth Signals
    if len(cat_month["year_month"].unique()) >= 2:
        unique_months = sorted(cat_month["year_month"].unique())
        last, prev = unique_months[-1], unique_months[-2]

        curr_df = cat_month[cat_month.year_month == last]
        prev_df = cat_month[cat_month.year_month == prev]

        growth = curr_df.merge(prev_df, on="category", suffixes=("_curr", "_prev")).fillna(0)
        growth["change_%"] = (
            (growth["amount_curr"] - growth["amount_prev"])
            / growth["amount_prev"].replace(0, 1)
            * 100
        )

        up = growth.sort_values("change_%", ascending=False).head(1)
        down = growth.sort_values("change_%", ascending=True).head(1)

        c2_.metric(
            "🔴 Highest Increase (Bad)",
            up.iloc[0]["category"],
            f"{up.iloc[0]['change_%']:.2f}% ↑",
        )
        c3_.metric(
            "🟢 Biggest Drop (Saving)",
            down.iloc[0]["category"],
            f"{down.iloc[0]['change_%']:.2f}% ↓",
        )
    else:
        c2_.metric("🔴 Highest Increase (Bad)", "Not enough data")
        c3_.metric("🟢 Biggest Drop (Saving)", "Not enough data")

    # =============== DEEP INSIGHTS =======================
    st.write("### 🧠 Category Intelligence Metrics")

    m1, m2, m3, m4 = st.columns(4)

    if len(cat_month) > 0:
        variance_cat = (
            cat_month.groupby("category")["amount"].var().sort_values(ascending=False)
        )

        m1.metric(
            "💡 Avg Spend/Category/Month",
            f"₹{cat_month.groupby('category')['amount'].mean().mean():,.0f}",
        )

        if len(variance_cat) > 0:
            m2.metric(
                "📈 Most Volatile Category",
                variance_cat.idxmax(),
                f"{variance_cat.max():.0f} variance",
            )
            m3.metric(
                "📉 Most Stable Category",
                variance_cat.idxmin(),
                f"{variance_cat.min():.0f} variance",
            )
    else:
        m1.metric("💡 Avg Spend/Category/Month", "Not enough data")
        m2.metric("📈 Most Volatile Category", "-")
        m3.metric("📉 Most Stable Category", "-")

    # Consistency Score (monthly level)
    monthly_expenses = month_totals
    if len(monthly_expenses) > 1:
        variance_month = monthly_expenses.pct_change().abs()
        if variance_month.max() > 0:
            consistency_score = (
                1 - (variance_month.mean() / variance_month.max())
            ) * 100
        else:
            consistency_score = 100.0
        m4.metric("🧠 Consistency Score", f"{consistency_score:.1f}%")
    else:
        m4.metric("🧠 Consistency Score", "Not enough data 📉")

    # ========= Category Share Table =============
    st.write("### 📊 Spend Share Breakdown")
    if len(cat_summary) > 0:
        share_df = cat_summary.reset_index().rename(columns={"amount": "Total Spend"})
        share_df["Share %"] = (share_df["Total Spend"] / total_spend * 100).round(2)
        st.dataframe(share_df, width="stretch")
    else:
        st.info("No category spend data to display.")

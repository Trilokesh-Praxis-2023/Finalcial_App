import streamlit as st
import pandas as pd
from datetime import datetime

# ---------- Income Function ----------
def get_income(date):
    base = datetime(2024,10,1)
    date = pd.to_datetime(date)
    diff = (date.year-base.year)*12 + (date.month-base.month)
    return 12000 if diff==0 else 14112 if diff==1 else 24400


# ============================== MAIN KPI RENDER ===============================
def render_kpis(filtered: pd.DataFrame, df: pd.DataFrame, MONTHLY_BUDGET: float):
    if filtered.empty:
        st.warning("⚠ No data available — adjust filters or import expenses.")
        return
    
    f = filtered.copy()
    today = pd.to_datetime("today").date()
    f["period"] = pd.to_datetime(f["period"])

    # ---------------- Core Data ----------------
    total_spend = f["amount"].sum()
    today_spend = f[f["period"].dt.date==today]["amount"].sum()
    lifetime_spend = df["amount"].sum()

    current_month_key = f["year_month"].max()
    current_month = f[f["year_month"]==current_month_key]
    current_month_total = current_month["amount"].sum()
    avg_monthly = f.groupby("year_month")["amount"].sum().mean() or 0

    # Weekly
    f["week"]=f["period"].dt.isocalendar().week
    weekly=f.groupby("week")["amount"].sum()
    current_week=weekly.iloc[-1] if len(weekly)>0 else 0
    prev_week=weekly.iloc[-2] if len(weekly)>1 else 0
    wow=((current_week-prev_week)/prev_week*100) if prev_week>0 else 0

    # ====================================================================
    st.subheader("📊 Financial KPI Overview")
    c1,c2,c3,c4 = st.columns(4)

    with c1:
        st.markdown("<div class='kpi-card'>",unsafe_allow_html=True)
        st.metric("💸 Total Spend", f"₹{total_spend:,.0f}")
        st.markdown("</div>",unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='kpi-card'>",unsafe_allow_html=True)
        st.metric("📆 This Month", f"₹{current_month_total:,.0f}")
        st.markdown("</div>",unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='kpi-card'>",unsafe_allow_html=True)
        st.metric("📅 Today", f"₹{today_spend:,.0f}")
        st.markdown("</div>",unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='kpi-card'>",unsafe_allow_html=True)
        st.metric("📅 Avg Monthly", f"₹{avg_monthly:,.0f}")
        st.markdown("</div>",unsafe_allow_html=True)


    # ====================================================================
    st.subheader("📈 Spend Trend & Momentum")
    t1,t2,t3,t4 = st.columns(4)

    lifetime_used=(total_spend/lifetime_spend*100) if lifetime_spend>0 else 0
    t1.metric("📊 Lifetime Used", f"{lifetime_used:.1f}%")

    month_tot=f.groupby("year_month")["amount"].sum()
    if len(month_tot)>0:
        best_m=month_tot.idxmax();best_amt=month_tot.max()
        t2.metric("🔥 Peak Month", f"{best_m}", f"₹{best_amt:,.0f}")
    else: t2.metric("🔥 Peak Month", "-")

    t3.metric("📅 Week Spend", f"₹{current_week:,.0f}")
    t4.metric("🔄 WoW Change", f"{wow:.1f}%",delta_color="inverse")

    
    # ====================================================================
    st.subheader("🏷 Category Strength + Daily Pattern")
    r1,r2,r3,r4=st.columns(4)

    prev_m=month_tot.iloc[-2] if len(month_tot)>1 else 0
    mom=((current_month_total-prev_m)/prev_m*100) if prev_m>0 else 0
    r1.metric("📆 MoM Change", f"{mom:.1f}%")

    cat=f.groupby("category")["amount"].sum()
    r2.metric("🏆 Top Category", cat.idxmax() if len(cat)>0 else "-")
    r3.metric("🪫 Low Category", cat.idxmin() if len(cat)>0 else "-")

    daily=f.groupby("period")["amount"].sum()
    r4.metric("📅 Avg/Day",f"₹{daily.mean():,.0f}")


    # ====================================================================
    st.subheader("💰 Income vs Expense (Financial Health)")
    i1,i2,i3,i4=st.columns(4)

    expected=get_income(current_month_key)
    balance=expected-current_month_total
    save_rate=(balance/expected*100) if expected>0 else 0
    spent_pct=(current_month_total/expected*100) if expected>0 else 0

    status="🟢 Safe" if spent_pct<70 else "🟡 High" if spent_pct<100 else "🔴 Risk"

    i1.metric("💰 Income Expected", f"₹{expected:,.0f}")
    i2.metric("📊 Balance Left", f"₹{balance:,.0f}")
    i3.metric("💾 Savings %", f"{save_rate:.1f}%")
    i4.metric("⚡ Income Spent", f"{spent_pct:.1f}%",status)


    # ====================================================================
    st.subheader("💼 Budget Survival Tracking")

    spent=current_month_total
    left=MONTHLY_BUDGET-spent
    today_day=today.day
    total_days=pd.Period(today,freq="M").days_in_month
    remain=max(total_days-today_day,1)
    daily_allow=left/remain

    b1,b2,b3=st.columns(3)

    b1.metric("💰 Budget Left",f"₹{left:,.0f}")
    b2.metric("📆 Days Left",remain)
    b3.metric("⚡ Daily Allowed",f"₹{daily_allow:,.0f}/day")


    # ====================================================================
    st.subheader("🧠 Smart Spend Suggestions")
    st.write("🔹 Reduce highest category to improve savings")
    st.write("🔹 Maintain daily budget to stay safe")
    st.write("🔹 Avoid spikes — track spending pattern weekly")


    # ====================================================================
    st.subheader("📊 Category Share Insights")
    if len(cat)>0:
        share=cat.reset_index().rename(columns={"amount":"Total"})
        share["%"]=share["Total"]/(total_spend)*100
        st.dataframe(share,width="container")
    else: st.info("Not enough category data")



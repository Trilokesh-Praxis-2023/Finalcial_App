import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prophet import Prophet
from io import BytesIO
import altair as alt
import os
from sqlalchemy import text


# =================================================
# 🔹 INITIAL SETUP + LOAD ENV
# =================================================
load_dotenv()
st.set_page_config(page_title="Finance Tracker", layout="wide")
st.title("💰 Personal Finance Tracker")

DATABASE_URL = os.getenv("DATABASE_URL") or \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

APP_PASSWORD = os.getenv("APP_PASSWORD")   # MUST exist in .env
engine = create_engine(DATABASE_URL)

CATEGORIES = ["Rent","Recharge","Transport","Food","Other","Household","Health",
              "Apparel","Social Life","Beauty","Gift","Education"]
MONTHLY_BUDGET = 18000


# =================================================
# 📥 LOAD DATA — CACHED & OPTIMIZED
# =================================================
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM finance_data", engine)
    df.columns = [c.lower() for c in df.columns]

    df['period'] = pd.to_datetime(df['period'], errors='coerce')
    df['year'] = df['period'].dt.year
    df['year_month'] = df['period'].dt.to_period("M").astype(str)

    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
    if "income" in df.columns:
        df['income'] = pd.to_numeric(df['income'], errors='coerce').fillna(0)

    return df


# =================================================
# 🔐 PASSWORD CHECK — FULL APP LOCK
# =================================================
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")
if password != APP_PASSWORD:
    st.warning("🔒 Access Restricted — Enter Correct Password to Continue")
    st.stop()  # 🚫 NO dashboard beyond this point visible


# =================================================
# 🔥 AUTH PASSED → LOAD DATA
# =================================================
df = load_data()
st.success("🔓 Access Granted")



# =================================================
# ➕ ADD EXPENSE ENTRY WITH MONTH + % + RUNNING TOTAL (FINAL FIX)
# =================================================
with st.expander("➕ Add Expense"):
    with st.form("expense_form"):
        d = st.date_input("Date")
        acc = st.text_input("Account / UPI / Card")
        cat = st.selectbox("Category", CATEGORIES)
        amt = st.number_input("Amount", min_value=0.0, step=1.0)
        submit = st.form_submit_button("💾 Save Entry")

    if submit:

        month_value = pd.to_datetime(d).strftime("%Y-%m")

        current_total = df["amount"].sum() if not df.empty else 0
        new_running_total = current_total + float(amt)

        row_percent = (float(amt) / new_running_total) * 100

        try:
            df_new = pd.DataFrame([{
                "period": pd.to_datetime(d),
                "accounts": acc,
                "category": cat,
                "amount": float(amt),
                "month": month_value,
                "percent_row": row_percent,        # 💥 FIXED — no % symbol
                "running_total": new_running_total # 💥 FIXED — no space
            }])

            df_new.to_sql("finance_data", engine, if_exists="append", index=False)
            load_data.clear()
            st.success("✔ Expense Saved Successfully")

        except Exception as e:
            st.error(f"❌ Upload Failed:\n{e}")



# =================================================
# 🔍 FILTER PANEL
# =================================================
st.sidebar.subheader("🔎 Filters")

f_year  = st.sidebar.multiselect("Year", sorted(df.year.unique()), default=list(df.year.unique()))
f_month = st.sidebar.multiselect("Month", sorted(df.year_month.unique()))
f_cat   = st.sidebar.multiselect("Category", sorted(df.category.unique()))
f_acc   = st.sidebar.multiselect("Account", sorted(df.accounts.unique()))

filtered = df.copy()
if f_year:  filtered = filtered[filtered.year.isin(f_year)]
if f_month: filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:   filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:   filtered = filtered[filtered.accounts.isin(f_acc)]



# =================================================
# 📊 ADVANCED KPI DASHBOARD (SMART GROUPED + RESPONSIVE)
# =================================================

# ---------- PRE-CALCULATIONS ----------
today = pd.to_datetime("today").date()
today_spend = filtered[filtered["period"].dt.date == today]["amount"].sum()

total_spend = filtered["amount"].sum()        # ⭐ filtered spend
lifetime_spend = df["amount"].sum()           # ⭐ total all-time spend

current_month_key = filtered["year_month"].max()
current_month = filtered[filtered["year_month"] == current_month_key]
current_month_total = current_month["amount"].sum()

month_fmt = lambda m: pd.to_datetime(m).strftime("%b %Y")

# Weekly breakdown
filtered["week"] = filtered["period"].dt.isocalendar().week
filtered["year_week"] = filtered["period"].dt.strftime("%Y-W%U")
weekly_spend = filtered.groupby("year_week")["amount"].sum()

current_week_key = weekly_spend.index.max()
current_week_total = weekly_spend.get(current_week_key, 0)

previous_week_total = weekly_spend.iloc[-2] if len(weekly_spend) > 1 else 0
wow_change = ((current_week_total - previous_week_total) / previous_week_total * 100) if previous_week_total > 0 else 0


# =================================================
# 🔹 ROW 1 — CORE SPEND HEALTH
# =================================================
c1, c2, c3, c4 = st.columns(4)

c1.metric("💸 Total Spend (Filtered)", f"₹{total_spend:,.0f}")
c2.metric("📆 Current Month Spend", f"₹{current_month_total:,.0f}")
c3.metric("📅 Today's Spend", f"₹{today_spend:,.0f}")

avg_monthly = filtered.groupby("year_month")["amount"].sum().mean()
c4.metric("📅 Avg Monthly Spend", f"₹{avg_monthly:,.0f}")


# =================================================
# 🔹 ROW 2 — MOMENTUM & TREND DIRECTION
# =================================================
t1, t2, t3, t4 = st.columns(4)

lifetime_used_pct = (total_spend / lifetime_spend * 100) if lifetime_spend > 0 else 0
t1.metric("📊 Lifetime Spend % Used", f"{lifetime_used_pct:.1f}%")

month_totals = filtered.groupby("year_month")["amount"].sum()
if len(month_totals) > 0:
    best_month = month_totals.idxmax()
    best_month_amt = month_totals.max()
    t2.metric("🔥 Peak Month", month_fmt(best_month), f"₹{best_month_amt:,.0f}")
else:
    t2.metric("🔥 Peak Month", "-")

t3.metric("📅 Current Week Spend", f"₹{current_week_total:,.0f}")
t4.metric("🔄 WoW Change", f"{wow_change:.1f}%", delta_color="inverse")


# =================================================
# 🔹 ROW 3 — CATEGORY STRENGTH & DAILY PATTERN
# =================================================
r3c1, r3c2, r3c3, r3c4 = st.columns(4)

# MoM change
prev_month = month_totals.iloc[-2] if len(month_totals) > 1 else 0
mom_change = ((current_month_total - prev_month) / prev_month * 100) if prev_month > 0 else 0
r3c1.metric("📆 MoM Change", f"{mom_change:.1f}%")

# Category performance
cat_sum = filtered.groupby("category")["amount"].sum()
r3c2.metric("🏆 Top Category", cat_sum.idxmax() if len(cat_sum) > 0 else "-")
r3c3.metric("🪫 Lowest Category", cat_sum.idxmin() if len(cat_sum) > 0 else "-")

daily_avg = filtered.groupby("period")["amount"].sum().mean()
r3c4.metric("📅 Avg Daily Spend", f"₹{daily_avg:,.0f}")


# =================================================
# 🔹 ROW 4 — INCOME vs EXPENSE IMPACT
# =================================================
i1, i2, i3, i4 = st.columns(4)

from datetime import datetime
def get_income(date):
    base = datetime(2024,10,1)
    date = pd.to_datetime(date)
    diff = (date.year-base.year)*12 + (date.month-base.month)
    return 12000 if diff==0 else 14112 if diff==1 else 24400

expected_income = get_income(current_month_key)
i1.metric("💰 Income Expected", f"₹{expected_income:,.0f}")

balance = expected_income - current_month_total
i2.metric("📊 Balance Left", f"₹{balance:,.0f}", "🟢" if balance>0 else "🔴")

save_rate = (balance/expected_income*100) if expected_income>0 else 0
i3.metric("💾 Savings Rate %", f"{save_rate:.1f}%")

expense_ratio = current_month_total/expected_income*100
indicator = "🟢 Safe" if expense_ratio<70 else "🟡 High" if expense_ratio<100 else "🔴 Risk"
i4.metric("⚡ % Income Spent", f"{expense_ratio:.1f}%", indicator)


# =================================================
# 🔹 ROW 5 — ACTIVITY / LIFETIME HEALTH
# =================================================
a1, a2, a3, a4 = st.columns(4)

a1.metric("📆 Active Days Logged", f"{filtered['period'].nunique()} days")

# historical income
monthly_full = df.groupby("year_month")["amount"].sum()
income_history = [get_income(m) for m in monthly_full.index]

total_income = sum(income_history)
a2.metric("💰 Total Estimated Income", f"₹{total_income:,.0f}")

lifetime_savings = total_income - lifetime_spend
a3.metric("🏦 Lifetime Savings", f"₹{lifetime_savings:,.0f}", 
          "🟢" if lifetime_savings>0 else "🔴")

income_burn = (lifetime_spend/total_income*100) if total_income>0 else 0
a4.metric("🔥 Lifetime Income Burn %", f"{income_burn:.1f}%", 
          "🟢 Good" if income_burn<75 else "🟡 High" if income_burn<100 else "🔴 Critical")

# =================================================
# 🔹 ROW 6 — Budget Left + Daily Spend Limit
# =================================================

fixed_A = 11600
fixed_B = 1900
total_fixed = fixed_A + fixed_B
flex_budget = MONTHLY_BUDGET - total_fixed   # money left to spend freely

# days passed in month
today_day = today.day
days_in_month = pd.to_datetime(today.replace(day=28) + pd.Timedelta(days=4)).day
days_left = max(days_in_month - today_day, 1)

per_day_budget = flex_budget / days_left

spent_so_far = current_month_total
budget_left = MONTHLY_BUDGET - spent_so_far

burn_pct = (spent_so_far / MONTHLY_BUDGET * 100) if MONTHLY_BUDGET>0 else 0


b1, b2, b3, b4 = st.columns(4)

b1.metric("🏦 Monthly Budget", f"₹{MONTHLY_BUDGET:,}")
b2.metric("🧾 Fixed Expense (Monthly)", f"₹{total_fixed:,}")

b3.metric("🔸 Flexible Left to Spend", 
         f"₹{flex_budget:,}", 
         "🟢 OK" if flex_budget>0 else "🔴 Over Fixed Budget")

b4.metric("📆 Daily Spend Limit", 
         f"₹{per_day_budget:,.0f}/day",
         "🟢 Safe" if per_day_budget>200 else "🟡 Tight" if per_day_budget>0 else "🔴 ZERO")


# -------- Budget Left vs Expense --------

r7c1, r7c2 = st.columns(2)

r7c1.metric("💰 Budget Remaining This Month", 
           f"₹{budget_left:,.0f}", 
           "🟢 Healthy" if budget_left>5000 else "🟡 Low" if budget_left>0 else "🔴 Overspent")

r7c2.metric("🔥 Budget Burn %", 
           f"{burn_pct:.1f}%", 
           "🟢 Good" if burn_pct<60 else "🟡 High" if burn_pct<100 else "🔴 Exceeded")





# ================================================================
# 📊 KPI DRILLDOWN – FULL SUITE + LABELS ON CHART (FILTER SAFE)
# ================================================================
st.subheader("📈 Trend Exploration Dashboard (with Values Visible)")

source = filtered.copy()

# ============ MONTHLY REBUILD BASED ON FILTER ============
monthly = source.groupby("year_month")["amount"].sum().reset_index()
monthly["year_month"] = pd.to_datetime(monthly["year_month"])
monthly = monthly.sort_values("year_month")

monthly["income"]  = [get_income(m) for m in monthly["year_month"]]
monthly["savings"] = monthly["income"] - monthly["amount"]

# =========================================================
# 1️⃣ Monthly Spend Trend — VALUES OVER LINE
# =========================================================
with st.expander("💸 Monthly Spend Trend — Values Displayed"):

    line = alt.Chart(monthly).mark_line(point=True, color="#29B6F6").encode(
        x="year_month:T", y="amount:Q"
    )

    labels = alt.Chart(monthly).mark_text(
        dy=-12, fontSize=12, color="yellow", fontWeight="bold"
    ).encode(
        x="year_month:T", y="amount:Q", text="amount:Q"
    )

    st.altair_chart(line + labels, use_container_width=True)



# =========================================================
# 2️⃣ Month-on-Month Comparison — BAR + LABELS
# =========================================================
with st.expander("📆 Month-on-Month Spend — Value Bars"):

    bars = alt.Chart(monthly).mark_bar(size=38, color="#26D67D").encode(
        x="year_month:T", y="amount:Q"
    )

    texts = alt.Chart(monthly).mark_text(
        dy=-10, fontSize=12, fontWeight="bold", color="white"
    ).encode(
        x="year_month:T", y="amount:Q", text="amount:Q"
    )

    st.altair_chart(bars + texts, use_container_width=True)



# =========================================================
# 3️⃣ Rolling 3-Month Smoothed Trend — LABELLED
# =========================================================
with st.expander("📅 Rolling 3-Month Spend Trend"):

    monthly["roll"] = monthly["amount"].rolling(3).mean()

    line = alt.Chart(monthly).mark_line(point=True, color="#FFC107").encode(
        x="year_month:T", y="roll:Q"
    )

    labels = alt.Chart(monthly).mark_text(
        dy=-10, fontSize=11, color="white"
    ).encode(
        x="year_month:T", y="roll:Q", text="roll:Q"
    )

    st.altair_chart(line + labels, use_container_width=True)



# =========================================================
# 4️⃣ Category Trend Over Time — LOG & Labels
# =========================================================
with st.expander("🏷 Category Trend Timeline"):

    cat = source.groupby(["year_month","category"])["amount"].sum().reset_index()
    cat["year_month"] = pd.to_datetime(cat["year_month"])

    line = alt.Chart(cat).mark_line(point=True).encode(
        x="year_month:T",
        y=alt.Y("amount:Q", scale=alt.Scale(type="log")),
        color="category:N"
    )

    labels = alt.Chart(cat).mark_text(
        dy=-10, fontSize=10
    ).encode(
        x="year_month:T", y="amount:Q", text="amount:Q", color="category:N"
    )

    st.altair_chart(line + labels, use_container_width=True)



# =========================================================
# 5️⃣ Income vs Expense vs Savings — LABELLED
# =========================================================
with st.expander("💰 Income vs Expense vs Savings (Monthly)"):

    melt = monthly.melt("year_month", value_vars=["amount","income","savings"])

    line = alt.Chart(melt).mark_line(point=True).encode(
        x="year_month:T", y="value:Q", color="variable:N"
    )

    labels = alt.Chart(melt).mark_text(
        dy=-10, fontSize=10
    ).encode(
        x="year_month:T", y="value:Q", text="value:Q", color="variable:N"
    )

    st.altair_chart(line + labels, use_container_width=True)



# =========================================================
# 6️⃣ Savings Trend — VALUE SHOWN
# =========================================================
with st.expander("🧾 Net Savings Monthly Trend"):

    area = alt.Chart(monthly).mark_area(color="#00C853", opacity=0.5).encode(
        x="year_month:T", y="savings:Q"
    )

    labels = alt.Chart(monthly).mark_text(
        dy=-10, fontSize=11, color="white"
    ).encode(
        x="year_month:T", y="savings:Q", text="savings:Q"
    )

    st.altair_chart(area + labels, use_container_width=True)



# =========================================================
# 7️⃣ Category Spend Share — With % Labels
# =========================================================
with st.expander("📊 Category Spend Share Distribution"):

    share = source.groupby("category")["amount"].sum().reset_index()
    share["percent"] = (share["amount"]/share["amount"].sum()*100).round(1)

    bars = alt.Chart(share).mark_bar(color="#FFCA28").encode(
        x="category:N", y="percent:Q"
    )

    labels = alt.Chart(share).mark_text(
        dy=-8, fontSize=11, fontWeight="bold"
    ).encode(
        x="category:N", y="percent:Q", text="percent:Q"
    )

    st.altair_chart(bars + labels, use_container_width=True)



# =========================================================
# 8️⃣ Best vs Worst Month Summary Card
# =========================================================
with st.expander("🏆 Best vs Worst Month Summary"):

    best = monthly.loc[monthly["amount"].idxmax()]
    worst = monthly.loc[monthly["amount"].idxmin()]

    st.success(f"🥇 Best Month → {best.year_month:%b %Y}  |  ₹{best.amount:,.0f}")
    st.error  (f"🥀 Worst Month → {worst.year_month:%b %Y} |  ₹{worst.amount:,.0f}")



# =========================================================
# 9️⃣ Spend Volatility / Stability Score
# =========================================================
with st.expander("🌡 Expense Stability (Volatility Index)"):

    if len(monthly)>2:
        vol = monthly["amount"].pct_change().abs().mean()*100
        stability = max(0,100-vol)
        st.metric("Stability Score", f"{stability:.1f}%")
        st.caption("Lower volatility = more consistent control 🚀")
    else:
        st.info("Need at least 3 months of data.")



# =========================================================
# 🔟 Survival Duration If Income Stops
# =========================================================
with st.expander("🛡 Survival Duration (If Income Stops)"):

    if monthly["amount"].mean()>0:
        burn = monthly["amount"].mean()
        surplus = monthly["income"].sum()-monthly["amount"].sum()
        st.metric("Estimated Survival", f"{surplus/burn:.1f} months")
    else:
        st.info("Not enough data to estimate.")

# =========================================================
# EXTRA INSIGHTS BELOW 🔥
# =========================================================

# 📈 CUMULATIVE SPENDING OVER TIME
with st.expander("📈 Cumulative Spending Curve"):
    monthly["cumulative"] = monthly["amount"].cumsum()
    st.altair_chart(
        alt.Chart(monthly).mark_line(point=True).encode(
            x="year_month:T", y="cumulative:Q", tooltip=["year_month","cumulative"]
        ),
        use_container_width=True
    )

# 📦 SPENDING DISTRIBUTION OUTLIER DETECTION
with st.expander("📦 Outlier Spread (Boxplot)"):

    st.altair_chart(
        alt.Chart(source).mark_boxplot(color="#8E44AD").encode(
            x="category:N", y="amount:Q"
        ),
        use_container_width=True
    )

# 🌦 SEASONAL SPENDING BEHAVIOR
with st.expander("🌦 Seasonal Spend Pattern (Month of Year)"):

    season = source.copy()
    season["m"] = season["period"].dt.month
    month_sum = season.groupby("m")["amount"].sum().reset_index()

    bars = alt.Chart(month_sum).mark_bar(color="#5DADE2").encode(
        x="m:N", y="amount:Q"
    )

    labels = alt.Chart(month_sum).mark_text(
        dy=-10,fontSize=11
    ).encode(
        x="m:N", y="amount:Q", text="amount:Q"
    )

    st.altair_chart(bars+labels, use_container_width=True)



# =================================================
# 🧠 SMART SPEND REDUCTION ADVISOR (FILTER SAFE)
# =================================================
st.subheader("🧠 Smart Spend Reduction Suggestions")

suggestions = []

# ========== SAFE CATEGORY GROUPING ==========
cat_group = filtered.groupby("category")["amount"].sum()

if len(cat_group) > 0:
    max_cat = cat_group.idxmax()
    max_cat_val = cat_group.max()
else:
    max_cat, max_cat_val = "None", 0


# ========== SAFE DAILY + MONTHLY BASELINE ==========
month_days = max(1, datetime.now().day)
ideal_daily = expected_income/30 if expected_income > 0 else 0


# ========== SAVINGS & RATIO SAFE CHECK ==========
ratio = (current_month_total/expected_income*100) if expected_income>0 else 0
save_rate = ((expected_income-current_month_total)/expected_income*100) if expected_income>0 else 0


# =====================================================
# RULE ENGINE — Now Fully Safe With Filtering
# =====================================================

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
    suggestions.append(f"💡 Reduce **{max_cat}** by ~15% → Save ~₹{max_cat_val*0.15:,.0f}")

# 4️⃣ Daily Spend Health
if daily_avg > ideal_daily > 0:
    suggestions.append(f"⚡ Daily spending too high → Maintain < ₹{ideal_daily:,.0f}/day")
else:
    suggestions.append("👍 Daily spend is stable and healthy.")

# 5️⃣ Spike Detection (Safe Mode)
if len(cat_group) > 0:
    mean_sp = cat_group.mean()
    for c,v in cat_group.items():
        if v > mean_sp * 1.4:
            suggestions.append(f"⚡ {c} spending jumped unusually — track & reduce habit.")


# 📝 DISPLAY — No errors even on 1 month filter
if suggestions:
    for s in suggestions:
        st.write(s)
else:
    st.info("No suggestions — filtered data too small to analyze.")





# ======================================================
# 📊 CATEGORY PERFORMANCE & GROWTH ANALYTICS (SMART+EXPANDED)
# ======================================================
st.subheader("📊 Category Performance & Growth Analytics (Smart Signals)")

# TOTAL spend by category
cat_summary = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
cat_month = filtered.groupby(["year_month","category"])["amount"].sum().reset_index()

c1, c2, c3 = st.columns(3)

# 1️⃣ Top Category Contribution
top_cat = cat_summary.idxmax()
top_cat_val = cat_summary.max()
share_top = (top_cat_val/total_spend*100) if total_spend>0 else 0
c1.metric("🥇 Top Category by Spend", top_cat, f"{share_top:.2f}% share")

# 2️⃣ & 3️⃣ Trend Growth Signals
if len(cat_month.year_month.unique()) >= 2:

    last, prev = cat_month.year_month.max(), sorted(cat_month.year_month.unique())[-2]

    curr_df = cat_month[cat_month.year_month == last]
    prev_df = cat_month[cat_month.year_month == prev]

    growth = curr_df.merge(prev_df, on="category", suffixes=("_curr","_prev")).fillna(0)
    growth["change_%"] = ((growth["amount_curr"] - growth["amount_prev"]) /
                          growth["amount_prev"].replace(0,1)) * 100

    # Fastest Rise (Bad → Red)
    up = growth.sort_values("change_%", ascending=False).head(1)
    c2.metric("🔴 Highest Increase (Bad)", up.iloc[0]["category"], f"{up.iloc[0]['change_%']:.2f}% ↑")

    # Best Drop (Good → Green)
    down = growth.sort_values("change_%", ascending=True).head(1)
    c3.metric("🟢 Biggest Drop (Saving)", down.iloc[0]["category"], f"{down.iloc[0]['change_%']:.2f}% ↓")

else:
    c2.metric("🔴 Increase", "Not enough data")
    c3.metric("🟢 Drop", "Not enough data")


# =============== NEW DEEP INSIGHTS =======================
st.write("### 🧠 Category Intelligence Metrics")

m1, m2, m3, m4 = st.columns(4)

# Variance = how unstable a category spend is
variance = cat_month.groupby("category")["amount"].var().sort_values(ascending=False)

m2.metric("📈 Most Volatile Category", variance.idxmax(), f"{variance.max():.0f} variance")
m3.metric("📉 Most Stable Category", variance.idxmin(), f"{variance.min():.0f} variance")

# Average spend per category per month
avg_cat_per_month = cat_month.groupby("category")["amount"].mean().sort_values(ascending=False)
m1.metric("💡 Avg Spend/Category/Month", f"₹{avg_cat_per_month.mean():,.0f}")

# ============================================================
# Consistency Score → Measures spending stability month-to-month
# Score closer to 100 = More disciplined spending
# ============================================================

monthly_expenses = filtered.groupby("year_month")["amount"].sum()

if len(monthly_expenses) > 1:
    variance = monthly_expenses.pct_change().abs()  # month-to-month fluctuation

    # normalized score → less volatility = higher consistency
    if variance.max() > 0:
        consistency_score = (1 - (variance.mean()/variance.max())) * 100
    else:
        consistency_score = 100  # perfect stability case

    m4.metric("🧠 Consistency Score", f"{consistency_score:.1f}%")

else:
    m4.metric("🧠 Consistency Score", "Not enough data 📉")



# ========= Category Share Table =============
st.write("### 📊 Spend Share Breakdown")
share_df = cat_summary.reset_index().rename(columns={"amount":"Total Spend"})
share_df["Share %"] = (share_df["Total Spend"]/total_spend*100).round(2)
st.dataframe(share_df, width="stretch")









# =================================================
# 📄 VIEW TRANSACTIONS + EXPORT + REFRESH (UPDATED + SORTED)
# =================================================
st.subheader("📄 Transactions")

# 🔄 Refresh Button
if st.button("🔄 Refresh Table"):
    load_data.clear()
    st.rerun()

# Format Display
df_display = filtered.copy()

# Convert period → date only
if "period" in df_display.columns:
    df_display["period"] = pd.to_datetime(df_display["period"]).dt.date

# ============================
# 🔥 Sort by latest first
# ============================
if "period" in df_display.columns:
    df_display = df_display.sort_values("period", ascending=False)


# Optional reorder for clean UI
order_cols = ["period","accounts","category","amount","month","percent_row","running_total"]
df_display = df_display[[c for c in order_cols if c in df_display.columns]]

# Display
st.dataframe(df_display, width="stretch", height=300)

# ============================
# DOWNLOAD BUTTONS
# ============================
csv = df_display.to_csv(index=False).encode("utf-8")
st.download_button("📄 Download CSV", csv, "transactions.csv")

buf = BytesIO()
with pd.ExcelWriter(buf) as writer:
    df_display.to_excel(writer, index=False)
st.download_button("📊 Download Excel", buf.getvalue(), "transactions.xlsx")


# =================================================
# ❌ DELETE A TRANSACTION (AUTO REFRESH + CLEAN DATE + SORTED)
# =================================================
st.subheader("🗑 Delete Transaction")

try:
    # Load DB with row index
    df_del = pd.read_sql("SELECT *, ROW_NUMBER() OVER () AS row_id FROM finance_data", engine)

    # Convert period → date only (fix display)
    if "period" in df_del.columns:
        df_del["period"] = pd.to_datetime(df_del["period"]).dt.date

    # Sort latest first
    df_del = df_del.sort_values("period", ascending=False)

    # Display
    df_del_display = df_del[["row_id","period","accounts","category","amount"]]
    st.dataframe(df_del_display, height=250, width="stretch")

    delete_id = st.number_input("Enter Row ID to Delete", min_value=1, step=1)

    if st.button("Delete Selected Record"):
        del_row = df_del[df_del["row_id"] == delete_id]

        if del_row.empty:
            st.error("⚠ Invalid ID — no matching record found.")
        else:
            with engine.connect() as conn:
                conn.execute(text("""
                    DELETE FROM finance_data
                    WHERE period = :p
                    AND accounts = :a
                    AND category = :c
                    AND amount = :m
                """), {
                    "p": del_row.iloc[0]["period"],
                    "a": del_row.iloc[0]["accounts"],
                    "c": del_row.iloc[0]["category"],
                    "m": del_row.iloc[0]["amount"]
                })
                conn.commit()

            st.success("🗑 Record Deleted Successfully!")
            load_data.clear()
            st.rerun()     # 🔥 auto refresh

except Exception as e:
    st.error(f"❌ Failed to load transaction table:\n{e}")



# =================================================
# 🔮 FORECASTING SECTION (MONTH + DAY)
# =================================================
st.divider()
st.header("🔮 Forecasting & AI Predictions")

if st.button("Generate Forecast"):
    
    # ==========================
    # MONTHLY FORECAST (Existing + Improved)
    # ==========================
    st.subheader("📅 Monthly Forecast (Next 6 Months)")

    f_month = filtered.groupby("year_month")["amount"].sum().reset_index()

    if len(f_month) < 3:
        st.warning("⚠ Need at least 3 months of data for monthly forecasting.")
    else:
        f_month["ds"] = pd.to_datetime(f_month.year_month)
        f_month.rename(columns={"amount": "y"}, inplace=True)

        m_model = Prophet()
        m_model.fit(f_month[["ds","y"]])

        future_m = m_model.make_future_dataframe(periods=6, freq="ME")
        forecast_m = m_model.predict(future_m)

        st.dataframe(
            forecast_m.tail(6)[["ds","yhat","yhat_lower","yhat_upper"]]
            .rename(columns={"ds":"Month","yhat":"Predicted"})
        )

        fig_m = m_model.plot(forecast_m)
        st.pyplot(fig_m)


    # ==========================
    # 🔥 DAY-WISE FORECAST
    # ==========================
    st.subheader("📆 Daily Forecast (Next 30 Days)")

    f_day = filtered.groupby("period")["amount"].sum().reset_index()

    if len(f_day) < 7:
        st.warning("⚠ Need at least 7 days of data for daily forecasting.")
    else:
        f_day["ds"] = pd.to_datetime(f_day["period"])
        f_day.rename(columns={"amount":"y"}, inplace=True)

        d_model = Prophet(daily_seasonality=True)  # enable day pattern detection
        d_model.fit(f_day[["ds","y"]])

        future_d = d_model.make_future_dataframe(periods=30, freq="D")
        forecast_d = d_model.predict(future_d)

        st.dataframe(
            forecast_d.tail(30)[["ds","yhat","yhat_lower","yhat_upper"]]
            .rename(columns={"ds":"Date","yhat":"Predicted"})
        )

        fig_d = d_model.plot(forecast_d)
        st.pyplot(fig_d)

        # Day-wise trend available
        st.line_chart(forecast_d.set_index("ds")["yhat"].tail(30))

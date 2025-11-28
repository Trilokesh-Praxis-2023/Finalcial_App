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
st.title("💰 Personal Finance Tracker (Secure & Optimized)")

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
        amt = st.number_input("Amount", min_value=1.0, step=1.0)
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
# 📊 ADVANCED KPI DASHBOARD (FULL + WEEKLY + CURRENT MONTH + DAILY)
# =================================================

# ========== PRE-CALCULATIONS ==========
today = pd.to_datetime("today").date()
today_spend = filtered[filtered["period"].dt.date == today]["amount"].sum()

current_month_key = filtered["year_month"].max()
current_month = filtered[filtered["year_month"] == current_month_key]
current_month_total = current_month["amount"].sum()

# Weekly Processing
filtered["week"] = filtered["period"].dt.isocalendar().week
filtered["year_week"] = filtered["period"].dt.strftime("%Y-W%U")
weekly_spend = filtered.groupby("year_week")["amount"].sum()

current_week_key = weekly_spend.index.max()
current_week_total = weekly_spend.loc[current_week_key] if current_week_key in weekly_spend else 0
previous_week_total = weekly_spend.iloc[-2] if len(weekly_spend) > 1 else 0
wow_change = ((current_week_total - previous_week_total) / previous_week_total * 100) if previous_week_total > 0 else 0


# ==========  ROW 1 — PRIMARY FINANCIAL SNAPSHOT ==========
row1 = st.columns(5)

total_spend = filtered["amount"].sum()
row1[0].metric("💸 Total Spend", f"₹{total_spend:,.0f}")

avg_monthly = filtered.groupby("year_month")["amount"].sum().mean()
row1[1].metric("📅 Avg Monthly Spend", f"₹{avg_monthly:,.0f}")

avg_category = filtered.groupby("category")["amount"].mean().mean()
row1[2].metric("🏷 Avg Category Expense", f"₹{avg_category:,.0f}")

row1[3].metric("📆 Current Month Spend", f"₹{current_month_total:,.0f}")
row1[4].metric("📅 Today's Spend", f"₹{today_spend:,.0f}")


# ==========  ROW 2 — MONTH + WEEK PERFORMANCE ==========
k5, k6, k7, k8 = st.columns(4)

lifetime_total = df["amount"].sum()
percent_of_total = (total_spend / lifetime_total * 100) if lifetime_total>0 else 0
k5.metric("📊 % of Lifetime Spend Used", f"{percent_of_total:.2f}%")

best_month = filtered.groupby("year_month")["amount"].sum().idxmax()
best_month_amt = filtered.groupby("year_month")["amount"].sum().max()
k6.metric("🔥 Highest Expense Month", f"{best_month}: ₹{best_month_amt:,.0f}")

k7.metric("📅 Current Week Spend", f"₹{current_week_total:,.0f}")

# WEEK CHANGE SIGNAL (RED if Up 🔴 , GREEN if Down 🟢)
k8.metric("🔄 Week-over-Week Change",
           f"{wow_change:.2f}%",
           delta_color="inverse")  # green ↓ good, red ↑ bad

# =================================================
# ==========  ROW 3 — CATEGORY KPIs  ==========
# =================================================
k9, k10, k11, k12 = st.columns(4)

# --- Month on Month Change ---
prev_month = filtered.groupby("year_month")["amount"].sum().iloc[-2] if len(filtered)>1 else 0
mom_change = ((current_month_total - prev_month) / prev_month * 100) if prev_month>0 else 0
k9.metric("📆 MoM Spend Change", f"{mom_change:.2f}%")

# --- Top Category Spend ---
max_cat = filtered.groupby("category")["amount"].sum().idxmax()
max_cat_val = filtered.groupby("category")["amount"].sum().max()
k10.metric("🏆 Top Category", f"{max_cat}: ₹{max_cat_val:,.0f}")

# --- Lowest Category Spend ---
min_cat = filtered.groupby("category")["amount"].sum().idxmin()
min_cat_val = filtered.groupby("category")["amount"].sum().min()
k11.metric("🪫 Lowest Category", f"{min_cat}: ₹{min_cat_val:,.0f}")

# --- Daily Average Spend ---
daily_avg = filtered.groupby("period")["amount"].sum().mean()
k12.metric("📅 Avg Daily Spend", f"₹{daily_avg:,.0f}")



# =================================================
# ==========  ROW 4 — INCOME VS EXPENSE KPIs ==========
# =================================================
i1, i2, i3, i4 = st.columns(4)

# --- Income Expectation (Based on Your EDATE Logic) ---
from datetime import datetime
def get_expected_income(current_period):
    base = datetime(2024,10,1)
    current_period = pd.to_datetime(current_period)
    month_diff = (current_period.year-base.year)*12 + (current_period.month-base.month)
    return 12000 if month_diff==0 else 14112 if month_diff==1 else 24400

expected_income = get_expected_income(current_month_key)
i1.metric("💰 Expected Income", f"₹{expected_income:,.0f}")

# --- Income vs Expense Balance ---
income_balance = expected_income - current_month_total
status_emoji = "🟢 Saved" if income_balance>0 else "🔴 Overspent"
i2.metric("📊 Income Balance", f"₹{income_balance:,.0f}", status_emoji)

# --- Savings Rate % ---
savings_rate = (income_balance/expected_income*100) if expected_income>0 else 0
i3.metric("💾 Savings Rate %", f"{savings_rate:.1f}%" )

# --- Expense/Income Ratio Health ---
ratio = (current_month_total/expected_income*100) if expected_income>0 else 0
signal = "🟢 Healthy" if ratio < 70 else "🟡 Warning" if ratio < 100 else "🔴 Critical"
i4.metric("⚡ Spend % of Income", f"{ratio:.1f}%", signal)



# ==========  ROW 4 — LOGGED ACTIVITY ==========
row4 = st.columns(1)
days_count = filtered["period"].nunique()
row4[0].metric("📆 Total Days Logged", f"{days_count} days")


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

# Consistency Score → lower variation = better discipline
consistency_score = (1 - variance/variance.max())*100 if variance.max()>0 else 100
m4.metric("🧠 Consistency Score", f"{consistency_score.mean():.1f}%")


# ========= Category Share Table =============
st.write("### 📊 Spend Share Breakdown")
share_df = cat_summary.reset_index().rename(columns={"amount":"Total Spend"})
share_df["Share %"] = (share_df["Total Spend"]/total_spend*100).round(2)
st.dataframe(share_df, width="stretch")


# ================== NORMALIZED CATEGORY TREND CHART ==================
st.write("### 📈 Category Trend Over Time — Normalized for Comparison")
pivot = cat_month.pivot(index="year_month", columns="category", values="amount").fillna(0)
normalized = pivot.apply(lambda x:(x-x.min())/(x.max()-x.min() if x.max()!=x.min() else 1))
st.line_chart(normalized, width="stretch", height=320)
st.caption("📍 Trend only — values scaled 0-1 for visibility.")






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
# 📊 ANALYTICS DASHBOARD
# =================================================
st.divider()
st.header("📊 Insights & Analysis")


# 1️⃣ Monthly Spend Trend
m = filtered.groupby("year_month")["amount"].sum().reset_index()
if not m.empty:
    st.subheader("📅 Monthly Spending Trend")

    chart = (
        alt.Chart(m)
        .mark_line(point=True)
        .encode(
            x="year_month",
            y="amount"
        )
        .properties(width="container")   # 🔥 NEW responsive width (no warnings)
    )

    st.altair_chart(chart)   # ← no width/use_container_width here


# =================================================
# 💰 Monthly Budget Monitor (Updated + Sorted + Clean Format)
# =================================================
st.divider()
st.header("💰 Monthly Budget Monitor")

# Group monthly spend
b = filtered.groupby("year_month")["amount"].sum().reset_index()

# Convert YYYY-MM → Pretty Month Format (Nov 2025)
b["Month"] = pd.to_datetime(b["year_month"]).dt.strftime("%b %Y")

# Status Evaluation
b["Status"] = b["amount"].apply(lambda x: "🚨 Over Budget" if x > MONTHLY_BUDGET else "🟢 Within Limit")
b["Remaining / Excess"] = b["amount"].apply(
    lambda x: f"-₹{x-MONTHLY_BUDGET:,.0f}" if x>MONTHLY_BUDGET else f"+₹{MONTHLY_BUDGET-x:,.0f}"
)

# Sort latest month first
b = b.sort_values("year_month", ascending=False)

# Display clean table
st.dataframe(
    b[["Month","amount","Status","Remaining / Excess"]],
    width="stretch", height=260
)



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

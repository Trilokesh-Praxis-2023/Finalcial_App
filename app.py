import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prophet import Prophet
from io import BytesIO
import altair as alt
import os

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
# ➕ ADD EXPENSE ENTRY
# =================================================
with st.expander("➕ Add Expense"):
    with st.form("expense_form"):
        d = st.date_input("Date")
        acc = st.text_input("Account / UPI / Card")
        cat = st.selectbox("Category", CATEGORIES)
        amt = st.number_input("Amount", min_value=1.0, step=1.0)
        submit = st.form_submit_button("💾 Save Entry")

    if submit:
        pd.DataFrame([{"period":d,"accounts":acc,"category":cat,"amount":amt}])\
            .to_sql("finance_data",engine,if_exists="append",index=False)
        load_data.clear()
        st.success("Expense saved ✔")


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
# 📄 VIEW TRANSACTIONS + EXPORT
# =================================================
st.subheader("📄 Transactions")
st.dataframe(filtered, width="stretch", height=300)

csv = filtered.to_csv(index=False).encode("utf-8")
st.download_button("📄 Download CSV", csv, "transactions.csv")

buf = BytesIO()
with pd.ExcelWriter(buf) as writer: filtered.to_excel(writer, index=False)
st.download_button("📊 Download Excel", buf.getvalue(), "transactions.xlsx")


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




# 2️⃣ Category Spending
st.subheader("🏷 Category Spending")

cat_sorted = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)

st.bar_chart(cat_sorted)



# =================================================
# 💰 BUDGET ENFORCEMENT
# =================================================
st.divider()
st.header("💰 Monthly Budget Monitor")

b = filtered.groupby("year_month")["amount"].sum().reset_index()
b["Status"] = b["amount"].apply(lambda x: "🚨 Over" if x>MONTHLY_BUDGET else "🟢 OK")

st.dataframe(b)



# =================================================
# 💸 INCOME + SAVINGS SYSTEM
# =================================================
st.divider()
st.header("💸 Income + Savings Dashboard")

with st.expander("➕ Add Income"):
    with st.form("income_form"):
        inc_d = st.date_input("Income Date")
        inc_amt = st.number_input("Income Amount", min_value=1000.0)
        inc_save = st.form_submit_button("💾 Save Income")
    if inc_save:
        pd.DataFrame([{"period":inc_d,"income":inc_amt}])\
            .to_sql("finance_data",engine,if_exists="append",index=False)
        load_data.clear()
        st.success("Income saved ✔")


if "income" in df.columns:
    income = df.groupby("year_month")["income"].sum()
    expense = df.groupby("year_month")["amount"].sum()
    st.subheader("📈 Income vs Expense")
    st.line_chart(pd.DataFrame({"Income":income,"Expense":expense}))
    st.success(f"🔥 Total Savings: ₹{(income-expense).sum():,.0f}")

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

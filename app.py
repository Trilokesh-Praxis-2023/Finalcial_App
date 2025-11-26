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
st.bar_chart(filtered.groupby("category")["amount"].sum())


# 3️⃣ Account Distribution
st.subheader("👥 Accounts Usage Breakdown")
st.bar_chart(filtered.groupby("accounts")["amount"].sum())


# =================================================
# 💰 BUDGET ENFORCEMENT
# =================================================
st.divider()
st.header("💰 Monthly Budget Monitor")

b = filtered.groupby("year_month")["amount"].sum().reset_index()
b["Status"] = b["amount"].apply(lambda x: "🚨 Over" if x>MONTHLY_BUDGET else "🟢 OK")

st.dataframe(b)

# Alerts
for _,r in b.iterrows():
    (st.error if r.amount>MONTHLY_BUDGET else st.success)(
        f"{r.year_month}: {r.Status} — ₹{r.amount:,.0f}"
    )


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
# 🔮 FORECASTING (Prophet — Runs on Demand)
# =================================================
st.divider()
st.header("🔮 Expense Forecast AI (Next 6 Months)")

if st.button("Run Forecast Model"):
    fdf = df.groupby("year_month")["amount"].sum().reset_index()
    fdf["ds"] = pd.to_datetime(fdf.year_month)
    fdf.rename(columns={"amount":"y"}, inplace=True)

    model = Prophet().fit(fdf[["ds","y"]])
    future = model.make_future_dataframe(6,"M")
    forecast = model.predict(future)

    st.dataframe(forecast.tail(6)[["ds","yhat","yhat_lower","yhat_upper"]])
    st.pyplot(model.plot(forecast))

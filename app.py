import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from prophet import Prophet
from io import BytesIO
import altair as alt
import os
from datetime import datetime

# 🚀 Imported KPI Dashboards
from kpi_dashboard import render_kpis, get_income
from kpi_drilldown import render_kpi_suite


# ============================================================
# ⬛ PAGE CONFIG + TITLE
# ============================================================
load_dotenv()
st.set_page_config(page_title="💰 Finance Analytics", page_icon="📊", layout="wide")

st.markdown("<h1 class='title-main'>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='subtitle'>Track • Analyze • Forecast • Optimize</h5>", unsafe_allow_html=True)
st.write("")


# ===========================
# 💎 LOAD CUSTOM CSS
# ===========================
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error("❗ styles.css not found")


# ============================================================
# 📦 DATABASE CONNECTION
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL") or \
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"

APP_PASSWORD = os.getenv("APP_PASSWORD") 
engine = create_engine(DATABASE_URL)


# ============================================================
# 📥 DATA FETCH (CACHED)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM finance_data", engine)
    df.columns = df.columns.str.lower()
    df['period'] = pd.to_datetime(df['period'], errors='coerce')
    df['year'] = df.period.dt.year
    df['year_month'] = df.period.dt.to_period("M").astype(str)
    df['amount'] = df.amount.astype(float)
    return df


# ============================================================
# 🔐 PASSWORD GATE
# ============================================================
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")
if password != APP_PASSWORD:
    st.warning("🔒 Enter correct password to continue")
    st.stop()

st.success("🔓 Access Granted")


# ============================================================
# 📥 LOAD DB DATA
# ============================================================
df = load_data()


# ============================================================
# 🔎 FILTERS PANEL
# ============================================================
st.sidebar.markdown("<h3>🔍 Smart Filters</h3>", unsafe_allow_html=True)

f_year  = st.sidebar.multiselect("📆 Year", sorted(df.year.unique()))
f_month = st.sidebar.multiselect("🗓 Month", sorted(df.year_month.unique()))
f_cat   = st.sidebar.multiselect("🏷 Category", sorted(df.category.unique()))
f_acc   = st.sidebar.multiselect("💳 Account", sorted(df.accounts.unique()))

filtered = df.copy()
if f_year:  filtered = filtered[filtered.year.isin(f_year)]
if f_month: filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:   filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:   filtered = filtered[filtered.accounts.isin(f_acc)]




# ============================================================
# ➕ ADD EXPENSE ENTRY — FORM UI
# ============================================================
st.markdown("<h3>➕ Add Expense Entry</h3>", unsafe_allow_html=True)
with st.expander("Add Expense Form"):

    with st.form("expense_form", clear_on_submit=True):
        colA, colB = st.columns(2)

        with colA:
            d   = st.date_input("📅 Date")
            cat = st.selectbox("📂 Category", 
                ["Rent","Recharge","Transport","Food","Other","Household","Health",
                 "Apparel","Social Life","Beauty","Gift","Education"])

        with colB:
            acc = st.text_input("🏦 Account / UPI / Card")
            amt = st.number_input("💰 Amount", min_value=0.0)

        submit_exp = st.form_submit_button("💾 Save Entry")

    if submit_exp:
        try:
            monthly_total = df.amount.sum()
            new_total     = monthly_total + amt
            percent       = (amt/new_total)*100

            add_row = pd.DataFrame([{ 
                "period": pd.to_datetime(d), "accounts": acc, "category": cat,
                "amount": amt, "month": str(d)[:7], "percent_row": percent,
                "running_total": new_total 
            }])

            add_row.to_sql("finance_data", engine, index=False, if_exists="append")
            load_data.clear()
            st.success(f"Added ₹{amt} to {cat}")
            st.balloons()

        except Exception as e:
            st.error("❌ Database insert failed")
            st.code(e)


# ============================================================
# 📊 KPI MODULE DASHBOARDS
# ============================================================
st.markdown("<h3>📈 KPI Snapshot</h3>", unsafe_allow_html=True)
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=18000)


st.markdown("<h3>📉 Advanced KPI Drilldown</h3>", unsafe_allow_html=True)
render_kpi_suite(filtered, get_income)


# ============================================================
# 📄 TRANSACTION TABLE + EXPORT
# ============================================================
st.markdown("<h3>📄 Transactions</h3>", unsafe_allow_html=True)

df_show = filtered.copy()
df_show["period"] = df_show["period"].dt.date
df_show = df_show.sort_values("period", ascending=False)

st.dataframe(df_show, use_container_width=True, height=250)

csv = df_show.to_csv(index=False).encode()
st.download_button("📥 Export CSV", csv, "finance_data.csv")


# ============================================================
# ❌ DELETE TRANSACTION
# ============================================================
st.markdown("<h3>🗑 Delete Transaction</h3>", unsafe_allow_html=True)

try:
    df_del = pd.read_sql("SELECT *, ROW_NUMBER() OVER() AS id FROM finance_data", engine)
    df_del["period"] = pd.to_datetime(df_del["period"]).dt.date
    df_del = df_del.sort_values("period", ascending=False)

    st.dataframe(df_del[["id","period","accounts","category","amount"]], height=200)

    del_id = st.number_input("Row ID to Delete", step=1, min_value=1)
    if st.button("🗑 Delete"):
        row = df_del[df_del.id==del_id]
        if not row.empty:
            with engine.connect() as conn:
                conn.execute(text("""
                    DELETE FROM finance_data
                    WHERE period=:p AND accounts=:a AND category=:c AND amount=:m
                """), {
                    "p": row.iloc[0]["period"],
                    "a": row.iloc[0]["accounts"],
                    "c": row.iloc[0]["category"],
                    "m": row.iloc[0]["amount"]
                })
                conn.commit()
            load_data.clear()
            st.success("Deleted Successfully")
            st.rerun()
        else:
            st.error("⚠ Invalid ID")

except Exception as e:
    st.error("Could not load delete records")
    st.code(e)

# ============================================================
# 🔮 FORECASTING MODULE — MONTHLY + DAILY PREDICTION
# ============================================================
st.markdown("<h3>🔮 Forecasting & Prediction</h3>", unsafe_allow_html=True)

# ----------------------------------------------------------
# MONTHLY FORECAST — NEXT 6 MONTHS
# ----------------------------------------------------------
if st.button("📅 Predict Next 6 Months"):
    
    monthly_series = filtered.groupby("year_month")["amount"].sum().reset_index()

    if len(monthly_series) < 3:
        st.warning("⚠ Need minimum 3 months of data for monthly forecasting.")
    else:
        monthly_series["ds"] = pd.to_datetime(monthly_series.year_month)
        monthly_series.rename(columns={"amount": "y"}, inplace=True)

        m_model = Prophet()
        m_model.fit(monthly_series[["ds","y"]])

        future_m = m_model.make_future_dataframe(periods=6, freq="ME")
        forecast_m = m_model.predict(future_m)

        st.success("📈 6-Month Forecast Generated!")
        st.line_chart(forecast_m.set_index("ds")["yhat"])
        st.dataframe(forecast_m.tail(6)[["ds","yhat","yhat_lower","yhat_upper"]]
                     .rename(columns={"ds":"Month","yhat":"Prediction"}))


st.markdown("---")  # divider line for premium separation


# ----------------------------------------------------------
# DAILY FORECAST — NEXT 30 DAYS
# ----------------------------------------------------------
if st.button("📆 Predict Next 30 Days (Daily)"):

    daily_series = filtered.groupby("period")["amount"].sum().reset_index()

    if len(daily_series) < 7:
        st.warning("⚠ Need minimum 7 days of data for daily forecasting.")
    else:
        daily_series["ds"] = pd.to_datetime(daily_series["period"])
        daily_series.rename(columns={"amount":"y"}, inplace=True)

        d_model = Prophet(daily_seasonality=True)
        d_model.fit(daily_series[["ds","y"]])

        future_d = d_model.make_future_dataframe(periods=30, freq="D")
        forecast_d = d_model.predict(future_d)

        st.success("📆 30-Day Daily Forecast Ready!")
        st.line_chart(forecast_d.set_index("ds")["yhat"])
        st.dataframe(forecast_d.tail(30)[["ds","yhat","yhat_lower","yhat_upper"]]
                     .rename(columns={"ds":"Date","yhat":"Predicted"}))

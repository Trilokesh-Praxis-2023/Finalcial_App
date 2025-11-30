import streamlit as st
# Inject GOLD WAVE LOADER HTML+CSS
st.markdown("""
<div id="gold-loader">
    <div class="gold-wave"></div>
    <div class="gold-wave"></div>
    <div class="gold-wave"></div>
</div>
""", unsafe_allow_html=True)

import pandas as pd
import numpy as np
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
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


import threading, time, requests, os

def keep_alive():
    url = os.getenv("APP_URL")  # add your Streamlit URL in .env
    while True:
        try:
            requests.get(url)
        except:
            pass
        time.sleep(300)  # ping every 5 mins

threading.Thread(target=keep_alive, daemon=True).start()

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
# 🔮 FORECASTING MODULE — FAST + OPTIMIZED + CLEAN
# ============================================================

st.markdown('<h1 class="page-title">📊 Personal Finance Dashboard</h1>', unsafe_allow_html=True)



FIXED_RENT = 11600  # 🔸 fixed cost never forecasted, always added later


# ============================================================
# 📌 DAILY FORECAST (Prophet — Log Smoothed + Confidence Bands)
# ============================================================
def predict_daily(df, periods=30):

    df = df.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    if len(daily) < 7:
        return None, "⚠ Need at least 7 days of data for daily forecast."

    daily["y_log"] = np.log1p(daily["amount"])

    model = Prophet(weekly_seasonality=True, daily_seasonality=False)
    model.fit(daily.rename(columns={"period": "ds", "y_log": "y"})[["ds","y"]])

    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast = model.predict(future)

    # reverse log-scaling back to ₹
    forecast["yhat"] = np.expm1(forecast["yhat"])
    return daily.rename(columns={"period": "ds","amount":"y"}), forecast


# ============================================================
# 📊 Plot Forecast Shared Visual (History + Curve + Confidence)
# ============================================================
def plot_forecast(hist, forecast, title):

    base = alt.Chart(hist).mark_line(point=True, color="#4FC3F7").encode(
        x="ds:T", y="y:Q"
    )

    band = alt.Chart(forecast).mark_area(opacity=0.18, color="#FFD95A").encode(
        x="ds:T", y="yhat_upper:Q", y2="yhat_lower:Q"
    )

    line = alt.Chart(forecast).mark_line(color="#FFC107", strokeWidth=2.4).encode(
        x="ds:T", y="yhat:Q"
    ).properties(title=title)

    st.altair_chart(base + band + line, use_container_width=True)


# ============================================================
# 🤖 MONTHLY FORECAST — XGBoost ML (Variable Spend Only)
# ============================================================
def predict_monthly_ml(filtered, future_months=6):

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("year_month")

    if len(monthly) < 6:
        st.warning("⚠ Need at least 6 months for ML forecasting.")
        return

    # feature engineering
    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))              # time trend

    # learn only variable spend (rent removed)
    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    X = monthly[["month","year","t"]]
    y = monthly["variable"]

    # ML training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=420, learning_rate=0.06,
        max_depth=5, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.1, reg_lambda=1.3, objective="reg:squarederror"
    )
    model.fit(X_train, y_train)

    # evaluate
    mae = mean_absolute_error(y_test, model.predict(X_test))
    st.info(f"📊 Forecast MAE: **₹{mae:,.0f}** (lower = better)")

    # forecast future
    future = pd.DataFrame({
        "year_month": pd.date_range(
            start=monthly["year_month"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=future_months, freq="MS"
        )
    })
    future["month"] = future["year_month"].dt.month
    future["year"]  = future["year_month"].dt.year
    future["t"]     = range(len(monthly), len(monthly)+future_months)

    future["var_forecast"] = model.predict(future[["month","year","t"]]).clip(0)
    future["Total_Predicted"] = future["var_forecast"] + FIXED_RENT

    # plot
    combined = pd.concat([
        monthly.rename(columns={"year_month":"Month","amount":"Actual"})[["Month","Actual"]],
        future.rename(columns={"year_month":"Month","Total_Predicted":"Forecast"})[["Month","Forecast"]]
    ])

    chart = (
        alt.Chart(combined).mark_line(point=True, color="#FFC300").encode(
            x="Month:T", y="Actual:Q"
        ) +
        alt.Chart(combined).mark_line(point=True, color="#00E676", strokeDash=[4,3]).encode(
            x="Month:T", y="Forecast:Q"
        )
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)


# ============================================================
# 🔘 ACTION BUTTONS
# ============================================================
if st.button("🤖 Predict Next 6 Months (ML Smart Forecast)"):
    predict_monthly_ml(filtered)

if st.button("📅 Predict Next 30 Days (Daily Prophet)"):
    result = predict_daily(filtered)
    if result[0] is None:
        st.warning(result[1])
    else:
        hist, fc = result
        plot_forecast(hist, fc, "Daily Spend Forecast")

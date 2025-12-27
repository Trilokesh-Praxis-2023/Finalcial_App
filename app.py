# ============================================================
# 💰 PERSONAL FINANCE INTELLIGENCE DASHBOARD — app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import threading, time, requests
from datetime import datetime
from io import BytesIO

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from prophet import Prophet

# ============================================================
# 📦 IMPORT INTERNAL MODULES (utils MUST have __init__.py)
# ============================================================
from utils.kpi_dashboard import render_kpis, get_income
from utils.kpi_drilldown import render_kpi_suite
from utils.forecasting_ml import forecasting_ui


# ============================================================
# 🌀 GOLD LOADER
# ============================================================
st.markdown("""
<div id="gold-loader">
    <div class="gold-wave"></div>
    <div class="gold-wave"></div>
    <div class="gold-wave"></div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# 🛡 KEEP ALIVE (STREAMLIT CLOUD)
# ============================================================
def keep_alive():
    url = os.getenv("APP_URL")
    if not url:
        return
    while True:
        try:
            requests.get(url, timeout=5)
        except:
            pass
        time.sleep(300)

threading.Thread(target=keep_alive, daemon=True).start()


# ============================================================
# ⬛ PAGE CONFIG
# ============================================================
load_dotenv()

st.set_page_config(
    page_title="💰 Finance Analytics",
    page_icon="📊",
    layout="wide"
)

st.markdown("<h1 class='title-main'>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='subtitle'>Track • Analyze • Forecast • Optimize</h5>", unsafe_allow_html=True)
st.write("")


# ============================================================
# 🎨 LOAD CUSTOM CSS
# ============================================================
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("styles.css not found — continuing without custom theme")


# ============================================================
# 📦 DATABASE CONNECTION
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}"
    f"@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

APP_PASSWORD = os.getenv("APP_PASSWORD")
engine = create_engine(DATABASE_URL)


# ============================================================
# 📥 DATA FETCH (CACHED)
# ============================================================
@st.cache_data
def load_data():
    df = pd.read_sql("SELECT * FROM finance_data", engine)
    df.columns = df.columns.str.lower()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)
    df["amount"] = df.amount.astype(float)
    return df


# ============================================================
# 🔐 PASSWORD GATE
# ============================================================
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")
if APP_PASSWORD and password != APP_PASSWORD:
    st.warning("🔒 Enter correct password to continue")
    st.stop()

st.success("🔓 Access Granted")


# ============================================================
# 📥 LOAD DATA
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
if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:
    filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]


# ============================================================
# ➕ ADD EXPENSE ENTRY
# ============================================================
st.markdown("<h3>➕ Add Expense Entry</h3>", unsafe_allow_html=True)

with st.expander("Add Expense Form"):

    if st.button("🔄 Refresh data"):
        load_data.clear()
        st.rerun()

    with st.form("expense_form", clear_on_submit=True):
        colA, colB = st.columns(2)

        with colA:
            d = st.date_input("📅 Date")
            cat = st.selectbox(
                "📂 Category",
                ["Rent","Recharge","Transport","Food","Other","Household","Health",
                 "Apparel","Social Life","Beauty","Gift","Education","Party","Party_Weekend"],
                index=3
            )

        with colB:
            acc = st.text_input("🏦 Account / UPI / Card", value="UPI")
            amt = st.number_input("💰 Amount", min_value=0.0, value=10.0)

        submit = st.form_submit_button("💾 Save Entry")

    if submit:
        try:
            add_row = pd.DataFrame([{
                "period": pd.to_datetime(d),
                "accounts": acc,
                "category": cat,
                "amount": amt
            }])
            add_row.to_sql("finance_data", engine, index=False, if_exists="append")
            load_data.clear()
            st.success(f"Added ₹{amt} to {cat}")
            st.balloons()
            st.rerun()
        except Exception as e:
            st.error("❌ Database insert failed")
            st.code(e)


# ============================================================
# 📊 KPI DASHBOARD
# ============================================================
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)

st.markdown("<h3>📉 Advanced KPI Drilldown</h3>", unsafe_allow_html=True)
render_kpi_suite(filtered, get_income)


# ============================================================
# 📄 TRANSACTIONS TABLE
# ============================================================
st.markdown("<h3>📄 Transactions</h3>", unsafe_allow_html=True)

df_show = filtered.copy()
df_show["Period"] = df_show["Period"].dt.date
df_show = df_show.sort_values("Period", ascending=False)

st.dataframe(df_show, use_container_width=True, height=260)

csv = df_show.to_csv(index=False).encode()
st.download_button("📥 Export CSV", csv, "finance_data.csv")


# ============================================================
# 🗑 DELETE TRANSACTION
# ============================================================
st.markdown("<h3>🗑 Delete Transaction</h3>", unsafe_allow_html=True)

try:
    df_del = pd.read_sql("SELECT *, ROW_NUMBER() OVER() AS id FROM finance_data", engine)
    df_del["period"] = pd.to_datetime(df_del["period"]).dt.date
    df_del = df_del.sort_values("period", ascending=False)

    st.dataframe(df_del[["id","period","accounts","category","amount"]], height=220)

    del_id = st.number_input("Row ID to Delete", step=1, min_value=1)
    if st.button("🗑 Delete"):
        row = df_del[df_del.id == del_id]
        if not row.empty:
            with engine.begin() as conn:
                conn.execute(
                    text("""
                        DELETE FROM finance_data
                        WHERE period=:p AND accounts=:a AND category=:c AND amount=:m
                    """),
                    {
                        "p": row.iloc[0]["period"],
                        "a": row.iloc[0]["accounts"],
                        "c": row.iloc[0]["category"],
                        "m": row.iloc[0]["amount"]
                    }
                )
            load_data.clear()
            st.success("Deleted Successfully")
            st.rerun()
        else:
            st.error("⚠ Invalid ID")

except Exception as e:
    st.error("Could not load delete records")
    st.code(e)


# ============================================================
# 🔮 FORECASTING UI
# ============================================================
st.markdown("<h2 class='page-title'>🔮 AI Forecasting Module</h2>", unsafe_allow_html=True)

if filtered is not None and not filtered.empty:
    forecasting_ui(filtered)
else:
    st.warning("⚠ No data available for forecasting.")

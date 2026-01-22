import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from utils.github_storage import read_csv, write_csv
from utils.kpi_dashboard import render_kpis, get_income
from utils.kpi_drilldown import render_kpi_suite
from utils.forecasting_ml import forecasting_ui

load_dotenv()

st.set_page_config(page_title="💰 Finance Analytics", page_icon="📊", layout="wide")

st.markdown("<h1>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)

APP_PASSWORD = os.getenv("APP_PASSWORD")
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")
if APP_PASSWORD and password != APP_PASSWORD:
    st.stop()

# -----------------------------------------------------------
# LOAD DATA FROM GITHUB CSV
# -----------------------------------------------------------
@st.cache_data
def load_data():
    return read_csv()

df = load_data()

# -----------------------------------------------------------
# FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Filters")

f_year  = st.sidebar.multiselect("Year", sorted(df.year.unique()))
f_month = st.sidebar.multiselect("Month", sorted(df.year_month.unique()))
f_cat   = st.sidebar.multiselect("Category", sorted(df.category.unique()))
f_acc   = st.sidebar.multiselect("Account", sorted(df.accounts.unique()))

filtered = df.copy()
if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:
    filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

# -----------------------------------------------------------
# ADD EXPENSE
# -----------------------------------------------------------
st.markdown("### ➕ Add Expense")

with st.form("expense_form", clear_on_submit=True):
    d = st.date_input("Date")
    cat = st.selectbox("Category", df.category.unique())
    acc = st.text_input("Account", value="UPI")
    amt = st.number_input("Amount", min_value=0.0, value=10.0)
    submit = st.form_submit_button("Save")

if submit:
    new_row = pd.DataFrame([{
        "period": pd.to_datetime(d),
        "accounts": acc,
        "category": cat,
        "amount": amt
    }])

    df_new = pd.concat([df, new_row], ignore_index=True)
    write_csv(df_new, f"Added ₹{amt} in {cat}")
    load_data.clear()
    st.success("Added successfully")
    st.rerun()

# -----------------------------------------------------------
# KPIs
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)
render_kpi_suite(filtered, get_income)

# -----------------------------------------------------------
# TABLE
# -----------------------------------------------------------
st.markdown("### 📄 Transactions")
df_show = filtered.copy()
df_show["period"] = df_show["period"].dt.date
st.dataframe(df_show.sort_values("period", ascending=False), height=260)

# -----------------------------------------------------------
# DELETE
# -----------------------------------------------------------
st.markdown("### 🗑 Delete Transaction")

df_del = df.copy().reset_index()
df_del["period"] = df_del["period"].dt.date

st.dataframe(df_del[["index","period","accounts","category","amount"]])

del_id = st.number_input("Row ID", min_value=0)

if st.button("Delete"):
    df_new = df_del.drop(index=del_id).drop(columns=["index"])
    write_csv(df_new, f"Deleted row {del_id}")
    load_data.clear()
    st.success("Deleted")
    st.rerun()

# -----------------------------------------------------------
# FORECAST
# -----------------------------------------------------------
st.markdown("## 🔮 Forecasting")
if not filtered.empty:
    forecasting_ui(filtered)

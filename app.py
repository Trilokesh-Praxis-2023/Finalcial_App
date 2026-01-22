import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv

from utils.github_storage import read_csv, write_csv
from utils.kpi_dashboard import render_kpis, get_income
from utils.kpi_drilldown import render_kpi_suite
from utils.forecasting_ml import forecasting_ui

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="💰 Finance Analytics",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------------------------------
# LOAD CUSTOM CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("<h1 class='title-main'>💰 Personal Finance Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 class='subtitle'>Track • Analyze • Forecast • Optimize</h5>", unsafe_allow_html=True)

# -----------------------------------------------------------
# PASSWORD
# -----------------------------------------------------------
APP_PASSWORD = os.getenv("APP_PASSWORD")
password = st.sidebar.text_input("🔑 Enter Access Password", type="password")

if APP_PASSWORD and password != APP_PASSWORD:
    st.stop()

st.success("🔓 Access Granted")

# -----------------------------------------------------------
# LOAD DATA FROM GITHUB CSV
# -----------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    return read_csv()

def refresh():
    load_data.clear()
    st.rerun()

df = load_data()

# ✅ Always keep latest transaction on top
df = df.sort_values("period", ascending=False).reset_index(drop=True)

# -----------------------------------------------------------
# SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### 🔍 Smart Filters")

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
st.markdown("<h3>➕ Add Expense Entry</h3>", unsafe_allow_html=True)

with st.expander("Add Expense Form"):

    if st.button("🔄 Refresh data"):
        refresh()

    with st.form("expense_form", clear_on_submit=True):
        d = st.date_input("📅 Date")

        # Default Category = Food
        categories = sorted(df.category.unique())
        default_cat_index = categories.index("Food") if "Food" in categories else 0
        cat = st.selectbox("📂 Category", categories, index=default_cat_index)

        acc = st.text_input("🏦 Account / UPI / Card", value="UPI")

        # Default Amount = 11
        amt = st.number_input("💰 Amount", min_value=0.0, value=11.0)

        submit = st.form_submit_button("💾 Save Entry")

    if submit:
        dt = pd.to_datetime(d)

        # ---------------- Derived columns ----------------
        year = dt.year
        month = dt.strftime("%B")
        year_month = dt.strftime("%Y-%m")

        # ✅ Order-proof running total
        last_total = (
            df["running_total"].max()
            if "running_total" in df.columns and not df.empty
            else 0
        )
        running_total = last_total + amt

        new_row = pd.DataFrame([{
            "period": dt,
            "accounts": acc,
            "category": cat,
            "amount": amt,
            "year": year,
            "month": month,
            "year_month": year_month,
            "running_total": running_total
        }])

        df_new = pd.concat([df, new_row], ignore_index=True)

        write_csv(df_new, f"Added ₹{amt} in {cat}")
        st.success(f"Added ₹{amt} to {cat}")
        st.balloons()
        refresh()

# -----------------------------------------------------------
# KPI DASHBOARD
# -----------------------------------------------------------
render_kpis(filtered=filtered, df=df, MONTHLY_BUDGET=20000)
render_kpi_suite(filtered, get_income)

# -----------------------------------------------------------
# TRANSACTIONS TABLE
# -----------------------------------------------------------
st.markdown("<h3>📄 Transactions</h3>", unsafe_allow_html=True)

df_show = filtered.copy()
df_show["period"] = df_show["period"].dt.date
df_show = df_show.sort_values("period", ascending=False)

st.dataframe(df_show, use_container_width=True, height=260)

csv = df_show.to_csv(index=False).encode()
st.download_button("📥 Export CSV", csv, "finance_data.csv")

# -----------------------------------------------------------
# DELETE TRANSACTION
# -----------------------------------------------------------
st.markdown("<h3>🗑 Delete Transaction</h3>", unsafe_allow_html=True)

df_del = df.copy().reset_index()
df_del["period"] = df_del["period"].dt.date

st.dataframe(df_del[["index", "period", "accounts", "category", "amount"]], height=220)

del_id = st.number_input("Row ID to Delete", min_value=0, step=1)

if st.button("🗑 Delete"):
    df_new = df_del.drop(index=del_id).drop(columns=["index"])
    write_csv(df_new, f"Deleted row {del_id}")
    st.success("Deleted Successfully")
    refresh()

# -----------------------------------------------------------
# FORECASTING
# -----------------------------------------------------------
st.markdown("<h2 class='page-title'>🔮 AI Forecasting Module</h2>", unsafe_allow_html=True)

if not filtered.empty:
    forecasting_ui(filtered)
else:
    st.warning("⚠ No data available for forecasting.")

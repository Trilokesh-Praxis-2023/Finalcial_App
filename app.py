import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from prophet import Prophet
from io import BytesIO
import altair as alt
import os
from sqlalchemy import text
import os
from kpi_dashboard import render_kpis, get_income
from datetime import datetime



# ===========================
# 💎 Load CSS Theme (Glass UI)
# ===========================

css_path = os.path.join(".streamlit", "styles.css")   # auto-correct path

if os.path.isfile(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.error("❗ styles.css NOT FOUND — place it inside .streamlit/styles.css")
    st.info("Expected path → .streamlit/styles.css")




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
# 🔥 ADD EXPENSE ENTRY — Clean UI + Better Structure + UX Upgrades
# =================================================
st.markdown("### ➕ Add New Expense Record")

with st.expander("Add Expense"):
    
    with st.form("expense_form", clear_on_submit=True):
        st.write("Enter your transaction details below 👇")

        col1, col2 = st.columns(2)
        with col1:
            d = st.date_input("📅 Date")
            cat = st.selectbox("📂 Category", CATEGORIES)

        with col2:
            acc = st.text_input("🏦 Account / UPI / Card")
            amt = st.number_input("💰 Amount", min_value=0.0, step=1.0)

        st.markdown("---")
        submit = st.form_submit_button("💾 Save Expense", use_container_width=True)

    if submit:
        # --- Month Processing
        month_value = pd.to_datetime(d).strftime("%Y-%m")

        # --- Running Total Logic
        current_total = df["amount"].sum() if not df.empty else 0
        new_running_total = current_total + float(amt)
        row_percent = (float(amt) / new_running_total) * 100

        try:
            df_new = pd.DataFrame([{
                "period"        : pd.to_datetime(d),
                "accounts"      : acc.title(),      # 🔥 Cleaner Text
                "category"      : cat,
                "amount"        : float(amt),
                "month"         : month_value,
                "percent_row"   : round(row_percent, 2),
                "running_total" : new_running_total
            }])

            df_new.to_sql("finance_data", engine, if_exists="append", index=False)
            load_data.clear()

            st.success(f"✔ Saved — ₹{amt:.0f} added to **{cat}** ({row_percent:.2f}%)")
            st.balloons()  # 🎉 Feel-good UX

        except Exception as e:
            st.error(f"❌ Failed to Upload:\n{e}")



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
# 📊 KPI DASHBOARD IMPORTED FROM MODULE
# =================================================

render_kpis(
    filtered=filtered,
    df=df,
    MONTHLY_BUDGET=MONTHLY_BUDGET
)





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

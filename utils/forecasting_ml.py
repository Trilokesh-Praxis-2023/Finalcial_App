# ============================================================
# 📁 forecasting_ml.py
# ML Forecasting (Daily + Monthly) + Auto-Predict + Performance Evaluation
# ============================================================

import os, joblib
import pandas as pd
import streamlit as st
import altair as alt
import xgboost as xgb
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# GLOBAL CONFIG
FIXED_RENT = 11600 + 588 + 470     # your updated rent
MODEL_DIR = "models"
DAILY_MODEL_PATH   = f"{MODEL_DIR}/daily_forecast_model.pkl"
MONTHLY_MODEL_PATH = f"{MODEL_DIR}/monthly_forecast_model.pkl"


# ============================================================
# 📆 DAILY MODEL TRAINING
# ============================================================
def train_daily_model(filtered):
    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    if len(daily) < 10:
        st.warning("⚠ Need at least 10 days of data to train Daily Model.")
        return

    daily["day"]   = daily["period"].dt.day
    daily["dow"]   = daily["period"].dt.dayofweek
    daily["month"] = daily["period"].dt.month
    daily["t"]     = range(len(daily))

    X = daily[["day", "dow", "month", "t"]]
    y = daily["amount"].clip(lower=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.05,
        max_depth=6, subsample=0.85, colsample_bytree=0.9,
        reg_alpha=1.2, reg_lambda=1.3, objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, DAILY_MODEL_PATH)

    st.success(f"📆 Daily Model Trained — MAE: ₹{mae:,.0f} (Saved)")


# ============================================================
# 📆 DAILY FORECASTING (Next 30 Days)
# ============================================================
def predict_daily_ml(filtered, future_days=30):
    if not os.path.exists(DAILY_MODEL_PATH):
        st.error("⚠ Daily Model Missing. Train First.")
        return

    model = joblib.load(DAILY_MODEL_PATH)

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    last_date = daily["period"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)

    future = pd.DataFrame({"period": future_dates})
    future["day"]   = future["period"].dt.day
    future["dow"]   = future["period"].dt.dayofweek
    future["month"] = future["period"].dt.month
    future["t"]     = range(len(daily), len(daily) + future_days)

    future["Forecast"] = model.predict(future[["day","dow","month","t"]]).clip(0)

    hist = daily.rename(columns={"period": "Date", "amount": "Actual"})
    future = future.rename(columns={"period": "Date"})

    chart = (
        alt.Chart(hist).mark_line(point=True, color="#4FC3F7")
        .encode(x="Date:T", y="Actual:Q")
        +
        alt.Chart(future).mark_line(point=True, color="#FFC107", strokeDash=[4,3])
        .encode(x="Date:T", y="Forecast:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)



# ============================================================
# 📏 DAILY MODEL PERFORMANCE (Backtest)
# ============================================================
def evaluate_daily_model(filtered):
    if not os.path.exists(DAILY_MODEL_PATH):
        st.error("⚠ Train Daily Model First")
        return

    model = joblib.load(DAILY_MODEL_PATH)

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    daily["day"]   = daily["period"].dt.day
    daily["dow"]   = daily["period"].dt.dayofweek
    daily["month"] = daily["period"].dt.month
    daily["t"]     = range(len(daily))

    daily["Predicted"] = model.predict(daily[["day","dow","month","t"]])
    daily["Error"] = daily["amount"] - daily["Predicted"]

    mae  = abs(daily["Error"]).mean()
    mape = (abs(daily["Error"]) / daily["amount"].replace(0, 1)).mean() * 100

    st.subheader("📏 Daily Model Performance")
    st.write(f"**MAE:** ₹{mae:,.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    chart = (
        alt.Chart(daily.rename(columns={"period": "Date", "amount": "Actual"}))
        .mark_line(point=True, color="#42A5F5")
        .encode(x="Date:T", y="Actual:Q")
        +
        alt.Chart(daily.rename(columns={"period": "Date"}))
        .mark_line(point=True, color="#EF5350", strokeDash=[4,3])
        .encode(x="Date:T", y="Predicted:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(daily[["period","amount","Predicted","Error"]])



# ============================================================
# 📅 MONTHLY MODEL TRAINING
# ============================================================
def train_monthly_model(filtered):

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    if len(monthly) < 6:
        st.warning("⚠ Need at least 6 months of data.")
        return

    monthly = monthly.sort_values("year_month")
    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))

    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    X = monthly[["month","year","t"]]
    y = monthly["variable"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=450, learning_rate=0.06,
        max_depth=6, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.1, reg_lambda=1.3, objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MONTHLY_MODEL_PATH)

    st.success(f"📊 Monthly Model Trained — MAE: ₹{mae:,.0f} (Saved)")


# ============================================================
# 📅 MONTHLY FORECAST (Next 6 Months)
# ============================================================
def predict_monthly_ml(filtered, months=6):

    if not os.path.exists(MONTHLY_MODEL_PATH):
        st.error("⚠ Monthly Model Missing. Train First.")
        return

    model = joblib.load(MONTHLY_MODEL_PATH)

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    monthly = monthly.sort_values("year_month")
    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))
    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    future = pd.DataFrame({
        "year_month": pd.date_range(
            start=monthly["year_month"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=months, freq="MS"
        )
    })

    future["month"] = future["year_month"].dt.month
    future["year"]  = future["year_month"].dt.year
    future["t"]     = range(len(monthly), len(monthly)+months)

    future["variable_pred"] = model.predict(future[["month","year","t"]])
    future["Forecast"] = future["variable_pred"] + FIXED_RENT

    hist = monthly.rename(columns={"year_month":"Month","amount":"Actual"})
    future = future.rename(columns={"year_month":"Month"})

    chart = (
        alt.Chart(hist).mark_line(point=True, color="#FFC300")
        .encode(x="Month:T", y="Actual:Q")
        +
        alt.Chart(future).mark_line(point=True, color="#00E676", strokeDash=[4,3])
        .encode(x="Month:T", y="Forecast:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)



# ============================================================
# 📏 MONTHLY PERFORMANCE EVALUATION (Backtest)
# ============================================================
def evaluate_monthly_model(filtered):
    if not os.path.exists(MONTHLY_MODEL_PATH):
        st.error("⚠ Train Monthly Model First")
        return

    model = joblib.load(MONTHLY_MODEL_PATH)

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    monthly = monthly.sort_values("year_month")
    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))

    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    monthly["Pred_var"] = model.predict(monthly[["month","year","t"]])
    monthly["Predicted"] = monthly["Pred_var"] + FIXED_RENT

    monthly["Error"] = monthly["amount"] - monthly["Predicted"]

    mae  = abs(monthly["Error"]).mean()
    mape = (abs(monthly["Error"]) / monthly["amount"].replace(0, 1)).mean() * 100

    st.subheader("📊 Monthly Model Performance")
    st.write(f"**MAE:** ₹{mae:,.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    chart = (
        alt.Chart(monthly.rename(columns={"year_month":"Month","amount":"Actual"}))
        .mark_line(point=True, color="#FB8C00")
        .encode(x="Month:T", y="Actual:Q")
        +
        alt.Chart(monthly.rename(columns={"year_month":"Month"}))
        .mark_line(point=True, color="#43A047", strokeDash=[4,3])
        .encode(x="Month:T", y="Predicted:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(monthly[["year_month","amount","Predicted","Error"]])



# ============================================================
# 📅 CURRENT MONTH FORECAST (Daily Model)
# ============================================================
def predict_current_month_forecast(filtered):

    if not os.path.exists(DAILY_MODEL_PATH):
        st.error("⚠ Daily Model Missing — Train First.")
        return

    model = joblib.load(DAILY_MODEL_PATH)

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    today = pd.Timestamp.today().normalize()
    end_of_month = today + pd.offsets.MonthEnd(0)

    future_dates = pd.date_range(today, end_of_month)

    future = pd.DataFrame({"period": future_dates})
    future["day"]   = future["period"].dt.day
    future["dow"]   = future["period"].dt.dayofweek
    future["month"] = future["period"].dt.month
    future["t"]     = range(len(daily), len(daily) + len(future_dates))

    future["Forecast"] = model.predict(future[["day","dow","month","t"]]).clip(0)

    amount_spent = df[df["period"].dt.to_period("M") == today.to_period("M")]["amount"].sum()
    month_forecast_total = amount_spent + future["Forecast"].sum()

    st.subheader("📅 Current Month Forecast Summary")

    c1, c2, c3 = st.columns(3)
    c1.metric("💸 Spent So Far", f"₹{amount_spent:,.0f}")
    c2.metric("🔮 Remaining Forecast", f"₹{future['Forecast'].sum():,.0f}")
    c3.metric("📊 Total This Month", f"₹{month_forecast_total:,.0f}")

    hist_month = daily[
        daily["period"].dt.to_period("M") == today.to_period("M")
    ].rename(columns={"period":"Date","amount":"Actual"})

    future_chart = future.rename(columns={"period":"Date"})

    chart = (
        alt.Chart(hist_month).mark_line(point=True, color="#4CAF50")
        .encode(x="Date:T", y="Actual:Q")
        +
        alt.Chart(future_chart).mark_line(point=True, color="#F57F17", strokeDash=[4,3])
        .encode(x="Date:T", y="Forecast:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future_chart)



# ============================================================
# 🔘 UI: BEAUTIFUL AUTO FORECAST + MANUAL CONTROLS + PERFORMANCE
# ============================================================
def forecasting_ui(filtered):

    st.markdown("""
        <h2 style='color:#6C5CE7;'>🔮 Machine Learning Forecast Dashboard</h2>
        <p style='color:gray;'>Automated prediction system with XGBoost — daily + monthly forecasts.</p>
    """, unsafe_allow_html=True)

    # ============================================================
    # 🔄 AUTO-TRAIN IF MODEL NOT FOUND
    # ============================================================
    with st.expander("⚙️ Model Status & Auto-Training", expanded=False):
        if not os.path.exists(DAILY_MODEL_PATH):
            st.info("📌 Daily model missing → training automatically...")
            train_daily_model(filtered)
        else:
            st.success("📆 Daily Model Loaded ✔")

        if not os.path.exists(MONTHLY_MODEL_PATH):
            st.info("📌 Monthly model missing → training automatically...")
            train_monthly_model(filtered)
        else:
            st.success("📅 Monthly Model Loaded ✔")

    # ============================================================
    # 📊 AUTO FORECAST SECTION
    # ============================================================
    st.markdown("<h3 style='color:#0984e3;'>📆 Daily Forecast (Next 30 Days)</h3>", unsafe_allow_html=True)
    predict_daily_ml(filtered)

    st.markdown("<h3 style='color:#6c5ce7;'>📅 Current Month Forecast</h3>", unsafe_allow_html=True)
    predict_current_month_forecast(filtered)

    st.markdown("<h3 style='color:#00b894;'>📅 Monthly Forecast (Next 6 Months)</h3>", unsafe_allow_html=True)
    predict_monthly_ml(filtered)

    st.markdown("---")

    # ============================================================
    # 🔧 MANUAL CONTROLS
    # ============================================================
    st.markdown("""
        <h3 style='color:#e17055;'>⚙️ Manual Model Controls</h3>
        <p style='color:gray;'>Use these buttons if you want to manually retrain or re-run predictions.</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🛠 Retrain Daily Model"):
            train_daily_model(filtered)

        if st.button("📅 Re-run Daily Forecast"):
            predict_daily_ml(filtered)

    with col2:
        if st.button("🛠 Retrain Monthly Model"):
            train_monthly_model(filtered)

        if st.button("🤖 Re-run Monthly Forecast"):
            predict_monthly_ml(filtered)

    st.markdown("---")

    # ============================================================
    # 📈 PERFORMANCE EVALUATION
    # ============================================================
    st.markdown("""
        <h3 style='color:#6c5ce7;'>📈 Forecast Accuracy & Model Performance</h3>
        <p style='color:gray;'>View how well the model performed against past actual values.</p>
    """, unsafe_allow_html=True)

    with st.expander("📏 Daily Model Performance", expanded=False):
        evaluate_daily_model(filtered)

    with st.expander("📊 Monthly Model Performance", expanded=False):
        evaluate_monthly_model(filtered)

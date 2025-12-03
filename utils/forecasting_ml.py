# ============================================================
# 📁 forecasting_ml.py
# Full ML-Based Forecasting System (Daily + Monthly Models)
# Auto-train + Auto-predict on load
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
FIXED_RENT = 11600  + 588 + 470
MODEL_DIR = "models"
MONTHLY_MODEL_PATH = f"{MODEL_DIR}/monthly_forecast_model.pkl"
DAILY_MODEL_PATH   = f"{MODEL_DIR}/daily_forecast_model.pkl"


# ============================================================
# 📆 DAILY XGBoost FORECAST — Train + Save Model
# ============================================================
def train_daily_model(filtered):
    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    if len(daily) < 10:
        st.warning("⚠ Need at least 10 days to train Daily Model.")
        return

    # --------- FEATURES ---------
    daily["day"]   = daily["period"].dt.day
    daily["dow"]   = daily["period"].dt.dayofweek
    daily["month"] = daily["period"].dt.month
    daily["t"]     = range(len(daily))

    X = daily[["day", "dow", "month", "t"]]
    y = daily["amount"].clip(lower=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

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


def predict_daily_ml(filtered, future_days=30):

    if not os.path.exists(DAILY_MODEL_PATH):
        st.error("⚠ No Daily Model Found — Train First")
        return

    model = joblib.load(DAILY_MODEL_PATH)

    df = filtered.copy()
    df["period"] = pd.to_datetime(df["period"])
    daily = df.groupby("period")["amount"].sum().reset_index()

    # -------- FUTURE --------
    last_date = daily["period"].max()
    future_dates = pd.date_range(last_date + timedelta(days=1), periods=future_days)

    future = pd.DataFrame({"period": future_dates})
    future["day"]   = future["period"].dt.day
    future["dow"]   = future["period"].dt.dayofweek
    future["month"] = future["period"].dt.month
    future["t"]     = range(len(daily), len(daily) + future_days)

    future["Forecast"] = model.predict(future[["day", "dow", "month", "t"]]).clip(0)

    # VISUAL
    hist = daily.rename(columns={"period": "Date", "amount": "Actual"})
    future = future.rename(columns={"period": "Date"})

    chart = (
        alt.Chart(hist).mark_line(point=True, color="#4FC3F7").encode(
            x="Date:T", y="Actual:Q"
        ) +
        alt.Chart(future).mark_line(point=True, color="#FFC107", strokeDash=[4, 3]).encode(
            x="Date:T", y="Forecast:Q"
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)



# ============================================================
# 📅 MONTHLY FORECAST — Train + Predict (XGBoost)
# ============================================================
def train_monthly_model(filtered):

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    if len(monthly) < 6:
        st.warning("⚠ Need at least 6 months to train Monthly Model.")
        return

    monthly = monthly.sort_values("year_month")
    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))
    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    X = monthly[["month", "year", "t"]]
    y = monthly["variable"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

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


def predict_monthly_ml(filtered, months=6):

    if not os.path.exists(MONTHLY_MODEL_PATH):
        st.error("⚠ No Monthly Model Found — Train First")
        return

    model = joblib.load(MONTHLY_MODEL_PATH)

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("year_month")

    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))
    monthly["variable"] = (monthly["amount"] - FIXED_RENT).clip(lower=0)

    # Future
    future = pd.DataFrame({
        "year_month": pd.date_range(
            start=monthly["year_month"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=months, freq="MS"
        )
    })

    future["month"] = future["year_month"].dt.month
    future["year"]  = future["year_month"].dt.year
    future["t"]     = range(len(monthly), len(monthly) + months)

    future["variable_pred"] = model.predict(future[["month", "year", "t"]]).clip(0)
    future["Forecast"] = future["variable_pred"] + FIXED_RENT

    hist = monthly.rename(columns={"year_month": "Month", "amount": "Actual"})
    future = future.rename(columns={"year_month": "Month"})

    chart = (
        alt.Chart(hist).mark_line(point=True, color="#FFC300").encode(
            x="Month:T", y="Actual:Q"
        ) +
        alt.Chart(future).mark_line(point=True, color="#00E676", strokeDash=[4, 3]).encode(
            x="Month:T", y="Forecast:Q"
        )
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)



# ============================================================
# 🔘 UI SECTION — Auto Forecast + Buttons
# ============================================================
def forecasting_ui(filtered):

    st.subheader("🔮 Machine Learning Based Forecasting")

    # AUTO-TRAIN IF MODEL MISSING
    if not os.path.exists(DAILY_MODEL_PATH):
        st.info("📌 Daily model not found — training automatically...")
        train_daily_model(filtered)

    if not os.path.exists(MONTHLY_MODEL_PATH):
        st.info("📌 Monthly model not found — training automatically...")
        train_monthly_model(filtered)

    # AUTO FORECAST ON PAGE LOAD
    st.markdown("## 📆 Auto Daily Forecast (Next 30 Days)")
    predict_daily_ml(filtered)

    st.markdown("## 📅 Auto Monthly Forecast (Next 6 Months)")
    predict_monthly_ml(filtered)

    # Manual Options
    st.markdown("---")
    st.markdown("### ⚙️ Manual Re-train or Re-run")

    if st.button("🛠 Retrain Daily Model"):
        train_daily_model(filtered)

    if st.button("📅 Re-run Daily Forecast"):
        predict_daily_ml(filtered)

    if st.button("🛠 Retrain Monthly Model"):
        train_monthly_model(filtered)

    if st.button("🤖 Re-run Monthly Forecast"):
        predict_monthly_ml(filtered)

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
        reg_alpha=1.2, reg_lambda=1.3,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, DAILY_MODEL_PATH)

    st.success(f"📆 Daily Model Trained — MAE: ₹{mae:,.0f} (Saved)")


# ============================================================
# 📆 DAILY FORECAST (Next 30 Days)
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

    future["Forecast"] = model.predict(
        future[["day","dow","month","t"]]
    ).clip(0)

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
# 📏 DAILY MODEL PERFORMANCE
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
# 📅 MONTHLY MODEL TRAINING (NO FIXED RENT)
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

    X = monthly[["month","year","t"]]
    y = monthly["amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=450, learning_rate=0.06,
        max_depth=6, subsample=0.9, colsample_bytree=0.9,
        reg_alpha=1.1, reg_lambda=1.3,
        objective="reg:squarederror"
    )

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MONTHLY_MODEL_PATH)

    st.success(f"📊 Monthly Model Trained — MAE: ₹{mae:,.0f} (Saved)")


# ============================================================
# 📅 MONTHLY FORECAST
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

    future = pd.DataFrame({
        "year_month": pd.date_range(
            start=monthly["year_month"].iloc[-1] + pd.offsets.MonthBegin(),
            periods=months, freq="MS"
        )
    })

    future["month"] = future["year_month"].dt.month
    future["year"]  = future["year_month"].dt.year
    future["t"]     = range(len(monthly), len(monthly) + months)

    future["Forecast"] = model.predict(future[["month","year","t"]]).clip(0)

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
# 📏 MONTHLY PERFORMANCE
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

    monthly["Predicted"] = model.predict(monthly[["month","year","t"]])
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

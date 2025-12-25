# ============================================================
# 📁 forecasting_ml.py
# Daily ML Forecast + Monthly Rolling Mean (Stable & Reliable)
# ============================================================

import os, joblib
import pandas as pd
import streamlit as st
import altair as alt
import xgboost as xgb
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


# ============================================================
# GLOBAL CONFIG
# ============================================================
MODEL_DIR = "models"
DAILY_MODEL_PATH = f"{MODEL_DIR}/daily_forecast_model.pkl"


# ============================================================
# 📆 DAILY MODEL TRAINING (NO TREND LEAKAGE)
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

    X = daily[["day", "dow", "month"]]
    y = daily["amount"].clip(lower=0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_alpha=1.0,
        reg_lambda=1.2,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, DAILY_MODEL_PATH)

    st.success(f"📆 Daily Model Trained — MAE: ₹{mae:,.0f} (Saved)")


# ============================================================
# 📆 DAILY FORECAST (SAFE + CAPPED)
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
    future_dates = pd.date_range(
        last_date + timedelta(days=1),
        periods=future_days
    )

    future = pd.DataFrame({"period": future_dates})
    future["day"]   = future["period"].dt.day
    future["dow"]   = future["period"].dt.dayofweek
    future["month"] = future["period"].dt.month

    max_daily = daily["amount"].quantile(0.95)
    avg_daily = daily["amount"].mean()

    preds = model.predict(future[["day", "dow", "month"]])
    preds = preds.clip(0, max_daily)

    if preds.mean() > avg_daily * 2:
        preds = preds.clip(0, avg_daily * 1.5)

    future["Forecast"] = preds

    hist = daily.rename(columns={"period": "Date", "amount": "Actual"})
    future = future.rename(columns={"period": "Date"})

    chart = (
        alt.Chart(hist)
        .mark_line(point=True, color="#4FC3F7")
        .encode(x="Date:T", y="Actual:Q")
        +
        alt.Chart(future)
        .mark_line(point=True, color="#FFC107", strokeDash=[4,3])
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

    daily["Predicted"] = model.predict(
        daily[["day", "dow", "month"]]
    )

    daily["Error"] = daily["amount"] - daily["Predicted"]

    mae  = abs(daily["Error"]).mean()
    mape = (abs(daily["Error"]) / daily["amount"].replace(0, 1)).mean() * 100

    st.subheader("📏 Daily Model Performance")
    st.write(f"**MAE:** ₹{mae:,.2f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    st.dataframe(daily[["period","amount","Predicted","Error"]])


# ============================================================
# 📅 MONTHLY FORECAST — ROLLING MEAN (NO ML)
# ============================================================
def predict_monthly_rolling_mean(filtered, months=6, window=3):

    monthly = filtered.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("year_month")

    if len(monthly) < 2:
        st.warning("⚠ Not enough data for monthly forecast.")
        return

    win = min(window, len(monthly))

    rolling_value = (
        monthly["amount"]
        .rolling(win)
        .mean()
        .iloc[-1]
    )

    future_months = pd.date_range(
        start=monthly["year_month"].iloc[-1] + pd.offsets.MonthBegin(),
        periods=months,
        freq="MS"
    )

    future = pd.DataFrame({
        "Month": future_months,
        "Forecast": [rolling_value] * months
    })

    hist = monthly.rename(
        columns={"year_month": "Month", "amount": "Actual"}
    )

    chart = (
        alt.Chart(hist)
        .mark_line(point=True, color="#FFC300")
        .encode(x="Month:T", y="Actual:Q")
        +
        alt.Chart(future)
        .mark_line(point=True, color="#00E676", strokeDash=[4,3])
        .encode(x="Month:T", y="Forecast:Q")
    )

    st.altair_chart(chart, use_container_width=True)
    st.dataframe(future)

    st.caption(
        f"📌 Monthly forecast uses {win}-month rolling average (stable mode)"
    )


# ============================================================
# 🔮 FORECASTING UI (ENTRY POINT)
# ============================================================
def forecasting_ui(filtered):

    st.markdown("""
        <h3 style='color:#6C5CE7;'>🔮 Machine Learning Forecast Dashboard</h3>
        <p style='color:gray;'>
            Daily ML forecast + Monthly rolling average (production safe)
        </p>
    """, unsafe_allow_html=True)

    with st.expander("⚙️ Model Status & Auto-Training", expanded=False):

        if not os.path.exists(DAILY_MODEL_PATH):
            st.info("📌 Daily model missing → training automatically...")
            train_daily_model(filtered)
        else:
            st.success("📆 Daily Model Loaded ✔")

    st.markdown("### 📆 Daily Forecast (Next 30 Days)")
    predict_daily_ml(filtered)

    st.markdown("### 📅 Monthly Forecast (Rolling Average)")
    predict_monthly_rolling_mean(filtered)

    st.markdown("---")

    with st.expander("📏 Daily Model Performance", expanded=False):
        evaluate_daily_model(filtered)

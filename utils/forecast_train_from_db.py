# ============================================================
# 📁 forecast_train_from_github.py
# Load finance_data from GitHub CSV → Train & Save ML Models
# ============================================================

import os
import joblib
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from utils.github_storage import read_csv   # ✅ IMPORTANT

# ============================================================
# GLOBAL CONFIG
# ============================================================
MODEL_DIR = "models"
DAILY_MODEL_PATH   = os.path.join(MODEL_DIR, "daily_forecast_model.pkl")
MONTHLY_MODEL_PATH = os.path.join(MODEL_DIR, "monthly_forecast_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
load_dotenv()


# ============================================================
# 📥 LOAD DATA FROM GITHUB CSV
# ============================================================
def load_data():
    try:
        df = read_csv()   # ✅ FROM GITHUB
        df.columns = df.columns.str.lower()

        df["period"] = pd.to_datetime(df["period"], errors="coerce")
        df["amount"] = df["amount"].astype(float)
        df["year_month"] = df["period"].dt.to_period("M").astype(str)

        df = df.dropna(subset=["period"])

        print(f"✔ Loaded {len(df)} rows from GitHub finance_data")
        return df

    except Exception as e:
        print(f"❌ Error loading GitHub data: {e}")
        return pd.DataFrame()


# ============================================================
# 💾 SAVE MODEL
# ============================================================
def save_model(model, path):
    joblib.dump(model, path)
    print(f"💾 Model saved → {path}")


# ============================================================
# 📆 TRAIN DAILY MODEL
# ============================================================
def train_daily_model(data):

    daily = data.groupby("period")["amount"].sum().reset_index()

    if len(daily) < 10:
        print("⚠ Need at least 10 days of data to train Daily Model.")
        return

    daily["day"]   = daily["period"].dt.day
    daily["dow"]   = daily["period"].dt.dayofweek
    daily["month"] = daily["period"].dt.month

    cap = daily["amount"].quantile(0.95)
    daily["amount_capped"] = daily["amount"].clip(0, cap)

    X = daily[["day", "dow", "month"]]
    y = daily["amount_capped"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=120,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.5,
        reg_lambda=2.0,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    save_model(model, DAILY_MODEL_PATH)
    print(f"📆 Daily Model Trained — MAE: ₹{mae:,.2f}")


# ============================================================
# 📅 TRAIN MONTHLY MODEL
# ============================================================
def train_monthly_model(data):

    monthly = data.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])

    if len(monthly) < 6:
        print("⚠ Need at least 6 months of data to train Monthly Model.")
        return

    monthly = monthly.sort_values("year_month")

    monthly["month"] = monthly["year_month"].dt.month
    monthly["year"]  = monthly["year_month"].dt.year
    monthly["t"]     = range(len(monthly))

    X = monthly[["month", "year", "t"]]
    y = monthly["amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=False
    )

    model = xgb.XGBRegressor(
        n_estimators=350,
        learning_rate=0.06,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=1.1,
        reg_lambda=1.3,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    save_model(model, MONTHLY_MODEL_PATH)
    print(f"📊 Monthly Model Trained — MAE: ₹{mae:,.2f}")


# ============================================================
# ▶ MAIN
# ============================================================
if __name__ == "__main__":

    print("🔄 Loading finance_data from GitHub...")
    df = load_data()

    if df.empty:
        print("❌ No data found — cannot train models.")
        exit()

    print("\n==============================")
    print(" TRAINING DAILY MODEL ")
    print("==============================")
    train_daily_model(df)

    print("\n==============================")
    print(" TRAINING MONTHLY MODEL ")
    print("==============================")
    train_monthly_model(df)

    print("\n✅ All models trained and saved successfully!")

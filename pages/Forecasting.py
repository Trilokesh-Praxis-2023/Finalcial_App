import os
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from utils.github_storage import read_csv


# -----------------------------------------------------------
# LOAD CSS
# -----------------------------------------------------------
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.set_page_config(layout="wide")
st.title("AI Forecasting Intelligence")


# -----------------------------------------------------------
# FORECAST HELPERS
# -----------------------------------------------------------
def mean_absolute_error(actual, predicted):
    actual_arr = np.asarray(actual, dtype=float)
    pred_arr = np.asarray(predicted, dtype=float)
    return float(np.mean(np.abs(actual_arr - pred_arr)))


def seasonal_naive_forecast(train_series, horizon, season_length):
    values = train_series.to_numpy(dtype=float)

    if len(values) == 0:
        return np.zeros(horizon)

    if len(values) < season_length:
        return np.full(horizon, values[-1])

    last_season = values[-season_length:]
    repeats = int(np.ceil(horizon / season_length))
    return np.tile(last_season, repeats)[:horizon]


def holt_winters_forecast(train_series, horizon, season_length):
    series = train_series.astype(float)

    trend = "add" if len(series) >= 4 else None
    use_seasonality = len(series) >= season_length * 2
    seasonal = "add" if use_seasonality else None

    model_kwargs = {
        "trend": trend,
        "seasonal": seasonal,
        "initialization_method": "estimated",
    }

    if trend:
        model_kwargs["damped_trend"] = True

    if seasonal:
        model_kwargs["seasonal_periods"] = season_length

    try:
        model = ExponentialSmoothing(series, **model_kwargs)
        fitted = model.fit(optimized=True, use_brute=True)
        forecast = np.asarray(fitted.forecast(horizon), dtype=float)
        return np.clip(forecast, 0, None)
    except Exception:
        # Safe fallback for unstable edge-cases
        return np.clip(np.full(horizon, series.iloc[-1]), 0, None)


def build_future_index(last_timestamp, periods, freq):
    offset = pd.tseries.frequencies.to_offset(freq)
    start = last_timestamp + offset
    return pd.date_range(start=start, periods=periods, freq=freq)


def select_best_forecast_model(series, horizon, season_length, freq):
    holdout = min(max(2, horizon // 2), max(2, len(series) // 5))

    if len(series) <= holdout + 2:
        final_pred = seasonal_naive_forecast(series, horizon, season_length)
        future_idx = build_future_index(series.index.max(), horizon, freq)
        forecast = pd.Series(np.clip(final_pred, 0, None), index=future_idx, name="forecast")

        return {
            "model": "Seasonal Naive",
            "holdout": 0,
            "mae": None,
            "baseline_mae": None,
            "forecast": forecast,
        }

    train = series.iloc[:-holdout]
    test = series.iloc[-holdout:]

    naive_pred = seasonal_naive_forecast(train, holdout, season_length)
    naive_mae = mean_absolute_error(test, naive_pred)

    hw_pred = holt_winters_forecast(train, holdout, season_length)
    hw_mae = mean_absolute_error(test, hw_pred)

    if hw_mae <= naive_mae:
        selected = "Holt-Winters"
        selected_mae = hw_mae
        final_pred = holt_winters_forecast(series, horizon, season_length)
    else:
        selected = "Seasonal Naive"
        selected_mae = naive_mae
        final_pred = seasonal_naive_forecast(series, horizon, season_length)

    future_idx = build_future_index(series.index.max(), horizon, freq)
    forecast = pd.Series(np.clip(final_pred, 0, None), index=future_idx, name="forecast")

    return {
        "model": selected,
        "holdout": holdout,
        "mae": selected_mae,
        "baseline_mae": naive_mae,
        "forecast": forecast,
    }


def render_forecast_chart(history_series, forecast_series, x_title):
    hist_df = history_series.reset_index()
    hist_df.columns = ["Date", "Actual"]

    fc_df = forecast_series.reset_index()
    fc_df.columns = ["Date", "Forecast"]

    history_line = (
        alt.Chart(hist_df)
        .mark_line(color="#4FC3F7", point=True)
        .encode(
            x=alt.X("Date:T", title=x_title),
            y=alt.Y("Actual:Q", title="Amount"),
            tooltip=["Date:T", alt.Tooltip("Actual:Q", format=",.2f")],
        )
    )

    forecast_line = (
        alt.Chart(fc_df)
        .mark_line(color="#FFC107", point=True, strokeDash=[5, 4])
        .encode(
            x=alt.X("Date:T", title=x_title),
            y=alt.Y("Forecast:Q", title="Amount"),
            tooltip=["Date:T", alt.Tooltip("Forecast:Q", format=",.2f")],
        )
    )

    st.altair_chart(history_line + forecast_line, use_container_width=True)


# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
df = read_csv()
df["period"] = pd.to_datetime(df["period"])
df = df.sort_values("period", ascending=False).reset_index(drop=True)


# -----------------------------------------------------------
# COMPACT SIDEBAR FILTERS
# -----------------------------------------------------------
st.sidebar.markdown("### Filters")

c1, c2 = st.sidebar.columns(2)

with c1:
    f_year = st.multiselect("Year", sorted(df.year.unique()))
    f_acc = st.multiselect("Account", sorted(df.accounts.unique()))

with c2:
    f_month = st.multiselect("Month", sorted(df.year_month.unique()))
    f_cat = st.multiselect("Category", sorted(df.category.unique()))


# -----------------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------------
filtered = df.copy()

if f_year:
    filtered = filtered[filtered.year.isin(f_year)]
if f_month:
    filtered = filtered[filtered.year_month.isin(f_month)]
if f_cat:
    filtered = filtered[filtered.category.isin(f_cat)]
if f_acc:
    filtered = filtered[filtered.accounts.isin(f_acc)]

if filtered.empty:
    st.warning("No data available for forecasting.")
    st.stop()


# -----------------------------------------------------------
# CONTEXT KPIS (WHAT MODEL SEES)
# -----------------------------------------------------------
st.markdown("### Dataset Snapshot (Model Input View)")

total = filtered["amount"].sum()
months = filtered["year_month"].nunique()
avg_month = total / months if months else 0
txns = len(filtered)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Spend", f"INR {total:,.0f}")
k2.metric("Months Covered", months)
k3.metric("Avg / Month", f"INR {avg_month:,.0f}")
k4.metric("Transactions", txns)


# -----------------------------------------------------------
# RECENT TREND (LAST 6 MONTHS)
# -----------------------------------------------------------
st.markdown("### Recent Spending Trend")

monthly_preview = (
    filtered.groupby("year_month")["amount"]
    .sum()
    .sort_index()
)

st.line_chart(monthly_preview.tail(6))


# -----------------------------------------------------------
# CATEGORY CONTRIBUTION
# -----------------------------------------------------------
st.markdown("### Category Contribution")

cat_share = (
    filtered.groupby("category")["amount"]
    .sum()
    .sort_values(ascending=False)
)

st.bar_chart(cat_share)


# -----------------------------------------------------------
# FORECAST SECTION
# -----------------------------------------------------------
st.markdown("### Forecast Prediction (Backtested Time-Series)")
st.caption(
    "Model selection compares Seasonal Naive vs Holt-Winters on a holdout window and keeps the lower MAE model."
)

fc1, fc2 = st.columns(2)
with fc1:
    daily_horizon = st.slider("Daily Forecast Horizon", min_value=7, max_value=90, value=30, step=1)
with fc2:
    monthly_horizon = st.slider("Monthly Forecast Horizon", min_value=3, max_value=12, value=6, step=1)

# Daily series with full day continuity (missing days treated as zero spend)
daily_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="D"))
    .sum()
    .asfreq("D", fill_value=0.0)
)

# Monthly series with continuous monthly index
monthly_series = (
    filtered.set_index("period")
    .sort_index()["amount"]
    .groupby(pd.Grouper(freq="MS"))
    .sum()
    .asfreq("MS", fill_value=0.0)
)

st.markdown("#### Daily Forecast")
if len(daily_series) < 14:
    st.warning("Need at least 14 days of data for a reliable daily forecast.")
else:
    daily_result = select_best_forecast_model(
        series=daily_series,
        horizon=daily_horizon,
        season_length=7,
        freq="D",
    )

    d1, d2, d3 = st.columns(3)
    d1.metric("Selected Model", daily_result["model"])
    d2.metric(
        "Holdout MAE",
        "N/A" if daily_result["mae"] is None else f"INR {daily_result['mae']:,.0f}",
    )
    d3.metric("Validation Window", "N/A" if daily_result["holdout"] == 0 else f"{daily_result['holdout']} days")

    render_forecast_chart(daily_series, daily_result["forecast"], "Date")

    daily_table = daily_result["forecast"].reset_index()
    daily_table.columns = ["Date", "Forecast"]
    daily_table["Forecast"] = daily_table["Forecast"].round(2)
    st.dataframe(daily_table, use_container_width=True)

st.markdown("#### Monthly Forecast")
if len(monthly_series) < 6:
    st.warning("Need at least 6 months of data for a reliable monthly forecast.")
else:
    monthly_result = select_best_forecast_model(
        series=monthly_series,
        horizon=monthly_horizon,
        season_length=12,
        freq="MS",
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Selected Model", monthly_result["model"])
    m2.metric(
        "Holdout MAE",
        "N/A" if monthly_result["mae"] is None else f"INR {monthly_result['mae']:,.0f}",
    )
    m3.metric(
        "Validation Window",
        "N/A" if monthly_result["holdout"] == 0 else f"{monthly_result['holdout']} months",
    )

    render_forecast_chart(monthly_series, monthly_result["forecast"], "Month")

    monthly_table = monthly_result["forecast"].reset_index()
    monthly_table.columns = ["Month", "Forecast"]
    monthly_table["Forecast"] = monthly_table["Forecast"].round(2)
    st.dataframe(monthly_table, use_container_width=True)

    projected_total = monthly_result["forecast"].sum()
    st.metric("Projected Spend (Forecast Window)", f"INR {projected_total:,.0f}")

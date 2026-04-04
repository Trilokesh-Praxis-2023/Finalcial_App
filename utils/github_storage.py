import base64
import requests
import pandas as pd
from io import StringIO
import streamlit as st

# -----------------------------------------------------------
# CONFIG FROM STREAMLIT SECRETS
# -----------------------------------------------------------
GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]

OWNER = "Trilokesh-Praxis-2023"
REPO = "Finalcial_App"
BRANCH = "main"
FILE_PATH = "finance_data.csv"

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",  # IMPORTANT: token not Bearer
    "Accept": "application/vnd.github+json",
}

RECURRING_BIKE_EMI_START = pd.Timestamp(2026, 4, 1)
RECURRING_BIKE_EMI_AMOUNT = 5333.0
RECURRING_BIKE_EMI_CATEGORY = "bike_emi"
RECURRING_BIKE_EMI_ACCOUNT = "Auto Debit"


def apply_recurring_transactions(df: pd.DataFrame) -> pd.DataFrame:
    updated = df.copy()
    updated["period"] = pd.to_datetime(updated["period"], errors="coerce")

    today = pd.Timestamp.today().normalize()
    current_month_start = today.replace(day=1)

    if current_month_start < RECURRING_BIKE_EMI_START:
        return updated

    recurring_months = pd.date_range(
        start=RECURRING_BIKE_EMI_START,
        end=current_month_start,
        freq="MS",
    )

    existing_periods = set(
        updated.loc[
            updated["category"].astype(str).str.lower() == RECURRING_BIKE_EMI_CATEGORY,
            "period",
        ]
        .dropna()
        .dt.normalize()
    )

    last_running_total = (
        pd.to_numeric(updated["running_total"], errors="coerce").max()
        if "running_total" in updated.columns and not updated.empty
        else 0.0
    )
    last_running_total = 0.0 if pd.isna(last_running_total) else float(last_running_total)

    missing_rows = []
    for period in recurring_months:
        normalized_period = period.normalize()
        if normalized_period in existing_periods:
            continue

        missing_rows.append(
            {
                "period": normalized_period,
                "accounts": RECURRING_BIKE_EMI_ACCOUNT,
                "category": RECURRING_BIKE_EMI_CATEGORY,
                "amount": RECURRING_BIKE_EMI_AMOUNT,
                "month": normalized_period.strftime("%B"),
                "running_total": last_running_total,
                "year": normalized_period.year,
                "year_month": str(normalized_period.to_period("M")),
            }
        )

    if missing_rows:
        updated = pd.concat([updated, pd.DataFrame(missing_rows)], ignore_index=True)

    return updated


# -----------------------------------------------------------
# READ CSV
# -----------------------------------------------------------
def read_csv():
    r = requests.get(BASE_URL, headers=HEADERS)

    if r.status_code != 200:
        raise Exception(f"GitHub Read Failed: {r.status_code} - {r.text}")

    content = r.json()["content"]
    decoded = base64.b64decode(content).decode("utf-8")

    df = pd.read_csv(StringIO(decoded))
    df = apply_recurring_transactions(df)

    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)

    return df


# -----------------------------------------------------------
# WRITE CSV
# -----------------------------------------------------------
def write_csv(df, message="update csv"):
    # 1️⃣ Get latest SHA
    r = requests.get(BASE_URL, headers=HEADERS)

    if r.status_code != 200:
        raise Exception(f"GitHub SHA Fetch Failed: {r.status_code} - {r.text}")

    sha = r.json()["sha"]

    # 2️⃣ Convert DF to base64
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    encoded = base64.b64encode(csv_buffer.getvalue().encode()).decode()

    payload = {
        "message": message,
        "content": encoded,
        "sha": sha,
        "branch": BRANCH,
    }

    r = requests.put(BASE_URL, headers=HEADERS, json=payload)

    if r.status_code not in [200, 201]:
        raise Exception(f"GitHub Write Failed: {r.status_code} - {r.text}")

    return True

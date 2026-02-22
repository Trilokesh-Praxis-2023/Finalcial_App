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
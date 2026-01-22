import os
import base64
import requests
import pandas as pd
from io import StringIO
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

OWNER = "Trilokesh-Praxis-2023"
REPO = "Finalcial_App"
BRANCH = "main"
FILE_PATH = "finance_data.csv"

BASE_URL = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{FILE_PATH}"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}


# -----------------------------------------------------------
# READ CSV
# -----------------------------------------------------------
def read_csv():
    r = requests.get(BASE_URL, headers=HEADERS)
    r.raise_for_status()

    content = r.json()["content"]
    decoded = base64.b64decode(content).decode("utf-8")

    df = pd.read_csv(StringIO(decoded))
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)

    return df


# -----------------------------------------------------------
# WRITE CSV (THE CORRECT WAY)
# -----------------------------------------------------------
def write_csv(df, message="update csv"):
    # 1. Get current file SHA
    r = requests.get(BASE_URL, headers=HEADERS)
    r.raise_for_status()
    sha = r.json()["sha"]

    # 2. Convert df to base64
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    encoded = base64.b64encode(csv_buffer.getvalue().encode()).decode()

    # 3. PUT file
    payload = {
        "message": message,
        "content": encoded,
        "sha": sha,
        "branch": BRANCH,
    }

    r = requests.put(BASE_URL, headers=HEADERS, json=payload)
    r.raise_for_status()

    return True

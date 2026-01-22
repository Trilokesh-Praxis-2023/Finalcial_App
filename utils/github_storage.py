# utils/github_storage.py

import os
import base64
import requests
import pandas as pd
from io import StringIO

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
REPO = "Trilokesh-Praxis-2023/Finalcial_App"
FILE_PATH = "finance_data.csv"

API_URL = f"https://api.github.com/repos/{REPO}/contents/{FILE_PATH}"

HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}


def _get_sha_and_content():
    r = requests.get(API_URL, headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    sha = data["sha"]
    content = base64.b64decode(data["content"]).decode("utf-8")
    return sha, content


def read_csv():
    _, content = _get_sha_and_content()
    df = pd.read_csv(StringIO(content))
    df.columns = df.columns.str.lower()
    df["period"] = pd.to_datetime(df["period"])
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)
    df["amount"] = df.amount.astype(float)
    return df


def write_csv(df, msg="update finance data"):
    sha, _ = _get_sha_and_content()
    buffer = StringIO()
    df.to_csv(buffer, index=False)

    encoded = base64.b64encode(buffer.getvalue().encode()).decode()

    payload = {
        "message": msg,
        "content": encoded,
        "sha": sha
    }

    r = requests.put(API_URL, headers=HEADERS, json=payload)
    r.raise_for_status()

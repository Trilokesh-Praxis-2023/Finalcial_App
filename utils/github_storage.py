import os
import base64
import requests
import pandas as pd
from io import StringIO

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

OWNER = "Trilokesh-Praxis-2023"
REPO = "Finalcial_App"
BRANCH = "main"
FILE_PATH = "finance_data.csv"

HEADERS = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28"
}

BASE = f"https://api.github.com/repos/{OWNER}/{REPO}"


# -----------------------------------------------------------
# READ CSV
# -----------------------------------------------------------
def read_csv():
    url = f"{BASE}/contents/{FILE_PATH}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()

    content = base64.b64decode(r.json()["content"]).decode()
    df = pd.read_csv(StringIO(content))

    df.columns = df.columns.str.lower()
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    df["year"] = df.period.dt.year
    df["year_month"] = df.period.dt.to_period("M").astype(str)
    df["amount"] = df.amount.astype(float)

    return df


# -----------------------------------------------------------
# WRITE CSV USING GIT BLOBS (NO SIZE LIMIT)
# -----------------------------------------------------------
def write_csv(df, message="update finance data"):
    # 1. Get latest commit SHA
    ref_url = f"{BASE}/git/ref/heads/{BRANCH}"
    ref = requests.get(ref_url, headers=HEADERS).json()
    latest_commit_sha = ref["object"]["sha"]

    # 2. Get tree SHA
    commit_url = f"{BASE}/git/commits/{latest_commit_sha}"
    commit = requests.get(commit_url, headers=HEADERS).json()
    base_tree_sha = commit["tree"]["sha"]

    # 3. Create blob from CSV content
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)

    blob_url = f"{BASE}/git/blobs"
    blob = requests.post(
        blob_url,
        headers=HEADERS,
        json={
            "content": csv_buffer.getvalue(),
            "encoding": "utf-8"
        }
    ).json()
    blob_sha = blob["sha"]

    # 4. Create new tree
    tree_url = f"{BASE}/git/trees"
    tree = requests.post(
        tree_url,
        headers=HEADERS,
        json={
            "base_tree": base_tree_sha,
            "tree": [
                {
                    "path": FILE_PATH,
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob_sha
                }
            ]
        }
    ).json()
    new_tree_sha = tree["sha"]

    # 5. Create commit
    commit_url = f"{BASE}/git/commits"
    new_commit = requests.post(
        commit_url,
        headers=HEADERS,
        json={
            "message": message,
            "tree": new_tree_sha,
            "parents": [latest_commit_sha]
        }
    ).json()
    new_commit_sha = new_commit["sha"]

    # 6. Update branch ref to new commit
    requests.patch(
        ref_url,
        headers=HEADERS,
        json={"sha": new_commit_sha}
    )

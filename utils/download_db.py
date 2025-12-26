import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# ============================================================
# 🔐 Load Environment Variables
# ============================================================
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# ============================================================
# 🗄 Table to Download
# ============================================================
TABLE_NAME = "finance_data"

# ============================================================
# 📂 Output File Path
# ============================================================
OUTPUT_FILE = r"C:\Users\trilo\OneDrive\Desktop\finance_data_export.xlsx"

# ============================================================
# ⬇ Download DB → Excel
# ============================================================
def download_db_to_excel():
    engine = create_engine(DATABASE_URL)

    query = f"SELECT * FROM {TABLE_NAME}"
    df = pd.read_sql(query, engine)

    df.to_excel(OUTPUT_FILE, index=False)

    print(f"✅ {len(df)} rows downloaded to:\n{OUTPUT_FILE}")

# ============================================================
# ▶ Run
# ============================================================
if __name__ == "__main__":
    download_db_to_excel()

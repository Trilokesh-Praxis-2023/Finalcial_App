import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# Load environment vars
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL") or (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
)

# Connect
engine = create_engine(DATABASE_URL)

# Query full dataset
df = pd.read_sql("SELECT * FROM finance_data ORDER BY period DESC", engine)

# Export to CSV
csv_path = "finance_data_full.csv"
df.to_csv(csv_path, index=False)

# Export to Excel
excel_path = "finance_data_full.xlsx"
df.to_excel(excel_path, index=False)

print("✅ Export complete!")
print("CSV saved as:", csv_path)
print("Excel saved as:", excel_path)

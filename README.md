readme_text = """
# 💰 Personal Finance Tracker (Streamlit + PostgreSQL)

A lightweight finance management & expense visualization app that helps track spending, monitor budgets,
forecast future expenses using AI, and export reports — all in one dashboard.

---

## 🚀 Features

| Feature | Status |
|---|---|
| Add & store expenses in PostgreSQL | ✅ |
| Password-protected input access | 🔐 |
| Interactive dashboard with monthly filters | 📊 |
| Export expenses (CSV + Excel) | 📥 |
| Category-wise spend analysis | 🏷 |
| Account usage breakdown | 👥 |
| Monthly spend trends + MoM (%) | 📈 |
| Budget tracking alerts (₹18,000 default) | ⚠ |
| AI-based expense forecasting (Prophet) | 🔮 |

---

## 📦 Tech Stack

| Component | Used |
|---|---|
| Backend | Python, Streamlit |
| Database | PostgreSQL + SQLAlchemy |
| Visualization | Altair Charts |
| AI Forecasting | Prophet |
| Secrets Management | .env |

---

## 📂 Project Structure

finance-tracker/
│── app.py
│── requirements.txt
│── .env
│── .gitignore
│── README.md

---

## 🔐 .env Configuration

DB_USER=postgres
DB_PASSWORD=your_pg_password
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finance_db
APP_PASSWORD=your_secret_password_here

---

## ▶️ Run the App

pip install -r requirements.txt
streamlit run app.py

Visit: http://localhost:8501

---

## 🔮 Forecasting (Prophet Model)

Predicts next 6 months of expenses & warns when you may exceed budget.

pip install prophet

---

## 📥 Export

- Download CSV / Excel
- Great for yearly tax & monthly report tracking.

---

## 🛡 Security

✔ Password protected entry  
✔ Credentials stored in .env  
✔ .gitignore prevents leaks

---

## 🔥 Next Upgrades

| Feature | Can be added |
|---|---|
| AI category forecast | Future spend per category |
| PDF + Email Monthly Report | Export & send automatically |
| WhatsApp Alerts | Budget breach notification |
| Multi-user login | Role-based dashboards |

"""

st.text_area("📄 Project Documentation (README.md)", readme_text, height=600)

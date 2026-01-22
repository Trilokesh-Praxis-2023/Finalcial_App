import streamlit as st
import pandas as pd
from utils.github_storage import read_csv
import os

# Load global CSS
css_path = ".streamlit/styles.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.set_page_config(layout="wide")
st.title("🧠 Spending Behavior Analysis")

df = read_csv()
df["period"] = pd.to_datetime(df["period"])

# ---------------- Most Frequent Category ----------------
st.subheader("🔁 Most Frequent Category")
freq_cat = df["category"].value_counts().idxmax()
st.success(f"You spend most frequently on **{freq_cat}**")

# ---------------- Most Used Account ----------------
st.subheader("🏦 Most Used Account")
freq_acc = df["accounts"].value_counts().idxmax()
st.info(f"Most used payment method: **{freq_acc}**")

# ---------------- Spend Spikes ----------------
st.subheader("⚡ Spend Spike Detection")

daily = df.groupby("period")["amount"].sum().reset_index()
threshold = daily["amount"].mean() + 2 * daily["amount"].std()

spikes = daily[daily["amount"] > threshold]

if not spikes.empty:
    st.write("High spending days detected:")
    st.dataframe(spikes)
else:
    st.write("No abnormal spikes detected.")

# ---------------- Low Spend Streak ----------------
st.subheader("🧘 Low Spend Days")

low_days = daily[daily["amount"] < daily["amount"].mean() * 0.5]
st.write(f"{len(low_days)} low-spend days identified.")

# ---------------- Day Pattern ----------------
st.subheader("📆 Day-of-Week Spending Pattern")

df["dow"] = df["period"].dt.day_name()
dow_spend = df.groupby("dow")["amount"].mean()
st.bar_chart(dow_spend)

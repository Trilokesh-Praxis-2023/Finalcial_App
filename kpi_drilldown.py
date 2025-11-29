import streamlit as st
import pandas as pd
import altair as alt

# ===============================================================
# 🔥 MAIN KPI RENDER FUNCTION (CALL FROM app.py)
# ===============================================================
def render_kpi_suite(filtered, get_income):

    source = filtered.copy()

    # ====== MONTHLY CALCULATIONS ======
    monthly = source.groupby("year_month")["amount"].sum().reset_index()
    monthly["year_month"] = pd.to_datetime(monthly["year_month"])
    monthly = monthly.sort_values("year_month")

    monthly["income"]  = [get_income(m) for m in monthly["year_month"]]
    monthly["savings"] = monthly["income"] - monthly["amount"]


    st.subheader("📊 KPI Drilldown Suite — Complete Analytics")


    # =========================================================
    # 1️⃣ Monthly Spend Trend — VALUES SHOWN
    # =========================================================
    with st.expander("💸 Monthly Spend Trend — Value Displayed"):

        base = alt.Chart(monthly).encode(x="year_month:T", y="amount:Q")

        chart = (
            base.mark_line(point=True, color="#29B6F6")
            + base.mark_text(dy=-12, fontSize=12, color="yellow", fontWeight="bold")
              .encode(text="amount:Q")
        )

        st.altair_chart(chart, use_container_width=True)


    # =========================================================
    # 2️⃣ Month-on-Month Compare — BAR + LABELS
    # =========================================================
    with st.expander("📆 Month-on-Month Spend — Bar Chart"):

        base = alt.Chart(monthly).encode(x="year_month:T", y="amount:Q")

        chart = (
            base.mark_bar(size=38, color="#26D67D")
            + base.mark_text(dy=-10, fontSize=12, fontWeight="bold", color="white")
              .encode(text="amount:Q")
        )

        st.altair_chart(chart, use_container_width=True)


    # =========================================================
    # 3️⃣ Rolling 3-Month Trend (Smoothing)
    # =========================================================
    with st.expander("📅 Rolling 3-Month Spend Trend"):

        monthly["roll"] = monthly["amount"].rolling(3).mean()

        st.altair_chart(
            alt.Chart(monthly).mark_line(point=True, color="#FFC107")
            .encode(x="year_month:T", y="roll:Q"),
            use_container_width=True
        )


    # =========================================================
    # 4️⃣ Category Trend — LOG + LABELS
    # =========================================================
    with st.expander("🏷 Category Trend Timeline (Log Scale)"):

        cat = source.groupby(["year_month","category"])["amount"].sum().reset_index()
        cat["year_month"] = pd.to_datetime(cat["year_month"])

        line = alt.Chart(cat).mark_line(point=True).encode(
            x="year_month:T",
            y=alt.Y("amount:Q", scale=alt.Scale(type="log")),
            color="category:N"
        )

        labels = alt.Chart(cat).mark_text(
            dy=-10, fontSize=10
        ).encode(
            x="year_month:T", y="amount:Q", text="amount:Q", color="category:N"
        )

        st.altair_chart(line + labels, use_container_width=True)


    # =========================================================
    # 5️⃣ Income vs Expense vs Savings
    # =========================================================
    with st.expander("💰 Income vs Expense vs Savings (Multi-line)"):

        melt = monthly.melt("year_month", value_vars=["amount","income","savings"])

        line = alt.Chart(melt).mark_line(point=True).encode(
            x="year_month:T", y="value:Q", color="variable:N"
        )

        labels = alt.Chart(melt).mark_text(dy=-10,fontSize=10).encode(
            x="year_month:T", y="value:Q", text="value:Q", color="variable:N"
        )

        st.altair_chart(line + labels, use_container_width=True)


    # =========================================================
    # 6️⃣ Savings Trend
    # =========================================================
    with st.expander("🧾 Net Savings Monthly Trend"):

        area = alt.Chart(monthly).mark_area(color="#00C853", opacity=0.5).encode(
            x="year_month:T", y="savings:Q"
        )

        labels = alt.Chart(monthly).mark_text(
            dy=-10,fontSize=11,color="white"
        ).encode(x="year_month:T", y="savings:Q", text="savings:Q")

        st.altair_chart(area + labels, use_container_width=True)


    # =========================================================
    # 7️⃣ Category Share % Chart
    # =========================================================
    with st.expander("📊 Category Spend Share %"):

        share = source.groupby("category")["amount"].sum().reset_index()
        share["percent"] = (share["amount"]/share["amount"].sum()*100).round(1)

        bars = alt.Chart(share).mark_bar(color="#FFCA28").encode(
            x="category:N", y="percent:Q"
        )

        labels = alt.Chart(share).mark_text(
            dy=-8,fontSize=11,fontWeight="bold"
        ).encode(x="category:N", y="percent:Q", text="percent:Q")

        st.altair_chart(bars + labels, use_container_width=True)


    # =========================================================
    # 8️⃣ Best vs Worst Month Summary
    # =========================================================
    with st.expander("🏆 Best vs Worst Month Summary"):

        best = monthly.loc[monthly["amount"].idxmax()]
        worst = monthly.loc[monthly["amount"].idxmin()]

        st.success(f"🥇 Best → {best.year_month:%b %Y} | ₹{best.amount:,.0f}")
        st.error  (f"🥀 Worst → {worst.year_month:%b %Y} | ₹{worst.amount:,.0f}")


    # =========================================================
    # 9️⃣ Stability / Volatility Score
    # =========================================================
    with st.expander("🌡 Expense Stability Score"):

        if len(monthly)>2:
            vol = monthly["amount"].pct_change().abs().mean()*100
            stability = max(0,100-vol)
            st.metric("Stability Score", f"{stability:.1f}%")
            st.caption("Higher = more consistent spending")
        else:
            st.info("Need at least 3 months of data.")


    # =========================================================
    # 🔟 Survival Duration if Income Stops
    # =========================================================
    with st.expander("🛡 Survival Duration Estimate"):

        burn = monthly["amount"].mean()
        surplus = monthly["income"].sum() - monthly["amount"].sum()

        if burn > 0:
            st.metric("You Can Survive", f"{surplus/burn:.1f} months")
        else:
            st.info("Not enough data to estimate.")


    # =========================================================
    # EXTRA INSIGHTS
    # =========================================================

    # Cumulative Spending Curve
    with st.expander("📈 Cumulative Spending Curve"):
        monthly["cumulative"] = monthly["amount"].cumsum()
        st.altair_chart(
            alt.Chart(monthly).mark_line(point=True).encode(
                x="year_month:T", y="cumulative:Q"
            ),
            use_container_width=True
        )

    # Outlier Detection Boxplot
    with st.expander("📦 Outlier Distribution by Category"):
        st.altair_chart(
            alt.Chart(source).mark_boxplot(color="#8E44AD").encode(
                x="category:N", y="amount:Q"
            ),
            use_container_width=True
        )

    # Seasonal Pattern
    with st.expander("🌦 Seasonal Monthly Spend Pattern"):
        season = source.copy()
        season["m"] = season["period"].dt.month
        month_sum = season.groupby("m")["amount"].sum().reset_index()

        bars = alt.Chart(month_sum).mark_bar(color="#5DADE2").encode(x="m:N", y="amount:Q")
        labels = alt.Chart(month_sum).mark_text(dy=-10,fontSize=11).encode(x="m:N", y="amount:Q", text="amount:Q")

        st.altair_chart(bars+labels, use_container_width=True)


# ====================== END OF KPI MODULE ======================

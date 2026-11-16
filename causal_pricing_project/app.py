import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from causal_engine import (
    generate_sample_data, run_synthetic_control, estimate_dml,
    build_summary, save_dataset
)

st.set_page_config(page_title="Causal Pricing Impact", layout="wide")
st.title("Causal Inference & Synthetic Control Engine")
st.caption("Educational demonstration using synthetic regional-market data. Results are illustrative, not business estimates.")

with st.sidebar:
    st.header("Experiment settings")
    n_markets = st.slider("Number of markets", 6, 20, 10)
    n_weeks = st.slider("Weeks of data", 20, 80, 40)
    intervention_week = st.slider("Intervention week", 8, n_weeks - 5, min(25, n_weeks - 5))
    treated_market = st.selectbox("Treated market", [f"Market_{i}" for i in range(1, n_markets + 1)], index=0)
    price_change = st.slider("Pricing-change effect (simulated)", -20, 40, 12)
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42)

if st.button("Generate data and run analysis", type="primary"):
    df = generate_sample_data(
        n_markets=n_markets, n_weeks=n_weeks,
        intervention_week=intervention_week, treated_market=treated_market,
        treatment_effect=float(price_change), random_state=int(seed)
    )
    save_dataset(df, "data/market_transactions.csv")
    st.session_state["df"] = df
    st.session_state["settings"] = {
        "intervention_week": intervention_week,
        "treated_market": treated_market,
        "price_change": price_change
    }

if "df" not in st.session_state:

    st.stop()

df = st.session_state["df"]
settings = st.session_state["settings"]
sc = run_synthetic_control(df, settings["treated_market"], settings["intervention_week"])
dml = estimate_dml(df, settings["treated_market"], settings["intervention_week"])
summary = build_summary(sc, dml, settings["intervention_week"])

c1, c2, c3, c4 = st.columns(4)
c1.metric("Synthetic-control lift", f"{sc['average_post_effect']:.2f} revenue units/week")
c2.metric("Cumulative post-period lift", f"{sc['cumulative_effect']:.2f}")
c3.metric("Pre-period MSPE", f"{sc['pre_mspe']:.2f}")
c4.metric("DML treatment effect", f"{dml['ate']:.2f} revenue units/market-week")

st.subheader("Actual vs. synthetic counterfactual")
fig, ax = plt.subplots(figsize=(10, 4.5))
ax.plot(sc["weeks"], sc["actual"], label="Treated market actual", linewidth=2)
ax.plot(sc["weeks"], sc["synthetic"], label="Synthetic counterfactual", linewidth=2)
ax.axvline(settings["intervention_week"], linestyle="--", label="Intervention")
ax.set_xlabel("Week")
ax.set_ylabel("Revenue (simulated units)")
ax.legend()
ax.grid(alpha=0.25)
st.pyplot(fig)

st.subheader("Estimated weekly treatment effect")
effect_df = pd.DataFrame({
    "week": sc["weeks"],
    "actual_revenue": sc["actual"],
    "synthetic_revenue": sc["synthetic"],
    "estimated_effect": sc["effects"],
})
st.dataframe(effect_df, use_container_width=True, hide_index=True)

st.subheader("Double Machine Learning (DML)")
st.write(
    "DML estimates the average treatment effect after cross-fitting models for the outcome "
    "and treatment assignment. Here the treatment is treated-market × post-intervention."
)
st.json({
    "ATE": round(dml["ate"], 3),
    "95% confidence interval (approx.)": [
        round(dml["ci_lower"], 3), round(dml["ci_upper"], 3)
    ],
    "observations": dml["n"],
    "interpretation": dml["interpretation"]
})

st.subheader("Plain-language summary")
for line in summary:
    st.write(f"- {line}")

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("Download generated dataset (CSV)", data=csv,
                   file_name="market_transactions.csv", mime="text/csv")
st.download_button("Download synthetic-control results (CSV)",
                   data=effect_df.to_csv(index=False).encode("utf-8"),
                   file_name="synthetic_control_results.csv", mime="text/csv")

with st.expander("Data preview"):
    st.dataframe(df.head(20), use_container_width=True, hide_index=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from causal_engine import (
    generate_sample_data,
    run_synthetic_control,
    estimate_dml,
    build_summary,
    save_dataset,
)

# ---------------------------------------------------------
# Page setup and visual theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Causal Pricing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.7rem; padding-bottom: 2.5rem;}
    .hero {
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 16px;
        background: linear-gradient(115deg, rgba(70,110,180,.13), rgba(70,170,150,.08));
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; padding: 0; font-size: 2rem;}
    .hero p {margin: .45rem 0 0 0; opacity: .78;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px;
        padding: 12px 14px;
        background: rgba(128,128,128,.055);
    }
    div[data-testid="stTabs"] button {font-weight: 600;}
    .small-note {font-size: .88rem; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>📊 Causal Pricing Analytics</h1>
      <p>Explore market performance, estimate intervention effects, and inspect the evidence behind the results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.info(
    "This educational dashboard uses simulated market data. Estimates are illustrative, "
    "not evidence of the impact of a real pricing strategy."
)

# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Experiment setup")
    st.caption("Configure a simulated pricing intervention.")

    n_markets = st.slider("Number of markets", 6, 20, 10)
    n_weeks = st.slider("Weeks of data", 20, 80, 40)
    intervention_week = st.slider(
        "Intervention starts at week",
        8,
        n_weeks - 5,
        min(25, n_weeks - 5),
    )
    treated_market = st.selectbox(
        "Treated market",
        [f"Market_{i}" for i in range(1, n_markets + 1)],
    )
    treatment_effect = st.slider(
        "Simulated treatment effect",
        -20,
        40,
        12,
        help="The effect deliberately added to treated-market revenue after the intervention.",
    )
    seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1)

    st.divider()
    run_button = st.button(
        "🚀 Generate data & run analysis",
        type="primary",
        use_container_width=True,
    )
    if st.button("Reset dashboard", use_container_width=True):
        for key in ("df", "settings"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------
# Generate data
# ---------------------------------------------------------
if run_button:
    try:
        with st.spinner("Generating the market panel and estimating effects..."):
            new_df = generate_sample_data(
                n_markets=n_markets,
                n_weeks=n_weeks,
                intervention_week=intervention_week,
                treated_market=treated_market,
                treatment_effect=float(treatment_effect),
                random_state=int(seed),
            )
            new_settings = {
                "intervention_week": int(intervention_week),
                "treated_market": treated_market,
                "treatment_effect": float(treatment_effect),
                "n_markets": int(n_markets),
                "n_weeks": int(n_weeks),
                "seed": int(seed),
            }
            st.session_state["df"] = new_df
            st.session_state["settings"] = new_settings
            save_dataset(new_df, "data/market_transactions.csv")
        st.success("New experiment generated and saved to data/market_transactions.csv.")
    except Exception as exc:
        st.error(f"Could not generate the experiment: {exc}")
        st.stop()

if "df" not in st.session_state or "settings" not in st.session_state:
    st.subheader("Welcome")
    st.write(
        "Choose the experiment settings in the sidebar, then generate a dataset "
        "to populate the analytics dashboard."
    )
    st.markdown(
        """
        **Available analysis**
        - Market-level revenue and driver exploration
        - Treated market compared with a synthetic counterfactual
        - Weekly treatment-effect estimates
        - Double Machine Learning estimate and approximate confidence interval
        - Dataset quality checks and downloadable outputs
        """
    )
    st.stop()

df = st.session_state["df"].copy()
settings = st.session_state["settings"]

required_columns = {
    "market", "week", "revenue", "demand_index", "competitor_price",
    "promo_spend", "treated_market", "post", "treatment",
}
missing = required_columns.difference(df.columns)
if missing:
    st.error(f"Dataset is missing required columns: {sorted(missing)}")
    st.stop()

# ---------------------------------------------------------
# Analysis (using the existing project's causal engine)
# ---------------------------------------------------------
try:
    with st.spinner("Running Synthetic Control and Double Machine Learning..."):
        sc = run_synthetic_control(
            df, settings["treated_market"], settings["intervention_week"]
        )
        dml = estimate_dml(
            df, settings["treated_market"], settings["intervention_week"]
        )
        summary = build_summary(sc, dml, settings["intervention_week"])
except Exception as exc:
    st.error(f"Analysis could not be completed: {exc}")
    st.stop()

effect_df = pd.DataFrame(
    {
        "week": sc["weeks"],
        "actual_revenue": sc["actual"],
        "synthetic_revenue": sc["synthetic"],
        "estimated_effect": sc["effects"],
        "period": [
            "Before intervention" if week < settings["intervention_week"]
            else "After intervention"
            for week in sc["weeks"]
        ],
    }
)

# ---------------------------------------------------------
# KPI strip
# ---------------------------------------------------------
total_revenue = df["revenue"].sum()
average_revenue = df["revenue"].mean()
average_post_effect = sc["average_post_effect"]

st.caption(
    f"Current experiment  |  {settings['n_markets']} markets  |  "
    f"{settings['n_weeks']} weeks  |  Treated: {settings['treated_market']}  |  "
    f"Intervention: week {settings['intervention_week']}"
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total simulated revenue", f"{total_revenue:,.1f}")
k2.metric("Average revenue / market-week", f"{average_revenue:,.2f}")
k3.metric("Average post-period gap", f"{average_post_effect:,.2f}")
k4.metric("DML estimated effect", f"{dml['ate']:,.2f}")

with st.expander("How to read these indicators"):
    st.markdown(
        """
        - **Average post-period gap:** average treated-market revenue minus its synthetic comparison after intervention.
        - **DML estimated effect:** estimate returned by the project's cross-fitted Double Machine Learning routine.
        - **Pre-period MSPE:** pre-intervention mean squared prediction error; smaller values indicate a closer pre-period fit.
        - These measures are not interchangeable and should be interpreted with the study design in mind.
        """
    )

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
overview_tab, market_tab, causal_tab, data_tab = st.tabs(
    ["📈 Overview", "🏬 Market explorer", "🧪 Causal analysis", "🗂️ Data & exports"]
)

with overview_tab:
    st.subheader("Treated market: actual vs synthetic counterfactual")
    st.caption(
        "The synthetic counterfactual is constructed from weighted untreated donor markets "
        "using pre-intervention revenue."
    )

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(sc["weeks"], sc["actual"], linewidth=2.5, label="Actual treated market")
    ax.plot(sc["weeks"], sc["synthetic"], linewidth=2.5, label="Synthetic counterfactual")
    ax.axvline(
        settings["intervention_week"],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Intervention begins",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue (simulated units)")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    left, right = st.columns(2)
    with left:
        st.markdown("**Average revenue by market**")
        market_avg = (
            df.groupby("market", as_index=False)["revenue"]
            .mean()
            .sort_values("revenue")
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(market_avg["market"], market_avg["revenue"])
        ax.set_xlabel("Average weekly revenue")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("**Average revenue before and after intervention**")
        period_avg = (
            df.assign(
                period=df["post"].map({0: "Before", 1: "After"}),
                market_group=df["treated_market"].map({0: "Control markets", 1: "Treated market"}),
            )
            .groupby(["period", "market_group"], as_index=False)["revenue"]
            .mean()
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        period_order = ["Before", "After"]
        group_order = ["Control markets", "Treated market"]
        width = 0.35
        x = range(len(period_order))
        for idx, group in enumerate(group_order):
            vals = [
                period_avg.loc[
                    (period_avg["period"] == period)
                    & (period_avg["market_group"] == group),
                    "revenue",
                ].mean()
                for period in period_order
            ]
            ax.bar(
                [pos + (idx - 0.5) * width for pos in x],
                vals,
                width=width,
                label=group,
            )
        ax.set_xticks(list(x), period_order)
        ax.set_ylabel("Average revenue")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with market_tab:
    st.subheader("Explore market performance")
    all_markets = sorted(df["market"].unique().tolist())
    selected_markets = st.multiselect(
        "Markets to compare",
        options=all_markets,
        default=all_markets,
    )
    min_week = int(df["week"].min())
    max_week = int(df["week"].max())
    selected_weeks = st.slider(
        "Week range",
        min_value=min_week,
        max_value=max_week,
        value=(min_week, max_week),
    )

    filtered = df[
        df["market"].isin(selected_markets)
        & df["week"].between(selected_weeks[0], selected_weeks[1])
    ]

    if filtered.empty:
        st.warning("Select at least one market and a valid week range.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Selected observations", f"{len(filtered):,}")
        m2.metric("Selected average revenue", f"{filtered['revenue'].mean():,.2f}")
        m3.metric("Selected average demand", f"{filtered['demand_index'].mean():,.2f}")

        weekly = filtered.groupby(["week", "market"], as_index=False)["revenue"].mean()
        fig, ax = plt.subplots(figsize=(11, 4.5))
        for market in selected_markets:
            part = weekly[weekly["market"] == market]
            if not part.empty:
                ax.plot(part["week"], part["revenue"], linewidth=1.8, label=market)
        ax.set_xlabel("Week")
        ax.set_ylabel("Revenue")
        ax.set_title("Weekly revenue by selected market")
        ax.grid(alpha=0.22)
        ax.legend(ncol=3, fontsize=8)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Demand index vs revenue**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(filtered["demand_index"], filtered["revenue"], alpha=0.55)
            ax.set_xlabel("Demand index")
            ax.set_ylabel("Revenue")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with c2:
            st.markdown("**Competitor price vs revenue**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(filtered["competitor_price"], filtered["revenue"], alpha=0.55)
            ax.set_xlabel("Competitor price")
            ax.set_ylabel("Revenue")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown("**Market summary for current filters**")
        market_summary = (
            filtered.groupby("market", as_index=False)
            .agg(
                average_revenue=("revenue", "mean"),
                average_demand=("demand_index", "mean"),
                average_competitor_price=("competitor_price", "mean"),
                total_promo_spend=("promo_spend", "sum"),
                observations=("revenue", "count"),
            )
            .sort_values("average_revenue", ascending=False)
        )
        st.dataframe(market_summary, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download filtered market data",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_market_transactions.csv",
            mime="text/csv",
        )

with causal_tab:
    st.subheader("1. Synthetic Control")
    st.write(
        "Synthetic Control estimates a counterfactual by combining untreated donor markets. "
        "The donor weights in this project are fitted on pre-intervention revenue."
    )

    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(sc["weeks"], sc["actual"], linewidth=2.4, label="Actual")
    ax.plot(sc["weeks"], sc["synthetic"], linewidth=2.4, label="Synthetic")
    ax.axvline(settings["intervention_week"], color="gray", linestyle="--", label="Intervention")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.legend()
    ax.grid(alpha=0.22)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    s1, s2, s3 = st.columns(3)
    s1.metric("Average post-period effect", f"{sc['average_post_effect']:,.3f}")
    s2.metric("Cumulative post-period effect", f"{sc['cumulative_effect']:,.3f}")
    s3.metric("Pre-period MSPE", f"{sc['pre_mspe']:,.4f}")

    st.markdown("**Weekly treatment-effect gap (actual − synthetic)**")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    bar_colors = [
        "#4C78A8" if week < settings["intervention_week"] else "#F58518"
        for week in effect_df["week"]
    ]
    ax.bar(effect_df["week"], effect_df["estimated_effect"], color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(settings["intervention_week"], color="gray", linestyle="--")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue gap")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Inspect donor-market weights"):
        weights_df = pd.DataFrame(
            list(sc["donor_weights"].items()),
            columns=["donor_market", "weight"],
        ).sort_values("weight", ascending=False)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.caption("Weights are constrained to be non-negative and sum to one.")

    st.divider()
    st.subheader("2. Double Machine Learning")
    d1, d2, d3 = st.columns(3)
    d1.metric("Estimated ATE", f"{dml['ate']:.3f}")
    d2.metric("Approx. 95% CI lower", f"{dml['ci_lower']:.3f}")
    d3.metric("Approx. 95% CI upper", f"{dml['ci_upper']:.3f}")
    st.write(
        "The existing DML routine uses cross-fitted Random Forest models for the outcome "
        "and treatment assignment. Treatment is defined as treated-market × post-intervention."
    )
    st.info(dml["interpretation"])
    st.warning(
        "The confidence interval is approximate. This is an educational implementation "
        "using simulated observational panel data; it is not a substitute for a study-design "
        "review or robust statistical inference."
    )

    st.subheader("3. Plain-language summary")
    for item in summary:
        st.markdown(f"- {item}")

    st.download_button(
        "⬇️ Download weekly causal results",
        data=effect_df.to_csv(index=False).encode("utf-8"),
        file_name="synthetic_control_results.csv",
        mime="text/csv",
    )

with data_tab:
    st.subheader("Dataset overview")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns)}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum())}")
    d.metric("Markets", f"{df['market'].nunique()}")

    st.markdown("**Data quality profile**")
    quality = pd.DataFrame(
        {
            "column": df.columns,
            "data_type": [str(df[col].dtype) for col in df.columns],
            "missing_values": [int(df[col].isna().sum()) for col in df.columns],
            "unique_values": [int(df[col].nunique()) for col in df.columns],
        }
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.markdown("**Dataset preview**")
    st.dataframe(df.head(100), use_container_width=True, hide_index=True)

    st.markdown("**Export project outputs**")
    e1, e2 = st.columns(2)
    with e1:
        st.download_button(
            "⬇️ Download full market dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="market_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with e2:
        st.download_button(
            "⬇️ Download causal results",
            data=effect_df.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_control_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    report_text = f"""# Causal Pricing Analytics — Experiment Summary

- Treated market: {settings['treated_market']}
- Intervention week: {settings['intervention_week']}
- Markets: {settings['n_markets']}
- Weeks: {settings['n_weeks']}
- Simulated treatment-effect setting: {settings['treatment_effect']}
- Total simulated revenue: {total_revenue:.3f}
- Average revenue per market-week: {average_revenue:.3f}
- Synthetic-control average post-period effect: {sc['average_post_effect']:.3f}
- Synthetic-control cumulative post-period effect: {sc['cumulative_effect']:.3f}
- Synthetic-control pre-period MSPE: {sc['pre_mspe']:.5f}
- DML estimated ATE: {dml['ate']:.5f}
- Approximate DML confidence interval: [{dml['ci_lower']:.5f}, {dml['ci_upper']:.5f}]

## Interpretation
This report summarizes a simulated educational experiment. Estimates are illustrative
and should not be interpreted as evidence from a real company or real pricing intervention.
"""
    st.download_button(
        "⬇️ Download experiment summary (Markdown)",
        data=report_text.encode("utf-8"),
        file_name="experiment_summary.md",
        mime="text/markdown",
    )

st.divider()
st.caption(
    "Causal Pricing Analytics | Simulated educational project | "
    "Interpret estimated effects cautiously."
)

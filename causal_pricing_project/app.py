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
# Page configuration and styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Causal Pricing Analytics",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.15rem;
        font-weight: 750;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        color: #718096;
        margin-bottom: 1.2rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 14px 16px;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">📊 Causal Pricing Analytics</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Market performance • Synthetic control • '
    'Double Machine Learning</div>',
    unsafe_allow_html=True,
)
st.info(
    "Educational demonstration using simulated regional-market data. "
    "Results are illustrative and should not be treated as real business estimates."
)

# ---------------------------------------------------------
# Sidebar: experiment configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Experiment settings")
    st.caption("Change the settings, then run the experiment.")

    n_markets = st.slider("Number of markets", 6, 20, 10)
    n_weeks = st.slider("Weeks of data", 20, 80, 40)
    intervention_week = st.slider(
        "First intervention week",
        8,
        n_weeks - 5,
        min(25, n_weeks - 5),
    )
    treated_market = st.selectbox(
        "Treated market",
        [f"Market_{i}" for i in range(1, n_markets + 1)],
        index=0,
    )
    price_change = st.slider(
        "Simulated treatment effect",
        -20,
        40,
        12,
        help="This is the effect deliberately added to the simulated treated market after intervention.",
    )
    seed = st.number_input(
        "Random seed", min_value=0, max_value=100000, value=42, step=1
    )

    run_button = st.button(
        "🚀 Generate data and run analysis",
        type="primary",
        use_container_width=True,
    )

    if st.button("Clear current results", use_container_width=True):
        for key in ("df", "settings"):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------
# Generate data and keep it in session state
# ---------------------------------------------------------
if run_button:
    try:
        with st.spinner("Generating market data and estimating treatment effects..."):
            generated_df = generate_sample_data(
                n_markets=n_markets,
                n_weeks=n_weeks,
                intervention_week=intervention_week,
                treated_market=treated_market,
                treatment_effect=float(price_change),
                random_state=int(seed),
            )
            st.session_state["df"] = generated_df
            st.session_state["settings"] = {
                "intervention_week": int(intervention_week),
                "treated_market": treated_market,
                "price_change": float(price_change),
                "n_markets": int(n_markets),
                "n_weeks": int(n_weeks),
                "seed": int(seed),
            }
            # Save a copy to the project's data folder, as in the original app.
            save_dataset(generated_df, "data/market_transactions.csv")
        st.success("Data generated and saved to data/market_transactions.csv.")
    except Exception as exc:
        st.error(f"Could not generate the experiment: {exc}")
        st.stop()

if "df" not in st.session_state or "settings" not in st.session_state:
    st.title("Welcome")
    st.write(
        "Use the controls in the sidebar and select **Generate data and run analysis** "
        "to create a simulated market experiment."
    )
    st.markdown(
        """
        **This dashboard will show:**
        - Market-level and weekly revenue performance
        - Demand and competitor-price patterns
        - Actual versus synthetic counterfactual revenue
        - Estimated treatment effects from Synthetic Control and DML
        """
    )
    st.stop()

df = st.session_state["df"].copy()
settings = st.session_state["settings"]

# ---------------------------------------------------------
# Validate input and run the project's existing analysis
# ---------------------------------------------------------
required_columns = {
    "market", "week", "revenue", "demand_index", "competitor_price",
    "promo_spend", "treated_market", "post", "treatment",
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(f"The dataset is missing required columns: {sorted(missing_columns)}")
    st.stop()

try:
    with st.spinner("Calculating Synthetic Control and Double Machine Learning..."):
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

# ---------------------------------------------------------
# Top-level KPIs
# ---------------------------------------------------------
total_revenue = df["revenue"].sum()
average_revenue = df["revenue"].mean()
market_count = df["market"].nunique()
average_post_effect = sc["average_post_effect"]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total simulated revenue", f"{total_revenue:,.1f}")
k2.metric("Average revenue / market-week", f"{average_revenue:,.2f}")
k3.metric("Markets in experiment", f"{market_count}")
k4.metric(
    "Synthetic-control post lift",
    f"{average_post_effect:,.2f}",
    help="Average treated-market revenue minus its synthetic comparison after intervention.",
)

k5, k6, k7 = st.columns(3)
k5.metric("Cumulative post-period difference", f"{sc['cumulative_effect']:,.2f}")
k6.metric("Pre-period MSPE", f"{sc['pre_mspe']:,.3f}")
k7.metric("DML estimated effect", f"{dml['ate']:,.2f}")

st.caption(
    f"Treated market: {settings['treated_market']}  •  "
    f"Intervention begins: week {settings['intervention_week']}  •  "
    f"Simulated effect setting: {settings['price_change']:.1f}"
)

# ---------------------------------------------------------
# Dashboard tabs
# ---------------------------------------------------------
overview_tab, market_tab, causal_tab, data_tab = st.tabs(
    ["📈 Overview", "🏬 Market & drivers", "🧪 Causal analysis", "🗂️ Data & downloads"]
)

with overview_tab:
    left, right = st.columns([1.65, 1])

    with left:
        st.subheader("Weekly revenue: treated market vs synthetic comparison")
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(
            sc["weeks"], sc["actual"],
            label=f"{settings['treated_market']} actual",
            linewidth=2.5,
        )
        ax.plot(
            sc["weeks"], sc["synthetic"],
            label="Synthetic counterfactual",
            linewidth=2.5,
        )
        ax.axvline(
            settings["intervention_week"],
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="Intervention begins",
        )
        ax.set_xlabel("Week")
        ax.set_ylabel("Revenue (simulated units)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.subheader("Revenue by market")
        market_summary = (
            df.groupby("market", as_index=False)["revenue"]
            .mean()
            .sort_values("revenue", ascending=True)
        )
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(market_summary["market"], market_summary["revenue"])
        ax.set_xlabel("Average weekly revenue")
        ax.set_ylabel("")
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.subheader("Revenue before and after intervention")
    period_summary = (
        df.assign(period=df["post"].map({0: "Before intervention", 1: "After intervention"}))
        .assign(group=df["treated_market"].map({0: "Control markets", 1: "Treated market"}))
        .groupby(["period", "group"], as_index=False)["revenue"]
        .mean()
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    periods = ["Before intervention", "After intervention"]
    groups = ["Control markets", "Treated market"]
    width = 0.34
    x = range(len(periods))
    for idx, group in enumerate(groups):
        values = [
            period_summary.loc[
                (period_summary["period"] == period)
                & (period_summary["group"] == group),
                "revenue",
            ].mean()
            for period in periods
        ]
        ax.bar(
            [v + (idx - 0.5) * width for v in x],
            values,
            width=width,
            label=group,
        )
    ax.set_xticks(list(x), periods)
    ax.set_ylabel("Average revenue")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with market_tab:
    st.subheader("Explore market performance")
    selected_markets = st.multiselect(
        "Choose markets",
        options=sorted(df["market"].unique()),
        default=sorted(df["market"].unique()),
    )
    filtered = df[df["market"].isin(selected_markets)]

    if filtered.empty:
        st.warning("Select at least one market to display the charts.")
    else:
        market_week = (
            filtered.groupby(["week", "market"], as_index=False)["revenue"].mean()
        )
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for market in selected_markets:
            part = market_week[market_week["market"] == market]
            ax.plot(part["week"], part["revenue"], linewidth=1.8, label=market)
        ax.set_xlabel("Week")
        ax.set_ylabel("Revenue")
        ax.set_title("Weekly revenue by selected market")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
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

        st.subheader("Market-level summary")
        market_table = (
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
        st.dataframe(market_table, use_container_width=True, hide_index=True)

with causal_tab:
    st.subheader("Synthetic Control: actual vs counterfactual")
    st.write(
        "The synthetic comparison is a weighted combination of untreated donor markets. "
        "Weights are fitted using pre-intervention revenue."
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(sc["weeks"], sc["actual"], label="Actual treated market", linewidth=2.5)
    ax.plot(sc["weeks"], sc["synthetic"], label="Synthetic counterfactual", linewidth=2.5)
    ax.axvline(
        settings["intervention_week"],
        color="gray",
        linestyle="--",
        label="Intervention begins",
    )
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

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
    st.subheader("Weekly gap: actual minus synthetic")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    colors = [
        "#4C78A8" if week < settings["intervention_week"] else "#F58518"
        for week in effect_df["week"]
    ]
    ax.bar(effect_df["week"], effect_df["estimated_effect"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(settings["intervention_week"], color="gray", linestyle="--")
    ax.set_xlabel("Week")
    ax.set_ylabel("Revenue gap")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("Synthetic-control donor weights"):
        weights_df = pd.DataFrame(
            list(sc["donor_weights"].items()),
            columns=["donor_market", "weight"],
        ).sort_values("weight", ascending=False)
        st.dataframe(weights_df, use_container_width=True, hide_index=True)
        st.caption("Weights are constrained to be non-negative and sum to one.")

    st.subheader("Double Machine Learning")
    d1, d2, d3 = st.columns(3)
    d1.metric("Estimated ATE", f"{dml['ate']:.3f}")
    d2.metric("Approx. 95% CI lower", f"{dml['ci_lower']:.3f}")
    d3.metric("Approx. 95% CI upper", f"{dml['ci_upper']:.3f}")
    st.write(
        "The DML routine uses cross-fitted Random Forest models for the outcome "
        "and treatment assignment. The treatment is defined as treated-market × "
        "post-intervention."
    )
    st.info(dml["interpretation"])
    st.caption(
        "The confidence interval is approximate. This educational implementation "
        "uses simulated observational panel data; it is not a substitute for a "
        "study design review or robust statistical inference."
    )

    st.subheader("Plain-language interpretation")
    for line in summary:
        st.markdown(f"- {line}")

with data_tab:
    st.subheader("Dataset overview")
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{len(df.columns)}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum())}")

    st.subheader("Data quality")
    quality = pd.DataFrame(
        {
            "column": df.columns,
            "data_type": [str(df[col].dtype) for col in df.columns],
            "missing_values": [int(df[col].isna().sum()) for col in df.columns],
            "unique_values": [int(df[col].nunique()) for col in df.columns],
        }
    )
    st.dataframe(quality, use_container_width=True, hide_index=True)

    st.subheader("Data preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    st.subheader("Download results")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download market dataset",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="market_transactions.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "⬇️ Download synthetic-control results",
            data=effect_df.to_csv(index=False).encode("utf-8"),
            file_name="synthetic_control_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.divider()
st.caption(
    "Causal Pricing Analytics | Simulated educational project | "
    "Interpret estimated effects cautiously."
)

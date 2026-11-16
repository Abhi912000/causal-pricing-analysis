from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def generate_sample_data(
    n_markets=10, n_weeks=40, intervention_week=25,
    treated_market="Market_1", treatment_effect=12.0, random_state=42
):
    """Create reproducible panel data; intervention_week is the first treated week."""
    if n_markets < 3:
        raise ValueError("Use at least 3 markets (one treated and two donors).")
    if not 2 <= intervention_week < n_weeks:
        raise ValueError("Intervention week must be between week 2 and the final week.")
    market_names = [f"Market_{i}" for i in range(1, n_markets + 1)]
    if treated_market not in market_names:
        raise ValueError("treated_market must match one of the selected markets.")

    rng = np.random.default_rng(random_state)
    market_baselines = rng.normal(100, 15, size=n_markets)
    market_trends = rng.normal(0.7, 0.18, size=n_markets)
    rows = []
    for idx, market in enumerate(market_names):
        market_shock = rng.normal(0, 2, size=n_weeks)
        for week in range(n_weeks):
            seasonality = 5 * np.sin(2 * np.pi * week / 12)
            demand_index = 100 + 0.35 * week + 4 * np.cos(2 * np.pi * week / 10) + rng.normal(0, 3)
            competitor_price = 20 + 0.06 * week + rng.normal(0, 1.2)
            promo_spend = max(0, 10 + 0.2 * week + rng.normal(0, 2))
            post = int(week >= intervention_week)
            treated = int(market == treated_market)
            effect = treatment_effect if (treated and post) else 0.0
            revenue = (
                market_baselines[idx] + market_trends[idx] * week + seasonality
                + 0.22 * (demand_index - 100) + 1.8 * competitor_price
                + 0.65 * promo_spend + market_shock[week] + rng.normal(0, 4)
                + effect
            )
            rows.append({
                "market": market, "week": week, "revenue": round(float(revenue), 3),
                "demand_index": round(float(demand_index), 3),
                "competitor_price": round(float(competitor_price), 3),
                "promo_spend": round(float(promo_spend), 3),
                "treated_market": treated, "post": post,
                "treatment": treated * post
            })
    return pd.DataFrame(rows)


def save_dataset(df, path="data/market_transactions.csv"):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(target, index=False)


def run_synthetic_control(df, treated_market, intervention_week):
    """Find non-negative donor weights summing to 1, fit on pre-period revenue."""
    required = {"market", "week", "revenue"}
    if not required.issubset(df.columns):
        raise ValueError(f"Dataset must include columns: {sorted(required)}")
    pivot = df.pivot(index="week", columns="market", values="revenue").sort_index()
    if treated_market not in pivot.columns:
        raise ValueError(f"Market not found: {treated_market}")
    donors = [m for m in pivot.columns if m != treated_market]
    pre = pivot.index < intervention_week
    post = ~pre
    if pre.sum() < 3 or post.sum() < 1:
        raise ValueError("Need at least 3 pre-intervention weeks and 1 post week.")

    y_pre = pivot.loc[pre, treated_market].to_numpy()
    X_pre = pivot.loc[pre, donors].to_numpy()

    def objective(weights):
        return np.mean((y_pre - X_pre @ weights) ** 2)

    result = minimize(
        objective,
        x0=np.full(len(donors), 1 / len(donors)),
        method="SLSQP",
        bounds=[(0, 1)] * len(donors),
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}],
        options={"maxiter": 2000, "ftol": 1e-10}
    )
    if not result.success:
        raise RuntimeError(f"Synthetic-control optimization failed: {result.message}")
    weights = result.x
    actual = pivot[treated_market].to_numpy()
    synthetic = pivot[donors].to_numpy() @ weights
    effects = actual - synthetic
    pre_idx = pivot.index.to_numpy() < intervention_week
    post_idx = ~pre_idx
    pre_mspe = mean_squared_error(actual[pre_idx], synthetic[pre_idx])
    post_effects = effects[post_idx]
    return {
        "weeks": pivot.index.to_numpy(),
        "actual": actual,
        "synthetic": synthetic,
        "effects": effects,
        "average_post_effect": float(np.mean(post_effects)),
        "cumulative_effect": float(np.sum(post_effects)),
        "pre_mspe": float(pre_mspe),
        "donor_weights": dict(zip(donors, weights.tolist())),
        "donors": donors
    }


def estimate_dml(df, treated_market, intervention_week, n_splits=5, random_state=42):
    """
    Cross-fitted partially linear DML estimate.
    For teaching, treatment is treated_market × post; covariates are measured
    demand, competitor price, promotion spend, and week. In real studies, avoid
    controlling for variables caused by treatment.
    """
    data = df.copy()
    data["treated_market"] = (data["market"] == treated_market).astype(int)
    data["post"] = (data["week"] >= intervention_week).astype(int)
    data["treatment"] = data["treated_market"] * data["post"]
    features = ["week", "demand_index", "competitor_price", "promo_spend", "treated_market", "post"]
    X = data[features].to_numpy(dtype=float)
    y = data["revenue"].to_numpy(dtype=float)
    t = data["treatment"].to_numpy(dtype=float)

    # K-fold cross-fitting creates out-of-fold predictions for outcome and treatment.
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_hat = np.zeros(len(data))
    t_hat = np.zeros(len(data))
    for train_idx, test_idx in folds.split(X):
        model_y = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=random_state, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=120, min_samples_leaf=5, random_state=random_state + 1, n_jobs=-1)
        model_y.fit(X[train_idx], y[train_idx])
        model_t.fit(X[train_idx], t[train_idx])
        y_hat[test_idx] = model_y.predict(X[test_idx])
        t_hat[test_idx] = model_t.predict(X[test_idx])

    y_res = y - y_hat
    t_res = t - t_hat
    denom = np.sum(t_res ** 2)
    if denom < 1e-10:
        raise ValueError("Treatment residual has near-zero variation; DML cannot estimate an effect.")
    ate = float(np.sum(t_res * y_res) / denom)
    residuals = y_res - ate * t_res
    # Approximate standard error for the residualized slope.
    se = float(np.sqrt(np.mean((t_res * residuals) ** 2)) / (np.sqrt(np.mean(t_res ** 2)) * np.sqrt(len(data))))
    return {
        "ate": ate, "se": se, "ci_lower": ate - 1.96 * se, "ci_upper": ate + 1.96 * se,
        "n": int(len(data)),
        "interpretation": "Estimated adjusted effect; interpret cautiously because this is simulated observational panel data."
    }


def build_summary(sc, dml, intervention_week):
    best = sorted(sc["donor_weights"].items(), key=lambda item: item[1], reverse=True)[:3]
    weights_text = ", ".join(f"{name}: {weight:.1%}" for name, weight in best)
    return [
        f"The intervention starts at week {intervention_week}; weeks before it are used to fit the synthetic control.",
        f"The treated market's average post-intervention revenue difference versus its synthetic comparison is {sc['average_post_effect']:.2f} units per week.",
        f"The cumulative post-intervention difference is {sc['cumulative_effect']:.2f} revenue units.",
        f"Pre-intervention MSPE is {sc['pre_mspe']:.2f}; lower values mean a closer pre-period fit, but there is no universal pass threshold.",
        f"Largest synthetic-control donor weights: {weights_text}.",
        f"The cross-fitted DML estimate is {dml['ate']:.2f} units. Compare it with the synthetic-control estimate rather than treating either estimate as definitive."
    ]

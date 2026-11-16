"""Optional command-line run: generates sample data and prints analysis results."""
from causal_engine import generate_sample_data, run_synthetic_control, estimate_dml, save_dataset

df = generate_sample_data()
save_dataset(df)
sc = run_synthetic_control(df, "Market_1", 25)
dml = estimate_dml(df, "Market_1", 25)
print("Synthetic-control average post effect:", round(sc["average_post_effect"], 3))
print("Cumulative post effect:", round(sc["cumulative_effect"], 3))
print("Pre-period MSPE:", round(sc["pre_mspe"], 3))
print("DML ATE:", round(dml["ate"], 3))
print("DML approximate 95% CI:", round(dml["ci_lower"], 3), "to", round(dml["ci_upper"], 3))
print("Donor weights:", {k: round(v, 3) for k, v in sc["donor_weights"].items()})

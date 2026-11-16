
from pathlib import Path
import pandas as pd

# This script is inside the powerbi_data folder
SCRIPT_DIR = Path(__file__).resolve().parent

# Find the original data folder
DATA_DIR = SCRIPT_DIR.parent / "data"

# Export files into the same folder as this script
OUTPUT_DIR = SCRIPT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load original CSV
input_file = DATA_DIR / "market_transactions.csv"
df = pd.read_csv(input_file)

# 1. Complete transaction data
df.to_csv(OUTPUT_DIR / "market_transactions.csv", index=False)

# 2. Weekly market summary
weekly_summary = (
    df.groupby(["market", "week"], as_index=False)
    .agg(
        average_revenue=("revenue", "mean"),
        average_demand=("demand_index", "mean"),
        average_competitor_price=("competitor_price", "mean"),
        total_promo_spend=("promo_spend", "sum"),
        treatment=("treatment", "max")
    )
)

weekly_summary.to_csv(
    OUTPUT_DIR / "weekly_market_summary.csv",
    index=False
)

# 3. Treatment summary
treatment_summary = (
    df.groupby(["treated_market", "post"], as_index=False)
    .agg(
        average_revenue=("revenue", "mean"),
        average_demand=("demand_index", "mean"),
        average_competitor_price=("competitor_price", "mean"),
        average_promo_spend=("promo_spend", "mean"),
        observations=("revenue", "count")
    )
)

treatment_summary.to_csv(
    OUTPUT_DIR / "treatment_summary.csv",
    index=False
)

print("Power BI export completed successfully!")
print("Files saved in:", OUTPUT_DIR)

for file in OUTPUT_DIR.glob("*.csv"):
    print("-", file.name)



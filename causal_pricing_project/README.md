# Causal Inference & Synthetic Control for Algorithmic Pricing

A beginner-friendly, runnable implementation of Project 1 from the supplied Zaalima Development project brief. It demonstrates:
- simulated regional-market revenue data and competitor/demand covariates;
- a treated market and intervention date;
- synthetic control with non-negative donor weights summing to one;
- pre-intervention mean squared prediction error (MSPE);
- a cross-fitted Double Machine Learning (DML) estimate;
- a Streamlit dashboard with actual-vs-counterfactual charts and CSV downloads.

**Important:** This is an educational prototype using generated data. It does not prove a real pricing effect. A production-grade causal study needs validated data, a defensible causal graph, treatment-assignment assumptions, placebo/sensitivity checks, and careful uncertainty estimates.

## 1. Requirements
- Windows 10/11
- Python 3.10–3.12 recommended
- Visual Studio Code with the Python extension

## 2. Open in VS Code
Extract the ZIP, then in VS Code choose **File → Open Folder** and select `causal_pricing_project`.

Open **Terminal → New Terminal**.

## 3. Create a virtual environment (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```
If `py` is not recognized, install Python from python.org and tick **Add Python to PATH**.

## 4. Run the app
```powershell
python -m streamlit run app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser. Set experiment settings in the sidebar and click **Generate data and run analysis**.

## 5. What you should see
- KPI values for synthetic-control average lift, cumulative lift, pre-period MSPE, and DML estimate.
- A line chart comparing treated-market revenue with the synthetic counterfactual.
- A weekly effect table and a plain-language summary.
- Download buttons for the generated panel data and the synthetic-control results.
- The generated dataset is also saved as `data/market_transactions.csv`.

## 6. How the data is structured
Each row is one market-week:
- `market`, `week`: market identifier and time index
- `revenue`: outcome
- `demand_index`, `competitor_price`, `promo_spend`: covariates
- `treated_market`: 1 for the selected treated market
- `post`: 1 from the intervention week onward
- `treatment`: `treated_market × post`

## 7. How the analysis works
1. **Generate panel data.** Market baselines, trends, seasonality, demand, competitor price, promotions, and noise create varied revenue trajectories.
2. **Synthetic control.** Optimize donor-market weights using only pre-intervention outcomes. The weights are constrained to be between 0 and 1 and sum to 1.
3. **Counterfactual.** Apply the learned weights to donor markets across all weeks to estimate what the treated market might have done without the intervention.
4. **Effect.** Actual revenue minus synthetic revenue is the estimated weekly gap. Average and cumulative post-period gaps are reported.
5. **MSPE.** Mean squared prediction error in the pre-period measures how closely the synthetic market matches the treated market before intervention.
6. **DML.** Cross-fitting predicts outcome and treatment using random forests, residualizes both, and estimates a residualized treatment coefficient.

## 8. Important limitations
- The data is simulated and deliberately includes a known treatment effect.
- The DML estimator is a compact teaching implementation; its confidence interval is approximate, and treatment is rare in this panel. For formal inference, use a validated implementation such as EconML or DoubleML and a design appropriate to panel data.
- Synthetic control relies on a good pre-treatment match and assumptions that donor markets are not affected by the intervention.
- In a real project, add a causal DAG, placebo interventions, donor-pool sensitivity, pre-trend diagnostics, and robustness checks before making pricing decisions.
- Do not interpret the demo's generated result as an actual business finding.

## 9. GitHub daily commits
The supplied project brief asks teams to create a GitHub repository and commit work daily. Example:
```powershell
git init
git add .
git commit -m "Initial causal pricing analytics project"
```
Then create a repository on GitHub and follow GitHub's instructions to add the remote and push. Never commit credentials or private customer data.

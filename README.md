# Causal Inference & Synthetic Control for Algorithmic Pricing

## Project Overview

This project focuses on applying Causal Inference and Synthetic Control techniques to analyze the impact of algorithmic pricing strategies.

The project aims to estimate the effect of pricing interventions using observational data and compare outcomes between treated and control groups.

It provides a Python-based application for generating or analyzing data and examining estimated treatment effects.

## Objectives

- Understand the impact of algorithmic pricing strategies.
- Apply causal inference techniques to observational data.
- Implement the Synthetic Control methodology.
- Estimate treatment effects.
- Analyze and interpret the results through Python.

## Technologies Used

- **Python** – Main programming language
- **Pandas** – Data manipulation and analysis
- **NumPy** – Numerical computations
- **Streamlit** – Interactive web application
- **SQL** – Data querying (where applicable)
- **Git & GitHub** – Version control and project management

## Project Structure

```text
causal_pricing_project/
│
├── app.py
├── causal_engine.py
├── run_analysis.py
├── requirements.txt
├── README.md
├── run_project.bat
└── .gitignore

data/
└── market_transactions.csv
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Abhi912000/causal-pricing-analysis.git
```

### 2. Navigate to the Project Folder

```bash
cd causal-pricing-analysis/causal_pricing_project
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

### Run the Streamlit Application

```bash
streamlit run app.py
```

The application should open in your browser. If it does not, open:

http://localhost:8501

### Run the Analysis Script

```bash
python run_analysis.py
```

## Main Project Components

### 1. Application Interface

`app.py` provides the interactive application interface.

### 2. Causal Engine

`causal_engine.py` contains the causal analysis implementation.

### 3. Analysis Script

`run_analysis.py` is used to execute the analysis independently.

### 4. Dataset

`market_transactions.csv` contains the market transaction data used by the project.

## Expected Learning Outcomes

- Understanding causal inference concepts.
- Practical implementation of synthetic control methods.
- Working with Python data analysis libraries.
- Analyzing treatment effects using observational data.
- Building an interactive data analytics application.

## Limitations

The results depend on the quality of the input data and the assumptions of the causal inference methodology.

Estimated effects should be interpreted carefully, particularly when using simulated observational data.

## Author

**Ganapathi Sai Abhishek Grandhe**

## Project Category

Data Analytics | Causal Inference | Algorithmic Pricing | Python

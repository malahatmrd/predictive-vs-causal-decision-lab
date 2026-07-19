# Predictive ML vs Causal Decision Intelligence Lab

A reproducible machine-learning project demonstrating why accurate outcome prediction does not necessarily produce the best intervention policy.

## Project Goal

This repository compares predictive modeling with causal decision-making using simulated customer-retention data with known potential outcomes.

The project implements:

- Predictive churn modeling
- Confounding analysis
- Propensity-score estimation
- Naive treatment-effect estimation
- Inverse Probability Weighting (IPW)
- T-learner estimation
- Augmented Inverse Probability Weighting (AIPW)
- Cross-fitting
- Average Treatment Effect (ATE)
- Conditional Average Treatment Effect (CATE)
- Policy-value evaluation
- Placebo and omitted-confounder robustness checks

## Research Question

Why can a predictive model with acceptable classification performance still produce a poor intervention policy, and how can causal estimators improve decision value?

## Planned Notebook Workflow

1. `00_environment_check.ipynb`
2. `01_causal_data_generation.ipynb`
3. `02_predictive_baseline.ipynb`
4. `03_ate_estimators.ipynb`
5. `04_cate_and_policy.ipynb`
6. `05_robustness.ipynb`
7. `06_final_report.ipynb`

## Repository Structure

```text
predictive-vs-causal-decision-lab/
├── configs/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── notebooks/
├── reports/
│   ├── figures/
│   └── tables/
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── visualization/
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Environment

- Python 3.11
- NumPy
- Pandas
- SciPy
- scikit-learn
- Matplotlib
- Jupyter
- Pytest

## Installation

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Project Status

Project structure and isolated Python environment initialized.

## Scientific Basis

Inspired by the distinction between prediction, intervention, generalization, and causality discussed in Pedro Domingos' article:

*A Few Useful Things to Know About Machine Learning*.
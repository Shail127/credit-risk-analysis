# Credit Risk Analysis: A Machine Learning Approach

Multiclass classification system that predicts loan outcomes (**Approved / Pending / Rejected**) and assesses borrower default risk using demographic, financial, transactional, and loan-specific data.

## Overview

Traditional credit scoring (logistic regression, rule-based scorecards) struggles to capture the non-linear patterns in borrower behavior. This project compares four machine learning approaches — **Random Forest, XGBoost, LightGBM, and a Backpropagation Neural Network** — to classify loan applicants and quantify their probability of default.

## Dataset

- **5,000** customer records, **46** original attributes (reduced to ~17 after preprocessing)
- Categories: demographic attributes, income/account financial indicators, loan-related variables, engineered ratios, identifiers
- Target: `loan_status` — 0 = Approved (2,241), 1 = Pending (1,400), 2 = Rejected (1,359)

## Feature Engineering

- **DTI (Debt-to-Income) ratio** — EMI estimated via standard amortization formula, converted to a % of monthly income, then bucketed into Low/Moderate/High/Very High risk
- **Composite credit score (300–900 scale)** — weighted combination of income, balance, deposits/withdrawals, loan terms, and behavioral indicators, mapped to CIBIL/FICO-style ratings
- Log-transformed income, loan-to-investment ratio, investment-to-income ratio, monthly installment, scaled high-variance features
- Interactive lookup modules simulating a loan-officer dashboard: enter a `customer_id` to retrieve DTI, risk category, credit score, and rating

## Methodology

- 80/20 stratified train-test split
- Feature scaling applied where needed (tree-based models trained on raw values)
- Heuristic hyperparameter tuning (manual, iterative — not exhaustive grid search) for all four models
- Evaluation: Accuracy, Precision/Recall/F1 per class, Confusion Matrix, macro-averaged one-vs-rest AUC

## Results

| Model | Accuracy | Macro AUC (OVR) | Notes |
|---|---|---|---|
| **XGBoost** | **73.8%** | **0.876** | Best overall — strongest risk separation |
| LightGBM | 72.7% | 0.872 | Fastest training, competitive accuracy |
| Random Forest | 72.0% | 0.862 | Stable baseline, handles imbalance well |
| Neural Network (Backprop) | 72.25% | 0.871 | Captures complex non-linear interactions |

All four classes improved measurably after heuristic tuning (e.g., XGBoost AUC rose from 0.808 → 0.876).

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · TensorFlow/Keras · Pandas · Matplotlib · Seaborn

## Repository Structure

```
├── CreditRiskAnalysis.ipynb        # Full analysis notebook (EDA, preprocessing, modeling, evaluation)
├── Cleaned_CreditRisk_Data.csv     # Preprocessed dataset
└── README.md
```

## Key Takeaways

- Ensemble boosting methods (XGBoost, LightGBM) outperform bagging and neural approaches on this dataset, but the margins are narrow
- Engineered financial-stress features (DTI, credit score, loan-to-investment ratio) meaningfully improve discriminative power over raw attributes alone
- Class imbalance (Approved >> Pending/Rejected) remains the main limiter on Pending/Rejected precision-recall

## Limitations & Future Work

- Dataset lacks historical credit/behavioral time-series data
- Tuning was heuristic rather than exhaustive (GridSearch/Bayesian optimization could improve results further)
- No cost-sensitive evaluation (misclassifying a defaulter is costlier than rejecting a good borrower)
- Next steps: SHAP-based explainability, SMOTE/cost-sensitive learning for imbalance, time-series behavioral modeling

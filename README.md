# CodeAlpha — Credit Scoring Model

Machine Learning Internship | Task 1

A classification model that predicts an individual's creditworthiness from financial data, comparing three algorithms across standard credit-risk metrics.

## Objective

Predict whether an individual is creditworthy using features drawn from their financial history — income, debt, payment history, credit utilization, and more.

## Approach

Three classification algorithms are trained and compared on the same data:

- Logistic Regression
- Decision Tree
- Random Forest

Each is evaluated on **Precision, Recall, F1-Score, and ROC-AUC** — the metrics that matter for credit decisions, where both false approvals and false rejections carry real cost.

## Pipeline

1. Data loading
2. Exploratory data analysis (class balance, correlations)
3. Feature engineering (debt-to-income ratio, income per account)
4. Train/test split (80/20, stratified) + feature scaling
5. Train three models
6. Evaluate and compare metrics
7. Confusion matrix, ROC curves, and feature importance

## Results

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.873 | 0.890 | 0.881 | 0.951 |
| Random Forest | 0.837 | 0.857 | 0.847 | 0.933 |
| Decision Tree | 0.812 | 0.820 | 0.816 | 0.885 |

The strongest predictors — payment history, credit utilization, and debt-to-income — align with what real lenders weigh most heavily.

## Dataset

The notebook generates a realistic synthetic credit dataset so it runs anywhere with no download. To use a real dataset instead (e.g. the UCI German Credit dataset), load it into a DataFrame named `df` with a binary `creditworthy` target column — the rest of the notebook runs unchanged.

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook CodeAlpha_CreditScoring.ipynb
```

Run all cells top to bottom.

## Tech Stack

Python · scikit-learn · pandas · NumPy · Matplotlib · seaborn

---

*Completed as part of the CodeAlpha Machine Learning Internship.*

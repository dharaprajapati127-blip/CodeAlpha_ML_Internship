# CodeAlpha — Machine Learning Internship

A collection of machine learning projects completed during the CodeAlpha Machine Learning Internship. Each project is a complete, self-contained classification pipeline with exploratory analysis, model comparison, and evaluation.

## Projects

| # | Project | Notebook | Domain |
|---|---------|----------|--------|
| 1 | Credit Scoring Model | `CodeAlpha_CreditScoring.ipynb` | Financial risk |
| 2 | Disease Prediction from Medical Data | `CodeAlpha_DiseasePrediction.ipynb` | Healthcare |

---

## Task 1 — Credit Scoring Model

**Objective:** Predict an individual's creditworthiness from financial data.

**Approach:** Compare Logistic Regression, Decision Tree, and Random Forest on Precision, Recall, F1-Score, and ROC-AUC.

**Pipeline:** Data -> EDA -> feature engineering (debt-to-income, income per account) -> 80/20 stratified split + scaling -> train three models -> evaluate -> confusion matrix, ROC curves, feature importance.

**Dataset:** A realistic synthetic credit dataset (income, debt, payment history, credit utilization, etc.) so the notebook runs anywhere. Swap in the UCI German Credit dataset by loading it into a `df` with a binary `creditworthy` target.

**Results:**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.873 | 0.890 | 0.881 | 0.951 |
| Random Forest | 0.837 | 0.857 | 0.847 | 0.933 |
| Decision Tree | 0.812 | 0.820 | 0.816 | 0.885 |

Strongest predictors -- payment history, credit utilization, and debt-to-income -- match what real lenders weigh most heavily.

---

## Task 2 — Disease Prediction from Medical Data

**Objective:** Predict the likelihood of disease from patient diagnostic data.

**Approach:** Compare Logistic Regression, Random Forest, SVM, and XGBoost on Precision, Recall, F1-Score, and ROC-AUC, with emphasis on recall for malignant cases.

**Pipeline:** Data -> EDA -> 80/20 stratified split + scaling -> train four models -> evaluate -> confusion matrix, ROC curves, feature importance.

**Dataset:** Breast Cancer Wisconsin (Diagnostic) -- a real medical dataset from the UCI ML Repository, built into scikit-learn. 569 patients, 30 diagnostic features, binary target (malignant vs benign).

**Results:**

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.986 | 0.986 | 0.986 | 0.995 |
| SVM | 0.986 | 0.986 | 0.986 | 0.995 |
| Random Forest | 0.959 | 0.972 | 0.966 | 0.993 |
| XGBoost | 0.946 | 0.972 | 0.959 | 0.994 |

Most important features -- worst-case radius, perimeter, and concavity -- align with clinical intuition about tumor characteristics.

> *Note:* an educational project; a real diagnostic tool would require clinical validation and regulatory approval before medical use.

---

## How to Run

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
jupyter notebook
```

Open either notebook and run all cells top to bottom.

## Tech Stack

Python · scikit-learn · XGBoost · pandas · NumPy · Matplotlib · seaborn

---

*Completed as part of the CodeAlpha Machine Learning Internship.*

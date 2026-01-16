# Credit Approval – Logistic Regression (Toy Model)

## Problem
This project implements a simple binary classification model to predict credit approval based on applicant age and income.  
It is a toy example designed to demonstrate a clean end-to-end ML workflow rather than production performance.

## Dataset
The dataset is a small, synthetic table with the following features:
- age (numeric)
- income (numeric)

Target:
- approved (binary: 0 = rejected, 1 = approved)

The dataset is intentionally minimal to focus on methodology.

## Model
A Logistic Regression classifier from scikit-learn is used.

Logistic regression is chosen because:
- it is a standard baseline for binary classification
- it is interpretable
- it is commonly used in credit scoring as a benchmark model

## Workflow
1. Data loading and basic cleaning
2. Feature / target separation
3. Train-test split
4. Model training
5. Evaluation on held-out test data

## Results
Model accuracy is evaluated on the test set.
Given the small dataset size, results are not statistically meaningful and are used for demonstration only.

## Limitations
- Extremely small dataset
- No feature engineering
- No cross-validation
- Not suitable for real credit decisions

## How to Run
```bash
pip install -r requirements.txt


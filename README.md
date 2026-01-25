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

Credit Approval Modeling with Cost-Sensitive Machine Learning
Problem Statement

In consumer lending, machine learning models are often evaluated using accuracy or AUC.
However, credit decisions are asymmetric:

Approving a bad borrower (false positive) leads to direct capital loss

Rejecting a good borrower (false negative) leads to opportunity cost

Treating both errors equally results in suboptimal and risky lending decisions.
This project demonstrates how to align machine learning models with financial decision-making by optimizing for expected cost rather than accuracy.

Dataset

A synthetic credit dataset was generated to reflect realistic lending features:

age

income

debt_to_income

credit_score

loan_amount

target: approved (binary)

The dataset intentionally includes noise, overlap, and class imbalance to mirror real-world credit data.

Cost Assumptions

To reflect business reality, asymmetric costs were defined:

False Positive (approving a bad borrower): 100

False Negative (rejecting a good borrower): 10

All models are evaluated using expected financial loss, not accuracy.

Models Evaluated

The following models were trained and evaluated using the same dataset and train–test split:

Logistic Regression (baseline)

K-Nearest Neighbours

XGBoost

Neural Network (MLP)

Each model was evaluated using:

confusion matrix

expected cost

cost per applicant

Results (Before Threshold Tuning)
Model	Cost per Applicant
Logistic Regression	26.89
KNN	7.64
XGBoost	5.55
Neural Network	5.62

XGBoost achieved the lowest expected cost and strongest control over false positives, which dominate financial risk.


Threshold Optimization

Machine learning models output probabilities, not decisions.
Instead of retraining models, the decision threshold was optimized to minimize expected cost.

By sweeping thresholds from 0 to 1 and evaluating expected loss at each point:

False positives were sharply reduced

False negatives increased moderately

Total expected loss decreased significantly

Final Deployment Policy (Optimized XGBoost)


After threshold tuning:

False Positives reduced from 157 to 12

Expected Cost reduced from 16,640 to 7,120

Cost per Applicant reduced from 5.55 to 2.37

This represents a ~57% reduction in expected loss without retraining the model.



Key Takeaways:

Accuracy is a poor metric for credit decisions

Cost-sensitive evaluation is essential in finance

Threshold tuning often delivers more value than model complexity

Tree-based models outperform neural networks on tabular credit data

Business policy should live in the decision threshold, not the model

Project Structure

credit-approval-ml/
├── src/
│ ├── train.py
│ ├── evaluate.py
├── notebooks/
│ ├── 01-exploration.ipynb
│ ├── 04_knn_model.ipynb
│ ├── 05_xgboost_model.ipynb
│ ├── 06_neural_network.ipynb
│ └── 07_xgboost_threshold_tuning.ipynb

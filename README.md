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

Credit Approval Modeling with Cost-Sensitive Machine Learning
Problem Statement

In consumer lending, machine learning models are often evaluated using accuracy or AUC.
However, credit decisions are asymmetric:

# Approving a bad borrower (false positive) leads to direct capital loss

# Rejecting a good borrower (false negative) leads to opportunity cost

Treating both errors equally results in suboptimal and risky lending decisions.
This project demonstrates how to align machine learning models with financial decision-making by optimizing for expected cost rather than accuracy.

Dataset - A synthetic credit dataset was generated to reflect realistic lending features:

age

income

debt_to_income

credit_score

loan_amount

target: approved (binary)

The dataset intentionally includes noise, overlap, and class imbalance to mirror real-world credit data.

# Cost Assumptions

To reflect business reality, asymmetric costs were defined:

False Positive (approving a bad borrower): 100

False Negative (rejecting a good borrower): 10

All models are evaluated using expected financial loss, not accuracy.

#Models Evaluated

The following models were trained and evaluated using the same dataset and train–test split:

Logistic Regression (baseline)

K-Nearest Neighbours

XGBoost

Neural Network (MLP)

Each model was evaluated using:

confusion matrix

expected cost

cost per applicant

# Results (Before Threshold Tuning)
Model	Cost per Applicant
Logistic Regression	26.89
KNN	7.64
XGBoost	5.55
Neural Network	5.62

XGBoost achieved the lowest expected cost and strongest control over false positives, which dominate financial risk.


# Threshold Optimization

Machine learning models output probabilities, not decisions.
Instead of retraining models, the decision threshold was optimized to minimize expected cost.

By sweeping thresholds from 0 to 1 and evaluating expected loss at each point:

False positives were sharply reduced

False negatives increased moderately

Total expected loss decreased significantly

Final Deployment Policy (Optimized XGBoost)


# After threshold tuning:

False Positives reduced from 157 to 12

Expected Cost reduced from 16,640 to 7,120

Cost per Applicant reduced from 5.55 to 2.37

This represents a ~57% reduction in expected loss without retraining the model.



## Key Takeaways:

Accuracy is a poor metric for credit decisions

Cost-sensitive evaluation is essential in finance

Threshold tuning often delivers more value than model complexity

Tree-based models outperform neural networks on tabular credit data

Business policy should live in the decision threshold, not the model



# Project Structure

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




##PHASE II - Experimenting with more data, reading and training on images and  optimising the policy further

# Problem Statement

This project simulates a real-world credit approval system where decisions are evaluated not by accuracy alone, but by economic cost of errors.

In lending:

False positives (approving bad borrowers) are far more costly than

False negatives (rejecting good borrowers)

The objective is therefore to minimize expected monetary loss, not maximize classification accuracy.


# Modeling Philosophy

Instead of treating credit approval as a pure ML problem, this repo approaches it as a decision system with:

cost asymmetry

policy constraints

human-in-the-loop signals

abstention and override logic

Each notebook adds one layer of realism and tests whether it reduces cost per applicant.



# Experiments Overview
Notebook	Description	Result
08	Cost-sensitive XGBoost (tabular baseline)	Strong baseline
09	Add structured text (location, officer notes)	Added noise
09.2	Text ablation	Confirmed no marginal gain
10	LLM-style policy signals	Significant cost reduction
10.2	Threshold + policy tuning	Best result
11	Image-based collateral signals	No improvement



# Key Metric

All models are evaluated using:

Expected Cost = (FP × cost_fp) + (FN × cost_fn)


With:

cost_fp = 100

cost_fn = 10

This reflects real underwriting economics.


## Final Results
Approach	Cost / Applicant
Tabular ML only	~2.78
+ Text features	Worse
+ Images	No improvement
Tabular + Policy (LLM-style)	~2.73 (best)


## Key Insight

More data modalities do not guarantee better decisions.
Weak or noisy signals increase complexity without reducing economic risk.

The largest gains came not from adding data, but from policy-aware decision logic layered on top of strong tabular models.



## Why This Matters

This mirrors how modern fintech lenders operate:

ML for scoring

policy for control

humans for edge cases

The repo is designed to be extendable to:

real dataset

production pipelines

governance frameworks

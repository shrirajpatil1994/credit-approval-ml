# Credit Approval Modeling with Cost-Sensitive Machine Learning

## Overview

This repository implements a **cost-sensitive credit approval system** that mirrors how real-world lenders evaluate risk.  
Instead of optimizing for accuracy or AUC, models are evaluated based on the **economic cost of decision errors**.

The project is structured in **three phases**, progressing from a simple baseline to a realistic, policy-aware decision system using tabular, textual, and image data.

---

## Core Problem

In consumer lending, classification errors are asymmetric:

- **False Positives (approving bad borrowers)** lead to direct capital loss  
- **False Negatives (rejecting good borrowers)** lead to opportunity cost  

Treating these errors equally results in poor financial decisions.

**Objective:**  
> Minimize *expected monetary loss*, not classification error.

---

## Cost Framework

All experiments use the same asymmetric cost assumptions:

- False Positive cost = **100**
- False Negative cost = **10**

Evaluation metric: Expected Cost = (FP × cost_fp) + (FN × cost_fn)


All reported results are based on **expected cost and cost per applicant**, not accuracy.

---

# Phase I — Baseline Credit Modeling

### Goal
Establish a clean, end-to-end ML workflow and demonstrate why accuracy is insufficient for credit decisions.

### Dataset
A synthetic credit dataset with realistic overlap and noise:

**Features**
- age  
- income  
- debt_to_income  
- credit_score  
- loan_amount  

**Target**
- approved (binary)

### Models Evaluated
- Logistic Regression (baseline)
- K-Nearest Neighbours
- XGBoost
- Neural Network (MLP)

### Key Result (Before Threshold Tuning)

| Model | Cost per Applicant |
|------|--------------------|
| Logistic Regression | 26.89 |
| KNN | 7.64 |
| XGBoost | 5.55 |
| Neural Network | 5.62 |

**Insight:**  
Tree-based models significantly outperform linear and neural models on tabular credit data when evaluated by cost.

---

# Phase II — Cost Optimization & Policy Control

### Goal
Reduce financial loss **without changing the model**, by optimizing the decision policy.

### Threshold Optimization
Instead of retraining, the decision threshold is swept from 0 → 1 to directly minimize expected cost.

### Results (XGBoost)

- False Positives reduced from **157 → 12**
- Expected Cost reduced from **16,640 → 7,120**
- Cost per Applicant reduced from **5.55 → 2.37**

**~57% reduction in expected loss without retraining**

### Key Insight
> In credit systems, **policy tuning often delivers more value than model complexity**.

---

# Phase III — Multimodal Signals & Decision Systems

### Goal
Test whether additional data modalities improve economic outcomes.

### Experiments

| Notebook | Description | Outcome |
|--------|------------|--------|
| 09 | Structured text (location, officer notes) | Added noise |
| 09.2 | Text ablation study | No marginal gain |
| 10 | LLM-style policy signal | Significant improvement |
| 10.2 | Policy + threshold tuning | **Best result** |
| 11 | Image-based collateral signals | No improvement |

### Modalities Tested
- **Tabular data** (core signal)
- **Textual data** (location descriptors, officer notes)
- **Image data** (collateral/property images)

### Final Results (After Phase III)

| Approach | Cost / Applicant |
|--------|------------------|
| Tabular ML only | ~2.78 |
| + Text features | Worse |
| + Image features | No improvement |
| **Tabular + Policy (LLM-style)** | **~2.73 (best)** |

---

## Key Insights

- Accuracy is a poor metric for credit decisions  
- Cost-sensitive evaluation is essential in finance  
- Threshold tuning often outperforms adding new models  
- Tree-based models dominate tabular credit data  
- **Weak signals increase noise and cost, even with more data**
- Policy logic delivers more value than raw model complexity  

---

## Why This Matters

This architecture mirrors real fintech underwriting systems:

- ML models for scoring  
- Policy layers for control and compliance  
- Human-in-the-loop escalation for edge cases  

The repository is designed to be extensible to:
- real lender datasets
- production scoring pipelines
- governance and risk frameworks

---

## Project Structure

```text
credit-approval-ml/
├── data/
│   ├── raw/
│   │   └── credit_dataset.csv
│   └── processed/
│       ├── credit_tabular.csv
│       ├── credit_tab_text.csv
│       └── credit_tab_text_conditioned.csv
│
├── notebooks/
│   ├── 01-exploration.ipynb
│   ├── 08-cost-sensitive-xgboost.ipynb
│   ├── 09-text-features.ipynb
│   ├── 09.2-text-ablation.ipynb
│   ├── 10-llm-policy.ipynb
│   ├── 10.2-llm-policy-tuned.ipynb
│   └── 11-image-signal.ipynb
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── config.py
│
└── requirements.txt

---

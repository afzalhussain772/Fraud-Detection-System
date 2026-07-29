# Fraud Detection System for Digital Payments and Banking Transactions

A machine learning system that classifies banking and digital payment transactions as legitimate or fraudulent. Built as a binary classification pipeline using scikit-learn, with a focus on handling the severe class imbalance that real-world fraud detection always involves.

## Overview

Every digital payment platform — credit cards, mobile wallets, online banking — processes far more legitimate transactions than fraudulent ones, typically less than 2% of all activity. That imbalance is what makes fraud detection a genuinely hard problem: a model that simply predicts "legitimate" for everything will still score 98% accuracy while catching zero fraud.

This project builds and evaluates two classifiers — Logistic Regression and Random Forest — trained to catch that minority class without drowning in false positives.

## Dataset

Real transaction data from financial institutions is confidential and generally unavailable outside partnership agreements, so this project uses a synthetically generated dataset (`generate_data.py`) built to mirror the statistical shape of real fraud data: a ~98/2 class split, and fraud cases skewed toward specific, realistic patterns (odd hours, newer accounts, higher amounts, unusual distance from the account holder's typical location).

| Feature | Description |
|---|---|
| `amount` | Transaction amount |
| `transaction_hour` | Hour of day the transaction occurred (0–23) |
| `account_age_days` | Age of the account in days |
| `num_transactions_last_24h` | Number of transactions in the preceding 24 hours |
| `distance_from_home_km` | Distance between the transaction location and the account holder's home |
| `is_foreign_transaction` | Whether the transaction originated abroad (0/1) |
| `is_online` | Whether the transaction was made online (0/1) |
| `class` | Target label — 0 = legitimate, 1 = fraud |

## Pipeline

1. **Load data** — read the transaction CSV into a DataFrame.
2. **Preprocess** — scale features with `StandardScaler` so no single feature dominates on the basis of magnitude alone.
3. **Split** — 75/25 train/test split, stratified on the target to preserve the fraud ratio in both sets.
4. **Train** — two models, both trained with `class_weight="balanced"` to counteract the imbalance:
   - **Logistic Regression** — a fast, interpretable baseline.
   - **Random Forest** — an ensemble of decision trees, generally stronger on non-linear patterns.
5. **Evaluate** — accuracy alone is not informative on imbalanced data, so evaluation centers on:
   - **Precision** — of the transactions flagged as fraud, how many actually were.
   - **Recall** — of all actual fraud cases, how many were caught.
   - **ROC-AUC** — overall separability between the two classes.
6. **Feature importance** — which inputs the Random Forest relied on most heavily when making its predictions.

## Usage

```bash
pip install -r requirements.txt
python generate_data.py       # generates the synthetic dataset -> data/transactions.csv
python fraud_detection.py     # trains both models and runs evaluation
```

**Output:**
- Console: dataset summary, per-model precision/recall/F1, ROC-AUC
- `results/evaluation_plots.png` — confusion matrix and ROC curve comparison
- `results/feature_importance.png` — ranked feature importances from the Random Forest

## Interpreting the results

- **Confusion matrix** — breaks predictions into true positives, false positives, and false negatives. False negatives (fraud predicted as legitimate) are the costliest error type in this domain, since that's the fraud that slips through.
- **ROC curve** — the closer the curve hugs the top-left corner, the better the model separates the two classes. An AUC of 1.0 is a perfect separator.
- **Feature importance** — in this dataset, `transaction_hour` and `distance_from_home_km` turned out to be the strongest predictors, consistent with the intuition that fraud tends to cluster at odd hours and away from a cardholder's usual location.

## Notes on the approach

**Why not just optimize for accuracy?** With a 98/2 class split, accuracy is close to meaningless — a model can score high while missing every fraud case. Precision, recall, and ROC-AUC give a much more honest picture of performance on the minority class.

**Why `class_weight="balanced"`?** Without it, both models would gravitate toward the majority class simply because it dominates the training data. This setting increases the penalty for misclassifying the minority (fraud) class during training.

**Random Forest vs. Logistic Regression.** Logistic Regression assumes a roughly linear relationship between features and outcome — good for a fast baseline. Random Forest, as an ensemble of trees, captures non-linear interactions between features (e.g., "large amount *and* odd hour *and* new account" matters more than any one feature alone), which is usually why it edges out the simpler model here.

**Limitations.** This is trained on synthetic data, so the results — while illustrative of the pipeline — shouldn't be read as a claim about real-world fraud rates. A production system would need real transaction history, additional signals (device fingerprinting, IP geolocation, merchant risk scores), and ongoing retraining as fraud patterns shift over time.

"""
generate_data.py
-----------------
Creates a synthetic (fake but realistic) dataset of digital payment /
banking transactions, similar in spirit to the well-known Kaggle
"Credit Card Fraud Detection" dataset.

Why synthetic data?
Real fraud datasets are usually private (banks don't share them) or
require a Kaggle account/API key to download. For an assignment, a
synthetic dataset with the SAME statistical properties (very few
fraud cases, unusual transaction patterns) works just as well to
demonstrate the whole Machine Learning pipeline.

Run this file once to create data/transactions.csv
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)

N_SAMPLES = 20000          # total transactions
FRAUD_RATIO = 0.015        # ~1.5% fraud -> realistic class imbalance

n_fraud = int(N_SAMPLES * FRAUD_RATIO)
n_normal = N_SAMPLES - n_fraud

# ---------- Normal (legit) transactions ----------
normal = pd.DataFrame({
    "amount": np.round(np.random.gamma(shape=2.0, scale=40, size=n_normal), 2),
    "transaction_hour": np.random.normal(loc=14, scale=4, size=n_normal).clip(0, 23).astype(int),
    "account_age_days": np.random.randint(30, 3000, size=n_normal),
    "num_transactions_last_24h": np.random.poisson(3, size=n_normal),
    "distance_from_home_km": np.abs(np.random.normal(5, 8, size=n_normal)),
    "is_foreign_transaction": np.random.choice([0, 1], size=n_normal, p=[0.95, 0.05]),
    "is_online": np.random.choice([0, 1], size=n_normal, p=[0.6, 0.4]),
    "class": 0
})

# ---------- Fraudulent transactions (different pattern) ----------
fraud = pd.DataFrame({
    "amount": np.round(np.random.gamma(shape=3.0, scale=150, size=n_fraud), 2),   # bigger amounts
    "transaction_hour": np.random.choice(range(0, 6), size=n_fraud),               # odd hours (late night)
    "account_age_days": np.random.randint(1, 400, size=n_fraud),                   # newer accounts
    "num_transactions_last_24h": np.random.poisson(9, size=n_fraud),               # burst of activity
    "distance_from_home_km": np.abs(np.random.normal(400, 300, size=n_fraud)),     # far from home
    "is_foreign_transaction": np.random.choice([0, 1], size=n_fraud, p=[0.3, 0.7]),
    "is_online": np.random.choice([0, 1], size=n_fraud, p=[0.15, 0.85]),
    "class": 1
})

data = pd.concat([normal, fraud], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle rows

os.makedirs("data", exist_ok=True)
data.to_csv("data/transactions.csv", index=False)

print(f"Dataset created: {len(data)} rows")
print(f"Fraud cases: {data['class'].sum()} ({data['class'].mean()*100:.2f}%)")
print("Saved to data/transactions.csv")

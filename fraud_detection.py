"""
fraud_detection.py
-------------------
A Fraud Detection System for digital payments and banking transactions.

Pipeline:
1. Load data
2. Explore data (basic stats)
3. Preprocess (scale features, split train/test)
4. Train two models: Logistic Regression & Random Forest
5. Evaluate using metrics that matter for imbalanced data
   (accuracy is misleading here -> we use precision, recall, F1, ROC-AUC)
6. Save a confusion matrix + ROC curve plot to results/

Run:
    python generate_data.py      # creates the dataset (run once)
    python fraud_detection.py    # trains & evaluates the models
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save files
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, roc_curve
)

os.makedirs("results", exist_ok=True)

# ---------- 1. Load data ----------
DATA_PATH = "data/transactions.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Run 'python generate_data.py' first."
    )

df = pd.read_csv(DATA_PATH)
print("=" * 55)
print("STEP 1: DATA LOADED")
print("=" * 55)
print(df.head())
print(f"\nShape: {df.shape}")
print(f"Fraud cases: {df['class'].sum()} out of {len(df)} "
      f"({df['class'].mean()*100:.2f}%)\n")

# ---------- 2. Features / target ----------
X = df.drop(columns=["class"])
y = df["class"]

# ---------- 3. Train/test split (stratify keeps the same fraud ratio in both sets) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------- 4. Scale features ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 5. Train models ----------
print("=" * 55)
print("STEP 2: TRAINING MODELS")
print("=" * 55)

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)

rf = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42, max_depth=10
)
rf.fit(X_train_scaled, y_train)

print("Both models trained: Logistic Regression, Random Forest\n")

# ---------- 6. Evaluate ----------
def evaluate(model, name):
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]

    print("-" * 55)
    print(f"MODEL: {name}")
    print("-" * 55)
    print(classification_report(y_test, preds, target_names=["Legit", "Fraud"]))
    auc = roc_auc_score(y_test, probs)
    print(f"ROC-AUC Score: {auc:.4f}\n")
    return preds, probs, auc

print("=" * 55)
print("STEP 3: EVALUATION")
print("=" * 55)
lr_preds, lr_probs, lr_auc = evaluate(log_reg, "Logistic Regression")
rf_preds, rf_probs, rf_auc = evaluate(rf, "Random Forest")

# ---------- 7. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix for the better model (Random Forest usually wins)
best_preds = rf_preds if rf_auc >= lr_auc else lr_preds
best_name = "Random Forest" if rf_auc >= lr_auc else "Logistic Regression"
cm = confusion_matrix(y_test, best_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title(f"Confusion Matrix - {best_name}")

# ROC curve comparison
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
axes[1].plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={lr_auc:.3f})")
axes[1].plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={rf_auc:.3f})")
axes[1].plot([0, 1], [0, 1], "k--", alpha=0.4)
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Curve Comparison")
axes[1].legend()

plt.tight_layout()
plt.savefig("results/evaluation_plots.png", dpi=150)
print("Saved plots to results/evaluation_plots.png")

# ---------- 8. Feature importance (Random Forest) ----------
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importance (Random Forest):")
print(importances)

plt.figure(figsize=(7, 4))
importances.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Which features matter most for detecting fraud?")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150)
print("Saved plot to results/feature_importance.png")

print("\nDONE. All results saved in the 'results/' folder.")

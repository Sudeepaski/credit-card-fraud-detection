"""
Credit Card Fraud Detection
Models:
1. Logistic Regression (with SMOTE)
2. Random Forest (without SMOTE, class_weight)
3. XGBoost (with and without SMOTE)
"""

# -----------------------------
# Imports
# -----------------------------
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------
DATA_PATH = "data/creditcard.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "Dataset not found. Please download creditcard.csv from Kaggle "
        "and place it inside the data/ folder."
    )

df = pd.read_csv(DATA_PATH)

# -----------------------------
# Prepare Data
# -----------------------------
X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# =====================================================
# 1️⃣ Logistic Regression (WITH SMOTE)
# =====================================================
print("\n--- Logistic Regression (WITH SMOTE) ---")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

lr = LogisticRegression(
    solver="liblinear",
    max_iter=5000,
    random_state=42
)

lr.fit(X_train_smote, y_train_smote)
y_pred_lr = lr.predict(X_test)

print(classification_report(y_test, y_pred_lr))
print("Precision:", precision_score(y_test, y_pred_lr))
print("Recall:", recall_score(y_test, y_pred_lr))
print("F1 Score:", f1_score(y_test, y_pred_lr))

# =====================================================
# 2️⃣ Random Forest (WITH SMOTE)
# =====================================================
print("\n--- Random Forest (WITH SMOTE) ---")

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_smote = RandomForestClassifier(
    n_estimators=100,          # keep smaller to reduce time
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

rf_smote.fit(X_train_smote, y_train_smote)

y_pred_rf_smote = rf_smote.predict(X_test)

print("Random Forest WITH SMOTE Results")
print(classification_report(y_test, y_pred_rf_smote))

print("Precision:", precision_score(y_test, y_pred_rf_smote))
print("Recall:", recall_score(y_test, y_pred_rf_smote))
print("F1 Score:", f1_score(y_test, y_pred_rf_smote))


# =====================================================
# 2️⃣ Random Forest (WITHOUT SMOTE)
# =====================================================
print("\n--- Random Forest (WITHOUT SMOTE) ---")

rf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print(classification_report(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf))
print("Recall:", recall_score(y_test, y_pred_rf))
print("F1 Score:", f1_score(y_test, y_pred_rf))

# =====================================================
# 3️⃣ XGBoost (WITH SMOTE)
# =====================================================
print("\n--- XGBoost (WITH SMOTE) ---")

pos_weight_smote = len(y_train_smote[y_train_smote == 0]) / len(
    y_train_smote[y_train_smote == 1]
)

xgb_smote = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight_smote,
    random_state=42,
    eval_metric="logloss"
)

xgb_smote.fit(X_train_smote, y_train_smote)
y_pred_xgb_smote = xgb_smote.predict(X_test)

print(classification_report(y_test, y_pred_xgb_smote))
print("Precision:", precision_score(y_test, y_pred_xgb_smote))
print("Recall:", recall_score(y_test, y_pred_xgb_smote))
print("F1 Score:", f1_score(y_test, y_pred_xgb_smote))

# =====================================================
# 3️⃣ XGBoost (WITHOUT SMOTE)
# =====================================================
print("\n--- XGBoost (WITHOUT SMOTE) ---")

pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_final = XGBClassifier(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric="logloss"
)

xgb_final.fit(X_train, y_train)
y_pred_xgb_final = xgb_final.predict(X_test)

print(classification_report(y_test, y_pred_xgb_final))
print("Precision:", precision_score(y_test, y_pred_xgb_final))
print("Recall:", recall_score(y_test, y_pred_xgb_final))
print("F1 Score:", f1_score(y_test, y_pred_xgb_final))

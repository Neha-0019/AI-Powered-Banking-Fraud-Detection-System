"""
Standalone script to train models without opening the notebook.
Run: python scripts/train_models.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATASET_PATH = os.path.join(ROOT, "dataset", "creditcard.csv")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Run: python scripts/generate_sample_data.py")
        print("Or download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount_scaled", "Time"]
    X = df[feature_cols]
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train: {len(X_train)}, Test: {len(X_test)}, Fraud %: {y.mean()*100:.2f}%")

    # Isolation Forest
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=min(0.01, y.mean() * 2), random_state=42)
    iso.fit(X_train)
    iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
    print(f"  Accuracy: {accuracy_score(y_test, iso_pred):.4f}, F1: {f1_score(y_test, iso_pred, zero_division=0):.4f}")

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}, F1: {f1_score(y_test, rf_pred, zero_division=0):.4f}")

    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "amount_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    print(f"\nModels saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()

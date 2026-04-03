"""
Training script for IEEE-CIS Fraud Detection dataset.
Handles large files by reading a subset and performing basic feature engineering.
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATASET_DIR = os.path.join(ROOT, "dataset")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# Define features to use
NUM_COLS = ['TransactionAmt', 'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'C1', 'C2', 'C13', 'C14', 'D1', 'D10']
CAT_COLS = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'DeviceType']

def main():
    trans_path = os.path.join(DATASET_DIR, "train_transaction.csv")
    id_path = os.path.join(DATASET_DIR, "train_identity.csv")

    if not os.path.exists(trans_path):
        print(f"Error: {trans_path} not found.")
        return

    print("Loading datasets (first 100,000 rows for speed and memory)...")
    train_trans = pd.read_csv(trans_path, nrows=100000)
    
    if os.path.exists(id_path):
        train_id = pd.read_csv(id_path, nrows=100000)
        df = pd.merge(train_trans, train_id, on='TransactionID', how='left')
    else:
        print("Identity file not found, training with transaction data only.")
        df = train_trans

    # Target
    y = df['isFraud']
    
    # Feature selection
    cols_to_use = NUM_COLS + CAT_COLS
    # Ensure columns exist in df
    available_cols = [c for c in cols_to_use if c in df.columns]
    X = df[available_cols].copy()

    print(f"Processing {len(X)} rows with {len(available_cols)} features...")

    # Handle Missing Values
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].fillna('Unknown')
        else:
            X[col] = X[col].fillna(X[col].median())

    # Label Encoding for Categorical
    encoders = {}
    for col in CAT_COLS:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}, Fraud ratio: {y.mean()*100:.2f}%")

    # Isolation Forest
    print("\nTraining Isolation Forest...")
    iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    iso.fit(X_train)
    iso_pred = np.where(iso.predict(X_test) == -1, 1, 0)
    print(f"  Accuracy: {accuracy_score(y_test, iso_pred):.4f}, F1: {f1_score(y_test, iso_pred):.4f}")

    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"  Accuracy: {accuracy_score(y_test, rf_pred):.4f}, F1: {f1_score(y_test, rf_pred):.4f}")

    # Save everything
    joblib.dump(iso, os.path.join(MODELS_DIR, "isolation_forest.pkl"))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(encoders, os.path.join(MODELS_DIR, "encoders.pkl"))
    joblib.dump(available_cols, os.path.join(MODELS_DIR, "feature_columns.pkl"))
    
    print(f"\nIEEE-CIS Models and metadata saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()

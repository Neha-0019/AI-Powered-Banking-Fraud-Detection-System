"""
AI-Powered Banking Fraud Detection Engine
=========================================
Detects fraudulent transactions using Isolation Forest and Random Forest models.
"""

import os
import numpy as np
import pandas as pd
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Model files
ISOLATION_FOREST_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
RANDOM_FOREST_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
FEATURE_COLS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")


def _load_models():
    """Load trained models and scaler. Returns (iso_forest, rf_model, scaler, encoders, feature_cols)."""
    if not os.path.exists(ISOLATION_FOREST_PATH):
        raise FileNotFoundError(
            "Models not found. Please run: scripts/train_ieee_models.py"
        )
    iso_forest = joblib.load(ISOLATION_FOREST_PATH)
    rf_model = joblib.load(RANDOM_FOREST_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    return iso_forest, rf_model, scaler, encoders, feature_cols


def _prepare_transaction(transaction_data: dict, feature_cols: list, scaler, encoders) -> pd.DataFrame:
    """
    Convert transaction dict/row into feature vector for prediction.
    """
    if isinstance(transaction_data, pd.DataFrame):
        row = transaction_data.iloc[0].copy()
    elif isinstance(transaction_data, pd.Series):
        row = transaction_data.copy()
    else:
        row = pd.Series(transaction_data).copy()

    # Build feature vector in correct order
    features = {}
    for col in feature_cols:
        val = row.get(col, np.nan)
        
        # Handle Categorical
        if col in encoders:
            le = encoders[col]
            val_str = str(val) if pd.notnull(val) else "Unknown"
            try:
                # Map to existing label or 'Unknown' equivalent
                if val_str in le.classes_:
                    features[col] = [le.transform([val_str])[0]]
                else:
                    # If unknown label, use first class as fallback or median
                    features[col] = [0] 
            except:
                features[col] = [0]
        else:
            # Handle Numerical
            features[col] = [float(val) if pd.notnull(val) else 0.0]

    X = pd.DataFrame(features)
    # Scale numerical features
    X_scaled = scaler.transform(X)
    return X_scaled


def predict_transaction(transaction_data, model="ensemble"):
    """
    Predict whether a transaction is fraudulent.
    """
    iso_forest, rf_model, scaler, encoders, feature_cols = _load_models()
    X = _prepare_transaction(transaction_data, feature_cols, scaler, encoders)

    result = {"prediction": "Legitimate", "fraud_probability": 0.0, "is_fraud": False, "model_used": model}

    if model == "isolation_forest":
        pred = iso_forest.predict(X)[0]
        scores = -iso_forest.score_samples(X)
        # Normalize score to 0-1 (approximate)
        prob = float(np.clip((scores[0] + 0.5), 0, 1)) 
        result["fraud_probability"] = prob
        result["is_fraud"] = pred == -1
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"

    elif model == "random_forest":
        prob = float(rf_model.predict_proba(X)[0, 1])
        result["fraud_probability"] = prob
        result["is_fraud"] = rf_model.predict(X)[0] == 1
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"

    else:  # ensemble
        # Isolation Forest prob
        iso_scores = -iso_forest.score_samples(X)
        iso_prob = float(np.clip((iso_scores[0] + 0.5), 0, 1))

        # Random Forest prob
        rf_prob = float(rf_model.predict_proba(X)[0, 1])

        # Average probability
        prob = (iso_prob + rf_prob) / 2
        result["fraud_probability"] = prob
        result["is_fraud"] = prob >= 0.5
        result["prediction"] = "Fraud" if result["is_fraud"] else "Legitimate"
        result["model_used"] = "ensemble"

    return result


# --- Example Usage ---
if __name__ == "__main__":
    # Sample IEEE-CIS transaction
    sample = {
        'TransactionAmt': 50.0,
        'ProductCD': 'W',
        'card1': 1000,
        'card2': 500,
        'card3': 150,
        'card4': 'visa',
        'card5': 226,
        'card6': 'debit',
        'P_emaildomain': 'gmail.com',
        'DeviceType': 'mobile',
        'C1': 1, 'C2': 1, 'C13': 1, 'C14': 1, 'D1': 0, 'D10': 0
    }

    try:
        out = predict_transaction(sample)
        print("Example IEEE-CIS prediction:")
        print(f"  Prediction: {out['prediction']}")
        print(f"  Fraud Probability: {out['fraud_probability']:.4f}")
        print(f"  Model: {out['model_used']}")
    except Exception as e:
        print(f"Error: {e}")

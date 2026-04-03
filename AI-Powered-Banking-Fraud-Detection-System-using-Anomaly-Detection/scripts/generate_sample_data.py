"""
Generate a small synthetic dataset for testing when Kaggle dataset is unavailable.
Use the real Kaggle dataset for actual training and evaluation.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)
OUTPUT_PATH = os.path.join(DATASET_DIR, "creditcard.csv")

def main():
    n = 10000  # Small sample for quick testing
    time = np.cumsum(np.random.exponential(500, n))
    amount = np.random.exponential(88, n)
    v_cols = {f"V{i}": np.random.randn(n) * 2 for i in range(1, 29)}

    # ~0.17% fraud
    fraud_idx = np.random.choice(n, size=int(n * 0.0017), replace=False)
    for i in fraud_idx:
        v_cols["V14"][i] += np.random.randn() * 3
        v_cols["V17"][i] += np.random.randn() * 2

    df = pd.DataFrame({"Time": time, "Amount": amount, **v_cols})
    df["Class"] = 0
    df.loc[fraud_idx, "Class"] = 1

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {OUTPUT_PATH} with {len(df)} rows, {df['Class'].sum()} fraud cases")

if __name__ == "__main__":
    main()

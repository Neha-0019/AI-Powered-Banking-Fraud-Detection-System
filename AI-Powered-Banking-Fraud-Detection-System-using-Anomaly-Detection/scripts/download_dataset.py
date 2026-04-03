"""
Download Credit Card Fraud Detection dataset from Kaggle.
Requires: pip install kaggle
Setup: Place kaggle.json (API credentials) in ~/.kaggle/
"""

import os
import subprocess
import sys

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

def main():
    try:
        subprocess.run([
            sys.executable, "-m", "kaggle", "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", DATASET_DIR,
            "--unzip"
        ], check=True)
        print(f"Dataset downloaded to {DATASET_DIR}")
    except subprocess.CalledProcessError:
        print("Kaggle CLI failed. Manual steps:")
        print("1. Install: pip install kaggle")
        print("2. Get API key from https://www.kaggle.com/settings")
        print("3. Place kaggle.json in ~/.kaggle/")
        print("4. Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"5. Extract creditcard.csv to {DATASET_DIR}")

if __name__ == "__main__":
    main()

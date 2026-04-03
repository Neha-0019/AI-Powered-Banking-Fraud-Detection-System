# 🛡️ AI-Powered Banking Fraud Detection System

A complete **Data Science mini-project** with AI/ML integration for detecting fraudulent banking transactions using anomaly detection and supervised learning. Suitable for **3rd year B.Tech CS portfolio** and academic submissions.

---

## 📋 Table of Contents

- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Project Report](#project-report)
- [Conclusion](#conclusion)

---

## Introduction

Financial fraud is a critical challenge for banks and payment providers. This project builds an **intelligent fraud detection system** that:

- Analyzes transaction patterns in real time
- Identifies suspicious behaviour using **Isolation Forest** and **Random Forest**
- Provides fraud probability scores
- Offers an interactive Streamlit dashboard for demos

---

## Dataset Description

**Source:** [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

| Column  | Description                                      |
|---------|--------------------------------------------------|
| Time    | Seconds elapsed between transactions             |
| V1–V28  | PCA-transformed features (anonymized)            |
| Amount  | Transaction amount                               |
| Class   | 0 = Normal, 1 = Fraud                            |

- **Rows:** ~284,000
- **Class imbalance:** ~0.17% fraud
- **Preprocessing:** Amount scaling, no missing values

---

## Project Structure

```
fraud_detection_ai/
│
├── dataset/
│   └── creditcard.csv          # Place dataset here
│
├── notebooks/
│   └── fraud_analysis.ipynb    # EDA + ML pipeline
│
├── models/
│   ├── isolation_forest.pkl
│   ├── random_forest.pkl
│   ├── amount_scaler.pkl
│   └── feature_columns.pkl
│
├── app/
│   └── streamlit_app.py        # Interactive dashboard
│
├── scripts/
│   └── download_dataset.py     # Kaggle download helper
│
├── visuals/                    # Saved charts
│
├── fraud_detection.py          # Prediction engine
├── requirements.txt
└── README.md
```

---

## Setup & Installation

### 1. Clone / Download the project

```bash
cd fraud_detection_ai
```

### 2. Create virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

**Option A – Kaggle CLI (recommended):**
```bash
pip install kaggle
# Place kaggle.json in ~/.kaggle/
python scripts/download_dataset.py
```

**Option B – Manual:**
1. Go to [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
2. Download and extract `creditcard.csv` to `dataset/`

**Option C – Synthetic sample (for quick testing):**
```bash
python scripts/generate_sample_data.py
```

### 5. Train models (run notebook)

```bash
jupyter notebook notebooks/fraud_analysis.ipynb
```

Run all cells to train models and save them in `models/`.

---

## Methodology

### 1. Data Preprocessing
- Load and inspect data
- Normalize `Amount` with StandardScaler
- Handle class imbalance (stratified split, class weights)

### 2. Exploratory Data Analysis
- Fraud vs Non-Fraud distribution
- Transaction amount histogram
- Correlation heatmap
- Fraud over time
- Feature importance

### 3. Machine Learning Pipeline
- **Isolation Forest:** anomaly detection (unsupervised)
- **Random Forest:** classification with `class_weight='balanced'`

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix

---

## Results

- **Isolation Forest:** Good at flagging anomalies without labels
- **Random Forest:** High precision/recall when trained on balanced data
- **Ensemble:** Combines both for robust predictions

---

## Usage

### Prediction engine (Python)

```python
from fraud_detection import predict_transaction

transaction = {
    "Time": 0,
    "Amount": 100.0,
    "V1": -1.36, "V2": -0.07, "V3": 2.54, "V4": 1.38, "V5": -0.34,
    "V6": 0.46, "V7": 0.24, "V8": 0.10, "V9": 0.36, "V10": -0.02,
    "V11": 0.28, "V12": -0.11, "V13": 0.07, "V14": 0.13, "V15": -0.19,
    "V16": 0.13, "V17": -0.02, "V18": 0.04, "V19": 0.26, "V20": -0.07,
    "V21": -0.13, "V22": -0.05, "V23": 0.07, "V24": 0.05, "V25": -0.06,
    "V26": -0.07, "V27": 0.09, "V28": -0.05,
}

result = predict_transaction(transaction, model="ensemble")
print(result["prediction"])          # "Fraud" or "Legitimate"
print(result["fraud_probability"])   # 0.0 to 1.0
```

### Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

---

## Project Report

### Introduction
This project uses ML to detect fraudulent credit card transactions from anonymized PCA features, time, and amount.

### Dataset
- Credit Card Fraud Detection dataset with ~284k rows and 31 columns.

### Methodology
- Preprocessing: scaling, stratified split
- EDA: distributions, correlations, fraud over time
- Models: Isolation Forest (anomaly) + Random Forest (supervised)
- Ensemble: average of both probabilities

### Results
- Both models flag fraud with reasonable precision/recall.
- Ensemble reduces variance and improves robustness.

### Conclusion
AI-based fraud detection can automate screening of transactions, reduce manual review, and lower fraud risk. Isolation Forest and Random Forest are effective for this task, especially in highly imbalanced settings.

---

## Conclusion

This system shows how **AI/ML can improve financial fraud detection** by:

1. **Automation:** Real-time scoring instead of manual review  
2. **Scalability:** Processing large transaction volumes  
3. **Accuracy:** Learning complex patterns from data  
4. **Flexibility:** Easy to retrain with new data  

Future improvements: SMOTE/oversampling, XGBoost, deep learning, and integration with live transaction streams.

---

## Tech Stack

- **Python 3.10+**
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Plotly
- **Models:** Isolation Forest, Random Forest
- **UI:** Streamlit

---

**Author:** B.Tech CS Student • **Use case:** Portfolio & Academic Project

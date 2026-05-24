# Credit Card Fraud Detection

A complete machine learning pipeline for detecting fraudulent credit card transactions, covering data preprocessing, class imbalance handling, classical classifiers, and neural networks.

---

## Overview

Credit card fraud detection is a canonical imbalanced classification problem: only **0.17%** of transactions are fraudulent. Standard accuracy metrics are misleading in this setting — a model that always predicts "no fraud" achieves 99.83% accuracy while catching nothing. This project addresses that challenge through two complementary resampling strategies and a systematic model comparison.

---

## Dataset

The project uses the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (`creditcard.csv`).

| Property | Value |
|---|---|
| Total transactions | 284,807 |
| Fraudulent | 492 (0.17%) |
| Features | V1–V28 (PCA-transformed), Amount, Time |
| Target | `Class` (0 = No Fraud, 1 = Fraud) |

> Features V1–V28 have already been anonymised via PCA. `Amount` and `Time` are in their original units and are scaled in preprocessing.

---

## Pipeline

### 1. Imports
All dependencies loaded: NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, imbalanced-learn, and TensorFlow/Keras.

### 2. Exploratory Data Analysis
- Shape, null-value check, class distribution
- Visualisation of the severe class imbalance

### 3. Scaling
`Amount` and `Time` are rescaled with **RobustScaler** (uses median + IQR, resistant to extreme transaction outliers). V1–V28 are left as-is (already PCA-scaled).

### 4. Stratified Split
The original dataset is split into train/test using `StratifiedKFold` **before any resampling**, preserving the real class ratio. The test set is never modified and is used only for final evaluation.

### 5. Random Undersampling
A balanced 50/50 subsample is created for training classical classifiers:
- All 492 fraud cases retained
- 492 random non-fraud cases kept
- Result: 984 samples, used for training only

### 6. Correlation Analysis
Correlation matrices are computed on the balanced dataset (the imbalanced version distorts correlations). Key features identified:
- **Negatively correlated with fraud:** V17, V14, V12, V10
- **Positively correlated with fraud:** V2, V4, V11, V19

### 7. Outlier Removal
Extreme outliers in V14, V12, and V10 are removed using the **IQR method** (threshold: 1.5×IQR), improving downstream model accuracy by ~3%.

### 8. Classical Classifiers + GridSearchCV
Four classifiers are tuned and evaluated on the undersampled data:

| Classifier | Tuned with |
|---|---|
| Logistic Regression | GridSearchCV (penalty, C) |
| K-Nearest Neighbors | GridSearchCV (n_neighbors, algorithm) |
| Support Vector Classifier | GridSearchCV (C, kernel) |
| Decision Tree | GridSearchCV (criterion, max_depth, min_samples_leaf) |

### 9. SMOTE Oversampling
Instead of discarding data, **SMOTE** generates synthetic minority-class examples by interpolating between existing fraud cases.

> ⚠️ SMOTE is applied **inside** a cross-validation pipeline (not before splitting) to prevent data leakage and inflated metrics.

### 10. Neural Networks
Two identical feedforward networks are trained and compared:

```
Input layer  (30 neurons, ReLU)
      ↓
Hidden layer (32 neurons, ReLU)
      ↓
Output layer (2 neurons, Softmax)
```

| | Model A | Model B |
|---|---|---|
| Training data | Undersampled (984 rows) | SMOTE (~280k balanced rows) |
| Optimizer | Adam (lr=0.001) | Adam (lr=0.001) |
| Loss | sparse_categorical_crossentropy | sparse_categorical_crossentropy |
| Epochs | 20 | 20 |

Both models are evaluated on the **original, untouched test set**.

---

## Key Results

| Metric | What it means here |
|---|---|
| **Recall** | Fraction of real frauds caught — the primary metric |
| **Precision** | Fraction of fraud alerts that are actually fraud |
| **F1** | Harmonic mean of Precision and Recall |
| **Accuracy** | Largely meaningless due to class imbalance |

The SMOTE-trained neural network generally achieves higher Recall with fewer false alarms than the undersampling-trained model.

---

## Core Trade-off

```
Higher Recall   →  catch more fraud  →  more false alarms (blocked legitimate cards)
Higher Precision →  fewer false alarms →  miss more real fraud
```

The right balance depends on business context. A bank may prefer maximum Recall (catch all fraud) even at the cost of some false positives.

---

## Key Lessons

| Topic | Key Point |
|---|---|
| Imbalance | 0.17% fraud → accuracy is meaningless; use Recall + F1 |
| Undersampling | Simple and fast, but discards 99%+ of legitimate data |
| SMOTE | Keeps all real data; synthetic examples generally improve results |
| Split first | Always split before resampling; test only on original data |
| SMOTE in CV | Apply SMOTE inside the pipeline — never before the split |

---

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
tensorflow
```

Install with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow
```

---

## Usage

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `/content/creditcard.csv` (or update the path in Part 2).
2. Open `credit_fraud_detection.ipynb` in Jupyter or Google Colab.
3. Run all cells sequentially — each part builds on the previous one.

---

## Project Structure

```
credit_fraud_detection.ipynb   # Main notebook (all 13 parts)
creditcard.csv                 # Dataset (download separately from Kaggle)
README.md                      # This file
```

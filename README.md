# 📊 Stock Return Direction Prediction

This repository contains a predictive modeling project aimed at **forecasting the next-month return direction of stocks** (Up/Down) using CRSP data and machine learning techniques.

---

## 1. Project Overview

- **Objective:** Predict whether a stock's next-month return is positive (Up) or negative (Down).  
- **Task Type:** Cross-sectional classification.  
- **Models Implemented:**
  - Baseline (Majority Class)
  - Logistic Regression
  - Random Forest
  - Neural Network

The project evaluates the performance of linear vs. non-linear models and analyzes feature importance for interpretability.

---

## 2. Data

### 2.1 Source

- **CRSP Monthly Stock File (MSF)** via **WRDS**
- Period: Jan 2018 – Dec 2024
- Sample: 50,000 records initially (after filtering: 12,929 rows)
- Variables:
  - `ret`: Monthly return
  - `prc`: Stock price
  - `vol`: Trading volume
  - `shrout`: Shares outstanding
- Additional mock fundamentals for demonstration: `roe` and `pb`.

### 2.2 Features

| Feature | Description |
|---------|-------------|
| momentum | Previous month return |
| volatility | 12-month rolling standard deviation of returns |
| ma_gap | Price vs. 20-month moving average gap |
| vol_change | Monthly volume change |
| log_mcap | Logarithm of market capitalization |
| roe_norm | Cross-sectional normalized ROE |
| pb | Price-to-book ratio |

- Target (`label`): 1 if next month's return > 0, 0 otherwise.
- Data split: **80% train / 20% test**
- StandardScaler applied to all features.

---

## 3. Modeling Approach

### 3.1 Baseline

- Predicts the majority class from the training set
- Serves as a benchmark for model comparison

### 3.2 Logistic Regression

- Linear classifier
- Provides interpretable coefficients
- Useful for understanding feature directions

### 3.3 Random Forest

- Non-linear ensemble model
- Captures feature interactions
- Provides feature importance metrics

### 3.4 Neural Network

- Two hidden layers (100 units each) with ReLU activation
- Cross-Entropy Loss optimized via SGD with momentum
- Tests non-linear predictive power

---

## 4. Results

### 4.1 Accuracy

| Model | Accuracy |
|-------|---------|
| Baseline | 0.560 |
| Logistic Regression | 0.560 |
| Random Forest | 0.569 |
| Neural Network | 0.558 |

- Random Forest slightly outperforms other models.
- Overall prediction accuracy is modest, reflecting the difficulty of forecasting stock returns.

### 4.2 Feature Importance

- **Random Forest:** Most important features are `volatility`, `log_mcap`, and `momentum`.
- **Permutation Importance:** `volatility` and `momentum` have the highest impact on prediction accuracy.
- **Logistic Coefficients:** Positive coefficients for `log_mcap` and `momentum`; negative coefficient for `volatility`.

### 4.3 Confusion Matrices

- RF, NN, and Logistic Regression mostly predict "Up" correctly.
- False positives for "Up" exist, consistent with noisy financial data.

### 4.4 Visualization

![Model Performance and Feature Importance](./figures/0aec1a1b-db42-44ea-beb1-cc80b81ecafc.png)

---

## 5. Conclusions

- Predicting next-month stock returns is **challenging**; linear and non-linear models yield modest gains over baseline.
- Momentum and volatility-related features are consistently the most predictive.
- Non-linear models (Random Forest) provide slight improvement over logistic regression.
- The project demonstrates the integration of **financial features**, **machine learning models**, and **model interpretability**.

### 5.1 Next Steps

- Integrate real Compustat fundamentals for richer feature sets
- Implement rolling-window cross-validation
- Explore additional models (e.g., XGBoost, LightGBM)
- Conduct portfolio-level backtesting for economic significance

---

## 6. Repository Structure


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

<img width="6000" height="3000" alt="stock_prediction_results" src="https://github.com/user-attachments/assets/cb75b05f-ca98-4993-bd88-64bc59732cd8" />

### 4.5 Results and Interpretation

### Figure 1. Model Accuracy Comparison
![part_1](https://github.com/user-attachments/assets/15b970bb-8572-4bc6-beb8-6eab55652b79)

**Results**
- All models achieve test accuracy in the range of 55%–57%.
- Random Forest records the highest accuracy among the four models.
- Logistic Regression and Neural Network perform similarly to the baseline.

**Interpretation**
- The narrow accuracy range indicates limited predictability of next-month stock returns.
- Non-linear models provide only marginal improvement over linear benchmarks.
- Model choice has a smaller impact than the intrinsic noise in return dynamics.

### Figure 2. Random Forest Feature Importance
![part_2](https://github.com/user-attachments/assets/eb1f6e84-aa76-47cf-a561-c6a54f21dd44)

**Results**
- Volatility is the most important feature in the Random Forest model.
- Log market capitalization and momentum follow in importance.
- ROE and price-to-book ratio rank among the least important features.

**Interpretation**
- The model relies primarily on risk-related and trading-based characteristics.
- Market capitalization captures cross-sectional heterogeneity beyond pure returns.
- Fundamental proxies contribute limited information in this setting.

### Figure 3. Permutation Importance
![part_3](https://github.com/user-attachments/assets/450bc987-0b9b-4363-b468-2880d6711197)

**Results**
- Permuting volatility and momentum leads to the largest drop in prediction accuracy.
- Log market capitalization remains positively important but weaker than the top two features.
- Permutation importance for ROE and PB is low and partially negative.

**Interpretation**
- Volatility and momentum provide robust marginal contributions to model performance.
- Fundamental variables contain limited predictive information, potentially because:
  - Simplified or mock fundamentals are used;
  - The monthly prediction horizon is unfavorable for fundamentals.
- Compared with Random Forest internal importance, permutation importance better reflects true predictive dependence.

### Figure 4. Logistic Regression Coefficients
![part_4](https://github.com/user-attachments/assets/889d6eda-eee3-4050-bf0b-a7e124dc480d)

**Results**
- Momentum and log market capitalization have positive coefficients.
- Volatility has a negative coefficient.
- Coefficients for ROE, PB, and MA gap are close to zero.

**Interpretation**
- Higher recent returns increase the probability of positive next-month returns.
- Higher volatility is associated with a lower likelihood of positive returns.
- Linear relationships between most features and return direction are weak.

### Figure 5. Random Forest Confusion Matrix
![part_5](https://github.com/user-attachments/assets/a5154f76-d3cc-49ec-85e2-4e0f5082e7b3)

**Results**
- The model correctly predicts a large number of positive-return observations.
- Misclassification of negative returns is relatively frequent.
- False positives exceed false negatives.

**Interpretation**
- The model exhibits a bias toward predicting positive returns.
- Class imbalance and asymmetric signal strength contribute to this pattern.
- Downside risk is more difficult to capture than upside movements.

### Figure 6. Neural Network Confusion Matrix
![part_6](https://github.com/user-attachments/assets/99979561-9a41-4aa3-8af4-e21c0466e7c4)

**Results**
- The Neural Network predicts positive returns in the majority of cases.
- Correct classification of negative returns is rare.
- Overall accuracy remains comparable to other models.

**Interpretation**
- Model complexity does not translate into improved discrimination.
- Weak predictive signals limit the effectiveness of high-capacity models.
- The network amplifies class imbalance rather than correcting it.

### Figure 7. Logistic Regression Confusion Matrix
![part_7](https://github.com/user-attachments/assets/6c6f62f9-0a53-4360-9de4-ebfe0e322ad2)

**Results**
- Predictions are more evenly distributed across classes compared to RF and NN.
- Both false positives and false negatives remain substantial.
- Overall accuracy is close to the baseline.

**Interpretation**
- Logistic Regression provides more stable and balanced predictions.
- Predictive power remains limited under a linear specification.
- Interpretability comes at the cost of performance gains.

In the complex task of predicting next-month stock return direction, all models achieve only slightly better than random accuracy (55%-57%), indicating extremely low predictability of returns at this horizon. The performance gains from model selection are far outweighed by inherent market noise. The effective predictive information concentrates mainly on volatility, momentum, and market capitalization—features related to technical factors and risk—while fundamental indicators such as ROE and book-to-market ratio contribute minimally, possibly due to simplified proxy variables or the unsuitability of fundamental logic for monthly horizons. Although nonlinear models (e.g., random forests) perform marginally better, both they and neural networks exhibit significant bias toward predicting the "positive return" class, suggesting class imbalance and the greater difficulty in capturing downside risk compared to upside opportunity. Overall, this prediction scenario is constrained by high noise and weak signals, with the robust patterns extracted by models mainly reflecting risk and trading behavior characteristics.

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


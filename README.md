# Stock Return Direction Prediction

## 1. Introduction

This project studies whether the **direction of next-month stock returns** can be predicted using firm-level characteristics constructed from CRSP data. The task is formulated as a **cross-sectional binary classification problem**, where each stock-month observation is used to predict whether the following month’s return is positive or negative.

Rather than aiming to design a trading strategy, the focus is on evaluating **predictive content**, comparing model performance, and understanding **which features models rely on when forming predictions**.

---

## 2. Data

### 2.1 Data Source

The dataset is obtained from the **CRSP Monthly Stock File (MSF)** via **WRDS**, covering the period from **January 2018 to December 2024**. Each observation represents a stock–month pair identified by a CRSP permanent number and a month-end date.

After data cleaning and feature construction, the final dataset contains **12,929 observations** and is stored as:


The dataset is not rendered directly in this README but can be fully reproduced using the provided code.

---

### 2.2 Raw Variables Collected

The following variables are directly collected from CRSP before any transformation or feature engineering:

| Variable | Description |
|--------|------------|
| permno | CRSP permanent stock identifier |
| date | Month-end date |
| ret | Monthly stock return |
| prc | Month-end stock price |
| vol | Monthly trading volume |
| shrout | Shares outstanding |

These variables form the basis for all constructed predictive features.

---

### 2.3 Data Preparation and Feature Engineering

Using the raw CRSP variables, several lagged features are constructed to capture recent performance, risk, liquidity, and firm size. All features rely only on information available at or before time *t*.

| Feature | Interpretation | Construction |
|------|---------------|-------------|
| momentum | Recent return trend | Previous month return |
| volatility | Return risk | 12-month rolling std of returns (lagged) |
| ma_gap | Price deviation | (Price − 20-month MA) / MA |
| vol_change | Liquidity change | Monthly % change in volume |
| log_mcap | Firm size | log(abs(price) × shares outstanding) |
| roe_norm | Relative profitability | ROE normalized cross-sectionally by date |
| pb | Valuation proxy | Price-to-book ratio |

After feature construction:
- Observations with missing or infinite values are removed
- Features are standardized using **StandardScaler**
- The dataset is split into **80% training and 20% testing**

---

### 2.4 Exploratory Data Analysis (EDA)

Exploratory analysis is conducted to understand the structure and limitations of the data prior to modeling.

Key observations include:
- The return direction label is **mildly imbalanced**, with positive returns occurring more frequently.
- Momentum and volatility display heavy tails and substantial dispersion.
- Fundamental proxies (ROE and PB) show weak univariate relationships with next-month returns.

<img width="1800" height="1200" alt="figure_E1_return_direction" src="https://github.com/user-attachments/assets/156aeb9e-057b-40e4-8e02-09fed0f2ef0e" />
 
<img width="3600" height="1200" alt="figure_E2_feature_distributions" src="https://github.com/user-attachments/assets/c711ee8b-8ab8-4cab-b353-15697b6854f0" />

<img width="3600" height="1200" alt="figure_E3_fundamentals_NextReturn" src="https://github.com/user-attachments/assets/9c8081a0-e92b-4d05-8cb6-330d6220b650" />


These patterns suggest that any predictive signal is likely to be weak and noisy, motivating comparison against a simple benchmark.

---

## 3. Models and Methodology

### 3.1 Prediction Framework and Baseline Reference

The task is framed as a **cross-sectional binary classification problem**. Model performance is evaluated using **out-of-sample accuracy** and confusion matrices.

A **baseline accuracy**, defined as the majority-class frequency in the training set, is used as a reference point. The baseline is **not treated as a predictive model**, but rather as a property of the data and a benchmark for evaluating model performance.

---

### 3.2 Logistic Regression

Logistic Regression serves as a linear benchmark model. It estimates a parametric relationship between features and the probability of a positive next-month return.

The primary advantage of this model is interpretability: coefficient signs and magnitudes provide direct insight into feature–outcome relationships under a linear specification.

---

### 3.3 Random Forest

The Random Forest model captures nonlinear relationships and interactions between features through an ensemble of decision trees.

This model is well-suited for detecting complex patterns in the data and provides built-in feature importance measures based on impurity reduction.

---

### 3.4 Neural Network

A feedforward Neural Network is implemented to test whether increased model flexibility improves predictive performance.

The architecture consists of:
- An input layer with 7 standardized features
- Two hidden layers with 100 neurons each
- ReLU activation functions
- An output layer with two nodes representing return direction classes

The network is trained using:
- Cross-entropy loss
- Stochastic Gradient Descent (SGD) with momentum
- Fixed learning rate and number of training epochs

This model serves as a test of whether high-capacity nonlinear transformations can extract additional signal from noisy financial data.

---

## 4. Feature Contribution and Interpretation Methodology

To understand **how predictions are formed**, multiple complementary interpretation methods are used:

1. **Logistic Regression Coefficients**  
   Measure linear marginal effects and provide directional interpretation.

2. **Random Forest Feature Importance**  
   Reflect how frequently features are used to split decision trees.

3. **Permutation Importance**  
   Measure the drop in accuracy when a feature’s values are randomly permuted, providing a model-agnostic assessment.

Using multiple approaches allows for more robust interpretation and mitigates the limitations of any single metric.

---

## 5. Results and Interpretation

<!-- PLACEHOLDER -->
**Figure R1. Model Accuracy Relative to Baseline**  
**Figure R2. Random Forest Feature Importance**  
**Figure R3. Permutation Importance**  
**Figure R4. Logistic Regression Coefficients**  
**Figure R5–R7. Confusion Matrices**

*(Results and interpretation to be added after final model evaluation.)*

---

## 6. Conclusion and Next Steps

<!-- PLACEHOLDER -->

---

## 7. Reproducibility

All results in this project can be reproduced using the provided data and code.

**Environment:**
- Python 3.x
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- PyTorch
- WRDS access for CRSP data

**Reproduction Steps:**
1. Obtain WRDS credentials and download CRSP MSF data.
2. Run the data preparation script to generate `stock_data_processed.csv`.
3. Execute the modeling script to train models and generate figures.
4. All figures are saved automatically to the `figures/` directory.

Random seeds are fixed where applicable to ensure reproducibility.

---

## Repository Structure




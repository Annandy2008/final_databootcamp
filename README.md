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
- The dataset is split into **80% training and 20% testing**. The dataset is split into training and testing samples using an ordered split after    sorting by stock and date. This preserves the temporal ordering within each firm while maintaining a cross-sectional prediction setting.

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
(The codes generate the graphs in the panel figure, separated below for better explanation)

<img width="2048" height="1024" alt="stock_prediction_results" src="https://github.com/user-attachments/assets/185183d5-f658-4909-9995-cc06cee9bc40" />

### 5.1 Predictive Performance Relative to Baseline

<img width="674" height="704" alt="image" src="https://github.com/user-attachments/assets/7e5b58f8-4088-4870-9b43-b621a3d6e035" />


Figure R1 compares out-of-sample classification accuracy across models, with the baseline representing the majority-class accuracy in the training sample.

All predictive models achieve test accuracy in a narrow range between approximately 55% and 57%. The Random Forest model attains the highest accuracy, while Logistic Regression and the Neural Network perform similarly to the baseline.

**Interpretation.**  
The modest improvement over the baseline highlights the difficulty of predicting next-month stock return direction. The limited dispersion in accuracy across models suggests that model choice has a relatively small impact compared to the intrinsic noise in short-horizon returns.

---

### 5.2 Feature Importance in the Random Forest Model

<img width="754" height="704" alt="image" src="https://github.com/user-attachments/assets/6817b0f7-e77f-4c4c-9b15-657eeafcc0a6" />

Figure R2 displays impurity-based feature importance from the Random Forest model.

Volatility emerges as the most important feature, followed by log market capitalization and momentum. Trading-related variables such as the moving-average gap and volume change exhibit moderate importance, while fundamental proxies (ROE and price-to-book ratio) rank lowest.

**Interpretation.**  
The dominance of volatility and momentum suggests that the Random Forest primarily relies on risk and recent return dynamics rather than firm fundamentals. Firm size captures cross-sectional heterogeneity beyond pure return information, contributing meaningfully to prediction.

---

### 5.3 Permutation Importance

<img width="754" height="718" alt="image" src="https://github.com/user-attachments/assets/a918057b-0a04-4a54-bdc1-266c88638b82" />

Permutation importance measures the change in model accuracy when individual features are randomly permuted.

Permuting volatility and momentum leads to the largest decline in accuracy, confirming their central role in prediction. Log market capitalization also exhibits positive importance, while ROE and price-to-book ratio display near-zero or slightly negative importance.

**Interpretation.**  
Compared to impurity-based importance, permutation importance provides a more robust assessment of predictive reliance. The results reinforce the conclusion that short-horizon predictability is driven primarily by volatility, momentum, and size, while simplified fundamental measures add little incremental information.

---

### 5.4 Logistic Regression Coefficients

<img width="744" height="718" alt="image" src="https://github.com/user-attachments/assets/8ddf5b33-b2c1-4541-b493-3c0c64f210a4" />

Figure R4 reports standardized coefficients from the Logistic Regression model.

Momentum and log market capitalization have positive coefficients, indicating that higher recent returns and larger firm size increase the probability of a positive next-month return. Volatility has a strongly negative coefficient, while other features have coefficients close to zero.

**Interpretation.**  
The signs of the coefficients align with financial intuition: momentum effects persist over short horizons, while higher volatility is associated with a lower likelihood of positive returns. The small magnitude of most coefficients reflects the weak linear relationship between firm characteristics and return direction.

---

### 5.5 Classification Errors and Confusion Matrices

<img width="2234" height="734" alt="image" src="https://github.com/user-attachments/assets/632545c5-46b5-46ef-8b74-01335e0d3448" />

Across all models, confusion matrices reveal a strong tendency to predict positive returns. Correct classification of negative returns is limited, particularly for the Neural Network, which exhibits the strongest bias toward the positive class.

**Interpretation.**  
The asymmetry in classification errors reflects both mild class imbalance and the greater difficulty of capturing downside movements. Increased model flexibility does not alleviate this issue; instead, the Neural Network amplifies the bias toward predicting positive returns, suggesting that higher capacity models struggle to extract meaningful downside signals from the data.

---

## 6. Conclusion and Next Steps

This project evaluates the predictability of next-month stock return direction using firm-level characteristics and a range of classification models. Across linear and nonlinear approaches, predictive performance improves only marginally relative to a simple baseline, underscoring the limited predictability of short-horizon stock returns.

Feature interpretation results are consistent across models and methodologies. Volatility, momentum, and firm size emerge as the most informative predictors, while simplified fundamental variables contribute little predictive power. These findings suggest that short-term return direction is driven primarily by risk and trading-related characteristics rather than fundamental valuation signals.

More flexible models, such as Random Forests and Neural Networks, do not substantially outperform simpler benchmarks. In particular, the Neural Network exhibits a strong bias toward predicting positive returns, indicating that increased model capacity does not overcome the noise inherent in monthly return dynamics.

Future work could extend this analysis by incorporating richer fundamental data, applying rolling or time-based cross-validation, or evaluating economic significance through portfolio-level backtesting. Exploring alternative objectives, such as probability calibration or downside-risk prediction, may also yield additional insights.


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


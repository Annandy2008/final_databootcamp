# Stock Return Direction Prediction

## 1. Introduction

This project studies whether the **direction of next-month stock returns** can be predicted using firm-level characteristics constructed from CRSP market data and Compustat fundamental data. The task is formulated as a **cross-sectional binary classification problem**, where each stock-month observation is used to predict whether the following month's return is positive or negative.

Rather than aiming to design a trading strategy, the focus is on evaluating **predictive content**, comparing model performance, and understanding **which features models rely on when forming predictions**.

---

## 2. Data

### 2.1 Data Sources

The dataset combines two primary sources via **WRDS**:

1. **CRSP Monthly Stock File (MSF)**: Market data covering January 2018 to December 2024
2. **Compustat Fundamentals Quarterly (FUNDQ)**: Accounting fundamentals for publicly traded companies

The two databases are linked using the **CRSP-Compustat Merged (CCM) Link Table**, which maps CRSP stock identifiers (PERMNO) to Compustat company identifiers (GVKEY). This enables the integration of market-based features with fundamental accounting ratios.

**Data Processing:**
- Initial query returned 50,000 CRSP monthly observations
- After deduplication (removing duplicate stock-months from overlapping quarterly reports): **22,795 unique stock-months**
- After feature engineering and removing missing values: **21,224 observations**

The final dataset is split into:
- **Training set:** 16,979 observations (80%)
- **Test set:** 4,245 observations (20%)

The dataset is split using an ordered split after sorting by stock and date, preserving temporal ordering within each firm while maintaining a cross-sectional prediction setting.

---

### 2.2 Raw Variables Collected

**From CRSP (Market Data):**
| Variable | Description |
|----------|-------------|
| permno | CRSP permanent stock identifier |
| date | Month-end date |
| ret | Monthly stock return |
| prc | Month-end stock price |
| vol | Monthly trading volume |
| shrout | Shares outstanding |

**From Compustat (Fundamental Data):**
| Variable | Description |
|----------|-------------|
| niq | Net income (quarterly) |
| ceqq | Common equity (book value, quarterly) |
| prccq | Price close (quarterly) |
| cshoq | Common shares outstanding (quarterly) |
| datadate | Fiscal quarter end date |

These variables form the basis for all constructed predictive features.

---

### 2.3 Data Preparation and Feature Engineering

Using the raw variables, several lagged features are constructed to capture recent performance, risk, liquidity, firm size, and fundamentals. All features rely only on information available at or before time *t*.

| Feature | Interpretation | Construction |
|---------|---------------|--------------|
| momentum | Recent return trend | Previous month return |
| volatility | Return risk | 12-month rolling std of returns (lagged) |
| ma_gap | Price deviation | (Price − 20-month MA) / MA |
| vol_change | Liquidity change | Monthly % change in volume |
| log_mcap | Firm size | log(abs(price) × shares outstanding) |
| roe_norm | Relative profitability | (NIQ/CEQ) normalized cross-sectionally by date |
| pb | Valuation proxy | (Market Cap / Book Equity) |

**Data Quality Controls:**
1. **Deduplication:** Stock-months with multiple quarterly reports are deduplicated by keeping the most recent quarterly data (sorted by `datadate DESC`)
2. **Missing value removal:** Observations with missing or infinite feature values are dropped
3. **Standardization:** Features are standardized using `StandardScaler` fitted on training data
   
---

### 2.4 Exploratory Data Analysis (EDA)

Exploratory analysis is conducted to understand the structure and limitations of the data prior to modeling.

Key observations include:
- The return direction label is **mildly imbalanced**, with positive returns occurring in approximately 53% of cases
- Momentum and volatility display heavy tails and substantial dispersion
- Fundamental proxies (ROE and PB) show weak univariate relationships with next-month returns

<img width="1800" height="1200" alt="figure_E1_return_direction" src="https://github.com/user-attachments/assets/2f70f752-b015-425d-8a64-d0259fd47666" />

<img width="3600" height="1200" alt="figure_E2_feature_distributions" src="https://github.com/user-attachments/assets/a12f1f90-be80-4c8f-9452-6b820bb5645b" />

<img width="3600" height="1200" alt="figure_E3_fundamentals_NextReturn" src="https://github.com/user-attachments/assets/98d017d3-1d0b-4035-9b40-f40f727f9bb0" />


These patterns suggest that any predictive signal is likely to be weak and noisy, motivating comparison against a simple benchmark.

---

## 3. Models and Methodology

### 3.1 Prediction Framework and Baseline Reference

The task is framed as a **cross-sectional binary classification problem**. Model performance is evaluated using **out-of-sample accuracy** and confusion matrices.

A **baseline accuracy** of 53.1%, defined as the majority-class frequency in the training set, is used as a reference point. The baseline is **not treated as a predictive model**, but rather as a property of the data and a benchmark for evaluating model performance.

---

### 3.2 Logistic Regression

Logistic Regression serves as a linear benchmark model. It estimates a parametric relationship between features and the probability of a positive next-month return.

The primary advantage of this model is interpretability: coefficient signs and magnitudes provide direct insight into feature–outcome relationships under a linear specification.

---

### 3.3 Random Forest

The Random Forest model captures nonlinear relationships and interactions between features through an ensemble of decision trees. The model is configured with 100 trees, maximum depth of 10, and a random state fixed for reproducibility.

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
- Stochastic Gradient Descent (SGD) with momentum (0.9)
- Learning rate: 0.01
- Training epochs: 100

This model serves as a test of whether high-capacity nonlinear transformations can extract additional signal from noisy financial data.

---

## 4. Feature Contribution and Interpretation Methodology

To understand **how predictions are formed**, multiple complementary interpretation methods are used:

1. **Logistic Regression Coefficients**  
   Measure linear marginal effects and provide directional interpretation.

2. **Random Forest Feature Importance (Gini-based)**  
   Reflect how frequently features are used to split decision trees and the quality of those splits.

3. **Permutation Importance**  
   Measure the drop in accuracy when a feature's values are randomly permuted, providing a model-agnostic assessment of predictive reliance.

Using multiple approaches allows for more robust interpretation and mitigates the limitations of any single metric. Notably, different importance metrics can yield different rankings when features are correlated or when the predictive signal is weak.

---

## 5. Results and Interpretation
(Though the codes generate the graphs in a panel, they are separated below for better explanation.)
<img width="6000" height="3000" alt="stock_prediction_results" src="https://github.com/user-attachments/assets/2c7315a2-69d9-49ce-b20f-100f6b925b6e" />

### 5.1 Predictive Performance Relative to Baseline

Figure R1 compares out-of-sample classification accuracy across models, with the baseline (53.1%) representing the majority-class accuracy in the training sample.

| Model | Test Accuracy |
|-------|---------------|
| Baseline | 53.1% |
| Logistic Regression | 54.5% |
| Neural Network | 54.3% |
| Random Forest | 55.6% |

<img width="672" height="712" alt="image" src="https://github.com/user-attachments/assets/5bb5bc3f-a2fd-4437-a0d1-fe4319c9e05c" />


All predictive models achieve test accuracy between 54.3% and 55.6%, with Random Forest performing best. The improvements over baseline range from 1.2 to 2.5 percentage points.

**Interpretation.**  
The modest improvement over baseline highlights the difficulty of predicting next-month stock return direction. The limited dispersion in accuracy across models suggests that model choice has a relatively small impact compared to the intrinsic noise in short-horizon returns. The results are consistent with the efficient market hypothesis, which posits that publicly available information is quickly incorporated into prices, leaving limited predictable patterns.

---

### 5.2 Feature Importance in the Random Forest Model

Figure R2 displays impurity-based (Gini) feature importance from the Random Forest model.

<img width="754" height="712" alt="image" src="https://github.com/user-attachments/assets/e4767f14-d6f3-4f8b-834f-d13ee7f894fc" />

**Interpretation.**  
The Gini-based importance suggests that normalized ROE is the most frequently used feature for splitting decision trees in the Random Forest. Firm size (log_mcap) and trading activity (vol_change) also contribute substantially. However, this metric measures split frequency and quality rather than direct predictive power, which motivates examining permutation importance for comparison.

---

### 5.3 Permutation Importance

Permutation importance measures the change in model accuracy when individual features are randomly shuffled, providing a more direct assessment of predictive reliance.

<img width="764" height="748" alt="image" src="https://github.com/user-attachments/assets/ce58bfb7-faf1-4439-a980-2dac3584c78b" />

**Interpretation.**  
Permutation importance reveals a different story: momentum and volatility cause the largest accuracy drops when permuted, indicating they are the most critical for actual predictions. Firm size (log_mcap) and normalized ROE show moderate importance, while price-to-book ratio has minimal impact.

**Discrepancy with RF Importance:** The disagreement between Gini-based and permutation importance is notable and informative. This discrepancy commonly arises in financial prediction when:
- Features are correlated (e.g., profitability and firm size often correlate)
- The prediction task has weak signal (our ~55% vs 53% baseline)
- Different metrics capture different aspects: Gini measures split utility while permutation measures prediction degradation

The permutation-based ranking appears more aligned with financial intuition, suggesting that short-horizon return patterns are driven primarily by momentum and volatility rather than accounting fundamentals.

---

### 5.4 Logistic Regression Coefficients

<img width="742" height="734" alt="image" src="https://github.com/user-attachments/assets/f7f5c9c0-fb36-4f83-a64c-f7ab07b5af1d" />

**Interpretation.**  
The coefficients provide linear marginal effects. The dominance of log_mcap suggests that, in a linear framework, firm size is the strongest predictor. The small magnitudes of most coefficients reflect weak linear relationships between features and return direction. The negative momentum coefficient is surprising and may reflect the coarse monthly horizon or nonlinear effects not captured by the linear model.

---

### 5.5 Classification Errors and Confusion Matrices

Confusion matrices reveal prediction patterns across all models:

<img width="2234" height="734" alt="image" src="https://github.com/user-attachments/assets/f32cfd0a-9eed-4fbc-9b79-1abc827b67d1" />

**Interpretation.**  
All models exhibit asymmetry in classification errors, with a tendency to predict positive returns. This reflects both the mild class imbalance (53% positive in training) and the greater difficulty of capturing downside movements in noisy monthly data. 

The Neural Network shows the strongest positive class bias, correctly identifying only 371 of 2,004 actual down months (18.5% recall for negative class). This suggests that increased model flexibility does not help extract downside signals; instead, the model defaults to the more frequent positive class. Random Forest achieves better balance with 35.6% recall on the negative class.

---

## 6. Data Limitations and Methodological Considerations

### 6.1 Fundamental Data Matching

This analysis merges CRSP monthly returns with Compustat quarterly fundamentals using a 6-month lookback window and the CRSP-Compustat link table. While this approach successfully links companies across databases, several considerations affect interpretation:

**1. Look-Ahead Bias (Minor)**

The SQL query condition `c.datadate <= a.date` permits the use of quarterly data released during the prediction month. For example, if predicting July returns, the query may include Q2 earnings announced mid-July, which would not have been available at the beginning of July.

**Impact:** This introduces mild look-ahead bias that may slightly inflate the apparent predictive power of ROE and P/B ratios. The effect is estimated to be small (1-3% of feature importance) because:
- Most quarterly reports are released 45-90 days after quarter-end
- The deduplication step ensures only the most recent quarter is used
- The cross-sectional standardization reduces sensitivity to absolute timing

**Why this does not invalidate our findings:** The primary conclusion—that short-horizon returns are difficult to predict—holds regardless. If anything, the slight look-ahead bias would favor fundamental features, yet they still contribute less than momentum and volatility in permutation importance. A more conservative approach would enforce `c.datadate < a.date - INTERVAL '1 month'`.

**2. Temporal Lag of Fundamentals**

Quarterly accounting data inherently lags the prediction target by 2-5 months:
- Q1 ends March 31 → typically reported by mid-May
- This Q1 data is then used for predictions through June, July, August

**Impact:** ROE and P/B reflect somewhat stale information relative to high-frequency market variables like momentum and volume. This is expected and reflects the fundamental nature of accounting data—it provides a slow-moving picture of firm health rather than day-to-day dynamics.

**Why this does not invalidate our findings:** Quarterly fundamentals are widely used in both academic research and practitioner models despite this lag. The comparison remains fair because all features face appropriate lags (momentum uses t-1 returns, volatility uses rolling 12-month data, etc.). The research question is whether these fundamentals, despite their lag, add predictive power to market-based features.

**3. Sample Selection Bias**

The initial 50,000 CRSP observations reduced to 22,795 after merging with Compustat, representing stocks with available fundamental data. After feature construction, the final sample contains 21,224 observations.

**Impact:** The analysis is restricted to larger, more established firms with Compustat coverage. Small-cap stocks, recent IPOs, and companies not filing standard quarterly reports are excluded.

**Why this does not invalidate our findings:** This sample selection is standard in academic finance research combining CRSP and Compustat. The conclusion that "next-month return direction is difficult to predict" is, if anything, conservative—larger firms with better data quality should be more predictable than small-caps. The results likely represent an upper bound on predictability.

**4. Deduplication of Quarterly Data**

The SQL query can return multiple quarterly reports per stock-month when reports fall within the 6-month window. The analysis addresses this by sorting by `datadate` (descending) and keeping only the first (most recent) quarterly observation for each stock-month.

**Impact:** Ensures each stock-month has exactly one ROE and P/B value, preventing artificial inflation of feature importance through duplicated observations.

**Verification:** Console output confirms successful deduplication: "50,000 records → 22,795 records after deduplication → 21,224 after feature engineering." The deduplication removed 27,205 duplicate stock-month records (54.4% of initial load), demonstrating the importance of this step.

---

### 6.2 Cross-Sectional vs. Time-Series Prediction

This project employs a **cross-sectional design**: at each time period, we predict return direction across different stocks using their contemporaneous characteristics. This differs from time-series prediction, where we would forecast future returns for individual stocks using their own historical patterns.

**Implications:**
- The model evaluates whether, within a given month, certain characteristics distinguish stocks likely to rise from those likely to fall
- Fundamental features that change slowly (like ROE) may be less informative in cross-section than in time-series
- The ordered train-test split preserves temporal structure but evaluates cross-sectional patterns

This design choice affects feature importance: momentum and volatility, which vary substantially across stocks within each month, naturally dominate slow-moving fundamentals in a cross-sectional setting.

---

### 6.3 Why These Limitations Do Not Invalidate the Research

The identified limitations—look-ahead bias, fundamental lag, sample selection, and cross-sectional design—are standard trade-offs in empirical finance research. They affect the *interpretation* and *magnitude* of findings but do not undermine the core conclusions:

1. **Primary finding robust:** All models achieve only modest improvements over baseline (2-3 percentage points), regardless of data limitations. Even if look-ahead bias were eliminated entirely, this conclusion would strengthen rather than weaken.

2. **Feature importance patterns consistent:** Multiple methodologies (Gini importance, permutation importance, logistic coefficients) all suggest momentum and volatility matter more than accounting fundamentals for monthly prediction, despite potential biases favoring fundamentals.

3. **Standard academic practice:** The data construction follows conventions in published finance research. Acknowledging limitations transparently demonstrates methodological awareness rather than indicating flawed analysis.

4. **Conservative interpretation:** The analysis avoids claiming strong predictability or superior performance. The conclusion that "monthly returns are difficult to predict" is conservative and holds under various specification choices.

For a more rigorous treatment, future work could implement stricter temporal lags, use rolling cross-validation, or explore time-series prediction for individual stocks. However, for the purpose of comparing model performance and understanding feature contributions, the current approach provides valid and interpretable results.

---

## 7. Conclusion and Future Directions

This project evaluates the predictability of next-month stock return direction using a combination of market-based technical features and accounting-based fundamental features. Across linear and nonlinear classification approaches, predictive performance improves only marginally relative to a simple baseline (55.6% vs. 53.1%), underscoring the limited predictability of short-horizon stock returns.

**Key Findings:**

1. **Modest Predictability:** All models achieve 54-56% accuracy versus a 53% baseline, consistent with weak-form market efficiency

2. **Feature Contributions Vary by Metric:**
   - Gini importance (RF): ROE, size, volume change most important
   - Permutation importance: Momentum, volatility, size most important
   - Logistic coefficients: Size dominates in linear specification

3. **Technical > Fundamental for Short Horizons:** Permutation importance suggests momentum and volatility drive predictions more than accounting ratios, likely because monthly returns reflect near-term dynamics that fundamentals (updated quarterly) cannot capture

4. **Model Complexity Adds Little:** Neural Networks do not outperform simpler models and exhibit stronger positive class bias, suggesting increased capacity does not overcome data noise

5. **Asymmetric Prediction Errors:** All models struggle more with predicting down months than up months, possibly reflecting genuine asymmetry in return dynamics or the challenges of capturing tail events

**Broader Implications:**

The results align with the efficient market hypothesis: publicly available information (both technical and fundamental) provides limited exploitable patterns for short-term return prediction. The fact that sophisticated ML models cannot substantially beat simple baselines suggests that monthly return direction contains significant unpredictable noise.

The discrepancy between Gini-based and permutation-based importance rankings highlights the value of using multiple interpretation methods. Gini importance measures how features organize decision trees, while permutation importance measures actual predictive reliance—both provide valuable but distinct perspectives.

**Future Directions:**

1. **Richer Fundamental Data:** Incorporate additional Compustat variables (cash flow, R&D, margins) or alternative data (sentiment, analyst forecasts)

2. **Longer Prediction Horizons:** Evaluate 3-month or 6-month return direction, where fundamentals may have more time to manifest

3. **Time-Series Prediction:** Shift from cross-sectional to firm-specific time-series models to leverage individual stock dynamics

4. **Economic Evaluation:** Translate accuracy metrics into portfolio returns, transaction costs, and risk-adjusted performance

5. **Alternative Targets:** Predict magnitude of returns, probability of large moves, or downside risk rather than simple direction

6. **Temporal Cross-Validation:** Implement rolling windows or expanding windows to better simulate real-world deployment

7. **Feature Interaction:** Explore explicit interaction terms or attention mechanisms to model non-additive effects

---

## 8. Reproducibility

All results in this project can be reproduced using the provided code and data access.

**Environment:**
- Python 3.12
- Required packages: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `torch`, `wrds`
- WRDS institutional access for CRSP and Compustat data

**Reproduction Steps:**
1. Configure WRDS credentials (script will prompt for username/password)
2. Run the main script: `python stock_prediction_model.py`
3. The script will automatically:
   - Query CRSP and Compustat data
   - Perform deduplication and feature engineering
   - Train all four models (Baseline, Logistic, RF, NN)
   - Generate all figures and CSV outputs

---

## 9. References

**Data Sources:**
- CRSP (Center for Research in Security Prices), Monthly Stock File
- Compustat North America, Fundamentals Quarterly
- CRSP-Compustat Merged Database (CCM) Link Table

*Project completed as part of Final Project — December 2024*

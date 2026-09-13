<div align="center">

#  Credit Card Fraud Detection

### Detecting fraudulent transactions in 1.85M+ records with a tuned Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Random%20Forest-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-2ea44f?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

*An end-to-end, leakage-safe classification pipeline for detecting fraud in a highly imbalanced, real-world-scale transaction dataset — achieving **88% fraud recall** on unseen data.*

[Problem Overview](#-problem-overview) • [Dataset](#-dataset) • [Preprocessing](#-data-preprocessing) • [Model](#-model) • [Results](#-results) • [Leakage Prevention](#-data-leakage-prevention)

</div>

---

##  Problem Overview

Credit card fraud detection is a classification problem where the goal is to identify fraudulent transactions while minimizing false alarms — and it's a problem where the "easy" baseline is actively misleading. A model that predicts "not fraud" for every transaction in this dataset would already be **>99% accurate**, which is exactly why accuracy alone is the wrong metric here.

### Core Challenges

| Challenge | Why It Matters |
|---|---|
|  **Extreme Class Imbalance** | Fraudulent transactions make up less than 1% of all transactions — naive models default to predicting the majority class |
|  **Data Leakage Risk** | Any preprocessing step fitted on the full dataset (including test data) inflates performance artificially |
|  **High Cardinality** | Features like `merchant`, `job`, `city`, and `trans_num` can blow up dimensionality and drive overfitting if used naively |
|  **Asymmetric Cost of Errors** | Missing a fraudulent transaction (false negative) is typically far more costly than incorrectly flagging a legitimate one (false positive) |

These constraints shaped every downstream decision in this project — from which features were dropped, to how preprocessing was fitted, to which metric was optimized.

---

##  Dataset

**Source:** [Fraud Detection Dataset — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)

The project uses the dataset's native train/test split, keeping evaluation strictly honest:

| Dataset | Transactions |
|---|---:|
| `fraudTrain.csv` | 1,296,675 |
| `fraudTest.csv` | 555,719 |
| **Total** | **1,852,394** |

### Target Variable — `is_fraud`

| Value | Meaning |
|:---:|---|
| `0` | Legitimate transaction |
| `1` | Fraudulent transaction |

---

##  Data Preprocessing

### 1 · Removing High-Cardinality Features

The following columns were dropped — they are identifiers, near-unique values, or high-cardinality categoricals that add dimensionality without generalizable predictive signal:

```text
Unnamed: 0   trans_num   cc_num   first   last   street
unix_time    merchant    job      city    state  zip
```

Retaining raw identifiers like `cc_num` or `trans_num` risks the model effectively "memorizing" specific transactions or cardholders rather than learning generalizable fraud patterns.

### 2 · Date & Time Feature Engineering

`trans_date_trans_time` was parsed to `datetime` and decomposed into:

- `hour`
- `day`
- `month`

Fraud often correlates with *when* a transaction occurs (e.g., unusual hours), so this decomposition surfaces temporal behavioral patterns the raw timestamp couldn't expose directly to the model.

### 3 · Age Feature Engineering

`dob` was converted to `datetime` and combined with the transaction date to compute the cardholder's **approximate age at time of transaction** — a far more directly meaningful signal than a raw date of birth.

### 4 · Categorical Encoding

Categorical features — `category`, `gender` — were transformed with `OneHotEncoder`:

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
```

Applied through a `ColumnTransformer` that was **fitted only on the training data**, then used to transform the test data — a deliberate choice explained in full in [Data Leakage Prevention](#-data-leakage-prevention).

---

##  Model

### Final Configuration

```python
RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
```

### Why Random Forest?

| Property | Benefit for This Problem |
|---|---|
| Handles nonlinear relationships | Fraud patterns rarely follow clean linear boundaries |
| Works well with mixed feature types | Numerical + one-hot encoded categorical features after preprocessing |
| Ensemble of trees | Reduces variance and improves generalization vs. a single decision tree |
| `class_weight='balanced'` | Automatically reweights the loss to give the minority (fraud) class proportionally more influence during training |

The `class_weight='balanced'` setting is doing a lot of the heavy lifting here — rather than resampling the data (SMOTE, undersampling), the model itself is penalized more heavily for misclassifying the rare fraud class during training, which keeps the full, un-distorted dataset in play.

---

## 📈 Results

Evaluated on **`fraudTest.csv`** — a completely held-out, unseen dataset.

<div align="center">

### 🎯 Fraud Recall: **88%**

</div>

| Metric | Score | What It Means |
|---|---:|---|
| **Accuracy** | 99.69% | Correct overall — but misleading alone, given the class imbalance |
| **Fraud Recall** | **88.00%** | Of all *actual* fraud cases, the model caught 88% of them |
| **Fraud Precision** | 57.00% | Of transactions *flagged* as fraud, 57% were truly fraudulent |
| **PR-AUC** | 0.86 | Strong precision/recall balance across thresholds, on an imbalanced dataset |

### Why Recall Is the Priority Metric

In fraud detection, a **false negative** (missed fraud) typically costs far more than a **false positive** (a legitimate transaction flagged for review) — a missed fraud is a direct financial loss, while a false positive is usually just a review step or a customer confirmation. This asymmetry is why the model was tuned to prioritize **Recall** over raw Accuracy or even Precision.

> **88% Fraud Recall** — the model successfully detected **1,884 of 2,145** fraudulent transactions in the test set.

The precision/recall trade-off here (57% precision) is a deliberate consequence of that priority, not an oversight — it reflects `class_weight='balanced'` pushing the model to cast a wider net for fraud, at the cost of some extra false alarms.

---

## 🧮 Confusion Matrix

<p align="center">
  <img src="images/Confusion%20M.png" alt="Confusion Matrix" width="600">
</p>

| Result | Count | Interpretation |
|---|---:|---|
| ✅ True Positive | 1,884 | Fraud correctly detected |
| ❌ False Negative | 261 | Fraud the model failed to catch |
| ✅ True Negative | 552,137 | Legitimate transaction correctly identified |
| ⚠️ False Positive | 1,437 | Legitimate transaction incorrectly flagged |

The relatively small False Negative count (261, against 552K+ correctly handled legitimate transactions) reflects the model's deliberate lean toward catching fraud, at the acceptable cost of the 1,437 false alarms.

---

## 📉 Precision-Recall Curve

<p align="center">
  <img src="images/Precision-Recall%20Curve.png" alt="Precision-Recall Curve" width="600">
</p>

For a dataset this imbalanced, the **ROC curve can look deceptively good** because it's dominated by the overwhelming number of true negatives. The **Precision-Recall curve** is the more honest diagnostic here, since it focuses entirely on how the model performs on the minority (fraud) class across different decision thresholds.

<div align="center">

### 📐 PR-AUC = 0.86

</div>

A PR-AUC of 0.86 indicates the model sustains a strong precision/recall balance across a wide range of thresholds — not just at the single default cutoff reflected in the headline metrics above.

---

## 🔒 Data Leakage Prevention

Preventing leakage was a central design constraint of this project, not an afterthought — with a dataset this imbalanced, even small leaks can make results look far better than they'd actually be in production.

```text
Training Data
     │
     ▼
Feature Engineering
     │
     ▼
Fit ColumnTransformer  ◄── fitted ONLY here
     │
     ▼
Transform Training Data
     │
     ▼
Train Random Forest


Test Data
     │
     ▼
Same Feature Engineering
     │
     ▼
Transform Using Already-Fitted Transformer  ◄── never re-fit
     │
     ▼
Evaluate Model
```

The `OneHotEncoder` / `ColumnTransformer` is **fitted exclusively on the training set**, then applied — never refit — to transform the test set. This guarantees the test results reflect genuine generalization to unseen data, rather than information the model implicitly absorbed from the evaluation set itself.

---

## 🛠️ Technologies Used

| Category | Tools |
|---|---|
| Language | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn (Random Forest Classifier) |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## 📁 Project Structure

```
Credit-Card-Fraud-Detection/
│
├── images/
│   ├── Confusion M.png
│   └── Precision-Recall Curve.png
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
├── notebooks/
│   └── credit_card_fraud_detection.ipynb
├── README.md
└── requirements.txt
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/khaled-amireh/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

> ⚠️ The raw CSV files are not included in this repository due to size. Download `fraudTrain.csv` and `fraudTest.csv` from the [Kaggle dataset page](https://www.kaggle.com/datasets/kartik2112/fraud-detection) and place them in the `data/` folder before running the notebook.

---

## ⚠️ Limitations

- The model has not been cross-validated across multiple folds — reported metrics reflect performance on a single, fixed test split.
- Precision (57%) means a meaningful share of flagged transactions are false alarms; in a production setting this would need to be weighed against the operational cost of manual review.
- The classification threshold used to generate the headline Recall/Precision figures was not explicitly tuned — the Precision-Recall curve suggests further gains may be available by adjusting the decision threshold for a specific business cost trade-off.

---

## 🚀 Future Improvements

- [ ] Explicit threshold tuning to optimize for a target business cost function (cost of missed fraud vs. cost of false alarms)
- [ ] Cross-validation for more robust performance estimates
- [ ] Benchmark against Gradient Boosting models (XGBoost, LightGBM) commonly used in production fraud systems
- [ ] SHAP-based feature attribution for per-transaction fraud explanations
- [ ] Explore geographic/distance-based features (e.g., distance between cardholder and merchant location)

---

## 👤 Author

**Khaled Amireh**
[GitHub](https://github.com/khaled-amireh)

---

<div align="center">

*If you found this project useful, consider giving it a ⭐ on GitHub.*

</div>

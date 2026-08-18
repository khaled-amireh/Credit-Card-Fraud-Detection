# Credit Card Fraud Detection using Random Forest

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-3776AB)
![License](https://img.shields.io/badge/License-MIT-green)

An end-to-end Machine Learning project for detecting fraudulent credit card transactions using a tuned **Random Forest Classifier**.

The project uses a highly imbalanced dataset containing approximately **1.85 million transactions** and achieves **88% fraud recall** on completely unseen test data.

---

## Table of Contents

- [Problem Overview](#-problem-overview)
- [Dataset](#-dataset)
- [Data Preprocessing](#️-data-preprocessing)
- [Model](#-model)
- [Results](#-results)
- [Confusion Matrix](#-confusion-matrix)
- [Precision-Recall Curve](#-precision-recall-curve)
- [Data Leakage Prevention](#-data-leakage-prevention)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [License](#-license)
- [Author](#-author)

---

## Problem Overview

Credit card fraud detection is a classification problem where the goal is to identify fraudulent transactions while minimizing false alarms.

The main challenges are:

- **Extreme Class Imbalance** — Fraudulent transactions represent less than 1% of all transactions.
- **Data Leakage Risk** — Preprocessing must be fitted only on the training data.
- **High Cardinality** — Features such as `merchant`, `job`, `city`, and `trans_num` can create unnecessary dimensionality and overfitting.
- **Fraud Detection Priority** — Missing a fraudulent transaction can be more costly than incorrectly flagging a legitimate transaction.

---

## Dataset

The project uses separate training and testing datasets:

| Dataset | Transactions |
|---|---:|
| `fraudTrain.csv` | 1,296,675 |
| `fraudTest.csv` | 555,719 |
| **Total** | **1,852,394** |

### Target Variable

The target column is `is_fraud`:

- `0` → Legitimate transaction
- `1` → Fraudulent transaction

---

## Data Preprocessing

### 1. Removing High-Cardinality Features

The following columns were removed because they are identifiers, highly unique values, or features that could increase dimensionality:

```text
Unnamed: 0
trans_num
cc_num
first
last
street
unix_time
merchant
job
city
state
zip
```

### 2. Date and Time Feature Engineering

The `trans_date_trans_time` column was converted to `datetime` and used to extract:

- `hour`
- `day`
- `month`

This was done because transaction time can contain useful behavioral patterns.

### 3. Age Feature Engineering

The `dob` column was converted to `datetime` and used together with the transaction date to calculate the cardholder's approximate age.

Using age instead of the raw date of birth makes the feature more meaningful for the model.

### 4. Categorical Encoding

Categorical features such as:

- `category`
- `gender`

were encoded using `OneHotEncoder`.

```python
OneHotEncoder(
    sparse_output=False,
    handle_unknown='ignore'
)
```

The `ColumnTransformer` was fitted only on the training data and then used to transform the test data.

This prevents data leakage and provides a more realistic evaluation.

---

## 🌲 Model

The final model is a Random Forest Classifier with the following parameters:

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

Random Forest was selected because it:

- Handles nonlinear relationships well.
- Works effectively with mixed feature types after preprocessing.
- Combines multiple decision trees to improve generalization.
- Supports `class_weight='balanced'`, which is useful for imbalanced classification problems.

The balanced class weight gives more importance to the minority fraud class.

---

## 📈 Results

The model was evaluated on the completely unseen `fraudTest.csv` dataset.

| Metric | Score |
|---|---:|
| Accuracy | 99.69% |
| Fraud Recall | 88.00% |
| Fraud Precision | 57.00% |
| PR-AUC | 0.86 |

### Why Recall Is Important

For fraud detection, **Recall** is one of the most important metrics because it measures how many actual fraudulent transactions were successfully detected.

The model achieved:

> **88% Fraud Recall** — successfully detecting 1,884 out of 2,145 fraudulent transactions.

---

## 🔍 Confusion Matrix

| Result | Transactions |
|---|---:|
| True Positives | 1,884 |
| False Negatives | 261 |
| True Negatives | 552,137 |
| False Positives | 1,437 |

### Interpretation

- **True Positive** — Fraud correctly detected.
- **False Negative** — Fraud that the model failed to detect.
- **True Negative** — Legitimate transaction correctly identified.
- **False Positive** — Legitimate transaction incorrectly flagged as fraud.

The model prioritizes detecting fraudulent transactions, which explains the relatively high recall and the trade-off with precision.

---

## 📉 Precision-Recall Curve

The Precision-Recall Curve is particularly useful for this project because the dataset is highly imbalanced.

The model achieved:

> **PR-AUC = 0.86**

A higher PR-AUC indicates that the model maintains a good balance between precision and recall across different classification thresholds.

---

## 🔒 Data Leakage Prevention

A major focus of the project was preventing data leakage.

The preprocessing workflow follows this approach:

```text
Training Data
     ↓
Feature Engineering
     ↓
Fit ColumnTransformer
     ↓
Transform Training Data
     ↓
Train Random Forest

Test Data
     ↓
Same Feature Engineering
     ↓
Transform Using Already-Fitted Transformer
     ↓
Evaluate Model
```

The encoder is **never** fitted on the test dataset.

This ensures that the test results represent the model's performance on previously unseen data.

---

## 📂 Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
│
├── LICENSE
└── README.md
```

> Adjust this section to match your actual repository layout.

---

## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/khaled-amireh/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

### Requirements

```text
pandas
numpy
scikit-learn
seaborn
matplotlib
```

---

## 🚀 Usage

1. Place `fraudTrain.csv` and `fraudTest.csv` inside the `data/` directory.
2. Run the preprocessing and training pipeline:

```bash
python src/train.py
```

3. Evaluate the trained model:

```bash
python src/evaluate.py
```

---

## 🔮 Future Improvements

- Experiment with gradient boosting models (XGBoost, LightGBM) for comparison.
- Apply SMOTE or other resampling techniques alongside class weighting.
- Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
- Deploy the model as a REST API for real-time fraud scoring.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Khaled Amireh**
GitHub: [@khaled-amireh](https://github.com/khaled-amireh)

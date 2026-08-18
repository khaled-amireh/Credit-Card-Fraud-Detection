"""
Credit Card Fraud Detection using Random Forest
=================================================
Converted from credit_card_fraud_detection.ipynb
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    PrecisionRecallDisplay,
)


# ============================================================
# Load Dataset
# ============================================================
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

drop_cols = ['is_fraud', 'Unnamed: 0', 'trans_num', 'cc_num', 'first', 'last',
             'street', 'unix_time', 'merchant', 'job', 'city', 'state', 'zip']

Xtrain_df = train_df.drop(drop_cols, axis=1)
ytrain_df = train_df['is_fraud']
Xtest_df = test_df.drop(drop_cols, axis=1)
ytest_df = test_df['is_fraud']


# ============================================================
# Exploratory Data Analysis (EDA)
# ============================================================
train_df.info()

train_df.describe()

pd.set_option('display.max_rows', None)
train_df.isnull().sum().sort_values(ascending=False)


# ============================================================
# Date & Time Feature Extraction
# ============================================================
for df in [Xtrain_df, Xtest_df]:
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day'] = df['trans_date_trans_time'].dt.day
    df['month'] = df['trans_date_trans_time'].dt.month
    df.drop(columns=['trans_date_trans_time'], inplace=True)

    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = 2026 - df['dob'].dt.year
    df.drop(columns=['dob'], inplace=True)


# ============================================================
# One-Hot Encoding
# ============================================================
categorical_cols = ['category', 'gender']

ct = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough',
    verbose_feature_names_out=False
)

Xtrain_df = pd.DataFrame(ct.fit_transform(Xtrain_df), columns=ct.get_feature_names_out(), index=Xtrain_df.index)
Xtest_df = pd.DataFrame(ct.transform(Xtest_df), columns=ct.get_feature_names_out(), index=Xtest_df.index)


# ============================================================
# Train the Model
# ============================================================
classifier = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
classifier.fit(Xtrain_df, ytrain_df)


# ============================================================
# Model Prediction
# ============================================================
ypred = classifier.predict(Xtest_df)
ypred_proba = classifier.predict_proba(Xtest_df)[:, 1]


# ============================================================
# Model Evaluation
# ============================================================

# --- Accuracy & Classification Report ---
print("Accuracy Score:", accuracy_score(ytest_df, ypred))
print("\nClassification Report:\n")
print(classification_report(ytest_df, ypred))

# --- Confusion Matrix ---
cm = confusion_matrix(ytest_df, ypred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# --- Precision-Recall Curve ---
precision_vals, recall_vals, _ = precision_recall_curve(ytest_df, ypred_proba)

pr_auc_score = auc(recall_vals, precision_vals)
print(f"PR-AUC Score: {pr_auc_score:.4f}")

display = PrecisionRecallDisplay(
    precision=precision_vals,
    recall=recall_vals
)

display.plot(color='purple')
plt.title(f'Precision-Recall Curve (PR-AUC = {pr_auc_score:.2f})')
plt.show()

"""
Data Acquisition, Cleaning, and Preprocessing
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ML Repository, via sklearn)
Author: Utkarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
pd.set_option('display.max_columns', 8)

# ============================================================
# STEP 1: DATA ACQUISITION
# ============================================================
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})
df = df.drop(columns=['target'])

print("Dataset shape:", df.shape)
print(df.head())
print(df.describe())
print("Class distribution:\n", df['diagnosis'].value_counts())

# ============================================================
# STEP 2: SIMULATE REALISTIC DATA QUALITY ISSUES
# (The original UCI dataset is clean; issues are introduced here
#  to demonstrate a full real-world cleaning pipeline.)
# ============================================================
raw = df.copy()
n = len(raw)

cols_missing = ['mean radius', 'mean texture', 'mean smoothness', 'worst area']
for col in cols_missing:
    idx = np.random.choice(raw.index, size=int(0.05 * n), replace=False)
    raw.loc[idx, col] = np.nan

raw = pd.concat([raw, raw.sample(n=6, random_state=1)], ignore_index=True)

err_idx = np.random.choice(raw.index, size=5, replace=False)
raw.loc[err_idx, 'mean area'] = -raw.loc[err_idx, 'mean area']

out_idx = np.random.choice(raw.index, size=4, replace=False)
raw.loc[out_idx, 'mean perimeter'] *= 10

# ============================================================
# STEP 3: DATA QUALITY ASSESSMENT
# ============================================================
print("\nMissing values:\n", raw.isnull().sum()[raw.isnull().sum() > 0])
print("Duplicate rows:", raw.duplicated().sum())
print("Negative 'mean area' entries:", (raw['mean area'] < 0).sum())

def iqr_outliers(series, factor=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    return series[(series < lower) | (series > upper)], lower, upper

for col in ['mean perimeter', 'mean radius', 'worst area']:
    outliers, lower, upper = iqr_outliers(raw[col].dropna())
    print(f"{col}: {len(outliers)} outliers (bounds {lower:.2f} - {upper:.2f})")

# ============================================================
# STEP 4: DATA CLEANING
# ============================================================
clean = raw.copy()

# 4.1 Remove duplicates
clean = clean.drop_duplicates()

# 4.2 Fix inconsistent (impossible negative) entries
clean['mean area'] = clean['mean area'].abs()

# 4.3 Cap extreme outliers (wide IQR rule to target likely data-entry errors only)
def cap_outliers_iqr(series, factor=3.0):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - factor * IQR, Q3 + factor * IQR
    return series.clip(lower=lower, upper=upper)

clean['mean perimeter'] = cap_outliers_iqr(clean['mean perimeter'])

# 4.4 Handle missing values (median imputation - robust to outliers/skew)
for col in cols_missing:
    clean[col] = clean[col].fillna(clean[col].median())

print("\nAfter cleaning:")
print("Shape:", clean.shape)
print("Missing values:", clean.isnull().sum().sum())
print("Duplicates:", clean.duplicated().sum())

# ============================================================
# STEP 5: FEATURE SCALING
# ============================================================
numeric_cols = clean.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
scaled = clean.copy()
scaled[numeric_cols] = scaler.fit_transform(clean[numeric_cols])

print("\n'mean radius' before scaling -> after scaling:")
print(f"  mean: {clean['mean radius'].mean():.2f} -> {scaled['mean radius'].mean():.2f}")
print(f"  std:  {clean['mean radius'].std():.2f} -> {scaled['mean radius'].std():.2f}")

# ============================================================
# STEP 6: SAVE FINAL OUTPUT
# ============================================================
scaled.to_csv("final_preprocessed_dataset.csv", index=False)
print("\nSaved final_preprocessed_dataset.csv - ready for downstream analysis/modeling.")

"""
Exploratory Data Analysis and Visualization
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ML Repository, via sklearn)
Uses the cleaned dataset produced by the Week 1 data_cleaning_pipeline.py
Author: Utkarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("cleaned_dataset.csv")  # output of Week 1 pipeline

# ============================================================
# STEP 1: INITIAL ANALYSIS - identify top separating features
# ============================================================
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
results = []
for col in numeric_cols:
    malignant = df[df['diagnosis'] == 'malignant'][col]
    benign = df[df['diagnosis'] == 'benign'][col]
    pooled_std = np.sqrt((malignant.std()**2 + benign.std()**2) / 2)
    cohens_d = (malignant.mean() - benign.mean()) / pooled_std
    t_stat, p_val = stats.ttest_ind(malignant, benign)
    results.append({'feature': col, 'cohens_d': cohens_d, 'p_value': p_val})

results_df = pd.DataFrame(results).sort_values('cohens_d', key=abs, ascending=False)
print(results_df.head(8))

# ============================================================
# STEP 2: DISTRIBUTION VISUALIZATION
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
malignant = df[df['diagnosis'] == 'malignant']['worst concave points']
benign = df[df['diagnosis'] == 'benign']['worst concave points']
ax.hist(benign, bins=25, alpha=0.6, label='Benign', color="#4C9F70")
ax.hist(malignant, bins=25, alpha=0.6, label='Malignant', color="#D9622B")
ax.set_xlabel("worst concave points"); ax.set_ylabel("Number of Patients")
ax.set_title("Distribution by Diagnosis"); ax.legend()
plt.savefig("distribution.png", dpi=150, bbox_inches='tight')

# ============================================================
# STEP 3: CORRELATION HEATMAP
# ============================================================
mean_features = [c for c in df.columns if c.startswith('mean')]
corr = df[mean_features].corr()
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr.columns))); ax.set_yticks(range(len(corr.columns)))
ax.set_xticklabels(corr.columns, rotation=45, ha='right')
ax.set_yticklabels(corr.columns)
plt.colorbar(im, ax=ax)
plt.savefig("correlation.png", dpi=150, bbox_inches='tight')

# ============================================================
# STEP 4: SCATTER PLOTS + CONDITIONAL PROBABILITY CHECK
# ============================================================
thresh = df['worst concave points'].median()
above = df[df['worst concave points'] > thresh]
below = df[df['worst concave points'] <= thresh]
print(f"Above median: {(above['diagnosis']=='malignant').mean()*100:.1f}% malignant")
print(f"Below median: {(below['diagnosis']=='malignant').mean()*100:.1f}% malignant")

# ============================================================
# STEP 5: ANOMALY CHECK - detect imputation artifacts
# ============================================================
median_val = df['mean radius'].median()
suspect = np.isclose(df['mean radius'], median_val, atol=0.001)
print(f"Rows at exact median 'mean radius' (likely imputed): {suspect.sum()}")

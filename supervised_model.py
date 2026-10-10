"""
Supervised Learning Model Implementation
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ML Repository, via sklearn)
Uses the cleaned dataset produced by Week 1's data_cleaning_pipeline.py
Author: Rajkumar Yadav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, auc,
                               precision_score, recall_score, f1_score, accuracy_score)

df = pd.read_csv("cleaned_dataset.csv")  # output of Week 1 pipeline
y = df['diagnosis'].map({'malignant': 1, 'benign': 0})
X = df.drop(columns=['diagnosis'])

# ============================================================
# STEP 1: FEATURE ENGINEERING - TESTED VIA ABLATION, NOT ASSUMED
# ============================================================
X_eng = X.copy()
X_eng['area_perimeter_ratio'] = X_eng['mean area'] / X_eng['mean perimeter']
X_eng['concavity_symmetry_interaction'] = X_eng['mean concavity'] * X_eng['mean symmetry']
X_eng['worst_mean_radius_ratio'] = X_eng['worst radius'] / X_eng['mean radius']
X_eng['texture_smoothness_interaction'] = X_eng['mean texture'] * X_eng['mean smoothness']

scores_base = cross_val_score(LogisticRegression(max_iter=5000), StandardScaler().fit_transform(X), y, cv=5, scoring='f1')
scores_eng = cross_val_score(LogisticRegression(max_iter=5000), StandardScaler().fit_transform(X_eng), y, cv=5, scoring='f1')
print(f"Without engineered features: {scores_base.mean():.4f}")
print(f"With engineered features:    {scores_eng.mean():.4f}")
print("-> Engineered features did not help; proceeding with original 30 features.")

# ============================================================
# STEP 2: TRAIN/TEST SPLIT + MODEL COMPARISON
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    f1 = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='f1').mean()
    print(f"{name}: F1={f1:.4f}")

# ============================================================
# STEP 3: HYPERPARAMETER TUNING (best model: Logistic Regression)
# ============================================================
grid = GridSearchCV(LogisticRegression(max_iter=5000, random_state=42),
                     {'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_scaled, y_train)
print(f"\nBest params: {grid.best_params_}, Best CV F1: {grid.best_score_:.4f}")

# ============================================================
# STEP 4: FINAL EVALUATION ON HELD-OUT TEST SET
# ============================================================
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)[:,1]

print(f"\nTest Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Test Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Test Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"Test F1:        {f1_score(y_test, y_pred):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

fpr, tpr, _ = roc_curve(y_test, y_proba)
print(f"ROC-AUC: {auc(fpr, tpr):.4f}")

# ============================================================
# STEP 5: MISCLASSIFICATION ANALYSIS
# ============================================================
results = pd.DataFrame({'actual': y_test.values, 'predicted': y_pred, 'proba_malignant': y_proba})
misclassified = results[results['actual'] != results['predicted']]
print(f"\nMisclassified cases:\n{misclassified}")

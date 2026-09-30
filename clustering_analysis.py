"""
Unsupervised Learning and Clustering Analysis
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset (UCI ML Repository, via sklearn)
Uses the cleaned dataset produced by Week 1's data_cleaning_pipeline.py
Author: Utkarsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

df = pd.read_csv("cleaned_dataset.csv")  # output of Week 1 pipeline
X = df.drop(columns=['diagnosis'])
y = df['diagnosis']

# ============================================================
# STEP 1: PREPROCESSING - QUANTIFY IMPACT OF SCALING
# ============================================================
km_unscaled = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X)
sil_unscaled = silhouette_score(X, km_unscaled.labels_)
ari_unscaled = adjusted_rand_score(y, km_unscaled.labels_)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
km_scaled = KMeans(n_clusters=2, random_state=42, n_init=10).fit(X_scaled)
sil_scaled = silhouette_score(X_scaled, km_scaled.labels_)
ari_scaled = adjusted_rand_score(y, km_scaled.labels_)

print(f"Unscaled: silhouette={sil_unscaled:.4f}, ARI={ari_unscaled:.4f}")
print(f"Scaled:   silhouette={sil_scaled:.4f}, ARI={ari_scaled:.4f}")
print("-> Scaled clustering agrees far better with real diagnosis despite lower silhouette")

# ============================================================
# STEP 2: DETERMINE OPTIMAL K
# ============================================================
silhouettes = []
for k in range(2, 9):
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))
best_k = list(range(2, 9))[np.argmax(silhouettes)]
print(f"\nBest k by silhouette score: {best_k}")

# ============================================================
# STEP 3: FINAL K-MEANS + PCA VISUALIZATION
# ============================================================
km = KMeans(n_clusters=2, random_state=42, n_init=10)
cluster_labels = km.fit_predict(X_scaled)
print(f"\nFinal silhouette: {silhouette_score(X_scaled, cluster_labels):.4f}")
print(f"Final ARI vs actual diagnosis: {adjusted_rand_score(y, cluster_labels):.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")

# ============================================================
# STEP 4: HIERARCHICAL CLUSTERING (comparison method)
# ============================================================
Z = linkage(X_scaled, method='ward')
hier_labels = fcluster(Z, t=2, criterion='maxclust')
print(f"\nHierarchical silhouette: {silhouette_score(X_scaled, hier_labels):.4f}")
print(f"Hierarchical ARI: {adjusted_rand_score(y, hier_labels):.4f}")

# ============================================================
# STEP 5: CLUSTER PROFILING
# ============================================================
df['cluster'] = cluster_labels
df.to_csv("clustered_dataset.csv", index=False)
print("\nSaved clustered_dataset.csv")
print(df.groupby('cluster')['diagnosis'].value_counts(normalize=True))

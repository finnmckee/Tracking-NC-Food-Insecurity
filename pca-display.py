# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("County Sheet Finalized.csv")

# Clean percentage and currency columns
def clean_column(col):
    return pd.to_numeric(col.astype(str).str.replace(r'[%$,]', '', regex=True).str.strip(), errors='coerce')

df['Overall Food Insecurity Rate'] = clean_column(df['Overall Food Insecurity Rate'])
df['# of Food Insecure Persons'] = clean_column(df[' # of Food Insecure Persons Overall '])
df['Black FI Rate'] = clean_column(df['Food Insecurity Rate among Black Persons (all ethnicities)'])
df['Hispanic FI Rate'] = clean_column(df['Food Insecurity Rate among Hispanic Persons (any race)'])
df['White FI Rate'] = clean_column(df['Food Insecurity Rate among White, non-Hispanic Persons'])
df['FI ≤ SP Threshold'] = clean_column(df['% FI ≤ SP Threshold'])
df['FI > SP Threshold'] = clean_column(df['% FI > SP Threshold'])
df['Annual Shortfall'] = clean_column(df[' Weighted Annual Food Budget Shortfall '])
df['Cost per Meal'] = clean_column(df['Cost per Meal'])
df['Weekly $ Needed'] = clean_column(df['Weighted Weekly $ Needed by FI'])

# Calculate Funding Gap
df['Funding Gap per Person'] = df['Annual Shortfall'] / df['# of Food Insecure Persons']

# Select relevant features
model_df = df[[
    'Overall Food Insecurity Rate',
    '# of Food Insecure Persons',
    'Black FI Rate',
    'Hispanic FI Rate',
    'White FI Rate',
    'FI ≤ SP Threshold',
    'FI > SP Threshold',
    'Annual Shortfall',
    'Cost per Meal',
    'Weekly $ Needed',
    'Funding Gap per Person'
]].dropna()

# Separate target and features
X = model_df.drop(columns=['Overall Food Insecurity Rate'])
y = model_df['Overall Food Insecurity Rate']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Random Forest for Feature Importance
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)
importances = rf.feature_importances_

# Display feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print("Feature Importance:")
print(feature_importance_df)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# PCA Summary Table
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

pca_summary = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained Variance': explained_variance,
    'Proportion of Variance': explained_variance_ratio,
    'Cumulative Proportion': cumulative_variance
})
print("\nPCA Summary:")
print(pca_summary)

# PCA Explained Variance Plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Use top 3 PCA components for clustering
X_pca_reduced = X_pca[:, :3]

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca_reduced)
silhouette_kmeans = silhouette_score(X_pca_reduced, kmeans_labels)
print(f"\nKMeans Silhouette Score: {silhouette_kmeans:.3f}")

# Hierarchical Clustering Dendrogram
linkage_matrix = linkage(X_pca_reduced, method='ward')
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, truncate_mode='level', p=5)
plt.title("Hierarchical Clustering Dendrogram (Truncated)")
plt.xlabel("Sample Index")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

# Silhouette Scores for Different Cluster Counts
silhouette_scores = {}
for k in range(2, 7):
    labels = fcluster(linkage_matrix, k, criterion='maxclust')
    silhouette_scores[k] = silhouette_score(X_pca_reduced, labels)

print("\nSilhouette Scores for Hierarchical Clustering:")
for k, score in silhouette_scores.items():
    print(f"{k} Clusters: {score:.3f}")

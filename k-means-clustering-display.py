# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load data
df = pd.read_csv("County Sheet Finalized.csv")

# Clean and select relevant features
def clean_column(col):
    return pd.to_numeric(col.astype(str).str.replace(r'[%$,]', '', regex=True).str.strip(), errors='coerce')

df['FI Rate Overall'] = clean_column(df['Overall Food Insecurity Rate'])
df['FI Rate Black'] = clean_column(df['Food Insecurity Rate among Black Persons (all ethnicities)'])
df['FI Rate Hispanic'] = clean_column(df['Food Insecurity Rate among Hispanic Persons (any race)'])
df['FI Rate White'] = clean_column(df['Food Insecurity Rate among White, non-Hispanic Persons'])
df['Percent FI Below SNAP'] = clean_column(df['% FI ≤ SP Threshold'])
df['Percent FI Above SNAP'] = clean_column(df['% FI > SP Threshold'])
df['Cost per Meal'] = clean_column(df['Cost per Meal'])
df['Weighted Weekly $ Needed by FI'] = clean_column(df['Weighted Weekly $ Needed by FI'])
df['Annual Budget Shortfall'] = clean_column(df[' Weighted Annual Food Budget Shortfall '])

# Select features
features = [
    'FI Rate Overall', 'FI Rate Black', 'FI Rate Hispanic', 'FI Rate White',
    'Percent FI Below SNAP', 'Percent FI Above SNAP', 'Cost per Meal',
    'Weighted Weekly $ Needed by FI'
]
X_reduced = df[features].dropna()
y = df.loc[X_reduced.index, 'Annual Budget Shortfall']

# Random Forest for Feature Importance
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=features).sort_values()

plt.figure(figsize=(8, 6))
importances.plot(kind='barh')
plt.title("Feature Importance – Annual Budget Shortfall")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reduced)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

# Use top 3 PCs
X_pca_top3 = X_pca[:, :3]

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca_top3)
X_reduced['Cluster'] = clusters

# Cluster scatterplot (PC1 vs PC2)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca_top3[:, 0], y=X_pca_top3[:, 1], hue=clusters, palette='Set1', s=100, alpha=0.7)
plt.title("K-Means Clustering (Based on Top 3 PCA Components)")
plt.xlabel("PC1 (47.2%)")
plt.ylabel("PC2 (26.3%)")
plt.grid(True)
plt.tight_layout()
plt.show()

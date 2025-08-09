from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Code to generate simulated funding data!


np.random.seed(42)  # for reproducibility
df['Funding Received'] = df['Annual Budget Shortfall'] * np.random.normal(loc=1.0, scale=0.2, size=len(df))

df['Funding Gap'] = df['Funding Received'] - df['Annual Budget Shortfall']

df[['County, State', 'Annual Budget Shortfall', 'Funding Received', 'Funding Gap']].head(10)

# Creating 'average gap' by county
avg_gap = df.groupby('County, State')['Funding Gap'].mean().reset_index().sort_values(by='Funding Gap')

# Show most underfunded counties (on average)
avg_gap.head(10)

avg_gap.sort_values(by='Funding Gap', ascending=False).head(10)

# Feature set
features = [
    'FI Rate Overall', 'FI Persons Overall',
    'FI Rate Black', 'FI Rate Hispanic', 'FI Rate White',
    'Percent FI Below SNAP', 'Percent FI Above SNAP',
    'Cost per Meal', 'Weighted Weekly $ Needed by FI' # Changed 'Weekly $ Needed' to 'Weighted Weekly $ Needed by FI'
]

X = df[features]
y = df['Annual Budget Shortfall']

# Model 1: Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
lr_r2 = r2_score(y_test, y_pred_lr)
lr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr))
{"Linear Regression R-Squared": lr_r2,
 "Linear Regression RMSE": lr_rmse}

# Model 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
rf_r2 = r2_score(y_test, y_pred_rf)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
{"Random Forest R-Sqaured": rf_r2,
 "Random Forest RMSE": rf_rmse}

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# Example corpus
documents = ["apple banana cherry date elderberry",
             "banana cherry date elderberry",
             "apple banana date"]

# Convert text to Term-Document Matrix (TDM) using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)

# Apply TruncatedSVD (LSA)
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X)

print("SVD components:", X_svd)

# Model 3: Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# 'Funding Status' column based on 'Funding Gap'
df['Funding Status'] = np.where(df['Funding Gap'] >= 0, 'Fully Funded', 'Underfunded')

y_class = df['Funding Status']
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Logistic Regression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_c, y_train_c)
log_pred = log_clf.predict(X_test_c)
log_report = classification_report(y_test_c, log_pred, output_dict=True)
{"Logistic Regression Classification Report": log_report}

# Model 3: Classification Models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
# 'Funding Status' column based on 'Funding Gap'
df['Funding Status'] = np.where(df['Funding Gap'] >= 0, 'Fully Funded', 'Underfunded')

y_class = df['Funding Status']
# Use the original feature set (X) from input 22
X = df[features]  # features is defined in input 22
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)

# Logistic Regression
log_clf = LogisticRegression(max_iter=1000)
log_clf.fit(X_train_c, y_train_c)
log_pred = log_clf.predict(X_test_c)
log_report = classification_report(y_test_c, log_pred, output_dict=True)
{"Logistic Regression Classification Report": log_report}

# Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
rf_pred = rf_clf.predict(X_test_c)
rf_class_report = classification_report(y_test_c, rf_pred, output_dict=True)
{"Random Forest Classification Report": rf_class_report}

# Model 4: KMeans Clustering

from sklearn.preprocessing import StandardScaler # Import StandardScaler
from sklearn.cluster import KMeans # Import KMeans
cluster_features = df[features + ['Funding Gap']]
X_scaled = StandardScaler().fit_transform(cluster_features)
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
{"Clusters Assigned": df['Cluster'].nunique()}

# PLOT 1: Actual vs Predicted (Random Forest)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', label='Perfect Prediction')
plt.xlabel("Actual Budget Shortfall")
plt.ylabel("Predicted Budget Shortfall")
plt.title("Random Forest Regressor: Actual vs Predicted")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2. Prediction Error Distribution
errors = y_pred_rf - y_test
plt.figure(figsize=(8, 5))
sns.histplot(errors, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of Prediction Errors (Random Forest)")
plt.xlabel("Prediction Error (Predicted - Actual)")
plt.tight_layout()
plt.show()

# Plot 3: Feature Importance - Regressor
reg_feature_importance = pd.Series(rf_model.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 6))
reg_feature_importance.plot(kind='barh')
plt.title("Feature Importance – Predicting Budget Shortfall")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# 4. Feature Importance - Classifier
clf_feature_importance = pd.Series(rf_clf.feature_importances_, index=features).sort_values()
plt.figure(figsize=(8, 6))
clf_feature_importance.plot(kind='barh', color='orange')
plt.title("Feature Importance – Predicting Over/Underfunded Status")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# Wanted to test Feature Importance without FI Persons Overall

features_reduced = [
    'FI Rate Overall',
    'FI Rate Black',
    'FI Rate Hispanic',
    'FI Rate White',
    'Percent FI Below SNAP',
    'Percent FI Above SNAP',
    'Cost per Meal',
    'Weighted Weekly $ Needed by FI'
]

X_reduced = df[features_reduced]
y_reduced = df['Annual Budget Shortfall']

# Train/test split
X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_reduced, y_reduced, test_size=0.2, random_state=42)

# New RF
rf_reduced = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reduced.fit(X_train_red, y_train_red)
y_pred_red = rf_reduced.predict(X_test_red)

# Reduced performance values
r2_red = r2_score(y_test_red, y_pred_red)
rmse_red = np.sqrt(mean_squared_error(y_test_red, y_pred_red))

# Plot new feature importance
reduced_importance = pd.Series(rf_reduced.feature_importances_, index=features_reduced).sort_values()
plt.figure(figsize=(8, 6))
reduced_importance.plot(kind='barh')
plt.title("Feature Importance – Budget Shortfall (Excluding FI Persons Overall)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# New model performance
r2_red, rmse_red

# Principal Component Analysis
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_reduced)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(explained_variance) + 1), explained_variance.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.tight_layout()
plt.show()

explained_variance

# Plot 5: Top 10 Underfunded & Overfunded
top_underfunded = df.sort_values(by='Funding Gap').head(10)
top_overfunded = df.sort_values(by='Funding Gap', ascending=False).head(10)
top_funding_gaps = pd.concat([top_underfunded, top_overfunded])
top_funding_gaps = top_funding_gaps[['County, State', 'Year', 'Funding Gap'] + [col for col in top_funding_gaps.columns if col not in ['County, State', 'Year', 'Funding Gap']]]
top_funding_gaps['County (Year)'] = top_funding_gaps['County, State'] + " (" + top_funding_gaps['Year'].astype(str) + ")"

plt.figure(figsize=(12, 8))
sns.barplot(data=top_funding_gaps, x='Funding Gap', y='County (Year)', palette='coolwarm')
plt.axvline(0, color='black', linestyle='--')
plt.title("Top 10 Overfunded and Underfunded Counties")
plt.xlabel("Funding Gap ($)")
plt.ylabel("County (Year)")
plt.tight_layout()
plt.show()

# PCA as Dataframe
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_reduced)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

explained_variance_df = pd.DataFrame({
    'Principal Component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
    'Explained Variance Ratio': pca.explained_variance_ratio_,
    'Cumulative Variance': np.cumsum(pca.explained_variance_ratio_)
})

print(explained_variance_df)

# Funding Gap Visual, but by Year (2019, 2022)
def plot_funding_gap(year):
    filtered_df = df[df['Year'] == year]  # Filter data for the given year
    top_underfunded = filtered_df.sort_values(by='Funding Gap').head(10)
    top_overfunded = filtered_df.sort_values(by='Funding Gap', ascending=False).head(10)
    top_funding_gaps = pd.concat([top_underfunded, top_overfunded])
    top_funding_gaps['County (Year)'] = top_funding_gaps['County, State'] + " (" + top_funding_gaps['Year'].astype(str) + ")"

    plt.figure(figsize=(12, 8))
    sns.barplot(data=top_funding_gaps, x='Funding Gap', y='County (Year)', palette='coolwarm')
    plt.axvline(0, color='black', linestyle='--')
    plt.title(f"Top 10 Overfunded and Underfunded Counties ({year})")
    plt.xlabel("Funding Gap ( $ in millions)")
    plt.ylabel("County (Year)")
    plt.tight_layout()
    plt.show()

# Create plots for 2019 and 2022
plot_funding_gap(2019)
plot_funding_gap(2022)

# Plot 6: Cluster Scatter Plot
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x='Funding Gap', y='FI Rate Overall', hue='Cluster', palette='Set2')
plt.title("County Clusters by Funding Gap & Insecurity Rate")
plt.xlabel("Funding Gap")
plt.ylabel("Food Insecurity Rate")
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

df.groupby('Cluster')[[
    'Funding Gap',
    'FI Rate Overall',
    'Cost per Meal',
    'Weighted Weekly $ Needed by FI',
    'Percent FI Below SNAP',
    'Percent FI Above SNAP'
]].mean().round(2)

filtered_df = df[df['Year'] == 2022]  # Filter for the year 2022
filtered_df[['County, State', 'Year', 'Cluster']].sort_values(by='Cluster')


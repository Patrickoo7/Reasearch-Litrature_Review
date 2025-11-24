# Lesson 4: Anomaly Detection & Outlier Handling 🎯

**Module 3: Data Engineering | Lesson 4 of 4**

Master techniques to detect and handle outliers - critical for robust ML models!

---

## Why Outlier Handling Matters?

**Outliers can:**
- Skew statistical analysis
- Reduce model accuracy
- Represent data quality issues OR valuable insights
- Break assumptions of many ML algorithms

**Real-world examples:**
- Fraud detection: Outliers = fraud (keep them!)
- House prices: $50M mansion in dataset of $200K homes (maybe remove)
- Sensor data: Erroneous reading = -999 (definitely remove)

**Key question: Are outliers errors or insights?**

---

## 1. Types of Outliers 📊

### Point Outliers

```python
import numpy as np
import matplotlib.pyplot as plt

# Normal data with outliers
data = np.concatenate([
    np.random.normal(100, 10, 100),  # Normal data
    [200, 220, 250]                   # Outliers
])

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(data, bins=30)
plt.title('Distribution with Outliers')

plt.subplot(1, 2, 2)
plt.boxplot(data)
plt.title('Box Plot')
plt.show()
```

### Contextual Outliers

```python
# Temperature data
# 30°C in summer = normal
# 30°C in winter = outlier!

dates = pd.date_range('2023-01-01', periods=365)
temps = np.random.normal(20, 10, 365)  # Base temperature
# Add seasonal pattern
temps += 15 * np.sin(2 * np.pi * np.arange(365) / 365)

# Add contextual outlier
temps[30] = 30  # Hot day in winter!

plt.figure(figsize=(12, 4))
plt.plot(dates, temps)
plt.scatter(dates[30], temps[30], color='red', s=100, label='Outlier')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.show()
```

### Collective Outliers

```python
# Sequence of normal values, but unusual together
# Example: Website traffic spike over weekend
```

---

## 2. Statistical Methods 📈

### Z-Score Method

**Assumption:** Data is normally distributed

```python
from scipy import stats

data = np.concatenate([
    np.random.normal(100, 15, 100),
    [200, 220, 180]
])

# Calculate z-scores
z_scores = np.abs(stats.zscore(data))

# Threshold: typically |z| > 3
threshold = 3
outliers = np.where(z_scores > threshold)[0]

print(f"Outliers detected: {len(outliers)}")
print(f"Outlier indices: {outliers}")
print(f"Outlier values: {data[outliers]}")

# Remove outliers
data_clean = data[z_scores <= threshold]
print(f"\nOriginal size: {len(data)}")
print(f"After removal: {len(data_clean)}")
```

**When to use:**
- Data is approximately normal
- Univariate analysis
- Quick check

**Limitations:**
- Assumes normality
- Sensitive to extreme outliers
- Univariate only

### Modified Z-Score (Robust)

```python
# Uses median instead of mean (more robust)
def modified_z_score(data):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z = 0.6745 * (data - median) / mad
    return modified_z

modified_z = np.abs(modified_z_score(data))
outliers = np.where(modified_z > 3.5)[0]

print(f"Modified Z-Score outliers: {len(outliers)}")
```

### IQR (Interquartile Range) Method

**Most common, works for skewed data!**

```python
import pandas as pd

df = pd.DataFrame({'value': data})

# Calculate IQR
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

# Define outlier bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound:.2f}")
print(f"Upper bound: {upper_bound:.2f}")

# Identify outliers
outliers = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"\nOutliers: {len(outliers)}")

# Remove outliers
df_clean = df[(df['value'] >= lower_bound) & (df['value'] <= upper_bound)]

# Visualize
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.boxplot(df['value'])
plt.title('With Outliers')

plt.subplot(1, 2, 2)
plt.boxplot(df_clean['value'])
plt.title('Without Outliers')
plt.show()
```

**When to use:**
- Skewed data
- No normality assumption
- Quick and robust

### Percentile Method

```python
# Remove extreme percentiles
lower_percentile = df['value'].quantile(0.01)  # 1st percentile
upper_percentile = df['value'].quantile(0.99)  # 99th percentile

df_clean = df[(df['value'] >= lower_percentile) & (df['value'] <= upper_percentile)]

print(f"Removed {len(df) - len(df_clean)} outliers")
```

---

## 3. Machine Learning Methods 🤖

### Isolation Forest ⭐ Best for High-Dimensional

**Idea:** Outliers are easier to isolate (fewer splits needed in tree)

```python
from sklearn.ensemble import IsolationForest

# Multi-dimensional data
from sklearn.datasets import make_classification
X, _ = make_classification(n_samples=300, n_features=5, n_redundant=0, random_state=42)

# Add outliers
outliers = np.random.uniform(low=-10, high=10, size=(20, 5))
X_with_outliers = np.vstack([X, outliers])

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.1,  # Expected outlier proportion
                             random_state=42,
                             n_estimators=100)

# Predict: 1 = inlier, -1 = outlier
predictions = iso_forest.fit_predict(X_with_outliers)

# Count outliers
n_outliers = np.sum(predictions == -1)
print(f"Outliers detected: {n_outliers}")

# Get outlier scores
scores = iso_forest.score_samples(X_with_outliers)
# More negative = more anomalous

# Visualize (2D projection)
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_with_outliers)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1],
           label='Inliers', alpha=0.6)
plt.scatter(X_pca[predictions == -1, 0], X_pca[predictions == -1, 1],
           label='Outliers', color='red', alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Isolation Forest Results')
plt.show()
```

**When to use:**
- High-dimensional data
- No assumptions about distribution
- Fast and scalable

### Local Outlier Factor (LOF)

**Idea:** Outliers have low density compared to neighbors

```python
from sklearn.neighbors import LocalOutlierFactor

# Fit LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
predictions = lof.fit_predict(X_with_outliers)

# LOF scores
lof_scores = lof.negative_outlier_factor_
# More negative = more anomalous

n_outliers = np.sum(predictions == -1)
print(f"LOF outliers detected: {n_outliers}")

# Visualize
X_pca = pca.fit_transform(X_with_outliers)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[predictions == 1, 0], X_pca[predictions == 1, 1],
           label='Inliers', alpha=0.6)
plt.scatter(X_pca[predictions == -1, 0], X_pca[predictions == -1, 1],
           label='Outliers', color='red', alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('Local Outlier Factor Results')
plt.show()
```

**When to use:**
- Density-based outlier detection
- Clusters of different densities
- Local anomalies

### One-Class SVM

**Idea:** Learn boundary around normal data

```python
from sklearn.svm import OneClassSVM

# Fit One-Class SVM
oc_svm = OneClassSVM(nu=0.1,  # Expected outlier proportion
                     kernel='rbf',
                     gamma='auto')

predictions = oc_svm.fit_predict(X_with_outliers)

n_outliers = np.sum(predictions == -1)
print(f"One-Class SVM outliers: {n_outliers}")
```

**When to use:**
- Clear boundary around normal data
- Non-linear boundaries needed

### DBSCAN Clustering

**Idea:** Outliers don't belong to any cluster

```python
from sklearn.cluster import DBSCAN

# Fit DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
clusters = dbscan.fit_predict(X_with_outliers)

# Outliers labeled as -1
n_outliers = np.sum(clusters == -1)
print(f"DBSCAN outliers: {n_outliers}")

# Visualize
X_pca = pca.fit_transform(X_with_outliers)

plt.figure(figsize=(10, 6))
plt.scatter(X_pca[clusters != -1, 0], X_pca[clusters != -1, 1],
           c=clusters[clusters != -1], cmap='viridis', alpha=0.6)
plt.scatter(X_pca[clusters == -1, 0], X_pca[clusters == -1, 1],
           label='Outliers', color='red', marker='x', s=100)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.title('DBSCAN Outlier Detection')
plt.show()
```

---

## 4. Multivariate Outlier Detection 🎯

### Mahalanobis Distance

**Distance accounting for correlations between features**

```python
from scipy.spatial.distance import mahalanobis

# Sample data
np.random.seed(42)
data = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=100)

# Add outliers
outliers = np.array([[5, 5], [-5, 5], [5, -5]])
data_with_outliers = np.vstack([data, outliers])

# Calculate Mahalanobis distance
mean = np.mean(data_with_outliers, axis=0)
cov = np.cov(data_with_outliers.T)
cov_inv = np.linalg.inv(cov)

distances = [mahalanobis(x, mean, cov_inv) for x in data_with_outliers]
distances = np.array(distances)

# Threshold using chi-square distribution
from scipy.stats import chi2
threshold = chi2.ppf(0.99, df=2)  # 99% confidence, 2 dimensions

outlier_mask = distances > threshold
n_outliers = np.sum(outlier_mask)

print(f"Mahalanobis outliers: {n_outliers}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(data_with_outliers[~outlier_mask, 0],
           data_with_outliers[~outlier_mask, 1],
           label='Inliers', alpha=0.6)
plt.scatter(data_with_outliers[outlier_mask, 0],
           data_with_outliers[outlier_mask, 1],
           label='Outliers', color='red', s=100)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Mahalanobis Distance Outliers')
plt.show()
```

---

## 5. Handling Outliers 🛠️

### Option 1: Remove

```python
# Remove based on IQR
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1

df_clean = df[
    (df['value'] >= Q1 - 1.5 * IQR) &
    (df['value'] <= Q3 + 1.5 * IQR)
]

print(f"Removed: {len(df) - len(df_clean)} rows")
```

**When to use:**
- Clear data quality issues
- Small proportion of outliers
- Outliers are errors, not insights

### Option 2: Cap (Winsorize)

```python
from scipy.stats.mstats import winsorize

# Cap at 5th and 95th percentiles
df['value_capped'] = winsorize(df['value'], limits=[0.05, 0.05])

# Or manual capping
lower = df['value'].quantile(0.05)
upper = df['value'].quantile(0.95)
df['value_capped_manual'] = df['value'].clip(lower, upper)

print("Before capping:")
print(df['value'].describe())
print("\nAfter capping:")
print(df['value_capped'].describe())
```

**When to use:**
- Want to keep all data
- Reduce outlier impact without removal
- Competition/production systems

### Option 3: Transform

```python
# Log transform (reduces right skew)
df['value_log'] = np.log1p(df['value'])  # log(1 + x)

# Square root transform
df['value_sqrt'] = np.sqrt(df['value'])

# Box-Cox transform (finds optimal power)
from scipy.stats import boxcox

transformed, lambda_param = boxcox(df['value'] + 1)  # +1 if values <= 0
df['value_boxcox'] = transformed

print(f"Optimal lambda: {lambda_param:.3f}")

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

df['value'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Original')

df['value_log'].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Log Transform')

df['value_sqrt'].hist(bins=50, ax=axes[1, 0])
axes[1, 0].set_title('Square Root Transform')

df['value_boxcox'].hist(bins=50, ax=axes[1, 1])
axes[1, 1].set_title('Box-Cox Transform')

plt.tight_layout()
plt.show()
```

**When to use:**
- Skewed distributions
- Need normality for model
- Feature engineering

### Option 4: Treat as Separate Segment

```python
# Create outlier flag
df['is_outlier'] = 0
outlier_mask = (df['value'] < Q1 - 1.5 * IQR) | (df['value'] > Q3 + 1.5 * IQR)
df.loc[outlier_mask, 'is_outlier'] = 1

# Use as feature
print(f"Outlier distribution:\n{df['is_outlier'].value_counts()}")

# Separate models for outliers and normal data
from sklearn.ensemble import RandomForestClassifier

# Model for normal data
X_normal = df[df['is_outlier'] == 0][features]
y_normal = df[df['is_outlier'] == 0]['target']

# Model for outliers
X_outlier = df[df['is_outlier'] == 1][features]
y_outlier = df[df['is_outlier'] == 1]['target']
```

**When to use:**
- Outliers have different behavior
- Valuable segment (fraud, VIP customers)
- Sufficient outlier samples

### Option 5: Robust Models

```python
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.ensemble import RandomForestRegressor

# Tree-based models are robust to outliers!
rf = RandomForestRegressor(n_estimators=100)

# Huber loss (robust to outliers)
huber = HuberRegressor()

# RANSAC (fits model on random subsets)
ransac = RANSACRegressor()

# Compare
models = {
    'Random Forest': rf,
    'Huber': huber,
    'RANSAC': ransac
}

for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} R²: {score:.3f}")
```

**When to use:**
- Can't remove outliers
- Want model to be naturally robust
- Tree-based models are best choice

---

## 6. Time Series Outlier Detection ⏰

```python
# Generate time series with anomalies
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=365)
values = 100 + np.cumsum(np.random.randn(365)) + 10 * np.sin(np.arange(365) * 2 * np.pi / 365)

# Add anomalies
values[100] = values[100] + 50
values[200] = values[200] - 40

ts = pd.DataFrame({'date': dates, 'value': values})
ts.set_index('date', inplace=True)

# Method 1: Rolling Z-Score
window = 30
rolling_mean = ts['value'].rolling(window=window, center=True).mean()
rolling_std = ts['value'].rolling(window=window, center=True).std()

z_scores = np.abs((ts['value'] - rolling_mean) / rolling_std)
anomalies = z_scores > 3

print(f"Anomalies detected: {anomalies.sum()}")

# Method 2: Seasonal Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts['value'], model='additive', period=30)
residuals = decomposition.resid.dropna()

# Outliers in residuals
threshold = 3 * residuals.std()
residual_anomalies = np.abs(residuals) > threshold

# Visualize
fig, axes = plt.subplots(4, 1, figsize=(15, 12))

ts['value'].plot(ax=axes[0])
axes[0].scatter(ts.index[anomalies], ts['value'][anomalies], color='red', s=100, label='Anomalies')
axes[0].set_title('Original with Anomalies')
axes[0].legend()

decomposition.trend.plot(ax=axes[1])
axes[1].set_title('Trend')

decomposition.seasonal.plot(ax=axes[2])
axes[2].set_title('Seasonal')

decomposition.resid.plot(ax=axes[3])
axes[3].scatter(residuals.index[residual_anomalies],
               residuals[residual_anomalies],
               color='red', s=100)
axes[3].set_title('Residuals with Anomalies')

plt.tight_layout()
plt.show()
```

---

## 7. Complete Pipeline 🎯

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest

class OutlierDetector:
    """Custom outlier detector for sklearn pipelines"""

    def __init__(self, method='isolation_forest', **kwargs):
        self.method = method
        self.kwargs = kwargs
        if method == 'isolation_forest':
            self.detector = IsolationForest(**kwargs)
        elif method == 'lof':
            from sklearn.neighbors import LocalOutlierFactor
            self.detector = LocalOutlierFactor(**kwargs)

    def fit(self, X, y=None):
        if self.method == 'lof':
            # LOF doesn't have separate fit
            pass
        else:
            self.detector.fit(X)
        return self

    def transform(self, X):
        # Remove outliers
        if self.method == 'lof':
            predictions = self.detector.fit_predict(X)
        else:
            predictions = self.detector.predict(X)

        mask = predictions == 1  # Keep inliers
        return X[mask]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

# Complete ML pipeline with outlier handling
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline([
    ('outlier_detection', OutlierDetector(method='isolation_forest', contamination=0.1)),
    ('scaler', RobustScaler()),  # Robust to outliers
    ('classifier', RandomForestClassifier())
])

# Note: Outlier removal changes sample size, so this is for demonstration
# In practice, detect outliers separately before pipeline
```

---

## Quick Reference 📖

**Method Selection Guide:**

| Data Type | Best Method | Why |
|-----------|-------------|-----|
| Univariate, Normal | Z-Score | Fast, simple |
| Univariate, Skewed | IQR | Robust, no assumptions |
| Multivariate, Low-dim | LOF | Captures local density |
| Multivariate, High-dim | Isolation Forest | Scalable, fast |
| Time Series | Rolling Stats + Decomposition | Accounts for trends |
| Mixed Distributions | Percentile | Simple, effective |

**Handling Strategy:**

```python
# Decision tree
if outliers_are_errors:
    remove_or_cap()
elif outliers_are_valuable:
    treat_as_separate_segment()
elif model_is_sensitive:
    transform_data() or use_robust_model()
else:
    keep_and_use_as_feature()
```

---

## Practice Exercises 🏋️

1. Detect outliers in 2D data using multiple methods, compare results
2. Build time series anomaly detector with rolling statistics
3. Create pipeline that handles outliers automatically
4. Compare model performance with/without outlier removal

<details>
<summary>Solutions</summary>

```python
# 1. Multiple methods comparison
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=1, random_state=42)
X = np.vstack([X, [[10, 10], [-10, -10], [10, -10]]])

methods = {
    'Isolation Forest': IsolationForest(contamination=0.01).fit_predict(X),
    'LOF': LocalOutlierFactor(contamination=0.01).fit_predict(X),
    'One-Class SVM': OneClassSVM(nu=0.01).fit_predict(X)
}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, pred) in zip(axes, methods.items()):
    ax.scatter(X[pred == 1, 0], X[pred == 1, 1], alpha=0.6, label='Inliers')
    ax.scatter(X[pred == -1, 0], X[pred == -1, 1], color='red', label='Outliers')
    ax.set_title(name)
    ax.legend()
plt.show()

# 2. Time series anomaly detector
def detect_ts_anomalies(ts, window=30, threshold=3):
    rolling_mean = ts.rolling(window=window, center=True).mean()
    rolling_std = ts.rolling(window=window, center=True).std()
    z_scores = np.abs((ts - rolling_mean) / rolling_std)
    return z_scores > threshold

# 3. Automatic outlier handling pipeline
class AutoOutlierHandler:
    def fit(self, X, y=None):
        self.iso_forest = IsolationForest(contamination=0.1)
        self.iso_forest.fit(X)
        return self

    def transform(self, X):
        predictions = self.iso_forest.predict(X)
        return X[predictions == 1]

# 4. Performance comparison
from sklearn.metrics import accuracy_score

# With outliers
model1 = RandomForestClassifier()
model1.fit(X_train, y_train)
score1 = model1.score(X_test, y_test)

# Without outliers
iso = IsolationForest(contamination=0.1)
mask = iso.fit_predict(X_train) == 1
model2 = RandomForestClassifier()
model2.fit(X_train[mask], y_train[mask])
score2 = model2.score(X_test, y_test)

print(f"With outliers: {score1:.3f}")
print(f"Without outliers: {score2:.3f}")
```
</details>

---

## Key Takeaways 💡

1. **Outliers ≠ always bad** - could be valuable insights
2. **IQR method** most common for univariate
3. **Isolation Forest** best for high-dimensional
4. **Never remove outliers blindly** - investigate first
5. **Tree-based models** naturally robust to outliers
6. **Capping > removal** if you want to keep all data
7. **Time series** needs special treatment (rolling stats)

---

**Module 3 Complete!** 🎉

You now master data engineering: SQL, Pandas, class imbalance, and outlier handling!

**Next Module:** [Module 4 - Feature Engineering & Evaluation →](../Module%204%20-%20Feature%20Engineering%20and%20Model%20Evaluation/Lesson%201%20-%20Feature%20Engineering%20and%20Selection.md)

---

**Congratulations!** You're equipped to handle messy real-world data! 🚀

# Lesson 1: Feature Engineering & Selection 🔧

**Module 4: Feature Engineering & Model Evaluation | Lesson 1 of 4**

Master feature engineering - the art that separates good ML practitioners from great ones!

---

## Why Feature Engineering?

**"Applied ML is basically feature engineering"** - Andrew Ng

**Impact:**
- Better features > Complex models
- Can improve accuracy by 10-30%
- Reduces training time
- Makes models more interpretable

**Real example:** Kaggle winners spend 80% time on features, 20% on models!

---

## 1. Creating Features 🎨

### From Numerical Features

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [50000, 80000, 100000, 120000],
    'years_employed': [2, 10, 20, 30]
})

# Binning / Discretization
df['age_group'] = pd.cut(df['age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle', 'Senior'])

# Polynomial features
df['age_squared'] = df['age'] ** 2
df['age_cubed'] = df['age'] ** 3

# Interactions
df['income_per_year'] = df['income'] / (df['years_employed'] + 1)
df['age_income'] = df['age'] * df['income']

# Log transform
df['income_log'] = np.log1p(df['income'])

# Ratios
df['income_to_age_ratio'] = df['income'] / df['age']
```

### From Categorical Features

```python
# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['age_group'], drop_first=True)

# Label Encoding (for ordinal)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['age_group_encoded'] = le.fit_transform(df['age_group'])

# Target Encoding (mean of target per category)
df['city_target_mean'] = df.groupby('city')['target'].transform('mean')

# Frequency Encoding
df['city_frequency'] = df.groupby('city')['city'].transform('count')

# Count per category
df['city_count'] = df['city'].map(df['city'].value_counts())
```

### From DateTime

```python
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['hour'] = df['date'].dt.hour
df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
df['quarter'] = df['date'].dt.quarter

# Cyclical encoding (for periodic features)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Time since event
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

# Is holiday
holidays = ['2023-01-01', '2023-12-25']
df['is_holiday'] = df['date'].dt.date.astype(str).isin(holidays).astype(int)
```

### From Text

```python
# Length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Character counts
df['num_uppercase'] = df['text'].str.count(r'[A-Z]')
df['num_digits'] = df['text'].str.count(r'\d')
df['num_special'] = df['text'].str.count(r'[^a-zA-Z0-9\s]')

# Presence of patterns
df['has_url'] = df['text'].str.contains(r'http', regex=True).astype(int)
df['has_email'] = df['text'].str.contains(r'@', regex=True).astype(int)
df['has_phone'] = df['text'].str.contains(r'\d{3}-\d{3}-\d{4}', regex=True).astype(int)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=100)
tfidf_features = tfidf.fit_transform(df['text'])
```

---

## 2. Aggregation Features 📊

### Group Statistics

```python
# Customer aggregations
customer_features = df.groupby('customer_id').agg({
    'amount': ['sum', 'mean', 'std', 'min', 'max', 'count'],
    'date': ['min', 'max'],
    'product_category': ['nunique']
}).reset_index()

customer_features.columns = ['_'.join(col).strip() for col in customer_features.columns.values]

# Time-based aggregations
df['amount_last_7_days'] = df.groupby('customer_id')['amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).sum()
)

# Lag features
df['prev_purchase_amount'] = df.groupby('customer_id')['amount'].shift(1)
df['prev_2_purchase_amount'] = df.groupby('customer_id')['amount'].shift(2)

# Rolling statistics
df['rolling_mean_7d'] = df.groupby('customer_id')['amount'].transform(
    lambda x: x.rolling(window=7, min_periods=1).mean()
)

# Expanding (cumulative)
df['cumsum_amount'] = df.groupby('customer_id')['amount'].cumsum()
df['purchase_number'] = df.groupby('customer_id').cumcount() + 1
```

---

## 3. Automated Feature Engineering 🤖

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

X = df[['age', 'income']].values

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print(f"Original features: {X.shape[1]}")
print(f"After polynomial: {X_poly.shape[1]}")
print(f"Feature names: {poly.get_feature_names_out()}")
```

### FeatureTools (AutoML Feature Engineering)

```python
import featuretools as ft

# Create entity set
es = ft.EntitySet(id='customer_data')

# Add entities
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers_df,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions_df,
    index='transaction_id',
    time_index='date'
)

# Add relationship
relationship = ft.Relationship(
    parent_dataframe_name='customers',
    parent_column_name='customer_id',
    child_dataframe_name='transactions',
    child_column_name='customer_id'
)
es = es.add_relationship(relationship)

# Deep feature synthesis
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2,
    verbose=True
)

print(f"Generated {len(feature_defs)} features")
```

---

## 4. Feature Selection 🎯

### Filter Methods (Fast)

```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Variance threshold (remove low-variance features)
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(threshold=0.1)
X_high_var = selector.fit_transform(X)

# Correlation with target
correlation = df.corr()['target'].abs().sort_values(ascending=False)
print(correlation.head(10))

# Statistical tests (ANOVA F-test)
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")

# Mutual information
selector = SelectKBest(mutual_info_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

### Wrapper Methods (Accurate but Slow)

```python
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

# Recursive Feature Elimination
model = RandomForestClassifier(n_estimators=50)
rfe = RFE(estimator=model, n_features_to_select=10, step=1)
rfe.fit(X, y)

selected_features = X.columns[rfe.support_]
feature_ranking = pd.DataFrame({
    'feature': X.columns,
    'ranking': rfe.ranking_
}).sort_values('ranking')

print(feature_ranking)

# Sequential Feature Selection
from sklearn.feature_selection import SequentialFeatureSelector

sfs = SequentialFeatureSelector(model, n_features_to_select=10, direction='forward')
sfs.fit(X, y)
selected_features = X.columns[sfs.get_support()]
```

### Embedded Methods (Best Balance)

```python
# L1 Regularization (Lasso)
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv=5)
lasso.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'coefficient': np.abs(lasso.coef_)
}).sort_values('coefficient', ascending=False)

# Select non-zero coefficients
selected_features = feature_importance[feature_importance['coefficient'] > 0]['feature']

# Tree-based feature importance
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top K
top_k = 10
selected_features = feature_importance.head(top_k)['feature']
```

### Permutation Importance (Best for Model Understanding)

```python
from sklearn.inspection import permutation_importance

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Permutation importance
result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': result.importances_mean,
    'std': result.importances_std
}).sort_values('importance', ascending=False)

print(importance_df.head(10))

# Visualize
import matplotlib.pyplot as plt

plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features')
plt.show()
```

---

## 5. Feature Scaling 📏

```python
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    MaxAbsScaler,
    Normalizer
)

# StandardScaler (z-score normalization)
# Mean = 0, Std = 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler (0-1 range)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# RobustScaler (robust to outliers)
# Uses median and IQR
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# MaxAbsScaler (-1 to 1, preserves sparsity)
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X)

# Normalizer (normalize samples, not features)
normalizer = Normalizer()
X_normalized = normalizer.fit_transform(X)
```

**When to scale:**
- Linear models (required)
- Neural networks (required)
- SVM (required)
- KNN (required)
- Tree-based (NOT needed)

---

## Complete Feature Engineering Pipeline 🔄

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Define feature types
numeric_features = ['age', 'income', 'years_employed']
categorical_features = ['city', 'gender']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline with model
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)
```

---

## Quick Reference 📖

**Feature Engineering Checklist:**

```
✅ Handle missing values
✅ Encode categorical variables
✅ Scale numerical features (if needed)
✅ Create interaction features
✅ Extract datetime components
✅ Aggregate group statistics
✅ Create lag/rolling features (time series)
✅ Remove low-variance features
✅ Select important features
✅ Validate no data leakage
```

**Common Pitfalls:**
- ❌ Fitting scaler on all data (use only train!)
- ❌ Using future data in features (data leakage)
- ❌ Too many features (curse of dimensionality)
- ❌ Not handling unseen categories

---

## Key Takeaways 💡

1. **Feature engineering > Model selection**
2. **Domain knowledge** helps create best features
3. **Interactions** often more predictive than individual features
4. **Always fit on train, transform on test**
5. **Tree models** don't need scaling
6. **Permutation importance** best for understanding
7. **Start simple** - add complexity gradually

---

**Next:** [Lesson 2 - Model Evaluation Metrics →](Lesson%202%20-%20Model%20Evaluation%20Metrics.md)

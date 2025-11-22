# Lesson 1: Linear Models 📈

**Module 2: Classical ML & Ensemble Methods | Lesson 1 of 4**

Master the foundation of supervised learning - Linear Models!

---

## Why Linear Models?

**Simple, interpretable, and surprisingly powerful:**
- Fast to train, even on large datasets
- Easy to interpret (understand feature importance)
- Baseline for comparison
- Work well when relationships are roughly linear
- Foundation for understanding complex models

---

## 1. Linear Regression 📉

**Goal:** Predict continuous values using a linear combination of features.

### The Math

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

where:
y = predicted value
β₀ = intercept (bias)
β₁, β₂, ... = coefficients (weights)
x₁, x₂, ... = features
ε = error term
```

**Matrix Form:**
```
y = Xβ + ε
```

### Simple Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simple dataset: house size → price
X = np.array([[500], [1000], [1500], [2000], [2500]])  # sq ft
y = np.array([150000, 250000, 350000, 450000, 550000])  # price

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Predictions
X_test = np.array([[1200], [1800]])
predictions = model.predict(X_test)
print(f"Predictions: {predictions}")
# [270000, 410000]

# Model parameters
print(f"Coefficient (slope): {model.coef_[0]:.2f}")  # 200
print(f"Intercept: {model.intercept_:.2f}")  # 50000
# Equation: price = 50000 + 200 × sq_ft

# Visualize
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Predicted')
plt.xlabel('Square Feet')
plt.ylabel('Price ($)')
plt.legend()
plt.show()
```

### Multiple Features

```python
import pandas as pd

# Multiple features
data = pd.DataFrame({
    'sqft': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 5, 8, 2, 1],
    'price': [250000, 350000, 450000, 550000, 650000]
})

X = data[['sqft', 'bedrooms', 'age']]
y = data['price']

model = LinearRegression()
model.fit(X, y)

# Coefficients
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.2f}")
# sqft: 180.00
# bedrooms: 15000.00
# age: -5000.00 (negative = older → cheaper)

# Predict
new_house = [[1800, 3, 6]]
price = model.predict(new_house)
print(f"Predicted price: ${price[0]:,.2f}")
```

### Evaluation Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: ${rmse:,.2f}")  # Root Mean Squared Error
print(f"MAE: ${mae:,.2f}")    # Mean Absolute Error
print(f"R²: {r2:.3f}")         # R-squared (0-1, higher = better)

# R² interpretation:
# 0.9 = Model explains 90% of variance
# 0.5 = Model explains 50% of variance
# <0 = Model worse than predicting mean
```

**Common Metrics:**
- **MSE (Mean Squared Error)**: Penalizes large errors
- **RMSE**: Same units as target, easier to interpret
- **MAE**: Average absolute error, robust to outliers
- **R² (R-squared)**: Proportion of variance explained (0-1)

---

## 2. Logistic Regression 🎯

**Goal:** Binary classification (predict probability of class 1).

### The Math

```
P(y=1|x) = 1 / (1 + e^(-(β₀ + β₁x₁ + ... + βₙxₙ)))

This is the sigmoid function: σ(z) = 1 / (1 + e^(-z))
```

**Sigmoid squashes output to [0, 1] range:**
```
z = -∞  → σ(z) = 0
z = 0   → σ(z) = 0.5
z = +∞  → σ(z) = 1
```

### Binary Classification Example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate binary classification dataset
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict class
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred[:10]}")
# [0 1 1 0 1 0 0 1 1 0]

# Predict probability
y_prob = model.predict_proba(X_test)
print(f"Probabilities (first 5):\n{y_prob[:5]}")
# [[0.82, 0.18],  # 82% class 0, 18% class 1
#  [0.23, 0.77],  # 23% class 0, 77% class 1
#  ...]

# Accuracy
from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")

print(classification_report(y_test, y_pred))
```

### Real-World Example: Email Spam Detection

```python
from sklearn.feature_extraction.text import CountVectorizer

# Sample emails
emails = [
    "Win free money now!",
    "Meeting scheduled for tomorrow",
    "Claim your prize immediately",
    "Project update attached",
    "Urgent: Act now to win",
    "Quarterly report ready"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=spam, 0=not spam

# Convert text to features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()

# Train
model = LogisticRegression()
model.fit(X, labels)

# Predict new email
new_emails = ["Free prize waiting for you", "Meeting notes from today"]
X_new = vectorizer.transform(new_emails).toarray()
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)

for email, pred, prob in zip(new_emails, predictions, probabilities):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    confidence = prob[pred] * 100
    print(f"{email}")
    print(f"  → {label} ({confidence:.1f}% confident)\n")
```

### Multi-class Classification

```python
from sklearn.datasets import load_iris

# Iris dataset: 3 classes (species of flowers)
iris = load_iris()
X, y = iris.data, iris.target  # 0, 1, 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression handles multi-class automatically
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Probabilities for all 3 classes
y_prob = model.predict_proba(X_test)
print(f"Class probabilities (first sample):\n{y_prob[0]}")
# [0.89, 0.10, 0.01] → 89% class 0, 10% class 1, 1% class 2

# Accuracy
print(f"Accuracy: {model.score(X_test, y_test):.3f}")
```

### Evaluation for Classification

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

# Binary classification
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probability of class 1

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
# [[TN, FP],
#  [FN, TP]]

# Metrics from confusion matrix
TN, FP, FN, TP = cm.ravel()
accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)  # Of predicted positives, how many correct?
recall = TP / (TP + FN)      # Of actual positives, how many found?
f1 = 2 * (precision * recall) / (precision + recall)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")

# ROC-AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")  # 0.5 = random, 1.0 = perfect

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

---

## 3. Regularization - Preventing Overfitting 🎯

**Problem:** Linear models with many features can overfit.

**Solution:** Add penalty to large coefficients.

### Ridge Regression (L2 Regularization)

**Adds squared magnitude of coefficients to loss:**

```
Loss = MSE + α × Σ(βᵢ²)

α (alpha) = regularization strength
- α = 0: No regularization (standard linear regression)
- α → ∞: All coefficients → 0
```

```python
from sklearn.linear_model import Ridge

# Create dataset with many features
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=50, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Standard Linear Regression (might overfit)
lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"Linear Reg - Train R²: {lr.score(X_train, y_train):.3f}")
print(f"Linear Reg - Test R²: {lr.score(X_test, y_test):.3f}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"Ridge - Train R²: {ridge.score(X_train, y_train):.3f}")
print(f"Ridge - Test R²: {ridge.score(X_test, y_test):.3f}")
# Ridge usually has lower train score but better test score!

# Compare coefficients
print(f"\nMax coefficient magnitude:")
print(f"Linear Reg: {np.max(np.abs(lr.coef_)):.2f}")
print(f"Ridge: {np.max(np.abs(ridge.coef_)):.2f}")
# Ridge has smaller coefficients
```

**Finding best alpha:**

```python
from sklearn.linear_model import RidgeCV

# Cross-validation to find best alpha
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train, y_train)

print(f"Best alpha: {ridge_cv.alpha_}")
print(f"Test R²: {ridge_cv.score(X_test, y_test):.3f}")
```

### Lasso Regression (L1 Regularization)

**Adds absolute magnitude of coefficients:**

```
Loss = MSE + α × Σ|βᵢ|
```

**Key difference:** Can shrink coefficients to exactly 0 → **Feature selection**!

```python
from sklearn.linear_model import Lasso

# Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

print(f"Lasso - Train R²: {lasso.score(X_train, y_train):.3f}")
print(f"Lasso - Test R²: {lasso.score(X_test, y_test):.3f}")

# Count non-zero coefficients
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"\nFeatures used: {n_nonzero} / {X.shape[1]}")
# Lasso automatically selected important features!

# Get important features
feature_importance = pd.DataFrame({
    'feature': range(X.shape[1]),
    'coefficient': lasso.coef_
})
important = feature_importance[feature_importance['coefficient'] != 0]
print(f"\nImportant features:\n{important}")
```

**Finding best alpha:**

```python
from sklearn.linear_model import LassoCV

lasso_cv = LassoCV(alphas=None, cv=5, random_state=42)
lasso_cv.fit(X_train, y_train)

print(f"Best alpha: {lasso_cv.alpha_:.5f}")
print(f"Test R²: {lasso_cv.score(X_test, y_test):.3f}")
```

### Elastic Net (L1 + L2)

**Combines Ridge and Lasso:**

```
Loss = MSE + α × (ρ × Σ|βᵢ| + (1-ρ) × Σ(βᵢ²))

ρ (rho) = L1 ratio
- ρ = 0: Pure Ridge
- ρ = 1: Pure Lasso
- 0 < ρ < 1: Mix of both
```

```python
from sklearn.linear_model import ElasticNet

# Elastic Net
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # 50% L1, 50% L2
elastic.fit(X_train, y_train)

print(f"Elastic Net - Test R²: {elastic.score(X_test, y_test):.3f}")
print(f"Non-zero coefficients: {np.sum(elastic.coef_ != 0)}")
```

**When to use what:**
- **Ridge**: Many features, all relevant
- **Lasso**: Many features, only some relevant (want feature selection)
- **Elastic Net**: Many correlated features, want feature selection

---

## 4. Comparison & When to Use

```python
import time

# Create larger dataset
X, y = make_regression(n_samples=10000, n_features=100, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
}

results = []
for name, model in models.items():
    # Train and time
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Evaluate
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    # Coefficient stats
    n_nonzero = np.sum(model.coef_ != 0)
    max_coef = np.max(np.abs(model.coef_))

    results.append({
        'Model': name,
        'Train R²': f"{train_r2:.4f}",
        'Test R²': f"{test_r2:.4f}",
        'Non-zero': n_nonzero,
        'Max |coef|': f"{max_coef:.2f}",
        'Time (s)': f"{train_time:.4f}"
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## 5. Practical Tips 💡

### Preprocessing for Linear Models

```python
from sklearn.preprocessing import StandardScaler

# Always standardize features for regularized models!
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Why? Regularization penalizes coefficients
# Features on different scales → unfair penalties
```

### Handling Categorical Features

```python
# One-hot encoding
data = pd.DataFrame({
    'size': [1200, 1500, 1800],
    'city': ['NYC', 'LA', 'NYC'],
    'price': [500000, 400000, 550000]
})

# Convert categorical to dummy variables
X = pd.get_dummies(data[['size', 'city']], drop_first=True)
# drop_first=True prevents multicollinearity
print(X)
#    size  city_NYC
# 0  1200         1
# 1  1500         0
# 2  1800         1
```

### Feature Engineering for Linear Models

```python
# Polynomial features for non-linear relationships
from sklearn.preprocessing import PolynomialFeatures

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x²

# Linear model fails
lr = LinearRegression()
lr.fit(X, y)
print(f"Linear R²: {lr.score(X, y):.3f}")  # Poor fit

# Add polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# [[1, 1, 1],   # [1, x, x²]
#  [1, 2, 4],
#  ...]

lr.fit(X_poly, y)
print(f"Polynomial R²: {lr.score(X_poly, y):.3f}")  # Perfect fit!
```

---

## Quick Reference 📖

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| **Linear Regression** | Continuous prediction | Fast, interpretable | Assumes linearity |
| **Logistic Regression** | Classification | Fast, probabilities | Assumes linearity |
| **Ridge** | Many features, all relevant | Prevents overfitting | Keeps all features |
| **Lasso** | Many features, some irrelevant | Feature selection | Might miss correlated features |
| **Elastic Net** | Many correlated features | Balance Ridge+Lasso | Two hyperparameters |

**Regularization Cheat Sheet:**
```
No regularization:     α = 0
Light regularization:  α = 0.1
Medium:                α = 1.0
Heavy:                 α = 10.0
Very heavy:            α = 100.0
```

---

## Practice Exercises 🏋️

1. Load diabetes dataset, predict disease progression with Linear Regression
2. Compare Ridge vs Lasso on high-dimensional data
3. Build spam classifier with Logistic Regression
4. Find optimal alpha using cross-validation

<details>
<summary>Solutions</summary>

```python
# 1. Diabetes prediction
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)
print(f"R²: {lr.score(X_test, y_test):.3f}")

# 2. Ridge vs Lasso
X, y = make_regression(n_samples=100, n_features=100, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=1.0)
lasso = Lasso(alpha=0.1)

ridge.fit(X_train_scaled, y_train)
lasso.fit(X_train_scaled, y_train)

print(f"Ridge non-zero: {np.sum(ridge.coef_ != 0)}")  # All
print(f"Lasso non-zero: {np.sum(lasso.coef_ != 0)}")  # Few

# 3. Spam classifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

categories = ['sci.med', 'comp.graphics']
newsgroups = fetch_20newsgroups(subset='train', categories=categories)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(newsgroups.data)
y = newsgroups.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)
print(f"Accuracy: {model.score(X_test, y_test):.3f}")

# 4. Optimal alpha
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=5)
ridge_cv.fit(X_train, y_train)
print(f"Best alpha: {ridge_cv.alpha_:.5f}")
```
</details>

---

## Key Takeaways 💡

1. **Linear models** are fast, interpretable, and great baselines
2. **Regularization** prevents overfitting (use Ridge/Lasso/ElasticNet)
3. **Lasso** does automatic feature selection (coefficients → 0)
4. **Always standardize** features for regularized models
5. **Logistic Regression** for classification, despite the name!
6. **Start simple** - linear models before complex ones

---

**Next:** [Lesson 2 - Tree-Based Models (XGBoost) →](Lesson%202%20-%20Tree-Based%20Models.md)

---

**Congratulations!** You now understand linear models - the foundation of ML! 🎉

# Lesson 3: Instance-Based & Probabilistic Models 🎯

**Module 2: Classical ML & Ensemble Methods | Lesson 3 of 4**

Master KNN, SVM, and Naive Bayes - simple yet powerful algorithms!

---

## Overview

Three fundamentally different approaches to ML:

**Instance-Based (KNN):** "You are the average of your neighbors"
**Support Vector Machines (SVM):** "Find the best boundary"
**Naive Bayes:** "Calculate probabilities using Bayes' Theorem"

---

## 1. K-Nearest Neighbors (KNN) 👥

**Simplest ML algorithm: Classify based on closest training examples.**

### The Concept

```
New point (?) → Find K nearest neighbors → Take majority vote

Example: K=3
Neighbors: [Cat, Cat, Dog]
Prediction: Cat (2 out of 3)
```

**No training phase!** Just stores data, computes at prediction time.

### Classification Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Create dataset
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Accuracy
print(f"Train Accuracy: {knn.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {knn.score(X_test, y_test):.3f}")

# Predict with probabilities
X_new = [[0, 0]]
prediction = knn.predict(X_new)
probabilities = knn.predict_proba(X_new)

print(f"Prediction: {prediction[0]}")
print(f"Probabilities: {probabilities[0]}")
```

### Visualizing Decision Boundary

```python
def plot_decision_boundary(model, X, y):
    # Create mesh
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predict on mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{model.__class__.__name__}')
    plt.show()

# Compare K=1 vs K=15
for k in [1, 5, 15]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print(f"K={k}: Test Accuracy = {knn.score(X_test, y_test):.3f}")
    plot_decision_boundary(knn, X_test, y_test)
```

**Observations:**
- K=1: Overfits (captures noise)
- K=15: Smoother boundary
- Optimal K: Balance between bias and variance

### Regression with KNN

```python
from sklearn.neighbors import KNeighborsRegressor

# Generate regression data
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN Regression
knn_reg = KNeighborsRegressor(n_neighbors=5)
knn_reg.fit(X_train, y_train)

# Predict
y_pred = knn_reg.predict(X_test)

# Metrics
from sklearn.metrics import mean_squared_error, r2_score
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.legend()
plt.show()
```

### Finding Optimal K

```python
from sklearn.model_selection import cross_val_score

# Try different K values
k_values = range(1, 31)
cv_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Plot
plt.plot(k_values, cv_scores)
plt.xlabel('K')
plt.ylabel('Cross-Validation Accuracy')
plt.title('Optimal K for KNN')
plt.show()

# Best K
best_k = k_values[np.argmax(cv_scores)]
print(f"Best K: {best_k} (Accuracy: {max(cv_scores):.3f})")
```

### Distance Metrics

```python
# Different distance metrics
metrics = ['euclidean', 'manhattan', 'chebyshev']

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train, y_train)
    print(f"{metric.capitalize()}: {knn.score(X_test, y_test):.3f}")

# Euclidean: √((x1-x2)² + (y1-y2)²) - default
# Manhattan: |x1-x2| + |y1-y2| - city block distance
# Chebyshev: max(|x1-x2|, |y1-y2|) - chessboard distance
```

### Scaling is Critical!

```python
from sklearn.preprocessing import StandardScaler

# Without scaling
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print(f"Without scaling: {knn.score(X_test, y_test):.3f}")

# With scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn.fit(X_train_scaled, y_train)
print(f"With scaling: {knn.score(X_test_scaled, y_test):.3f}")
# Usually better with scaling!
```

**Why?** KNN uses distance. Features with larger values dominate distance calculation.

---

## 2. Support Vector Machines (SVM) 🎯

**Find the hyperplane that maximizes margin between classes.**

### The Concept

```
Class 0: ●●●●
Class 1:     ○○○○
         |
    Best boundary maximizes distance to both classes
         ↑
    Called "maximum margin hyperplane"
```

**Support Vectors:** Points closest to boundary (most important).

### Linear SVM

```python
from sklearn.svm import SVC

# Linear kernel
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_train, y_train)

print(f"Train Accuracy: {svm_linear.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {svm_linear.score(X_test, y_test):.3f}")

# Support vectors
print(f"Number of support vectors: {len(svm_linear.support_)}")
print(f"Support vectors per class: {svm_linear.n_support_}")
```

### Non-Linear SVM (RBF Kernel)

**For non-linearly separable data:**

```python
# Generate non-linear data
from sklearn.datasets import make_circles
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear SVM (will fail)
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
print(f"Linear SVM: {svm_linear.score(X_test, y_test):.3f}")  # Poor!

# RBF (Radial Basis Function) kernel
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_train, y_train)
print(f"RBF SVM: {svm_rbf.score(X_test, y_test):.3f}")  # Much better!

# Visualize
plot_decision_boundary(svm_rbf, X_test, y_test)
```

### Hyperparameters: C and Gamma

**C (Regularization):**
- Small C: Wider margin, more errors allowed (high bias)
- Large C: Narrow margin, fewer errors (high variance)

**Gamma (Kernel coefficient):**
- Small gamma: Far reach, smooth boundary
- Large gamma: Close reach, complex boundary

```python
# Effect of C
C_values = [0.1, 1, 10, 100]

for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_train, y_train)
    print(f"C={C:>5}: Train={svm.score(X_train, y_train):.3f}, "
          f"Test={svm.score(X_test, y_test):.3f}")

# Effect of Gamma
gamma_values = [0.001, 0.01, 0.1, 1]

for gamma in gamma_values:
    svm = SVC(kernel='rbf', C=1, gamma=gamma)
    svm.fit(X_train, y_train)
    print(f"Gamma={gamma}: Train={svm.score(X_train, y_train):.3f}, "
          f"Test={svm.score(X_test, y_test):.3f}")
```

### Different Kernels

```python
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:
    svm = SVC(kernel=kernel, gamma='scale')
    svm.fit(X_train, y_train)
    print(f"{kernel.capitalize():>8} kernel: {svm.score(X_test, y_test):.3f}")

# linear: Linear separation
# poly: Polynomial decision boundary (degree parameter)
# rbf: Radial basis function (most common)
# sigmoid: Similar to neural network
```

### SVM Regression (SVR)

```python
from sklearn.svm import SVR

# Generate regression data
X, y = make_regression(n_samples=200, n_features=1, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### Tuning SVM

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.3f}")
print(f"Test score: {grid.score(X_test, y_test):.3f}")
```

### SVM Scaling

```python
# SVM is VERY sensitive to feature scaling!
from sklearn.pipeline import Pipeline

# Pipeline: scale → SVM
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', C=1, gamma='scale'))
])

pipeline.fit(X_train, y_train)
print(f"Pipeline Test Accuracy: {pipeline.score(X_test, y_test):.3f}")
```

---

## 3. Naive Bayes 📊

**Probabilistic classifier based on Bayes' Theorem.**

### Bayes' Theorem Recap

```
P(Class|Features) = P(Features|Class) × P(Class) / P(Features)

Posterior = Likelihood × Prior / Evidence
```

**"Naive" assumption:** Features are independent given the class.

### Types of Naive Bayes

1. **GaussianNB:** Features follow Gaussian (normal) distribution
2. **MultinomialNB:** Features are counts (text classification)
3. **BernoulliNB:** Features are binary (0/1)

### Gaussian Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB

# Classification dataset
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gaussian NB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

print(f"Train Accuracy: {gnb.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {gnb.score(X_test, y_test):.3f}")

# Predict probabilities
y_prob = gnb.predict_proba(X_test)
print(f"\nProbabilities for first test sample:")
print(f"Classes: {gnb.classes_}")
print(f"Probs: {y_prob[0]}")
```

### Multinomial Naive Bayes (Text Classification)

**Perfect for text data (word counts).**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Text data
texts = [
    "I love this movie it was fantastic",
    "Terrible movie waste of time",
    "Great film highly recommend",
    "Awful experience would not watch again",
    "Amazing movie best I've seen",
    "Horrible acting and plot"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# Convert text to word counts
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train
mnb = MultinomialNB()
mnb.fit(X, labels)

# Predict new reviews
new_reviews = [
    "Great movie loved it",
    "Terrible film avoid"
]
X_new = vectorizer.transform(new_reviews)
predictions = mnb.predict(X_new)
probabilities = mnb.predict_proba(X_new)

for review, pred, prob in zip(new_reviews, predictions, probabilities):
    sentiment = "Positive" if pred == 1 else "Negative"
    confidence = prob[pred] * 100
    print(f"\"{review}\"")
    print(f"  → {sentiment} ({confidence:.1f}% confident)\n")
```

### Real-World: Spam Classification

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Load email-like data
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)

X_train_text = newsgroups.data
y_train = newsgroups.target

newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
X_test_text = newsgroups_test.data
y_test = newsgroups_test.target

# Convert to features
vectorizer = TfidfVectorizer(max_features=1000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train Naive Bayes
mnb = MultinomialNB(alpha=1.0)  # alpha = smoothing parameter
mnb.fit(X_train, y_train)

print(f"Train Accuracy: {mnb.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {mnb.score(X_test, y_test):.3f}")

# Most informative features
feature_names = vectorizer.get_feature_names_out()
class_0_prob = mnb.feature_log_prob_[0]
class_1_prob = mnb.feature_log_prob_[1]

# Top words for each class
top_class_0 = np.argsort(class_0_prob)[-10:]
top_class_1 = np.argsort(class_1_prob)[-10:]

print("\nTop words for class 0:")
print([feature_names[i] for i in top_class_0])

print("\nTop words for class 1:")
print([feature_names[i] for i in top_class_1])
```

### Bernoulli Naive Bayes

**For binary features (presence/absence).**

```python
from sklearn.naive_bayes import BernoulliNB

# Binary features (word present or not)
vectorizer = CountVectorizer(binary=True)  # Binary counts
X = vectorizer.fit_transform(texts)

bnb = BernoulliNB()
bnb.fit(X, labels)

print(f"Accuracy: {bnb.score(X, labels):.3f}")
```

### Smoothing Parameter (Alpha)

```python
# Alpha controls smoothing (Laplace/Lidstone smoothing)
alphas = [0.1, 0.5, 1.0, 2.0, 5.0]

for alpha in alphas:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_train, y_train)
    print(f"Alpha={alpha}: Test Accuracy = {mnb.score(X_test, y_test):.3f}")

# alpha=1.0 (Laplace smoothing) is default
# Higher alpha = more smoothing = less overfitting
```

---

## 4. Comparison 📊

```python
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Generate dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM (Linear)': SVC(kernel='linear'),
    'SVM (RBF)': SVC(kernel='rbf', C=1, gamma='scale'),
    'Naive Bayes': GaussianNB()
}

import time

results = []
for name, model in models.items():
    # Use scaled data (except Naive Bayes doesn't need it)
    if 'Naive Bayes' in name:
        X_tr, X_te = X_train, X_test
    else:
        X_tr, X_te = X_train_scaled, X_test_scaled

    # Train
    start = time.time()
    model.fit(X_tr, y_train)
    train_time = time.time() - start

    # Predict
    start = time.time()
    y_pred = model.predict(X_te)
    pred_time = time.time() - start

    # Metrics
    train_acc = model.score(X_tr, y_train)
    test_acc = model.score(X_te, y_test)

    results.append({
        'Model': name,
        'Train Acc': f"{train_acc:.3f}",
        'Test Acc': f"{test_acc:.3f}",
        'Train Time': f"{train_time:.4f}s",
        'Pred Time': f"{pred_time:.4f}s"
    })

import pandas as pd
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## When to Use What? 🤔

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| **KNN** | Small datasets, simple baseline | No training, interpretable | Slow predictions, memory-intensive |
| **SVM (Linear)** | High-dimensional data (text) | Fast, works well | Only linear boundaries |
| **SVM (RBF)** | Non-linear boundaries | Powerful, flexible | Slow on large data, hard to tune |
| **Naive Bayes** | **Text classification** | **Very fast, works well** | Independence assumption |

**Decision Guide:**
- **Text classification** → Naive Bayes (MultinomialNB)
- **Small dataset, need interpretability** → KNN
- **Non-linear, small dataset** → SVM (RBF)
- **High-dimensional linear** → SVM (Linear)
- **Large dataset** → Try tree-based methods instead

---

## Practical Tips 💡

### 1. Always Scale for KNN and SVM

```python
# Pipeline ensures proper scaling
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC(kernel='rbf'))
])

pipeline.fit(X_train, y_train)
```

### 2. KNN for Anomaly Detection

```python
from sklearn.neighbors import NearestNeighbors

# Find outliers: points far from all neighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_train)

# Distance to 5th nearest neighbor
distances, indices = nn.kneighbors(X_train)
avg_distances = distances.mean(axis=1)

# Outliers: large average distance
threshold = np.percentile(avg_distances, 95)
outliers = avg_distances > threshold
print(f"Outliers detected: {outliers.sum()}")
```

### 3. Naive Bayes for Incremental Learning

```python
# Partial fit (online learning)
gnb = GaussianNB()

# Learn in batches
for i in range(0, len(X_train), 100):
    X_batch = X_train[i:i+100]
    y_batch = y_train[i:i+100]
    gnb.partial_fit(X_batch, y_batch, classes=np.unique(y))

# Useful for streaming data
```

### 4. Combine Models

```python
from sklearn.ensemble import VotingClassifier

# Ensemble of different models
ensemble = VotingClassifier(
    estimators=[
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('svm', SVC(kernel='rbf', probability=True)),
        ('nb', GaussianNB())
    ],
    voting='soft'  # Use predicted probabilities
)

ensemble.fit(X_train_scaled, y_train)
print(f"Ensemble Accuracy: {ensemble.score(X_test_scaled, y_test):.3f}")
```

---

## Quick Reference 📖

**KNN Hyperparameters:**
```python
KNeighborsClassifier(
    n_neighbors=5,        # Number of neighbors
    metric='euclidean',   # Distance metric
    weights='uniform'     # 'uniform' or 'distance'
)
```

**SVM Hyperparameters:**
```python
SVC(
    kernel='rbf',        # 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,              # Regularization (smaller = more regularization)
    gamma='scale'       # Kernel coefficient ('scale', 'auto', or float)
)
```

**Naive Bayes Hyperparameters:**
```python
MultinomialNB(
    alpha=1.0           # Smoothing parameter (0 = no smoothing)
)
```

---

## Practice Exercises 🏋️

1. Find optimal K for KNN using cross-validation
2. Compare SVM kernels on moon-shaped data
3. Build spam classifier with Multinomial NB
4. Create ensemble of KNN, SVM, and Naive Bayes

<details>
<summary>Solutions</summary>

```python
# 1. Optimal K
from sklearn.model_selection import cross_val_score

k_range = range(1, 31)
cv_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_range[np.argmax(cv_scores)]
print(f"Best K: {best_k}")

# 2. SVM kernels on moons
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for kernel in ['linear', 'poly', 'rbf']:
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    print(f"{kernel}: {svm.score(X_test, y_test):.3f}")

# 3. Spam classifier
from sklearn.feature_extraction.text import TfidfVectorizer

texts = ["Buy now!", "Meeting tomorrow", "Win prize!", "Project update"]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

mnb = MultinomialNB()
mnb.fit(X, labels)

# 4. Ensemble
ensemble = VotingClassifier([
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True)),
    ('nb', GaussianNB())
], voting='soft')

ensemble.fit(X_train_scaled, y_train)
print(f"Ensemble: {ensemble.score(X_test_scaled, y_test):.3f}")
```
</details>

---

## Key Takeaways 💡

1. **KNN:** Simple, no training, but slow predictions
2. **SVM:** Powerful for non-linear data, but needs tuning
3. **Naive Bayes:** **Best for text classification**, very fast
4. **Scaling critical** for KNN and SVM
5. **Choose Naive Bayes variant** based on data type
6. **SVM with RBF kernel** for non-linear boundaries
7. **KNN good for** small datasets and baselines

---

**Next:** [Lesson 4 - Ensemble Methods →](Lesson%204%20-%20Ensemble%20Methods.md)

---

**Congratulations!** You now understand instance-based and probabilistic models! 🎉

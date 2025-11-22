# Classical Machine Learning Algorithms 📊

Classic ML algorithms that every ML practitioner should know. These form the foundation before deep learning.

## 1. Linear Regression 📈

**Purpose:** Predict continuous values by fitting a straight line.

### Theory
Find the best line: `y = mx + b` (or `y = w₁x₁ + w₂x₂ + ... + b`)

**Cost Function (Mean Squared Error):**
```
J(w,b) = (1/2m) Σ(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
```

### When to Use
- Linear relationship between features and target
- Simple baseline model
- Interpretable results needed

### Implementation
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Make predictions
X_new = np.array([[6]])
prediction = model.predict(X_new)

# Model parameters
print(f"Slope (weight): {model.coef_[0]}")
print(f"Intercept (bias): {model.intercept_}")
```

### Assumptions
1. Linear relationship
2. Independence of errors
3. Homoscedasticity (constant variance)
4. Normality of errors
5. No multicollinearity (for multiple features)

### Variants
- **Ridge Regression:** L2 regularization (penalty on large weights)
- **Lasso Regression:** L1 regularization (can zero out features)
- **Elastic Net:** Combination of Ridge + Lasso

---

## 2. Logistic Regression 🎯

**Purpose:** Binary classification (yes/no, spam/not spam).

### Theory
Uses sigmoid function to output probabilities (0 to 1):

```
σ(z) = 1 / (1 + e⁻ᶻ)
where z = w·x + b
```

**Decision boundary:** If σ(z) ≥ 0.5 → class 1, else → class 0

### When to Use
- Binary classification
- Need probability estimates
- Linear decision boundary is acceptable
- Interpretable model required

### Implementation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=100, n_features=2,
                          n_redundant=0, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict class
prediction = model.predict([[1, 2]])

# Predict probability
probability = model.predict_proba([[1, 2]])
print(f"P(class=0): {probability[0][0]:.2f}")
print(f"P(class=1): {probability[0][1]:.2f}")
```

### Multiclass Extension
- **One-vs-Rest (OvR):** Train N binary classifiers
- **Multinomial:** Direct multiclass using softmax

---

## 3. Decision Trees 🌳

**Purpose:** Classification and regression using tree-like decisions.

### Theory
Split data recursively based on features to create a tree:

```
         [Root: Age < 30?]
        /                 \
    Yes                    No
    /                       \
[Income < 50k?]        [Bought = Yes]
  /         \
Yes         No
/             \
[No]        [Yes]
```

**Splitting Criteria:**
- **Gini Impurity:** `Gini = 1 - Σ(pᵢ)²`
- **Entropy:** `H = -Σ(pᵢ log₂ pᵢ)`
- **MSE:** For regression

### When to Use
- Non-linear relationships
- Mixed feature types (numerical + categorical)
- Interpretability important (can visualize tree)
- No need for feature scaling

### Implementation
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Train model
X = [[0, 0], [1, 1], [2, 2], [3, 3]]
y = [0, 0, 1, 1]

clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)

# Visualize tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=['X1', 'X2'])
plt.show()

# Feature importance
print("Feature importances:", clf.feature_importances_)
```

### Hyperparameters
- `max_depth`: Maximum tree depth
- `min_samples_split`: Minimum samples to split
- `min_samples_leaf`: Minimum samples in leaf
- `max_features`: Features to consider for split

### Pros & Cons
✅ Easy to interpret
✅ Handle non-linear relationships
✅ No feature scaling needed
❌ Prone to overfitting
❌ Unstable (small data changes → different tree)

---

## 4. Random Forests 🌲🌲🌲

**Purpose:** Ensemble of decision trees for better performance.

### Theory
**Bagging (Bootstrap Aggregating):**
1. Create N bootstrap samples (random sampling with replacement)
2. Train a decision tree on each sample
3. Average predictions (regression) or vote (classification)

**Random feature selection:** Each split considers random subset of features

### When to Use
- Better performance than single decision tree
- Reduce overfitting
- Feature importance needed
- Handle missing values

### Implementation
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20,
                          n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,      # Number of trees
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train, y_train)

# Evaluate
accuracy = rf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Feature importance
importances = rf.feature_importances_
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### Hyperparameters
- `n_estimators`: Number of trees (more is better, but slower)
- `max_depth`: Depth of each tree
- `max_features`: Features per split ('sqrt' for classification)
- `bootstrap`: Use bootstrap sampling (True/False)

---

## 5. K-Nearest Neighbors (KNN) 👥

**Purpose:** Classification/regression based on nearest neighbors.

### Theory
**Classification:** Majority vote of K nearest neighbors
**Regression:** Average of K nearest neighbors

**Distance Metrics:**
- Euclidean: `d = √Σ(xᵢ - yᵢ)²`
- Manhattan: `d = Σ|xᵢ - yᵢ|`
- Minkowski: General form

### When to Use
- Small to medium datasets
- Low dimensional data
- Non-linear decision boundaries
- Anomaly detection

### Implementation
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Data
X_train = [[0, 0], [1, 1], [2, 2], [3, 3]]
y_train = [0, 0, 1, 1]
X_test = [[1.5, 1.5]]

# Scale features (important for KNN!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Predict
prediction = knn.predict(X_test_scaled)
distances, indices = knn.kneighbors(X_test_scaled)
print(f"Nearest neighbors indices: {indices}")
print(f"Distances: {distances}")
```

### Choosing K
- Small K: More sensitive to noise
- Large K: Smoother decision boundaries
- Rule of thumb: K = √n (n = number of samples)
- Use cross-validation to find optimal K

### Pros & Cons
✅ Simple and intuitive
✅ No training phase
✅ Naturally handles multiclass
❌ Slow prediction for large datasets
❌ Sensitive to feature scaling
❌ Curse of dimensionality

---

## 6. Support Vector Machines (SVM) ⚖️

**Purpose:** Find optimal hyperplane for classification.

### Theory
Find hyperplane with **maximum margin** between classes.

**Kernel Trick:** Transform data to higher dimensions for non-linear boundaries:
- Linear: `K(x, y) = x·y`
- Polynomial: `K(x, y) = (x·y + c)ᵈ`
- RBF (Gaussian): `K(x, y) = e^(-γ||x-y||²)`

### When to Use
- Clear margin of separation
- High-dimensional data
- More features than samples
- Binary classification (primarily)

### Implementation
```python
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

# Non-linear data
X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

# Scale data (crucial for SVM!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linear SVM
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_scaled, y)

# RBF (Radial Basis Function) SVM
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_rbf.fit(X_scaled, y)

# Polynomial SVM
svm_poly = SVC(kernel='poly', degree=3, C=1.0)
svm_poly.fit(X_scaled, y)
```

### Hyperparameters
- `C`: Regularization (smaller = more regularization)
- `kernel`: Type of kernel ('linear', 'rbf', 'poly')
- `gamma`: Kernel coefficient (for RBF, poly)
- `degree`: Polynomial degree

### Pros & Cons
✅ Effective in high dimensions
✅ Memory efficient (uses support vectors)
✅ Versatile (different kernels)
❌ Slow for large datasets
❌ Sensitive to feature scaling
❌ Difficult to interpret

---

## 7. Naive Bayes 🎲

**Purpose:** Probabilistic classification using Bayes' theorem.

### Theory
**Bayes' Theorem:**
```
P(y|X) = P(X|y) · P(y) / P(X)

where:
P(y|X) = Posterior probability
P(X|y) = Likelihood
P(y) = Prior probability
P(X) = Evidence
```

**"Naive" Assumption:** Features are independent given the class.

### Variants
1. **Gaussian NB:** Continuous features (assumes normal distribution)
2. **Multinomial NB:** Discrete counts (text classification)
3. **Bernoulli NB:** Binary features

### When to Use
- Text classification (spam detection, sentiment analysis)
- Fast training and prediction needed
- Small datasets
- Baseline model

### Implementation
```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.datasets import make_classification

# Gaussian Naive Bayes (continuous features)
X, y = make_classification(n_samples=100, n_features=4)
gnb = GaussianNB()
gnb.fit(X, y)
predictions = gnb.predict(X)

# Multinomial Naive Bayes (count features - e.g., text)
from sklearn.feature_extraction.text import CountVectorizer

texts = ["I love this movie", "This movie is terrible",
         "Great film", "Awful movie"]
labels = [1, 0, 1, 0]  # 1=positive, 0=negative

vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts)

mnb = MultinomialNB()
mnb.fit(X_text, labels)

# Predict new text
new_text = vectorizer.transform(["I love this film"])
prediction = mnb.predict(new_text)
```

### Pros & Cons
✅ Fast and scalable
✅ Works well with high dimensions
✅ Requires little training data
✅ Handles missing values naturally
❌ Independence assumption rarely true
❌ Can be outperformed by other methods

---

## 8. K-Means Clustering 🎯

**Purpose:** Group data into K clusters (unsupervised).

### Theory
**Algorithm:**
1. Initialize K cluster centers randomly
2. Assign each point to nearest center
3. Update centers to mean of assigned points
4. Repeat 2-3 until convergence

**Objective:** Minimize within-cluster sum of squares (WCSS)

### When to Use
- Customer segmentation
- Image compression
- Feature engineering
- Exploratory data analysis

### Implementation
```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Generate data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
centers = kmeans.cluster_centers_

# Plot
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
plt.title('K-Means Clustering')
plt.show()

# Elbow method to find optimal K
wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()
```

### Choosing K
- **Elbow method:** Plot WCSS vs K, look for "elbow"
- **Silhouette score:** Measures cluster cohesion
- **Domain knowledge:** Use business logic

### Limitations
- Must specify K beforehand
- Sensitive to initialization
- Assumes spherical clusters
- Sensitive to outliers

---

## Algorithm Selection Guide 🗺️

### For Classification

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| Logistic Regression | Linear boundaries, interpretability | Non-linear patterns |
| Decision Tree | Interpretability, mixed features | Need high accuracy |
| Random Forest | General purpose, feature importance | Need speed, interpretability |
| SVM | High dimensions, clear margin | Large datasets, speed critical |
| KNN | Small datasets, simple | Large datasets, high dimensions |
| Naive Bayes | Text, fast baseline | Feature independence violated |

### For Regression

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| Linear Regression | Linear relationships, interpretability | Non-linear patterns |
| Ridge/Lasso | Multicollinearity, feature selection | Simple linear case |
| Decision Tree | Non-linear, interpretability | Need stability |
| Random Forest | General purpose, non-linear | Interpretability critical |
| SVM Regression | High dimensions | Large datasets |

### For Clustering

| Algorithm | Best For | Avoid When |
|-----------|----------|------------|
| K-Means | Spherical clusters, speed | Irregular shapes, unknown K |
| DBSCAN | Arbitrary shapes, noise | Varying density |
| Hierarchical | Dendrogram needed | Large datasets |

---

## Practical Tips 💡

1. **Start Simple:** Try logistic regression or decision tree first

2. **Ensemble Methods:** Random Forest often wins in Kaggle

3. **Scale Features:** Always for KNN, SVM; not needed for tree-based

4. **Regularization:** Use Ridge/Lasso to prevent overfitting

5. **Cross-Validation:** Always validate, don't trust single train/test split

6. **Feature Engineering:** Often more important than algorithm choice

7. **Try Multiple Algorithms:** Compare performance on your specific problem

## Next Steps 📚

Now that you know classical algorithms, learn:
- **[03_Data_Preprocessing.md](03_Data_Preprocessing.md)** - Prepare data properly
- **[06_Model_Evaluation.md](06_Model_Evaluation.md)** - Evaluate models correctly
- **[04_Neural_Networks_Basics.md](04_Neural_Networks_Basics.md)** - Move to deep learning

---

**Practice makes perfect!** Implement these algorithms from scratch to truly understand them. 🚀

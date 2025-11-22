# Lesson 2: Tree-Based Models 🌳

**Module 2: Classical ML & Ensemble Methods | Lesson 2 of 4**

Master Decision Trees, Random Forests, XGBoost, LightGBM - the industry workhorses!

---

## Why Tree-Based Models?

**The most popular ML algorithms in industry:**
- **No feature scaling needed** (unlike linear models)
- Handle non-linear relationships naturally
- Capture feature interactions automatically
- Work with mixed data types (numerical + categorical)
- **XGBoost/LightGBM win Kaggle competitions**
- Interpretable (can visualize decisions)

**Real-world dominance:** ~70% of tabular ML problems use tree-based models.

---

## 1. Decision Trees 🌲

**How it works:** Split data recursively to create pure groups.

### The Concept

```
Is Age > 30?
├─ Yes → Is Income > 50K?
│         ├─ Yes → Class: Buy (90% confidence)
│         └─ No  → Class: Don't Buy (80%)
└─ No  → Class: Don't Buy (95%)
```

Each split maximizes **information gain** or minimizes **impurity**.

### Classification Tree

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
tree.fit(X_train, y_train)

# Accuracy
print(f"Train Accuracy: {tree.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {tree.score(X_test, y_test):.3f}")

# Visualize tree
from sklearn.tree import plot_tree

plt.figure(figsize=(15, 10))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, fontsize=10)
plt.show()
```

### Regression Tree

```python
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing

# Load housing data
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
tree_reg = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score
y_pred = tree_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Feature Importance

```python
import pandas as pd

# Get feature importance
importances = tree.feature_importances_
feature_names = iris.feature_names

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(importance_df)

# Visualize
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()
```

### Overfitting Problem

```python
# Deep tree (overfits)
deep_tree = DecisionTreeClassifier(max_depth=None)  # No limit
deep_tree.fit(X_train, y_train)

print(f"Deep Tree - Train: {deep_tree.score(X_train, y_train):.3f}")
print(f"Deep Tree - Test: {deep_tree.score(X_test, y_test):.3f}")
# Train: 1.000 (perfect!)
# Test: 0.900 (worse than shallow tree!)

# Shallow tree (better generalization)
shallow_tree = DecisionTreeClassifier(max_depth=3)
shallow_tree.fit(X_train, y_train)

print(f"Shallow Tree - Train: {shallow_tree.score(X_train, y_train):.3f}")
print(f"Shallow Tree - Test: {shallow_tree.score(X_test, y_test):.3f}")
# Train: 0.975
# Test: 0.967 (better!)
```

**Key Hyperparameters:**
- `max_depth`: Maximum tree depth (prevent overfitting)
- `min_samples_split`: Minimum samples to split node
- `min_samples_leaf`: Minimum samples in leaf
- `max_features`: Features to consider for split

---

## 2. Random Forests 🌲🌳🌴

**Idea:** Train many trees on random subsets, average predictions.

**Why it works:**
- Many weak learners → strong learner
- Random subsets → diverse trees → less overfitting
- Averaging → reduces variance

### Classification

```python
from sklearn.ensemble import RandomForestClassifier

# Train random forest
rf = RandomForestClassifier(n_estimators=100,  # Number of trees
                           max_depth=10,
                           min_samples_split=5,
                           random_state=42,
                           n_jobs=-1)  # Use all CPU cores

rf.fit(X_train, y_train)

print(f"Train Accuracy: {rf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.3f}")

# Typically better than single tree!
```

### Regression

```python
from sklearn.ensemble import RandomForestRegressor

# Housing prices
rf_reg = RandomForestRegressor(n_estimators=100,
                              max_depth=20,
                              random_state=42,
                              n_jobs=-1)

rf_reg.fit(X_train, y_train)

y_pred = rf_reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Out-of-Bag (OOB) Score

**Built-in validation without separate test set!**

```python
rf_oob = RandomForestClassifier(n_estimators=100,
                                oob_score=True,  # Enable OOB
                                random_state=42)
rf_oob.fit(X_train, y_train)

print(f"OOB Score: {rf_oob.oob_score_:.3f}")
print(f"Test Score: {rf_oob.score(X_test, y_test):.3f}")
# OOB score approximates test score!
```

### Feature Importance (More Reliable)

```python
# Random Forest importance is more stable than single tree
importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances,
    'Std': std
}).sort_values('Importance', ascending=False)

print(importance_df)

# Plot with error bars
plt.barh(importance_df['Feature'], importance_df['Importance'],
         xerr=importance_df['Std'])
plt.xlabel('Importance')
plt.show()
```

---

## 3. Gradient Boosting 🚀

**Idea:** Train trees sequentially, each correcting previous errors.

**Process:**
1. Train tree on data
2. Calculate errors (residuals)
3. Train next tree to predict errors
4. Add to ensemble
5. Repeat

### Gradient Boosting Classifier

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(n_estimators=100,
                               learning_rate=0.1,
                               max_depth=3,
                               random_state=42)

gb.fit(X_train, y_train)

print(f"Train Accuracy: {gb.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {gb.score(X_test, y_test):.3f}")
```

### Learning Rate Effect

```python
# Higher learning rate = faster but less accurate
learning_rates = [0.01, 0.1, 0.5, 1.0]

for lr in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=100,
                                   learning_rate=lr,
                                   max_depth=3,
                                   random_state=42)
    gb.fit(X_train, y_train)
    print(f"LR={lr}: Test Accuracy = {gb.score(X_test, y_test):.3f}")

# Typical: low learning rate + more trees = better
```

---

## 4. XGBoost ⭐ Industry Standard

**eXtreme Gradient Boosting** - the Kaggle champion!

**Why XGBoost?**
- 10x faster than scikit-learn GradientBoosting
- Handles missing values automatically
- Built-in regularization (L1, L2)
- Parallel processing
- Tree pruning (smarter than depth-limited)
- Wins competitions!

### Installation

```bash
pip install xgboost
```

### Classification

```python
import xgboost as xgb

# Create DMatrix (XGBoost's data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'multi:softmax',  # Multi-class classification
    'num_class': 3,
    'max_depth': 5,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'eval_metric': 'mlogloss'
}

# Train
model = xgb.train(params, dtrain, num_boost_round=100)

# Predict
y_pred = model.predict(dtest)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.3f}")
```

### Scikit-learn API (Easier)

```python
from xgboost import XGBClassifier

# Easier interface
xgb_clf = XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softmax',
    random_state=42,
    n_jobs=-1
)

xgb_clf.fit(X_train, y_train)

print(f"Train Accuracy: {xgb_clf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {xgb_clf.score(X_test, y_test):.3f}")
```

### Regression

```python
from xgboost import XGBRegressor

xgb_reg = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    objective='reg:squarederror',
    random_state=42
)

xgb_reg.fit(X_train, y_train)

y_pred = xgb_reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Early Stopping (Prevent Overfitting)

```python
xgb_clf = XGBClassifier(n_estimators=1000, learning_rate=0.1)

xgb_clf.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           early_stopping_rounds=10,  # Stop if no improvement for 10 rounds
           verbose=False)

print(f"Best iteration: {xgb_clf.best_iteration}")
print(f"Best score: {xgb_clf.best_score:.3f}")
```

### Feature Importance

```python
from xgboost import plot_importance

# Multiple importance types
importance_types = ['weight', 'gain', 'cover']

for imp_type in importance_types:
    print(f"\n{imp_type.upper()} Importance:")
    plot_importance(xgb_clf, importance_type=imp_type, max_num_features=10)
    plt.title(f'Feature Importance ({imp_type})')
    plt.show()

# weight: Number of times feature used
# gain: Average gain when feature used
# cover: Average coverage of feature
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

xgb_clf = XGBClassifier(random_state=42)

grid_search = GridSearchCV(xgb_clf, param_grid, cv=5,
                          scoring='accuracy', n_jobs=-1, verbose=1)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")
```

---

## 5. LightGBM 🚄 Even Faster!

**Microsoft's gradient boosting framework.**

**Advantages over XGBoost:**
- **Faster training** (especially on large datasets)
- **Less memory usage**
- Better accuracy on some datasets
- Handles categorical features natively

### Installation

```bash
pip install lightgbm
```

### Classification

```python
import lightgbm as lgb

# Create Dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Parameters
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9
}

# Train
lgb_model = lgb.train(params, train_data,
                     num_boost_round=100,
                     valid_sets=[test_data],
                     early_stopping_rounds=10)

# Predict
y_pred = lgb_model.predict(X_test)
y_pred_class = np.argmax(y_pred, axis=1)

print(f"Accuracy: {np.mean(y_pred_class == y_test):.3f}")
```

### Scikit-learn API

```python
from lightgbm import LGBMClassifier

lgb_clf = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42,
    n_jobs=-1
)

lgb_clf.fit(X_train, y_train)

print(f"Train Accuracy: {lgb_clf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {lgb_clf.score(X_test, y_test):.3f}")
```

### Regression

```python
from lightgbm import LGBMRegressor

lgb_reg = LGBMRegressor(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31,
    random_state=42
)

lgb_reg.fit(X_train, y_train)

y_pred = lgb_reg.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")
```

### Categorical Features (Native Support)

```python
# LightGBM handles categorical features directly!
import pandas as pd

df = pd.DataFrame({
    'age': [25, 35, 45, 30, 50],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA'],  # Categorical
    'income': [50000, 80000, 90000, 70000, 100000]
})

# Mark categorical
df['city'] = df['city'].astype('category')

X = df[['age', 'city']]
y = df['income']

lgb_reg = LGBMRegressor(n_estimators=100)
lgb_reg.fit(X, y, categorical_feature=['city'])  # Specify categorical

# No need for one-hot encoding!
```

---

## 6. CatBoost 🐈 Handles Categories Best

**Yandex's gradient boosting - best for categorical features.**

**Key advantages:**
- **Best categorical feature handling**
- Minimal hyperparameter tuning needed
- Good default parameters
- Fast predictions

### Installation

```bash
pip install catboost
```

### Classification

```python
from catboost import CatBoostClassifier

cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=5,
    verbose=False,
    random_state=42
)

cat_clf.fit(X_train, y_train)

print(f"Train Accuracy: {cat_clf.score(X_train, y_train):.3f}")
print(f"Test Accuracy: {cat_clf.score(X_test, y_test):.3f}")
```

### With Categorical Features

```python
from catboost import CatBoostRegressor

# Dataset with categorical features
df = pd.DataFrame({
    'age': [25, 35, 45, 30, 50, 28, 40],
    'city': ['NYC', 'LA', 'NYC', 'SF', 'LA', 'SF', 'NYC'],
    'education': ['BS', 'MS', 'PhD', 'BS', 'MS', 'PhD', 'BS'],
    'income': [50000, 80000, 90000, 70000, 100000, 85000, 60000]
})

X = df[['age', 'city', 'education']]
y = df['income']

# Specify categorical features by name or index
cat_features = ['city', 'education']

cat_reg = CatBoostRegressor(iterations=100,
                           learning_rate=0.1,
                           verbose=False)

cat_reg.fit(X, y, cat_features=cat_features)

# Predict
new_data = pd.DataFrame({
    'age': [32],
    'city': ['NYC'],
    'education': ['MS']
})

prediction = cat_reg.predict(new_data)
print(f"Predicted income: ${prediction[0]:,.2f}")
```

### GPU Support (Super Fast!)

```python
# If GPU available
cat_clf_gpu = CatBoostClassifier(
    iterations=1000,
    task_type='GPU',  # Use GPU
    devices='0',      # GPU device ID
    verbose=False
)

cat_clf_gpu.fit(X_train, y_train)
# Much faster on large datasets!
```

---

## 7. Comparison: Which to Use? 🤔

```python
import time
from sklearn.datasets import make_classification

# Generate large dataset
X, y = make_classification(n_samples=10000, n_features=20,
                          n_informative=15, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, max_depth=5, n_jobs=-1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=-1, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, depth=5, verbose=False, random_state=42)
}

results = []
for name, model in models.items():
    # Train
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    # Predict
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start

    # Accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    results.append({
        'Model': name,
        'Train Acc': f"{train_acc:.4f}",
        'Test Acc': f"{test_acc:.4f}",
        'Train Time': f"{train_time:.3f}s",
        'Pred Time': f"{pred_time:.4f}s"
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

**Typical Results:**
```
Model              Train Acc  Test Acc  Train Time  Pred Time
Decision Tree      0.9500     0.8500    0.05s       0.001s
Random Forest      0.9800     0.9100    2.50s       0.050s
Gradient Boosting  0.9200     0.9000    15.0s       0.020s
XGBoost            0.9300     0.9200    1.20s       0.010s
LightGBM           0.9300     0.9250    0.80s       0.008s
CatBoost           0.9350     0.9200    2.00s       0.012s
```

---

## When to Use What? 📊

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **Decision Tree** | Quick baseline, interpretability | Fast, visual | Overfits easily |
| **Random Forest** | General-purpose, robust | Handles overfitting well | Slower predictions |
| **XGBoost** | **Structured data competitions** | **Fast, accurate, winner** | Hyperparameters matter |
| **LightGBM** | **Large datasets (100K+ rows)** | **Fastest training** | Can overfit small data |
| **CatBoost** | **Many categorical features** | **Best cat handling** | Slower than LightGBM |

**Rule of thumb:**
1. Start with **Random Forest** (robust, few hyperparameters)
2. Try **XGBoost** for better accuracy (tune hyperparameters)
3. Use **LightGBM** if data is large (>100K rows)
4. Use **CatBoost** if many categorical features

---

## Practical Tips 💡

### 1. Handling Imbalanced Data

```python
# Class weights
xgb_clf = XGBClassifier(scale_pos_weight=10)  # If minority class is 10x smaller

# Or use sample_weight
sample_weights = np.where(y_train == 1, 10, 1)
xgb_clf.fit(X_train, y_train, sample_weight=sample_weights)
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(xgb_clf, X, y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

### 3. Save/Load Models

```python
import joblib

# Save
joblib.dump(xgb_clf, 'xgb_model.pkl')

# Load
loaded_model = joblib.load('xgb_model.pkl')
predictions = loaded_model.predict(X_test)
```

### 4. Missing Values

```python
# XGBoost/LightGBM/CatBoost handle NaN automatically!
X_train_with_nan = X_train.copy()
X_train_with_nan[np.random.rand(*X_train.shape) < 0.1] = np.nan

# No preprocessing needed
xgb_clf.fit(X_train_with_nan, y_train)
```

---

## Quick Reference 📖

**Key Hyperparameters:**

| Parameter | XGBoost | LightGBM | CatBoost | Meaning |
|-----------|---------|----------|----------|---------|
| Trees | n_estimators | n_estimators | iterations | Number of trees |
| Depth | max_depth | max_depth | depth | Max tree depth |
| Learning | learning_rate | learning_rate | learning_rate | Step size |
| Leaves | max_leaves | num_leaves | - | Max leaf nodes |
| Regularization | reg_alpha, reg_lambda | reg_alpha, reg_lambda | l2_leaf_reg | L1/L2 penalty |

**Typical Good Starting Points:**
```python
{
    'n_estimators': 100-1000,
    'max_depth': 3-10,
    'learning_rate': 0.01-0.3,
    'subsample': 0.8-1.0,
    'colsample_bytree': 0.8-1.0
}
```

---

## Practice Exercises 🏋️

1. Compare Decision Tree vs Random Forest on iris dataset
2. Train XGBoost with early stopping on California housing
3. Use LightGBM with categorical features
4. Tune XGBoost hyperparameters with GridSearchCV

<details>
<summary>Solutions</summary>

```python
# 1. Tree vs Forest
from sklearn.datasets import load_iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

tree.fit(X_train, y_train)
rf.fit(X_train, y_train)

print(f"Tree: {tree.score(X_test, y_test):.3f}")
print(f"RF: {rf.score(X_test, y_test):.3f}")

# 2. XGBoost early stopping
housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

xgb_reg = XGBRegressor(n_estimators=1000, learning_rate=0.05)
xgb_reg.fit(X_train, y_train,
           eval_set=[(X_test, y_test)],
           early_stopping_rounds=20,
           verbose=False)

print(f"Best iteration: {xgb_reg.best_iteration}")

# 3. LightGBM categorical
df = pd.DataFrame({
    'num_feature': [1, 2, 3, 4, 5],
    'cat_feature': ['A', 'B', 'A', 'C', 'B'],
    'target': [10, 20, 15, 30, 25]
})

X = df[['num_feature', 'cat_feature']]
y = df['target']

lgb_model = LGBMRegressor(n_estimators=100)
lgb_model.fit(X, y, categorical_feature=['cat_feature'])

# 4. Hyperparameter tuning
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

grid = GridSearchCV(XGBClassifier(), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```
</details>

---

## Key Takeaways 💡

1. **Tree-based models** = industry workhorses for tabular data
2. **Random Forest** = great default choice (robust, few hyperparameters)
3. **XGBoost** = Kaggle winner, excellent accuracy
4. **LightGBM** = fastest for large datasets
5. **CatBoost** = best for categorical features
6. **Always try boosting** (XGBoost/LightGBM) for competitions
7. **Feature scaling not needed** for tree models
8. **Tune hyperparameters** for best results

---

**Next:** [Lesson 3 - Instance-Based & Probabilistic Models →](Lesson%203%20-%20Instance-Based%20and%20Probabilistic%20Models.md)

---

**Congratulations!** You now master tree-based models - the ML workhorses! 🎉

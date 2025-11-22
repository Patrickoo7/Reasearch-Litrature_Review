# Lesson 4: Ensemble Methods 🎭

**Module 2: Classical ML & Ensemble Methods | Lesson 4 of 4**

Master Bagging, Boosting, Stacking, and Blending - the secret to winning ML competitions!

---

## Why Ensemble Methods?

**"Wisdom of crowds" - Multiple weak learners combine to create a strong learner.**

**Example:** 100 models each 60% accurate → Ensemble 95% accurate!

**Real-world impact:**
- **Netflix Prize winner:** Ensemble of 100+ models
- **Kaggle winners:** Almost always use ensembles
- **Production ML:** Ensembles for reliability

---

## Core Idea

```
Single Model:        Model A → Prediction (70% accurate)

Ensemble:            Model A ─┐
                     Model B ─┤→ Combine → Final Prediction (85% accurate!)
                     Model C ─┘

Key: Models should be diverse (different errors)
```

---

## 1. Bagging (Bootstrap Aggregating) 🎒

**Strategy:** Train multiple models on random subsets of data, average results.

### How It Works

```
Original Data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Bootstrap Sample 1: [1, 2, 2, 4, 6, 7, 9, 9, 10]  → Model 1
Bootstrap Sample 2: [1, 3, 3, 4, 5, 7, 8, 9, 10]  → Model 2
Bootstrap Sample 3: [2, 3, 5, 5, 6, 7, 8, 8, 9]   → Model 3
...

Final Prediction:
- Classification: Majority vote
- Regression: Average
```

**Key: Sampling with replacement (same data point can appear multiple times)**

### Bagging Classifier

```python
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Single decision tree (baseline)
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
print(f"Single Tree Test Accuracy: {single_tree.score(X_test, y_test):.3f}")

# Bagging with 100 trees
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,      # Number of base models
    max_samples=0.8,       # Use 80% of data per model
    max_features=0.8,      # Use 80% of features per model
    bootstrap=True,        # Sample with replacement
    n_jobs=-1,            # Parallel processing
    random_state=42
)

bagging.fit(X_train, y_train)
print(f"Bagging Test Accuracy: {bagging.score(X_test, y_test):.3f}")
# Usually significant improvement!
```

### Bagging Regressor

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score

# Regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Single tree
single_tree = DecisionTreeRegressor(random_state=42)
single_tree.fit(X_train, y_train)
y_pred_single = single_tree.predict(X_test)
print(f"Single Tree RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_single)):.2f}")

# Bagging
bagging_reg = BaggingRegressor(
    base_estimator=DecisionTreeRegressor(),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

bagging_reg.fit(X_train, y_train)
y_pred_bagging = bagging_reg.predict(X_test)
print(f"Bagging RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_bagging)):.2f}")
print(f"Bagging R²: {r2_score(y_test, y_pred_bagging):.3f}")
```

### Random Forest is Bagging!

```python
from sklearn.ensemble import RandomForestClassifier

# Random Forest = Bagging + Random feature selection at each split
rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',   # Random feature selection
    bootstrap=True,        # Bootstrap sampling
    n_jobs=-1,
    random_state=42
)

rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {rf.score(X_test, y_test):.3f}")
```

### Out-of-Bag (OOB) Error

**Each model trained on ~63% of data, validated on remaining ~37%.**

```python
bagging_oob = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    oob_score=True,        # Enable OOB evaluation
    random_state=42
)

bagging_oob.fit(X_train, y_train)

print(f"OOB Score: {bagging_oob.oob_score_:.3f}")
print(f"Test Score: {bagging_oob.score(X_test, y_test):.3f}")
# OOB approximates test score without separate validation set!
```

---

## 2. Boosting 🚀

**Strategy:** Train models sequentially, each correcting previous errors.

### How It Works

```
Data: [x1, x2, x3, x4, x5, ...]

Round 1: Train Model 1 on data
         → Identify misclassified points
         → Give them higher weight

Round 2: Train Model 2 on re-weighted data
         → Focus on previously difficult points
         → Identify new errors

Round 3: Train Model 3 on re-weighted data
         ...

Final: Weighted combination of all models
```

**Key: Sequential learning, each model focuses on hard examples**

### AdaBoost (Adaptive Boosting)

```python
from sklearn.ensemble import AdaBoostClassifier

# AdaBoost with decision trees
ada = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),  # "Stumps"
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)

ada.fit(X_train, y_train)

print(f"AdaBoost Train Accuracy: {ada.score(X_train, y_train):.3f}")
print(f"AdaBoost Test Accuracy: {ada.score(X_test, y_test):.3f}")

# Feature importance
import pandas as pd
feature_importance = pd.DataFrame({
    'importance': ada.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 features:\n{feature_importance.head()}")
```

### Learning Rate Effect

```python
learning_rates = [0.1, 0.5, 1.0, 2.0]

for lr in learning_rates:
    ada = AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100,
        learning_rate=lr,
        random_state=42
    )
    ada.fit(X_train, y_train)
    print(f"LR={lr}: Test Accuracy = {ada.score(X_test, y_test):.3f}")

# Lower LR + more estimators = better generalization
```

### AdaBoost Regression

```python
from sklearn.ensemble import AdaBoostRegressor

ada_reg = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

ada_reg.fit(X_train, y_train)
y_pred = ada_reg.predict(X_test)

print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
```

### Gradient Boosting (More Powerful)

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,         # Use 80% of samples per tree
    random_state=42
)

gb.fit(X_train, y_train)

print(f"Gradient Boosting Train: {gb.score(X_train, y_train):.3f}")
print(f"Gradient Boosting Test: {gb.score(X_test, y_test):.3f}")
```

### XGBoost, LightGBM, CatBoost (Covered in Lesson 2)

**Modern gradient boosting implementations:**

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = {
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=100, random_state=42),
    'CatBoost': CatBoostClassifier(iterations=100, verbose=False, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:20s}: {model.score(X_test, y_test):.3f}")
```

---

## 3. Stacking (Stacked Generalization) 🥞

**Strategy:** Train multiple diverse models, use another model to combine them.

### How It Works

```
Training:
    Level 0 Models (Base Models):
    X_train → Model A → Predictions A
    X_train → Model B → Predictions B
    X_train → Model C → Predictions C

    Level 1 Model (Meta-Model):
    [Predictions A, B, C] → Meta-Model → Final Prediction

Prediction:
    X_test → Model A, B, C → [Pred A, B, C] → Meta-Model → Final
```

**Key: Meta-model learns how to best combine base models**

### Simple Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Base models (diverse!)
base_models = [
    ('dt', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('svm', SVC(probability=True, random_state=42))
]

# Meta-model (simple, like Logistic Regression)
meta_model = LogisticRegression()

# Stacking ensemble
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5  # Cross-validation to avoid overfitting
)

stacking.fit(X_train, y_train)

# Compare with individual models
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name.upper():5s}: {model.score(X_test, y_test):.3f}")

print(f"STACK: {stacking.score(X_test, y_test):.3f}")
# Usually better than any single model!
```

### Advanced Stacking with Multiple Levels

```python
# Level 0: Diverse base models
level0 = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=50, random_state=42)),
    ('lgb', LGBMClassifier(n_estimators=50, random_state=42))
]

# Level 1: Stacking
level1 = StackingClassifier(
    estimators=level0,
    final_estimator=LogisticRegression(),
    cv=5
)

level1.fit(X_train, y_train)
print(f"Multi-level Stacking: {level1.score(X_test, y_test):.3f}")
```

### Stacking Regression

```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Base regressors
base_regressors = [
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=50, random_state=42))
]

# Meta-regressor
meta_regressor = Ridge()

# Stacking
stacking_reg = StackingRegressor(
    estimators=base_regressors,
    final_estimator=meta_regressor,
    cv=5
)

stacking_reg.fit(X_train, y_train)
y_pred = stacking_reg.predict(X_test)

print(f"Stacking Regression RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Stacking Regression R²: {r2_score(y_test, y_pred):.3f}")
```

### Custom Stacking (Manual)

```python
from sklearn.model_selection import KFold

# Base models
models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    XGBClassifier(n_estimators=50, random_state=42),
    LGBMClassifier(n_estimators=50, random_state=42)
]

# Step 1: Generate meta-features using cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features_train = np.zeros((X_train.shape[0], len(models)))

for i, model in enumerate(models):
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        model.fit(X_tr, y_tr)
        meta_features_train[val_idx, i] = model.predict_proba(X_val)[:, 1]

# Step 2: Train base models on full training set
meta_features_test = np.zeros((X_test.shape[0], len(models)))

for i, model in enumerate(models):
    model.fit(X_train, y_train)
    meta_features_test[:, i] = model.predict_proba(X_test)[:, 1]

# Step 3: Train meta-model
meta_model = LogisticRegression()
meta_model.fit(meta_features_train, y_train)

# Step 4: Final predictions
y_pred = meta_model.predict(meta_features_test)
accuracy = np.mean(y_pred == y_test)
print(f"Custom Stacking Accuracy: {accuracy:.3f}")
```

---

## 4. Blending 🎨

**Similar to stacking, but simpler:**
- Split data into train/validation
- Train base models on train set
- Predict on validation set
- Train meta-model on validation predictions

**Difference from Stacking:** Uses hold-out set instead of cross-validation (faster, less data-efficient)

```python
# Split into train, validation, test
from sklearn.model_selection import train_test_split

X_train_base, X_val, y_train_base, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Step 1: Train base models on train_base
base_models = [
    RandomForestClassifier(n_estimators=50, random_state=42),
    XGBClassifier(n_estimators=50, random_state=42),
    LGBMClassifier(n_estimators=50, random_state=42)
]

# Step 2: Get predictions on validation set
val_predictions = np.zeros((X_val.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    model.fit(X_train_base, y_train_base)
    val_predictions[:, i] = model.predict_proba(X_val)[:, 1]

# Step 3: Train meta-model on validation predictions
meta_model = LogisticRegression()
meta_model.fit(val_predictions, y_val)

# Step 4: Predict on test set
test_predictions = np.zeros((X_test.shape[0], len(base_models)))

for i, model in enumerate(base_models):
    test_predictions[:, i] = model.predict_proba(X_test)[:, 1]

y_pred = meta_model.predict(test_predictions)
accuracy = np.mean(y_pred == y_test)
print(f"Blending Accuracy: {accuracy:.3f}")
```

---

## 5. Voting Classifier/Regressor 🗳️

**Simplest ensemble: Average predictions or take majority vote.**

### Hard Voting (Classification)

```python
from sklearn.ensemble import VotingClassifier

# Different types of models
voting_hard = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=50, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    voting='hard'  # Majority vote
)

voting_hard.fit(X_train, y_train)
print(f"Hard Voting Accuracy: {voting_hard.score(X_test, y_test):.3f}")
```

### Soft Voting (Uses Probabilities)

```python
voting_soft = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=50, use_label_encoder=False, random_state=42)),
        ('lgb', LGBMClassifier(n_estimators=50, random_state=42))
    ],
    voting='soft',   # Average probabilities
    weights=[2, 1, 1]  # Give RF twice the weight
)

voting_soft.fit(X_train, y_train)
print(f"Soft Voting Accuracy: {voting_soft.score(X_test, y_test):.3f}")
# Usually better than hard voting
```

### Voting Regressor

```python
from sklearn.ensemble import VotingRegressor

voting_reg = VotingRegressor([
    ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=50, random_state=42)),
    ('lgb', LGBMRegressor(n_estimators=50, random_state=42))
])

voting_reg.fit(X_train, y_train)
y_pred = voting_reg.predict(X_test)

print(f"Voting Regressor RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"Voting Regressor R²: {r2_score(y_test, y_pred):.3f}")
```

---

## 6. Complete Comparison 📊

```python
import time

# Prepare data
X, y = make_classification(n_samples=5000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# All ensemble methods
ensembles = {
    'Single Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
    'Bagging': BaggingClassifier(n_estimators=50, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=50, use_label_encoder=False, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=50, random_state=42),
    'Voting': VotingClassifier([
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=30, use_label_encoder=False, random_state=42))
    ], voting='soft'),
    'Stacking': StackingClassifier([
        ('rf', RandomForestClassifier(n_estimators=30, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=30, use_label_encoder=False, random_state=42))
    ], final_estimator=LogisticRegression(), cv=3)
}

results = []
for name, model in ensembles.items():
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    results.append({
        'Method': name,
        'Train Acc': f"{train_acc:.4f}",
        'Test Acc': f"{test_acc:.4f}",
        'Time (s)': f"{train_time:.2f}"
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

---

## When to Use What? 🤔

| Method | Best For | Pros | Cons | Complexity |
|--------|----------|------|------|------------|
| **Bagging** | Reduce variance | Simple, parallel | Less powerful | Low |
| **Random Forest** | General purpose | Robust, fast | Memory-intensive | Low |
| **AdaBoost** | Focus on errors | Good with stumps | Sensitive to noise | Medium |
| **Gradient Boosting** | High accuracy | Very powerful | Sequential, slow | Medium |
| **XGBoost/LightGBM** | **Competitions** | **Fast, accurate** | Hyperparameters | Medium |
| **Voting** | Quick ensemble | Simple, fast | Limited gains | Low |
| **Stacking** | **Max accuracy** | **Best results** | Complex, slow | High |
| **Blending** | Faster than stacking | Simple | Less data-efficient | Medium |

**Decision Guide:**
1. **Start simple:** Random Forest or XGBoost
2. **Need more accuracy:** Try Voting (soft)
3. **Competition/Critical app:** Stacking with diverse models
4. **Large dataset:** LightGBM or XGBoost
5. **Small dataset:** Stacking or Voting

---

## Best Practices 💡

### 1. Model Diversity is Key

```python
# Good: Different types of models
diverse_ensemble = [
    ('tree', RandomForestClassifier()),  # Tree-based
    ('linear', LogisticRegression()),    # Linear
    ('neighbor', KNeighborsClassifier()) # Instance-based
]

# Bad: Similar models
similar_ensemble = [
    ('rf', RandomForestClassifier()),
    ('et', ExtraTreesClassifier()),  # Very similar to RF
    ('gb', GradientBoostingClassifier())
]
```

### 2. Feature Engineering for Different Models

```python
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

# Pipeline 1: Tree-based (no scaling needed)
pipeline1 = RandomForestClassifier()

# Pipeline 2: Linear model (needs scaling)
pipeline2 = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', LogisticRegression())
])

# Stacking with different pipelines
stacking = StackingClassifier([
    ('rf', pipeline1),
    ('lr', pipeline2)
], final_estimator=LogisticRegression())
```

### 3. Cross-Validation for Stacking

```python
# Always use CV in stacking to prevent overfitting
stacking = StackingClassifier(
    estimators=[...],
    final_estimator=LogisticRegression(),
    cv=5  # IMPORTANT: Prevents meta-model overfitting
)
```

### 4. Save Ensembles

```python
import joblib

# Save
joblib.dump(stacking, 'stacking_model.pkl')

# Load
loaded_model = joblib.load('stacking_model.pkl')
predictions = loaded_model.predict(X_test)
```

### 5. Weighted Voting

```python
# Give better models more weight
voting = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier()),     # Best model
        ('rf', RandomForestClassifier()),
        ('lr', LogisticRegression())  # Weakest
    ],
    voting='soft',
    weights=[3, 2, 1]  # XGBoost gets 3x weight
)
```

---

## Real-World Example: Kaggle-Style Ensemble 🏆

```python
# Complete pipeline for competition

# 1. Base models (diverse)
base_models = [
    ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5)),
    ('lgb', LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31)),
    ('cat', CatBoostClassifier(iterations=200, learning_rate=0.05, depth=6, verbose=False)),
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10))
]

# 2. Train base models
for name, model in base_models:
    model.fit(X_train, y_train)
    print(f"{name.upper()}: {model.score(X_test, y_test):.4f}")

# 3. Level 1: Voting ensemble
voting = VotingClassifier(base_models, voting='soft')
voting.fit(X_train, y_train)
print(f"\nVoting: {voting.score(X_test, y_test):.4f}")

# 4. Level 2: Stacking
stacking = StackingClassifier(
    estimators=base_models,
    final_estimator=LogisticRegression(C=0.1),
    cv=5
)
stacking.fit(X_train, y_train)
print(f"Stacking: {stacking.score(X_test, y_test):.4f}")

# Typically: Stacking > Voting > Individual models
```

---

## Quick Reference 📖

**Ensemble Formulas:**

**Bagging:**
```
Final = (1/N) × Σ Model_i(x)  for regression
Final = mode(Model_1(x), ..., Model_N(x))  for classification
```

**Boosting:**
```
Final = Σ α_i × Model_i(x)
where α_i = weight of model i
```

**Stacking:**
```
Level 0: [Model_A(x), Model_B(x), Model_C(x)]
Level 1: Meta_Model([pred_A, pred_B, pred_C])
```

**Key Parameters:**

```python
# Bagging
BaggingClassifier(
    n_estimators=100,    # Number of models
    max_samples=1.0,     # Samples per model
    bootstrap=True       # Sample with replacement
)

# Stacking
StackingClassifier(
    estimators=[...],    # Base models
    final_estimator=..., # Meta-model
    cv=5                 # Cross-validation folds
)

# Voting
VotingClassifier(
    estimators=[...],
    voting='soft',       # 'hard' or 'soft'
    weights=None         # Model weights
)
```

---

## Practice Exercises 🏋️

1. Compare Bagging vs Random Forest performance
2. Build stacking ensemble with 3 diverse models
3. Create voting ensemble and tune weights
4. Implement custom blending manually

<details>
<summary>Solutions</summary>

```python
# 1. Bagging vs Random Forest
from sklearn.tree import DecisionTreeClassifier

bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)

bagging.fit(X_train, y_train)
rf.fit(X_train, y_train)

print(f"Bagging: {bagging.score(X_test, y_test):.3f}")
print(f"RF: {rf.score(X_test, y_test):.3f}")

# 2. Stacking with 3 diverse models
stacking = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=100)),
        ('lr', LogisticRegression(max_iter=1000)),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ],
    final_estimator=LogisticRegression(),
    cv=5
)
stacking.fit(X_train, y_train)
print(f"Stacking: {stacking.score(X_test, y_test):.3f}")

# 3. Voting with weights
from sklearn.model_selection import GridSearchCV

weights_grid = {
    'weights': [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
}

voting = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=50)),
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('lr', LogisticRegression())
    ],
    voting='soft'
)

grid = GridSearchCV(voting, weights_grid, cv=3)
grid.fit(X_train, y_train)
print(f"Best weights: {grid.best_params_}")

# 4. Custom blending
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2)

models = [XGBClassifier(), RandomForestClassifier(), LogisticRegression()]
val_preds = np.zeros((len(X_val), len(models)))

for i, model in enumerate(models):
    model.fit(X_tr, y_tr)
    val_preds[:, i] = model.predict_proba(X_val)[:, 1]

meta = LogisticRegression()
meta.fit(val_preds, y_val)

test_preds = np.zeros((len(X_test), len(models)))
for i, model in enumerate(models):
    test_preds[:, i] = model.predict_proba(X_test)[:, 1]

final_pred = meta.predict(test_preds)
print(f"Blending: {np.mean(final_pred == y_test):.3f}")
```
</details>

---

## Key Takeaways 💡

1. **Ensembles > Individual models** (almost always)
2. **Diversity is crucial** - combine different model types
3. **Bagging** reduces variance (Random Forest)
4. **Boosting** reduces bias (XGBoost, LightGBM)
5. **Stacking** typically gives best results (competitions)
6. **Voting** is simplest ensemble (good starting point)
7. **Always use CV** in stacking to prevent overfitting
8. **Real competitions** use 10+ model ensembles

---

**Module 2 Complete!** 🎉

You now master all classical ML algorithms and ensemble methods - the tools that win competitions and power production systems!

**Next Module:** [Module 3 - Data Engineering →](../Module%203%20-%20Data%20Engineering/Lesson%201%20-%20SQL%20and%20Data%20Manipulation.md)

---

**Congratulations!** You're now equipped with the most important ML algorithms! 🚀

# Lesson 3: Cross-Validation & Hyperparameter Tuning 🎛️

**Module 4: Feature Engineering & Model Evaluation | Lesson 3 of 4**

Master model validation and hyperparameter optimization - essential for robust ML!

---

## 1. Cross-Validation 🔄

### Why Cross-Validation?

**Problem:** Single train/test split can be lucky or unlucky

**Solution:** Multiple splits, average results

### K-Fold CV

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)

# 5-fold CV
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"CV Scores: {cv_scores}")
print(f"Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Custom CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    print(f"Fold {fold+1}: {score:.3f}")
```

### Stratified K-Fold (For Classification)

```python
from sklearn.model_selection import StratifiedKFold

# Maintains class distribution in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"Stratified CV F1: {cv_scores.mean():.3f}")
```

### Time Series CV

```python
from sklearn.model_selection import TimeSeriesSplit

# No shuffling! Respects time order
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Train uses past, test uses future
```

### Leave-One-Out CV (Small Datasets)

```python
from sklearn.model_selection import LeaveOneOut

loo = LeaveOneOut()
cv_scores = cross_val_score(model, X, y, cv=loo)
print(f"LOO CV: {cv_scores.mean():.3f}")
```

---

## 2. Hyperparameter Tuning 🎯

### Grid Search (Exhaustive)

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with CV
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.3f}")
print(f"Test score: {grid_search.score(X_test, y_test):.3f}")

# Best model
best_model = grid_search.best_estimator_
```

### Random Search (Faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distributions
param_distributions = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 20),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': uniform(0.1, 0.9)
}

# Random search
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(),
    param_distributions=param_distributions,
    n_iter=50,  # Number of random combinations
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best score: {random_search.best_score_:.3f}")
```

**Random Search > Grid Search:**
- Faster
- Explores more hyperparameter space
- Often finds better results

### Bayesian Optimization (Best)

```python
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# Define search space
search_spaces = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(3, 20),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 10),
    'max_features': Real(0.1, 1.0)
}

# Bayesian optimization
bayes_search = BayesSearchCV(
    estimator=RandomForestClassifier(),
    search_spaces=search_spaces,
    n_iter=50,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)

bayes_search.fit(X_train, y_train)

print(f"Best params: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_:.3f}")
```

---

## 3. XGBoost / LightGBM Tuning 🚀

### XGBoost

```python
from xgboost import XGBClassifier

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [0, 0.1, 1]
}

xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
```

### LightGBM

```python
from lightgbm import LGBMClassifier

param_grid = {
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

lgbm = LGBMClassifier()

random_search = RandomizedSearchCV(
    lgbm, param_grid, n_iter=20, cv=5, scoring='roc_auc', n_jobs=-1
)
random_search.fit(X_train, y_train)
```

---

## 4. Pipeline with CV 🔄

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA()),
    ('classifier', RandomForestClassifier())
])

# Tune pipeline parameters
param_grid = {
    'pca__n_components': [5, 10, 15, 20],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best pipeline params: {grid_search.best_params_}")
```

---

## 5. Early Stopping 🛑

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    early_stopping_rounds=10
)

xgb.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {xgb.best_iteration}")
print(f"Best score: {xgb.best_score}")
```

---

## 6. Nested CV (Unbiased Evaluation)

```python
from sklearn.model_selection import cross_val_score

# Outer CV: Evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV: Hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Grid search as inner CV
grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid={'n_estimators': [50, 100], 'max_depth': [5, 10]},
    cv=inner_cv,
    scoring='f1'
)

# Nested CV
nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv, scoring='f1')

print(f"Nested CV F1: {nested_scores.mean():.3f} (+/- {nested_scores.std():.3f})")
```

---

## Quick Reference 📖

**CV Methods:**
- **K-Fold**: General purpose (k=5 or 10)
- **Stratified K-Fold**: Classification (maintains class balance)
- **TimeSeriesSplit**: Time series (respects order)
- **LeaveOneOut**: Very small datasets

**Tuning Methods:**
- **Grid Search**: Small search space, exhaustive
- **Random Search**: Large search space, faster
- **Bayesian**: Best results, intelligent search

**Typical Hyperparameters:**

**XGBoost/LightGBM:**
```python
{
    'n_estimators': 100-1000,
    'max_depth': 3-10,
    'learning_rate': 0.01-0.3,
    'subsample': 0.7-1.0,
    'colsample_bytree': 0.7-1.0,
    'reg_alpha': 0-1,  # L1
    'reg_lambda': 0-1  # L2
}
```

**Random Forest:**
```python
{
    'n_estimators': 100-500,
    'max_depth': 10-30 or None,
    'min_samples_split': 2-10,
    'min_samples_leaf': 1-5,
    'max_features': 'sqrt' or 'log2'
}
```

---

## Key Takeaways 💡

1. **Always use CV** - never trust single train/test split
2. **Stratified CV** for classification
3. **TimeSeriesSplit** for time series
4. **Random Search > Grid Search** (usually)
5. **Bayesian optimization** for serious tuning
6. **Early stopping** saves time
7. **Nested CV** for unbiased evaluation
8. **Start with defaults** - tune if needed

---

**Next:** [Lesson 4 - A/B Testing & Experimentation →](Lesson%204%20-%20AB%20Testing%20and%20Experimentation.md)

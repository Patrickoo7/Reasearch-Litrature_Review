# Lesson 2: SHAP Values ⭐

**Module 9: Explainable AI | Lesson 2 of 4**

Master SHAP - the gold standard for model explanations!

---

## What is SHAP?

**SHapley Additive exPlanations:**
- Based on game theory
- Unified framework for any model
- Local + global explanations
- **Industry standard**

---

## 1. Installation & Setup 🔧

```bash
pip install shap
```

```python
import shap
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
```

---

## 2. TreeExplainer (Fast for Trees) 🌳

```python
# Create explainer
explainer = shap.TreeExplainer(model)

# Compute SHAP values
shap_values = explainer.shap_values(X)

# For binary classification, select positive class
if isinstance(shap_values, list):
    shap_values = shap_values[1]

print(f"SHAP values shape: {shap_values.shape}")  # (samples, features)
```

---

## 3. Visualizations 📊

### Summary Plot (Global)

```python
# Summary plot - shows feature importance
shap.summary_plot(shap_values, X, plot_type="bar")

# Detailed summary
shap.summary_plot(shap_values, X)
```

**Interpretation:**
- Red = high feature value
- Blue = low feature value
- Position = SHAP value (impact on prediction)

### Force Plot (Single Prediction)

```python
# Explain single prediction
i = 0  # First sample

shap.force_plot(
    explainer.expected_value[1],
    shap_values[i],
    X.iloc[i],
    matplotlib=True
)
```

**Shows:** How features push prediction from base value

### Waterfall Plot

```python
# Waterfall plot for single prediction
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[i],
        base_values=explainer.expected_value[1],
        data=X.iloc[i],
        feature_names=X.columns.tolist()
    )
)
```

### Dependence Plot

```python
# How feature affects predictions
feature_idx = 0  # First feature

shap.dependence_plot(
    feature_idx,
    shap_values,
    X,
    interaction_index='auto'  # Automatically find interaction
)
```

---

## 4. Global Feature Importance 🌍

```python
# Feature importance based on mean |SHAP|
feature_importance = np.abs(shap_values).mean(axis=0)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("Top 10 features:")
print(importance_df.head(10))

# Bar plot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.barh(importance_df['feature'][:10], importance_df['importance'][:10])
plt.xlabel('Mean |SHAP value|')
plt.title('Global Feature Importance (SHAP)')
plt.gca().invert_yaxis()
plt.show()
```

---

## 5. KernelExplainer (Model-Agnostic) 🎯

```python
# For any model (slower)
from sklearn.neural_network import MLPClassifier

# Train neural network
nn = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
nn.fit(X, y)

# Create explainer (sample background data)
background = shap.sample(X, 100)
explainer = shap.KernelExplainer(nn.predict_proba, background)

# Compute SHAP values (slower!)
shap_values = explainer.shap_values(X.iloc[:100])  # First 100 samples

# Visualize
shap.summary_plot(shap_values[1], X.iloc[:100])
```

---

## 6. Comparing Predictions 🔍

```python
# Compare two predictions
i, j = 0, 50

# Expected value
base_value = explainer.expected_value[1]

# SHAP values
shap_i = shap_values[i]
shap_j = shap_values[j]

# Predictions
pred_i = base_value + shap_i.sum()
pred_j = base_value + shap_j.sum()

print(f"Sample {i}: Prediction = {pred_i:.3f}")
print(f"Sample {j}: Prediction = {pred_j:.3f}")

# Most different features
diff = np.abs(shap_i - shap_j)
top_diff_features = X.columns[np.argsort(-diff)[:5]]

print(f"\nMost different features:")
for feat in top_diff_features:
    idx = X.columns.get_loc(feat)
    print(f"  {feat}:")
    print(f"    Sample {i}: value={X.iloc[i, idx]:.2f}, SHAP={shap_i[idx]:.3f}")
    print(f"    Sample {j}: value={X.iloc[j, idx]:.2f}, SHAP={shap_j[idx]:.3f}")
```

---

## 7. Complete Example: XGBoost 🚀

```python
import xgboost as xgb

# Train XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X, y)

# SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X)

# Summary
shap.summary_plot(shap_values, X)

# Force plot for first 100 samples
shap.force_plot(
    explainer.expected_value,
    shap_values[:100],
    X.iloc[:100],
    matplotlib=True
)

# Dependence plot with interaction
shap.dependence_plot(
    0,  # Feature index
    shap_values,
    X,
    interaction_index='auto'
)
```

---

## Quick Reference 📖

**SHAP Cheat Sheet:**

```python
import shap

# 1. Create explainer
explainer = shap.TreeExplainer(model)  # For trees
# or
explainer = shap.KernelExplainer(model.predict, background)  # Any model

# 2. Compute SHAP values
shap_values = explainer.shap_values(X)

# 3. Visualize
shap.summary_plot(shap_values, X)  # Global importance
shap.waterfall_plot(shap_values[0])  # Single prediction
shap.dependence_plot(0, shap_values, X)  # Feature effect
```

**Explainer Types:**
- **TreeExplainer:** Fast for tree models
- **KernelExplainer:** Model-agnostic (slow)
- **LinearExplainer:** For linear models
- **DeepExplainer:** For neural networks

---

## Key Takeaways 💡

1. **SHAP** = gold standard for explainability
2. **Based on game theory** (fair allocation)
3. **Works for any model**
4. **Local + global** explanations
5. **TreeExplainer** very fast
6. **Summary plot** most useful
7. **Always validate** model behavior with SHAP

---

**Next:** [Lesson 3 - LIME →](Lesson%203%20-%20LIME.md)

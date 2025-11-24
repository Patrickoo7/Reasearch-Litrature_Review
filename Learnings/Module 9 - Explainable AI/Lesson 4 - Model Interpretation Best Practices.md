# Lesson 4: Model Interpretation Best Practices 🎯

**Module 9: Explainable AI | Lesson 4 of 4**

Master best practices for explaining and debugging ML models!

---

## Complete Interpretation Workflow 🔄

```python
import shap
import lime
import lime.lime_tabular
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. Load and prepare data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# 2. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate
print(f"Accuracy: {model.score(X_test, y_test):.3f}")

# 4. Global interpretation (SHAP)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values[1], X_test, feature_names=data.feature_names)

# 5. Local interpretation (LIME)
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=data.feature_names,
    class_names=['malignant', 'benign'],
    mode='classification'
)

# Explain a few predictions
for i in range(3):
    exp = lime_explainer.explain_instance(
        X_test[i],
        model.predict_proba,
        num_features=5
    )
    print(f"\n=== Sample {i} ===")
    print(exp.as_list())
```

---

## Debugging with Explanations 🐛

```python
# Find misclassified samples
predictions = model.predict(X_test)
misclassified = np.where(predictions != y_test)[0]

print(f"Misclassified: {len(misclassified)} / {len(y_test)}")

# Investigate misclassifications
for idx in misclassified[:3]:
    print(f"\n=== Misclassified Sample {idx} ===")
    print(f"True: {data.target_names[y_test[idx]]}")
    print(f"Predicted: {data.target_names[predictions[idx]]}")

    # LIME explanation
    exp = lime_explainer.explain_instance(
        X_test[idx],
        model.predict_proba,
        num_features=10
    )

    print("Top features:")
    for feature, weight in exp.as_list()[:5]:
        print(f"  {feature}: {weight:.3f}")

    # SHAP waterfall
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[1][idx],
            base_values=explainer.expected_value[1],
            data=X_test[idx],
            feature_names=data.feature_names.tolist()
        )
    )
```

---

## Feature Interaction Analysis 🔗

```python
# Find interactions with SHAP
feature_idx = 0  # First feature

shap.dependence_plot(
    feature_idx,
    shap_values[1],
    X_test,
    feature_names=data.feature_names,
    interaction_index='auto'
)

# Manual interaction check
from sklearn.inspection import partial_dependence

features = [0, 1]  # Two features
pdp_result = partial_dependence(
    model, X_test, features,
    kind='average'
)

# 2D PDP
from sklearn.inspection import PartialDependenceDisplay

fig, ax = plt.subplots(figsize=(8, 6))
display = PartialDependenceDisplay.from_estimator(
    model, X_test, [features],
    feature_names=data.feature_names,
    ax=ax
)
plt.show()
```

---

## Model Comparison 📊

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Train multiple models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=10000)
}

# Compare interpretability
for name, model in models.items():
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    print(f"\n=== {name} ===")
    print(f"Accuracy: {accuracy:.3f}")

    # SHAP values
    if name == 'Logistic Regression':
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test[:100])

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Feature importance
    feature_importance = np.abs(shap_values).mean(axis=0)
    top_features = np.argsort(-feature_importance)[:5]

    print("Top 5 features:")
    for idx in top_features:
        print(f"  {data.feature_names[idx]}: {feature_importance[idx]:.3f}")
```

---

## Production Monitoring 🚨

```python
class ExplainableModel:
    """Wrapper for model with explanations"""

    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.explainer = shap.TreeExplainer(model)
        self.feature_names = feature_names
        self.X_train = X_train

    def predict_with_explanation(self, X, top_n=5):
        """Predict and explain"""
        # Predict
        predictions = self.model.predict_proba(X)

        # Explain
        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        results = []
        for i in range(len(X)):
            # Get top features
            feature_importance = np.abs(shap_values[i])
            top_indices = np.argsort(-feature_importance)[:top_n]

            explanation = {
                'prediction': predictions[i],
                'top_features': [
                    {
                        'name': self.feature_names[idx],
                        'value': X[i, idx],
                        'shap': shap_values[i, idx]
                    }
                    for idx in top_indices
                ]
            }

            results.append(explanation)

        return results

    def monitor_predictions(self, X, threshold=0.3):
        """Flag uncertain predictions"""
        predictions = self.model.predict_proba(X)

        # Find uncertain predictions
        uncertain = np.where(
            (predictions[:, 1] > threshold) &
            (predictions[:, 1] < 1 - threshold)
        )[0]

        print(f"Uncertain predictions: {len(uncertain)}")

        # Explain uncertain ones
        for idx in uncertain[:3]:
            print(f"\n=== Sample {idx} ===")
            print(f"Probability: {predictions[idx]}")

            explanation = self.predict_with_explanation(X[idx:idx+1])[0]
            print("Top features:")
            for feat in explanation['top_features']:
                print(f"  {feat['name']}: value={feat['value']:.2f}, SHAP={feat['shap']:.3f}")

# Usage
explainable_model = ExplainableModel(model, X_train, data.feature_names)

# Monitor
explainable_model.monitor_predictions(X_test)

# Predict with explanation
results = explainable_model.predict_with_explanation(X_test[:5])
```

---

## Documentation Template 📝

```python
def create_model_card(model, X_test, y_test, feature_names):
    """Create model documentation"""

    card = f"""
# Model Card

## Model Details
- Type: {model.__class__.__name__}
- Trained: {datetime.now().strftime('%Y-%m-%d')}

## Performance
- Accuracy: {model.score(X_test, y_test):.3f}

## Feature Importance
"""

    # Get SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Top features
    importance = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(-importance)[:10]

    card += "\nTop 10 Features:\n"
    for idx in top_indices:
        card += f"- {feature_names[idx]}: {importance[idx]:.3f}\n"

    card += """
## Interpretation
- Use SHAP for explanations
- Monitor predictions with confidence < 0.7
- Regular retraining required

## Limitations
- May not generalize to new populations
- Feature dependencies not captured
"""

    return card

# Generate
card = create_model_card(model, X_test, y_test, data.feature_names)
print(card)

# Save
with open('model_card.md', 'w') as f:
    f.write(card)
```

---

## Quick Reference 📖

**Interpretation Checklist:**

```
✅ Global feature importance (SHAP summary)
✅ Local explanations (LIME/SHAP waterfall)
✅ Investigate misclassifications
✅ Check feature interactions (dependence plots)
✅ Monitor uncertain predictions
✅ Document model behavior
✅ Regular audits
```

**Best Practices:**
1. **Multiple methods** (SHAP + LIME)
2. **Global + local** explanations
3. **Investigate errors** systematically
4. **Monitor in production**
5. **Document** findings
6. **Regular audits**

---

## Key Takeaways 💡

1. **Always explain** your models
2. **SHAP** for thorough analysis
3. **LIME** for quick checks
4. **Debug** with explanations
5. **Monitor** in production
6. **Document** model behavior
7. **Explainability ≠ accuracy** (trade-off)

---

**Module 9 Complete!** 🎉

**Next Module:** [Module 10 - Production ML & MLOps →](../Module%2010%20-%20Production%20ML%20and%20MLOps/Lesson%201%20-%20Model%20Deployment.md)

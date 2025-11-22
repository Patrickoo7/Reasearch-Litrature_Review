# Lesson 2: Model Evaluation Metrics 📊

**Module 4: Feature Engineering & Model Evaluation | Lesson 2 of 4**

Master evaluation metrics - choose the right metric for your problem!

---

## Why Metrics Matter?

**"What gets measured gets improved"**

**Wrong metric = Wrong model!**
- Optimizing accuracy when you need recall = disaster
- Using RMSE when you care about large errors = wrong focus

---

## 1. Classification Metrics 🎯

### Confusion Matrix

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 0, 1]

cm = confusion_matrix(y_true, y_pred)
print(cm)
# [[TN, FP],
#  [FN, TP]]

# Visualize
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```

### Accuracy, Precision, Recall

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Accuracy: (TP + TN) / Total
accuracy = accuracy_score(y_true, y_pred)

# Precision: TP / (TP + FP) - "Of predicted positives, how many correct?"
precision = precision_score(y_true, y_pred)

# Recall/Sensitivity: TP / (TP + FN) - "Of actual positives, how many found?"
recall = recall_score(y_true, y_pred)

# F1-Score: Harmonic mean of precision and recall
f1 = f1_score(y_true, y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1: {f1:.3f}")
```

**When to use:**
- **Accuracy**: Balanced classes
- **Precision**: False positives costly (spam detection)
- **Recall**: False negatives costly (disease detection)
- **F1**: Balance precision and recall

### ROC-AUC

```python
from sklearn.metrics import roc_auc_score, roc_curve

y_prob = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

import matplotlib.pyplot as plt
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**AUC = 0.5:** Random  
**AUC = 1.0:** Perfect

### Precision-Recall Curve

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
ap = average_precision_score(y_test, y_prob)

plt.plot(recall, precision, label=f'AP = {ap:.3f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

**Use for imbalanced data** (better than ROC-AUC)

### Multi-Class Metrics

```python
from sklearn.metrics import classification_report

y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 1, 2]

print(classification_report(y_true, y_pred))

# Cohen's Kappa (accounts for chance agreement)
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(y_true, y_pred)
```

---

## 2. Regression Metrics 📈

### MAE, MSE, RMSE

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

# Mean Absolute Error (same units as target)
mae = mean_absolute_error(y_true, y_pred)

# Mean Squared Error (penalizes large errors)
mse = mean_squared_error(y_true, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

# R² Score (proportion of variance explained)
r2 = r2_score(y_true, y_pred)

print(f"MAE: {mae:.3f}")
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")
```

**When to use:**
- **MAE**: Robust to outliers
- **MSE/RMSE**: Penalize large errors
- **R²**: Model explanation (0-1, higher = better)

### MAPE

```python
# Mean Absolute Percentage Error
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"MAPE: {mape(y_true, y_pred):.2f}%")
```

### Custom Metrics

```python
# Weighted error (business-driven)
def weighted_error(y_true, y_pred):
    errors = np.abs(y_true - y_pred)
    # Higher penalty for large values
    weights = y_true / y_true.mean()
    return np.mean(errors * weights)
```

---

## 3. Ranking Metrics 📊

### NDCG (Normalized Discounted Cumulative Gain)

```python
from sklearn.metrics import ndcg_score

# True relevance scores
y_true = [[1, 0, 1, 0, 0]]

# Predicted scores
y_score = [[0.9, 0.1, 0.8, 0.3, 0.2]]

ndcg = ndcg_score(y_true, y_score)
print(f"NDCG: {ndcg:.3f}")
```

---

## 4. Custom Business Metrics 💰

```python
def fraud_detection_cost(y_true, y_pred, cost_fn=100, cost_fp=1):
    """
    cost_fn: Cost of missing fraud
    cost_fp: Cost of false alarm
    """
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    total_cost = (FN * cost_fn) + (FP * cost_fp)
    return total_cost

# Example
cost = fraud_detection_cost(y_test, y_pred, cost_fn=100, cost_fp=5)
print(f"Total Cost: ${cost}")
```

---

## Quick Reference 📖

**Classification:**
```python
from sklearn.metrics import (
    accuracy_score,           # Balanced classes
    precision_score,          # Minimize false positives
    recall_score,             # Minimize false negatives
    f1_score,                 # Balance precision/recall
    roc_auc_score,           # Overall discrimination
    average_precision_score   # Imbalanced data
)
```

**Regression:**
```python
from sklearn.metrics import (
    mean_absolute_error,      # Robust, interpretable
    mean_squared_error,       # Penalize large errors
    r2_score,                 # Variance explained
    mean_absolute_percentage_error  # Percentage terms
)
```

**Decision Guide:**

| Problem | Metric | Why |
|---------|--------|-----|
| Fraud detection | Recall, Precision-Recall AUC | Catch frauds |
| Spam filter | Precision | Avoid blocking legit emails |
| Disease diagnosis | Recall | Don't miss diseases |
| Balanced classification | Accuracy, F1 | Simple |
| Imbalanced | Precision-Recall AUC | Better than ROC-AUC |
| House prices | RMSE, MAE | Interpretable error |
| Stock prices | MAPE | Percentage error |
| Ranking (search) | NDCG | Position matters |

---

## Key Takeaways 💡

1. **Never use accuracy alone** for imbalanced data
2. **Precision vs Recall** - understand tradeoff
3. **ROC-AUC** for balanced, **PR-AUC** for imbalanced
4. **RMSE** penalizes large errors more than **MAE**
5. **R²** shows model explanation power
6. **Custom metrics** for business problems
7. **Always match metric to business goal**

---

**Next:** [Lesson 3 - Cross-Validation & Hyperparameter Tuning →](Lesson%203%20-%20Cross-Validation%20and%20Hyperparameter%20Tuning.md)

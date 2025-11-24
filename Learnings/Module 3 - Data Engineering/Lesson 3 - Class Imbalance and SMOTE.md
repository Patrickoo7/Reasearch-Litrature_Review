# Lesson 3: Class Imbalance & SMOTE ⚖️

**Module 3: Data Engineering | Lesson 3 of 4**

Master techniques to handle imbalanced datasets - a critical real-world ML challenge!

---

## Why Class Imbalance Matters?

**Real-world datasets are often imbalanced:**
- Fraud detection: 0.1% fraud, 99.9% legitimate
- Disease diagnosis: 5% diseased, 95% healthy
- Churn prediction: 10% churn, 90% retain
- Spam detection: 20% spam, 80% legitimate

**Problem:** Models predict majority class only → High accuracy but useless!

```
Example: 99% legitimate transactions
Naive model: "Always predict legitimate"
Accuracy: 99% ✅
But catches 0% fraud! ❌
```

---

## 1. Detecting Imbalance 🔍

### Check Class Distribution

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Create imbalanced dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    weights=[0.95, 0.05],  # 95% class 0, 5% class 1
    random_state=42
)

# Check distribution
unique, counts = np.unique(y, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")
print(f"Class 0: {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
print(f"Class 1: {counts[1]} ({counts[1]/len(y)*100:.1f}%)")

# Imbalance ratio
print(f"Imbalance ratio: {counts[0]/counts[1]:.1f}:1")
```

### Visualize Imbalance

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar plot
pd.Series(y).value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Pie chart
pd.Series(y).value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Class Distribution')
plt.show()
```

---

## 2. Baseline Problem 📉

### Naive Model Performance

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train naive model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy looks good!
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")

# But confusion matrix reveals the problem
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Precision, Recall, F1 for minority class are poor!
```

### Better Metrics for Imbalanced Data

```python
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred):.3f}")
```

**Key Metrics:**
- **Precision:** Of predicted positives, how many are correct?
- **Recall:** Of actual positives, how many did we find?
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Overall discrimination ability

---

## 3. Solution 1: Class Weights ⚖️

**Give more importance to minority class during training.**

### Sklearn Models with class_weight

```python
# Logistic Regression with class weights
model = LogisticRegression(class_weight='balanced')  # Auto-compute weights
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("With Class Weights:")
print(classification_report(y_test, y_pred))

# Manual weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
weights_dict = dict(zip(np.unique(y_train), class_weights))
print(f"Class weights: {weights_dict}")

model = LogisticRegression(class_weight=weights_dict)
model.fit(X_train, y_train)
```

### Other Models with Class Weights

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest
rf = RandomForestClassifier(class_weight='balanced', n_estimators=100)
rf.fit(X_train, y_train)

# XGBoost (scale_pos_weight)
scale = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(scale_pos_weight=scale)
xgb.fit(X_train, y_train)

# Compare
models = {
    'LogReg (balanced)': LogisticRegression(class_weight='balanced'),
    'RF (balanced)': RandomForestClassifier(class_weight='balanced', n_estimators=50),
    'XGBoost (scaled)': XGBClassifier(scale_pos_weight=scale)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name}:")
    print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
    print(f"Recall: {recall_score(y_test, y_pred):.3f}")
```

---

## 4. Solution 2: Resampling 🔄

### Undersampling (Reduce Majority Class)

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print(f"Original: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After undersampling: {pd.Series(y_train_under).value_counts().to_dict()}")

# Train on balanced data
model = LogisticRegression()
model.fit(X_train_under, y_train_under)
y_pred = model.predict(X_test)

print("\nUndersampling Results:")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
```

**Pros:** Fast, simple
**Cons:** Loses data, might lose important patterns

### Oversampling (Duplicate Minority Class)

```python
from imblearn.over_sampling import RandomOverSampler

# Random oversampling
ros = RandomOverSampler(random_state=42)
X_train_over, y_train_over = ros.fit_resample(X_train, y_train)

print(f"Original: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After oversampling: {pd.Series(y_train_over).value_counts().to_dict()}")

# Train
model = LogisticRegression()
model.fit(X_train_over, y_train_over)
y_pred = model.predict(X_test)

print("\nOversampling Results:")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
```

**Pros:** No data loss
**Cons:** Overfitting (exact duplicates)

---

## 5. Solution 3: SMOTE ⭐ Best Practice

**SMOTE (Synthetic Minority Over-sampling Technique):**
- Creates synthetic examples by interpolating between existing minority samples
- Better than random oversampling (no exact duplicates)

### How SMOTE Works

```
1. For each minority sample:
2. Find K nearest neighbors (same class)
3. Randomly select one neighbor
4. Create new point between sample and neighbor

Example:
Point A: [1, 2]
Neighbor B: [3, 4]
Synthetic: [2, 3] (midpoint)
```

### Basic SMOTE

```python
from imblearn.over_sampling import SMOTE

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Original: {pd.Series(y_train).value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_smote).value_counts().to_dict()}")

# Train
model = LogisticRegression()
model.fit(X_train_smote, y_train_smote)
y_pred = model.predict(X_test)

print("\nSMOTE Results:")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
```

### SMOTE Variants

```python
from imblearn.over_sampling import (
    SMOTE,
    ADASYN,
    BorderlineSMOTE,
    SVMSMOTE
)

# Different SMOTE variants
smote_variants = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),  # Adaptive synthetic sampling
    'Borderline-SMOTE': BorderlineSMOTE(random_state=42),  # Focus on borderline cases
    'SVM-SMOTE': SVMSMOTE(random_state=42)  # Use SVM for selection
}

results = []
for name, sampler in smote_variants.items():
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)

    model = LogisticRegression()
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)

    results.append({
        'Method': name,
        'F1': f1_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
```

### SMOTE with Different Ratios

```python
# Control minority/majority ratio
smote_ratios = [0.3, 0.5, 0.7, 1.0]  # 1.0 = balanced

for ratio in smote_ratios:
    smote = SMOTE(sampling_strategy=ratio, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    model = LogisticRegression()
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)

    print(f"\nRatio {ratio}:")
    print(f"  Training distribution: {pd.Series(y_res).value_counts().to_dict()}")
    print(f"  F1-Score: {f1_score(y_test, y_pred):.3f}")
```

---

## 6. Solution 4: Combined Sampling 🎨

**Combine undersampling and oversampling for best results.**

### SMOTE + Tomek Links

```python
from imblearn.combine import SMOTETomek

# SMOTE + remove Tomek links (border points)
smote_tomek = SMOTETomek(random_state=42)
X_train_clean, y_train_clean = smote_tomek.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train_clean, y_train_clean)
y_pred = model.predict(X_test)

print("SMOTE + Tomek:")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
```

### SMOTE + ENN

```python
from imblearn.combine import SMOTEENN

# SMOTE + Edited Nearest Neighbors
smote_enn = SMOTEENN(random_state=42)
X_train_clean, y_train_clean = smote_enn.fit_resample(X_train, y_train)

model = LogisticRegression()
model.fit(X_train_clean, y_train_clean)
y_pred = model.predict(X_test)

print("SMOTE + ENN:")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")
```

---

## 7. Threshold Tuning 🎯

**Adjust classification threshold for better recall/precision tradeoff.**

### Default Threshold (0.5)

```python
from sklearn.metrics import precision_recall_curve

# Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Default threshold
y_pred_default = (y_prob >= 0.5).astype(int)
print(f"Threshold 0.5:")
print(f"  Precision: {precision_score(y_test, y_pred_default):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_default):.3f}")
```

### Optimal Threshold

```python
# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# F1-Score for each threshold
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"\nOptimal threshold: {optimal_threshold:.3f}")

# Predict with optimal threshold
y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
print(f"Precision: {precision_score(y_test, y_pred_optimal):.3f}")
print(f"Recall: {recall_score(y_test, y_pred_optimal):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred_optimal):.3f}")

# Plot precision-recall curve
plt.figure(figsize=(10, 6))
plt.plot(recalls, precisions, label='Precision-Recall curve')
plt.scatter(recalls[optimal_idx], precisions[optimal_idx],
           color='red', s=100, label=f'Optimal (threshold={optimal_threshold:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.show()
```

### Business-Driven Threshold

```python
# Prioritize recall (catch all frauds, even with false positives)
high_recall_threshold = 0.3
y_pred_high_recall = (y_prob >= high_recall_threshold).astype(int)

print(f"\nHigh Recall Threshold ({high_recall_threshold}):")
print(f"  Recall: {recall_score(y_test, y_pred_high_recall):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred_high_recall):.3f}")

# Prioritize precision (minimize false positives)
high_precision_threshold = 0.7
y_pred_high_precision = (y_prob >= high_precision_threshold).astype(int)

print(f"\nHigh Precision Threshold ({high_precision_threshold}):")
print(f"  Precision: {precision_score(y_test, y_pred_high_precision):.3f}")
print(f"  Recall: {recall_score(y_test, y_pred_high_precision):.3f}")
```

---

## 8. Complete Pipeline 🎯

### End-to-End Imbalanced Classification

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline as ImbPipeline  # Supports imbalanced-learn
from sklearn.model_selection import cross_val_score

# Pipeline with SMOTE
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression())
])

# Cross-validation (use stratified folds)
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')

print(f"Cross-validation F1-Score: {scores.mean():.3f} (+/- {scores.std():.3f})")

# Train on full training set
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("\nTest Set Performance:")
print(classification_report(y_test, y_pred))
```

### Comparing All Methods

```python
from sklearn.ensemble import RandomForestClassifier

# Define methods
methods = {
    'Baseline': (X_train, y_train, LogisticRegression()),
    'Class Weights': (X_train, y_train, LogisticRegression(class_weight='balanced')),
    'Undersampling': (*RandomUnderSampler(random_state=42).fit_resample(X_train, y_train),
                     LogisticRegression()),
    'Oversampling': (*RandomOverSampler(random_state=42).fit_resample(X_train, y_train),
                     LogisticRegression()),
    'SMOTE': (*SMOTE(random_state=42).fit_resample(X_train, y_train),
              LogisticRegression()),
    'SMOTE+Tomek': (*SMOTETomek(random_state=42).fit_resample(X_train, y_train),
                    LogisticRegression()),
}

results = []
for name, (X_tr, y_tr, model) in methods.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_test)

    results.append({
        'Method': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Balanced Acc': balanced_accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    })

results_df = pd.DataFrame(results)
print(results_df.round(3).to_string(index=False))
```

---

## 9. Real-World Example: Fraud Detection 🕵️

```python
# Simulate fraud detection dataset
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=30,
    n_informative=20,
    n_redundant=10,
    weights=[0.99, 0.01],  # 1% fraud
    flip_y=0.02,  # Add noise
    random_state=42
)

print(f"Fraud rate: {y.mean():.2%}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                     stratify=y, random_state=42)

# Best practice pipeline
from xgboost import XGBClassifier

pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(sampling_strategy=0.5, random_state=42)),  # 50% ratio
    ('classifier', XGBClassifier(
        scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
        max_depth=5,
        n_estimators=100,
        random_state=42
    ))
])

# Train
pipeline.fit(X_train, y_train)

# Predict with probability
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]

# Predict with optimal threshold
y_pred = (y_prob >= optimal_threshold).astype(int)

print("\nFraud Detection Results:")
print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Cost analysis (example)
cost_false_negative = 100  # Miss a fraud
cost_false_positive = 1    # False alarm

TN, FP, FN, TP = cm.ravel()
total_cost = (FN * cost_false_negative) + (FP * cost_false_positive)
print(f"\nTotal Cost: ${total_cost}")
```

---

## Quick Reference 📖

**Decision Guide:**

```
Small imbalance (60:40) → Class weights
Medium imbalance (80:20) → SMOTE or class weights
High imbalance (95:5) → SMOTE + class weights
Extreme imbalance (99:1) → SMOTE + threshold tuning + ensemble
```

**Best Practices:**
1. Always use **stratified splitting** (`stratify=y`)
2. Apply resampling **only to training data**
3. Evaluate with **F1-Score, Recall, Precision** (not just accuracy)
4. Use **cross-validation** with stratified folds
5. Tune **threshold** based on business needs
6. Consider **ensemble methods** for extreme imbalance

**Code Template:**

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(class_weight='balanced'))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

## Practice Exercises 🏋️

1. Create imbalanced dataset (95:5) and compare baseline vs SMOTE
2. Find optimal threshold for fraud detection scenario
3. Implement cost-sensitive learning
4. Build complete pipeline with SMOTE and XGBoost

<details>
<summary>Solutions</summary>

```python
# 1. Baseline vs SMOTE
X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Baseline
baseline = LogisticRegression()
baseline.fit(X_train, y_train)
print(f"Baseline F1: {f1_score(y_test, baseline.predict(X_test)):.3f}")

# SMOTE
X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
smote_model = LogisticRegression()
smote_model.fit(X_smote, y_smote)
print(f"SMOTE F1: {f1_score(y_test, smote_model.predict(X_test)):.3f}")

# 2. Optimal threshold
y_prob = smote_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_threshold = thresholds[np.argmax(f1_scores)]
print(f"Optimal threshold: {optimal_threshold:.3f}")

# 3. Cost-sensitive
from sklearn.tree import DecisionTreeClassifier

cost_fp = 1
cost_fn = 10
weight_ratio = cost_fn / cost_fp

model = DecisionTreeClassifier(class_weight={0: 1, 1: weight_ratio})
model.fit(X_train, y_train)

# 4. Complete pipeline
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('xgb', XGBClassifier(scale_pos_weight=10, n_estimators=100))
])
pipeline.fit(X_train, y_train)
print(f"Pipeline F1: {f1_score(y_test, pipeline.predict(X_test)):.3f}")
```
</details>

---

## Key Takeaways 💡

1. **Never use accuracy** for imbalanced data
2. **SMOTE** is industry standard for oversampling
3. **Class weights** + SMOTE often best combination
4. **Threshold tuning** critical for business objectives
5. **Always stratify** train/test split
6. **Apply resampling only to training set**
7. **F1-Score** balances precision and recall

---

**Next:** [Lesson 4 - Anomaly Detection & Outlier Handling →](Lesson%204%20-%20Anomaly%20Detection%20and%20Outlier%20Handling.md)

---

**Congratulations!** You can now handle imbalanced datasets like a pro! 🎉

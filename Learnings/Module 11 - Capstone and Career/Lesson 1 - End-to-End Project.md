# Lesson 1: End-to-End ML Project 🚀

**Module 11: Capstone & Career | Lesson 1 of 4**

Build a complete ML project from start to finish!

---

## Project Workflow 🔄

```
1. Problem Definition
2. Data Collection
3. Exploratory Data Analysis
4. Data Preprocessing
5. Feature Engineering
6. Model Selection & Training
7. Model Evaluation
8. Hyperparameter Tuning
9. Model Deployment
10. Monitoring & Maintenance
```

---

## Example Project: Customer Churn Prediction 📊

### 1. Problem Definition

**Goal:** Predict which customers will churn (leave) in the next month

**Metrics:**
- Primary: F1-Score (balanced metric)
- Secondary: Precision, Recall
- Business: Revenue saved

---

### 2. Data Collection

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('customer_data.csv')

print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Churn'].value_counts(normalize=True)}")
```

**Dataset Features:**
- Customer demographics (age, gender, location)
- Account info (tenure, contract type, payment method)
- Usage data (monthly charges, total charges, services used)
- Target: Churn (0 = stayed, 1 = left)

---

### 3. Exploratory Data Analysis

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')

# Churn rate
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Age distribution
axes[0, 0].hist([df[df['Churn']==0]['Age'], df[df['Churn']==1]['Age']],
                label=['No Churn', 'Churn'], bins=30, alpha=0.7)
axes[0, 0].set_xlabel('Age')
axes[0, 0].legend()

# Tenure vs Churn
axes[0, 1].boxplot([df[df['Churn']==0]['Tenure'], df[df['Churn']==1]['Tenure']])
axes[0, 1].set_xticklabels(['No Churn', 'Churn'])
axes[0, 1].set_ylabel('Tenure (months)')

# Monthly charges
axes[1, 0].scatter(df['Tenure'], df['MonthlyCharges'],
                   c=df['Churn'], alpha=0.5, cmap='coolwarm')
axes[1, 0].set_xlabel('Tenure')
axes[1, 0].set_ylabel('Monthly Charges')

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr = df[numeric_cols].corr()
sns.heatmap(corr, ax=axes[1, 1], cmap='coolwarm', center=0)

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=300)
plt.show()

# Statistical tests
from scipy.stats import ttest_ind

# Compare churned vs non-churned customers
for col in ['Age', 'Tenure', 'MonthlyCharges']:
    stat, p_value = ttest_ind(
        df[df['Churn']==0][col],
        df[df['Churn']==1][col]
    )
    print(f"{col}: t-stat={stat:.3f}, p-value={p_value:.4f}")
```

---

### 4. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Create a copy
df_processed = df.copy()

# Handle missing values
df_processed['TotalCharges'] = pd.to_numeric(
    df_processed['TotalCharges'],
    errors='coerce'
)
df_processed['TotalCharges'].fillna(
    df_processed['MonthlyCharges'],
    inplace=True
)

# Encode categorical variables
label_encoders = {}
categorical_cols = df_processed.select_dtypes(include=['object']).columns
categorical_cols = categorical_cols.drop('Churn')  # Exclude target

for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

# Encode target
df_processed['Churn'] = (df_processed['Churn'] == 'Yes').astype(int)

# Split features and target
X = df_processed.drop('Churn', axis=1)
y = df_processed['Churn']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train_scaled.shape}")
print(f"Test set: {X_test_scaled.shape}")
print(f"Churn rate in train: {y_train.mean():.3f}")
print(f"Churn rate in test: {y_test.mean():.3f}")
```

---

### 5. Feature Engineering

```python
# Create new features
df_processed['ChargePerTenure'] = df_processed['TotalCharges'] / (df_processed['Tenure'] + 1)
df_processed['AvgMonthlyCharge'] = df_processed['TotalCharges'] / (df_processed['Tenure'] + 1)
df_processed['IsNewCustomer'] = (df_processed['Tenure'] < 6).astype(int)
df_processed['IsHighValue'] = (df_processed['MonthlyCharges'] > df_processed['MonthlyCharges'].median()).astype(int)

# Interaction features
df_processed['Tenure_x_Contract'] = df_processed['Tenure'] * df_processed['Contract']
df_processed['Charges_x_PaymentMethod'] = df_processed['MonthlyCharges'] * df_processed['PaymentMethod']

print(f"Original features: {X.shape[1]}")
print(f"After feature engineering: {df_processed.shape[1]}")
```

---

### 6. Model Selection & Training

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\n=== {name} ===")

    # Train
    model.fit(X_train_scaled, y_train)

    # Predict
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluate
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    # Store results
    results[name] = {
        'model': model,
        'roc_auc': roc_auc_score(y_test, y_proba),
        'predictions': y_pred,
        'probabilities': y_proba
    }

    print(f"ROC-AUC: {results[name]['roc_auc']:.4f}")

# Compare models
comparison = pd.DataFrame({
    name: {'ROC-AUC': res['roc_auc']}
    for name, res in results.items()
}).T.sort_values('ROC-AUC', ascending=False)

print("\n=== Model Comparison ===")
print(comparison)
```

---

### 7. Model Evaluation

```python
from sklearn.metrics import roc_curve, precision_recall_curve

# Get best model
best_model_name = comparison.index[0]
best_model = results[best_model_name]['model']

print(f"Best model: {best_model_name}")

# Confusion matrix
cm = confusion_matrix(y_test, results[best_model_name]['predictions'])
print(f"\nConfusion Matrix:\n{cm}")

# Plot ROC curve
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# ROC
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['probabilities'])
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})")

axes[0].plot([0, 1], [0, 1], 'k--', label='Random')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves')
axes[0].legend()
axes[0].grid(True)

# Precision-Recall
for name, res in results.items():
    precision, recall, _ = precision_recall_curve(y_test, res['probabilities'])
    axes[1].plot(recall, precision, label=name)

axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].set_title('Precision-Recall Curves')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300)
plt.show()
```

---

### 8. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid search for best model
if best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    # Randomized search (faster)
    search = RandomizedSearchCV(
        XGBClassifier(random_state=42, eval_metric='logloss'),
        param_distributions=param_grid,
        n_iter=20,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train_scaled, y_train)

    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV score: {search.best_score_:.4f}")

    # Use best model
    final_model = search.best_estimator_

else:
    final_model = best_model

# Evaluate final model
y_pred_final = final_model.predict(X_test_scaled)
y_proba_final = final_model.predict_proba(X_test_scaled)[:, 1]

print("\n=== Final Model Performance ===")
print(classification_report(y_test, y_pred_final, target_names=['No Churn', 'Churn']))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_final):.4f}")
```

---

### 9. Model Interpretation

```python
import shap

# SHAP values
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns.tolist())

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== Top 10 Features ===")
print(feature_importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'][:10], feature_importance['importance'][:10])
plt.xlabel('Importance')
plt.title('Top 10 Features')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300)
plt.show()
```

---

### 10. Save Model & Artifacts

```python
import joblib
import json
from datetime import datetime

# Create model directory
import os
os.makedirs('models', exist_ok=True)

# Save model
model_path = f'models/churn_model_{datetime.now().strftime("%Y%m%d")}.joblib'
joblib.dump(final_model, model_path)

# Save scaler
scaler_path = 'models/scaler.joblib'
joblib.dump(scaler, scaler_path)

# Save label encoders
encoders_path = 'models/label_encoders.joblib'
joblib.dump(label_encoders, encoders_path)

# Save metadata
metadata = {
    'model_type': best_model_name,
    'train_date': datetime.now().isoformat(),
    'features': X.columns.tolist(),
    'performance': {
        'roc_auc': float(roc_auc_score(y_test, y_proba_final)),
        'test_size': len(y_test),
        'churn_rate': float(y_test.mean())
    },
    'hyperparameters': final_model.get_params() if hasattr(final_model, 'get_params') else {}
}

with open('models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Model saved to: {model_path}")
print(f"Scaler saved to: {scaler_path}")
print(f"Metadata saved to: models/metadata.json")
```

---

## Project Deliverables 📦

### 1. Code Structure

```
churn-prediction/
├── data/
│   ├── raw/
│   │   └── customer_data.csv
│   └── processed/
│       └── processed_data.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── load_data.py
│   │   └── preprocess.py
│   ├── features/
│   │   └── engineer.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── plots.py
├── models/
│   ├── churn_model_20240115.joblib
│   ├── scaler.joblib
│   └── metadata.json
├── tests/
│   ├── test_preprocessing.py
│   └── test_model.py
├── api/
│   └── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

### 2. README Template

```markdown
# Customer Churn Prediction

Predict customer churn using machine learning.

## Problem Statement
Predict which customers will churn in the next month to enable proactive retention.

## Data
- 7,043 customers
- 20 features (demographics, account, usage)
- Target: Churn (26.5% positive class)

## Approach
1. EDA and feature engineering
2. Tested 4 models (LR, RF, GB, XGB)
3. XGBoost achieved best performance (ROC-AUC: 0.87)
4. Deployed as FastAPI endpoint

## Results
- ROC-AUC: 0.87
- Precision: 0.72
- Recall: 0.68
- F1-Score: 0.70

## Key Findings
- Tenure is strongest predictor
- Month-to-month contracts have 3x churn rate
- New customers (<6 months) most at risk

## Usage
```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python src/models/train.py

# Run API
uvicorn api.app:app --reload

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

## Future Work
- Add deep learning model
- Implement real-time monitoring
- A/B test retention strategies
```

---

## Key Takeaways 💡

1. **End-to-end workflow** is critical
2. **EDA** informs feature engineering
3. **Multiple models** for comparison
4. **Hyperparameter tuning** improves performance
5. **Interpretability** builds trust
6. **Documentation** enables maintenance
7. **Version control** everything (Git + DVC)

---

**Next:** [Lesson 2 - Interview Preparation →](Lesson%202%20-%20Interview%20Preparation.md)

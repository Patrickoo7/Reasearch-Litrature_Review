# Lesson 3: The ML Workflow 🔄

**Module 1: Foundations | Lesson 3 of 4**

Learn the complete end-to-end process of a Machine Learning project - from problem definition to deployment.

---

## The Complete ML Workflow

```
1. Define Problem
      ↓
2. Collect Data
      ↓
3. Explore & Clean Data
      ↓
4. Prepare Data
      ↓
5. Choose Model
      ↓
6. Train Model
      ↓
7. Evaluate Model
      ↓
8. Tune Hyperparameters
      ↓
9. Deploy Model
      ↓
10. Monitor & Maintain
```

Let's go through each step with practical examples!

---

## Step 1: Define the Problem 🎯

**Most important step!** A well-defined problem is half-solved.

### Questions to Ask

**1. What are you trying to predict/achieve?**
```
❌ Bad: "Improve customer satisfaction"
✅ Good: "Predict if a customer will churn within 30 days"
```

**2. What type of ML problem is this?**
- Classification? Regression? Clustering?
- Supervised? Unsupervised? Reinforcement?

**3. How will success be measured?**
```
❌ Bad: "Make good predictions"
✅ Good: "Achieve 85% accuracy on test set" or
        "Reduce customer churn by 20%"
```

**4. What are the constraints?**
- Latency requirements? (real-time vs batch)
- Resource limits? (computation, storage)
- Interpretability needed?

### Example: Email Spam Detection

```
Problem: Classify emails as spam or not spam

Type: Supervised Learning - Binary Classification

Success Metric:
- Precision: 95% (avoid false positives)
- Recall: 90% (catch most spam)

Constraints:
- Must run in <100ms (real-time)
- Model size <10MB (mobile deployment)
```

---

## Step 2: Collect Data 📊

**Garbage in = Garbage out!** Data quality is crucial.

### Data Sources

**Existing Data:**
- Databases
- Log files
- APIs
- Public datasets (Kaggle, UCI ML Repository)

**New Data:**
- Surveys
- Sensors/IoT devices
- User interactions
- Web scraping (check legal!)

### How Much Data?

**Rule of Thumb:**
- Simple models: 100s to 1000s of examples
- Deep learning: 10,000s to millions

**Quality > Quantity:**
- 1000 high-quality examples better than 100,000 noisy ones

### Example Code

```python
import pandas as pd

# Load data from CSV
data = pd.read_csv('emails.csv')

# Load from database
import sqlite3
conn = sqlite3.connect('database.db')
data = pd.read_sql_query("SELECT * FROM emails", conn)

# Load from API
import requests
response = requests.get('https://api.example.com/data')
data = pd.DataFrame(response.json())

print(f"Collected {len(data)} samples")
```

---

## Step 3: Explore & Clean Data 🔍

**Understand your data before modeling!**

### Exploratory Data Analysis (EDA)

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('house_prices.csv')

# Basic info
print(df.info())
print(df.describe())

# Check first rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Visualize distributions
df.hist(figsize=(12, 10))
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Target distribution
df['price'].hist(bins=50)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()
```

### Data Cleaning Tasks

**1. Handle Missing Values**
```python
# Check missing
print(df.isnull().sum())

# Strategy 1: Remove rows
df_clean = df.dropna()

# Strategy 2: Fill with mean/median
df['age'].fillna(df['age'].median(), inplace=True)

# Strategy 3: Fill with mode (categorical)
df['category'].fillna(df['category'].mode()[0], inplace=True)
```

**2. Handle Outliers**
```python
# Detect outliers (IQR method)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers
df_clean = df[(df['price'] >= lower_bound) &
              (df['price'] <= upper_bound)]

# Or cap outliers
df['price'] = df['price'].clip(lower_bound, upper_bound)
```

**3. Handle Duplicates**
```python
# Check duplicates
print(f"Duplicates: {df.duplicated().sum()}")

# Remove duplicates
df_clean = df.drop_duplicates()
```

---

## Step 4: Prepare Data 🧹

**Transform data into ML-ready format.**

### Feature Engineering

**Create new features from existing ones:**
```python
# Example: House price prediction
df['price_per_sqft'] = df['price'] / df['square_feet']
df['total_rooms'] = df['bedrooms'] + df['bathrooms']
df['age'] = 2024 - df['year_built']
```

### Encode Categorical Variables

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding (for ordinal categories)
label_encoder = LabelEncoder()
df['size'] = label_encoder.fit_transform(df['size'])
# 'small' → 0, 'medium' → 1, 'large' → 2

# One-Hot Encoding (for nominal categories)
df_encoded = pd.get_dummies(df, columns=['color'])
# color: 'red'/'green'/'blue' →
# color_red: 0/1, color_green: 0/1, color_blue: 0/1
```

### Scale/Normalize Features

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standardization (mean=0, std=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Normalization (range 0-1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

### Split Data

```python
from sklearn.model_selection import train_test_split

# Split into features (X) and target (y)
X = df.drop('price', axis=1)
y = df['price']

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
```

**Common Splits:**
- 80/20 (train/test)
- 70/15/15 (train/validation/test)
- 60/20/20 (train/validation/test)

---

## Step 5: Choose Model 🤖

**Select appropriate algorithm for your problem.**

### Decision Guide

**For Classification:**
```python
# Start simple
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# If non-linear relationships
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()

# For high-dimensional data
from sklearn.svm import SVC
model = SVC()
```

**For Regression:**
```python
# Start simple
from sklearn.linear_model import LinearRegression
model = LinearRegression()

# If non-linear
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

# For complex patterns
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
```

**Pro Tip:** Start simple, then increase complexity!

---

## Step 6: Train Model 🏋️

**Teach the model using training data.**

```python
from sklearn.ensemble import RandomForestClassifier

# Create model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# Train
model.fit(X_train, y_train)

print("Training complete!")
```

**Training Process:**
1. Model learns patterns from training data
2. Adjusts internal parameters
3. Minimizes error on training set

---

## Step 7: Evaluate Model 📈

**Measure performance on unseen data.**

### For Classification

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions
y_pred = model.predict(X_test)

# Metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall: {recall_score(y_test, y_pred):.3f}")
print(f"F1-Score: {f1_score(y_test, y_pred):.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
```

### For Regression

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")

# Visualize predictions vs actual
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5)

print(f"Cross-validation scores: {scores}")
print(f"Mean: {scores.mean():.3f} (+/- {scores.std():.3f})")
```

---

## Step 8: Tune Hyperparameters ⚙️

**Optimize model settings for better performance.**

### Grid Search

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.3f}")

# Use best model
best_model = grid_search.best_estimator_
```

### Random Search (faster)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Define distributions
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'min_samples_split': randint(2, 20)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_dist,
    n_iter=100,  # Try 100 random combinations
    cv=5,
    random_state=42
)

random_search.fit(X_train, y_train)
print(f"Best parameters: {random_search.best_params_}")
```

---

## Step 9: Deploy Model 🚀

**Put model into production.**

### Save Model

```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
loaded_model = joblib.load('model.pkl')

# Predict with loaded model
prediction = loaded_model.predict(new_data)
```

### Create API (Flask Example)

```python
from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.json
    features = np.array(data['features']).reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Usage:**
```bash
# Start server
python app.py

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1200, 3, 10]}'

# Response: {"prediction": 250000}
```

---

## Step 10: Monitor & Maintain 📊

**Track performance and update model over time.**

### What to Monitor

**1. Model Performance**
```python
# Log predictions and actual values
import logging

logging.basicConfig(filename='predictions.log', level=logging.INFO)

def predict_and_log(features, actual=None):
    prediction = model.predict([features])[0]

    log_entry = {
        'timestamp': datetime.now(),
        'features': features,
        'prediction': prediction,
        'actual': actual
    }

    logging.info(log_entry)

    return prediction
```

**2. Data Drift**
```
Are new inputs different from training data?
→ May need to retrain
```

**3. Model Degradation**
```
Is accuracy decreasing over time?
→ Retrain with recent data
```

### When to Retrain

✅ Retrain when:
- Performance drops significantly
- New patterns emerge in data
- Business requirements change
- Periodically (e.g., monthly)

### A/B Testing

```
Deploy new model to 10% of users
↓
Compare with old model
↓
If better → roll out to 100%
If worse → keep old model
```

---

## Complete Workflow Example 🎯

**Problem:** Predict house prices

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# 1. Load data
df = pd.read_csv('houses.csv')

# 2. Explore
print(df.info())
print(df.describe())

# 3. Clean
df = df.dropna()
df = df.drop_duplicates()

# 4. Prepare
X = df[['sqft', 'bedrooms', 'bathrooms', 'age']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5-6. Choose & Train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 7. Evaluate
y_pred = model.predict(X_test)
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.3f}")

# 8. Tune (simplified)
# ... GridSearchCV here ...

# 9. Deploy
joblib.dump(model, 'house_price_model.pkl')

# 10. Monitor
# ... Logging and monitoring code ...
```

---

## Common Pitfalls & Best Practices ⚠️

### Pitfalls to Avoid

❌ **Data Leakage**
```python
# WRONG: Scale before split
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled)

# RIGHT: Split first, then scale
X_train, X_test = train_test_split(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use same scaler!
```

❌ **Overfitting**
```
Training accuracy: 99%
Test accuracy: 65%
→ Model memorized training data!
```

❌ **Not using validation set**
```
Train → Test directly
→ Might overfit hyperparameter tuning
```

### Best Practices

✅ **Use version control** (Git) for code and data
✅ **Document everything** (experiments, decisions)
✅ **Start simple** before going complex
✅ **Use cross-validation** for small datasets
✅ **Monitor in production**
✅ **Reproducibility** (set random seeds)

---

## Key Takeaways 💡

1. **ML is iterative**: Expect to go back and forth between steps
2. **Data quality matters most**: Spend time on steps 2-4
3. **Start simple**: Don't jump to complex models immediately
4. **Measure what matters**: Choose right metrics for your problem
5. **Deploy early**: Get real-world feedback quickly
6. **Monitor continuously**: Models degrade over time

---

## Practice Project 🏋️

**Build a complete ML project:**

1. Choose a dataset from [Kaggle](https://www.kaggle.com/datasets)
2. Follow all 10 steps
3. Document your process
4. Deploy as API or web app

**Suggested datasets:**
- Titanic Survival (classification)
- House Prices (regression)
- Iris Flowers (multi-class classification)

---

## What's Next?

You now understand the complete ML workflow! Next, we'll learn important **concepts and terminology** that you'll use throughout your ML journey.

**Next:** [Lesson 4 - Key Concepts and Terminology →](Lesson%204%20-%20Key%20Concepts%20and%20Terminology.md)

---

## Further Reading 📚

- [Google's ML Guide](https://developers.google.com/machine-learning/guides)
- [Kaggle Learn](https://www.kaggle.com/learn)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

---

**Congratulations!** You now know the complete end-to-end ML workflow! 🎉

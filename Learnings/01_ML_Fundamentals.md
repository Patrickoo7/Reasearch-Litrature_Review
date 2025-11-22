# Machine Learning Fundamentals 🌱

## What is Machine Learning?

**Machine Learning (ML)** is a subset of Artificial Intelligence that enables computers to learn from data without being explicitly programmed.

### Traditional Programming vs Machine Learning

**Traditional Programming:**
```
Data + Program → Output
```
Example: Calculate sum of numbers
```python
def sum_numbers(a, b):
    return a + b  # Explicit rules
```

**Machine Learning:**
```
Data + Output → Program (Model)
```
Example: Predict house prices
```python
model.fit(house_features, prices)  # Learn patterns
prediction = model.predict(new_house)
```

## Types of Machine Learning

### 1. Supervised Learning 📊

Learning from **labeled data** (input-output pairs).

**Use Cases:**
- Email spam detection (spam/not spam)
- Image classification (cat/dog/bird)
- House price prediction
- Medical diagnosis

**Types:**
- **Classification**: Predict discrete categories
  - Binary: spam/not spam
  - Multi-class: cat/dog/bird

- **Regression**: Predict continuous values
  - House prices
  - Temperature
  - Stock prices

**Example:**
```python
from sklearn.linear_model import LinearRegression

# Training data
X = [[1], [2], [3], [4]]  # Features (house size)
y = [100, 200, 300, 400]  # Labels (prices)

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[5]])  # Predict price for size 5
```

### 2. Unsupervised Learning 🔍

Learning from **unlabeled data** (find patterns).

**Use Cases:**
- Customer segmentation
- Anomaly detection
- Data compression
- Recommendation systems

**Types:**
- **Clustering**: Group similar data points
  - K-Means, DBSCAN, Hierarchical

- **Dimensionality Reduction**: Reduce features
  - PCA, t-SNE, UMAP

- **Association**: Find relationships
  - Market basket analysis

**Example:**
```python
from sklearn.cluster import KMeans

# Data without labels
X = [[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]]

# Find 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.labels_  # [0, 0, 1, 1, 0, 1]
```

### 3. Reinforcement Learning 🎮

Learning through **trial and error** with rewards/penalties.

**Use Cases:**
- Game playing (Chess, Go, Atari)
- Robotics
- Autonomous vehicles
- Resource optimization

**Components:**
- **Agent**: Learner/decision maker
- **Environment**: World the agent interacts with
- **Action**: What the agent can do
- **Reward**: Feedback signal
- **State**: Current situation

**Example Concept:**
```
Robot learning to walk:
- Try stepping forward → Fall → Negative reward → Learn
- Balance properly → Stay upright → Positive reward → Reinforce
```

### 4. Semi-Supervised Learning 🔄

Learning from **small labeled + large unlabeled data**.

**Use Cases:**
- When labeling is expensive
- Medical imaging (few labeled scans)
- Speech recognition

### 5. Self-Supervised Learning 🧠

Learning by creating **labels from the data itself**.

**Use Cases:**
- Language models (predict next word)
- Image models (predict rotated image)
- Video understanding

**Example:**
```
Text: "The cat sat on the ___"
Model learns to predict: "mat"
(Labels created automatically from text)
```

## Key ML Terminology

### 1. Features (Input Variables)
Measurable properties used for prediction.

**Example - House Price Prediction:**
- Size (sq ft)
- Number of bedrooms
- Location
- Age of house

### 2. Labels (Output Variables)
What we want to predict.

**Example:**
- House price: $500,000

### 3. Training Data
Data used to teach the model.

```python
# 80% of total data
X_train = [[1400, 3], [1600, 3], [1700, 4]]  # Features
y_train = [245000, 312000, 279000]           # Labels
```

### 4. Test Data
Data used to evaluate model performance.

```python
# 20% of total data (never seen during training)
X_test = [[1800, 4]]
y_test = [355000]
```

### 5. Model
Mathematical function that maps inputs to outputs.

```
f(x) = y
where x = features, y = prediction
```

### 6. Training (Learning)
Process of finding the best model parameters.

### 7. Prediction (Inference)
Using the trained model on new data.

### 8. Loss/Cost Function
Measures how wrong the model's predictions are.

**Example - Mean Squared Error:**
```python
loss = (1/n) * Σ(y_true - y_pred)²
```

### 9. Hyperparameters
Settings you configure before training.

**Examples:**
- Learning rate
- Number of layers
- Number of neurons
- Batch size

### 10. Parameters
Values the model learns during training.

**Examples:**
- Weights
- Biases

## The Machine Learning Workflow

### Step 1: Define the Problem 🎯
- What are you trying to predict?
- Is it classification or regression?
- What data do you have?

### Step 2: Collect Data 📊
- Gather relevant data
- Ensure data quality
- More data = better models (usually)

### Step 3: Explore & Visualize Data 🔍
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data.csv')
print(df.describe())  # Statistics
df.hist()  # Visualize distributions
plt.show()
```

### Step 4: Prepare Data 🧹
- Handle missing values
- Remove outliers
- Encode categorical variables
- Normalize/standardize
- Split into train/test sets

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 5: Choose & Train Model 🤖
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
```

### Step 6: Evaluate Model 📈
```python
from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}")
```

### Step 7: Tune Hyperparameters ⚙️
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
```

### Step 8: Deploy Model 🚀
- Save the model
- Create API/web service
- Monitor performance
- Retrain periodically

## When to Use Machine Learning?

### ✅ Good Use Cases
1. **Pattern exists but is complex**
   - Image recognition
   - Natural language understanding

2. **Cannot program explicit rules**
   - Spam detection (too many patterns)
   - Recommendation systems

3. **Adaptability needed**
   - User preferences change
   - Market conditions evolve

4. **Scale**
   - Too much data for humans to process
   - Need automated decisions

### ❌ Poor Use Cases
1. **Simple rule-based logic**
   - Calculate tax (use formula)
   - Sort numbers (use algorithm)

2. **No data available**
   - Need historical examples

3. **Explainability critical**
   - ML models can be "black boxes"
   - Consider simpler models if interpretability is key

4. **Perfect accuracy required**
   - ML has inherent uncertainty
   - Safety-critical systems need careful design

## Common ML Challenges

### 1. Overfitting 📈
Model learns training data **too well**, including noise.

**Signs:**
- High training accuracy, low test accuracy
- Model memorizes instead of learning patterns

**Solutions:**
- More training data
- Regularization
- Simpler model
- Cross-validation

### 2. Underfitting 📉
Model is **too simple** to capture patterns.

**Signs:**
- Low training and test accuracy
- Model is too basic

**Solutions:**
- More complex model
- More features
- Less regularization

### 3. Data Quality Issues 🗑️
- Missing values
- Outliers
- Incorrect labels
- Biased data

### 4. Curse of Dimensionality 📐
Too many features make data sparse.

**Solution:** Feature selection/reduction

### 5. Class Imbalance ⚖️
One class dominates (99% class A, 1% class B).

**Solutions:**
- Resample data
- Use appropriate metrics
- Class weights

## Key Takeaways 💡

1. **ML learns patterns from data** instead of following explicit rules

2. **Three main types**: Supervised, Unsupervised, Reinforcement Learning

3. **ML workflow**: Problem → Data → Prepare → Train → Evaluate → Deploy

4. **Balance is key**: Not too simple (underfitting), not too complex (overfitting)

5. **Data quality matters**: Garbage in = garbage out

6. **Start simple**: Try simple models before complex ones

7. **Evaluation is crucial**: Don't just look at training performance

## Practice Exercises 🏋️

1. **Identify the type:**
   - Predicting customer churn: ?
   - Grouping similar products: ?
   - Game-playing AI: ?

   <details>
   <summary>Answers</summary>
   - Classification (Supervised)
   - Clustering (Unsupervised)
   - Reinforcement Learning
   </details>

2. **Code Exercise:**
   Train a simple ML model on the Iris dataset:
   ```python
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.tree import DecisionTreeClassifier
   from sklearn.metrics import accuracy_score

   # Your code here
   ```

3. **Think:**
   - Why might a model perform well on training data but poorly on test data?
   - What would you do if you have only 100 labeled examples but 10,000 unlabeled ones?

## Next Steps 📚

Ready to dive deeper? Move on to:
- **[02_Classical_ML_Algorithms.md](02_Classical_ML_Algorithms.md)** - Learn specific algorithms
- **[03_Data_Preprocessing.md](03_Data_Preprocessing.md)** - Master data preparation

---

**Remember:** Machine Learning is a practical skill. The best way to learn is by doing! 🚀

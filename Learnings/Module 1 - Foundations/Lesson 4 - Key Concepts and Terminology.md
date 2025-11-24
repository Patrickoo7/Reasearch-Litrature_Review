# Lesson 4: Key Concepts and Terminology 📚

**Module 1: Foundations | Lesson 4 of 4**

Master the essential concepts and vocabulary you'll use throughout your ML journey. This is your ML dictionary!

---

## Core Terminology 🎯

### Features (Input Variables, X)

**Definition:** The input variables used to make predictions.

**Example - House Price Prediction:**
```python
Features (X):
- Square footage: 1500
- Number of bedrooms: 3
- Number of bathrooms: 2
- Age of house: 10 years
- Location (zip code): 94102

These are what we measure/know about the house
```

**Types of Features:**
- **Numerical**: Age (25), Price ($50.00), Temperature (72°F)
- **Categorical**: Color (red/blue), Size (S/M/L), City names
- **Binary**: IsSpam (Yes/No), HasGarage (True/False)
- **Ordinal**: Rating (1-5 stars), Education (High School/Bachelor/PhD)

**Feature Vector:**
```python
# Single sample represented as vector
house_features = [1500, 3, 2, 10, 94102]

# Multiple samples as matrix
X = [[1500, 3, 2, 10, 94102],
     [1200, 2, 1, 5, 94103],
     [2000, 4, 3, 2, 94104]]
```

---

### Labels (Target Variable, Output, y)

**Definition:** What we want to predict.

**Example:**
```python
# Classification labels
y = [1, 0, 1, 0]  # spam=1, not spam=0
y = ['cat', 'dog', 'bird', 'cat']  # multi-class

# Regression labels
y = [250000, 300000, 450000]  # house prices
y = [72.5, 85.2, 68.9]  # temperatures
```

---

### Training Data vs Test Data

**Training Data:** Used to teach the model
```python
X_train = [[features...], [features...], ...]
y_train = [label1, label2, ...]

# Model learns from this
model.fit(X_train, y_train)
```

**Test Data:** Used to evaluate the model (unseen during training)
```python
X_test = [[features...], [features...], ...]
y_test = [label1, label2, ...]

# Measure performance
accuracy = model.score(X_test, y_test)
```

**Validation Data:** Used for hyperparameter tuning
```python
# Common splits
70% Training
15% Validation
15% Test
```

**Why split?**
- Training: Learn patterns
- Validation: Choose best hyperparameters
- Test: Final unbiased evaluation

**Critical Rule:** NEVER use test data during training!

---

### Model (Hypothesis)

**Definition:** Mathematical function that maps inputs to outputs.

```python
# Simple model
def model(x):
    return weight * x + bias

# In reality
prediction = model.predict(features)
```

**What's inside a model:**
- **Parameters**: Learned from data (weights, biases)
- **Hyperparameters**: Set before training (learning rate, tree depth)

---

### Parameters vs Hyperparameters

**Parameters (learned from data):**
```python
# Linear Regression: y = mx + b
m = 2.5  # slope (parameter)
b = 10   # intercept (parameter)

# Neural Network
weights = [[0.5, -0.3, 0.8], ...]  # learned parameters
biases = [0.1, -0.2, ...]
```

**Hyperparameters (set by you):**
```python
model = RandomForestClassifier(
    n_estimators=100,        # hyperparameter
    max_depth=10,            # hyperparameter
    min_samples_split=5      # hyperparameter
)

# Or for neural networks
learning_rate = 0.001        # hyperparameter
batch_size = 32              # hyperparameter
num_epochs = 100             # hyperparameter
```

---

## The Big Concepts 🧠

### 1. Overfitting vs Underfitting

**The Goldilocks Problem:** Not too simple, not too complex, just right!

#### Underfitting (Too Simple)

```
Model is too simple to capture patterns

Training Error: High ❌
Test Error: High ❌

Example: Using linear model for non-linear data
```

**Visual:**
```
True pattern: Curved
Your model: Straight line
→ Misses important patterns
```

**Code Example:**
```python
# Underfitting: Too simple
model = LinearRegression()  # Linear model for non-linear data
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)  # 0.65 (poor)
test_score = model.score(X_test, y_test)      # 0.63 (poor)
# Both scores low → underfitting
```

#### Overfitting (Too Complex)

```
Model memorizes training data instead of learning patterns

Training Error: Very Low ✅
Test Error: High ❌

Example: Decision tree with no depth limit
```

**Visual:**
```
Model passes through every training point
→ Captures noise, not signal
→ Fails on new data
```

**Code Example:**
```python
# Overfitting: Too complex
model = DecisionTreeClassifier(max_depth=None)  # No limit!
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)  # 0.99 (too good!)
test_score = model.score(X_test, y_test)      # 0.72 (poor)
# Large gap → overfitting
```

#### Just Right (Good Fit)

```
Model captures true patterns without memorizing noise

Training Error: Low ✅
Test Error: Low ✅
Gap is small ✅

Example: Well-regularized model
```

**Code Example:**
```python
# Good fit: Just right
model = RandomForestClassifier(max_depth=10)  # Limited complexity
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)  # 0.88 (good)
test_score = model.score(X_test, y_test)      # 0.85 (good)
# Small gap, both scores good → good fit
```

**Solutions to Overfitting:**
- Get more training data
- Reduce model complexity
- Use regularization
- Use cross-validation
- Early stopping

**Solutions to Underfitting:**
- Use more complex model
- Add more features
- Reduce regularization

---

### 2. Bias-Variance Tradeoff

**The fundamental tradeoff in ML!**

**Bias:** Error from overly simple models
```
High bias → Underfitting
Assumes data is simpler than it is
```

**Variance:** Error from overly complex models
```
High variance → Overfitting
Too sensitive to training data fluctuations
```

**Visual Representation:**
```
High Bias, Low Variance
└─ Consistently wrong (underfitting)

Low Bias, High Variance
└─ Inconsistent, memorizes training data (overfitting)

Low Bias, Low Variance
└─ Sweet spot! ✅
```

**Example:**
```python
# High Bias (too simple)
linear_model = LinearRegression()
# Always makes same type of predictions
# Even if data is non-linear

# High Variance (too complex)
deep_tree = DecisionTreeClassifier(max_depth=100)
# Different tree structure for slightly different training data
# Unstable, overfits

# Balanced
rf = RandomForestClassifier(max_depth=10, n_estimators=100)
# Ensemble reduces variance
# Depth limit prevents overfitting
```

**Goal:** Minimize total error = Bias² + Variance + Noise

---

### 3. Training Error vs Generalization Error

**Training Error:** Performance on training data
```python
train_predictions = model.predict(X_train)
train_error = mse(y_train, train_predictions)
```

**Generalization Error (Test Error):** Performance on unseen data
```python
test_predictions = model.predict(X_test)
test_error = mse(y_test, test_predictions)
```

**What matters:** Generalization error!
- Training error can be misleading
- Model needs to work on new data

**Example:**
```
Model A: train_error=0.05, test_error=0.08  ✅ Good!
Model B: train_error=0.01, test_error=0.25  ❌ Overfitting!

Choose Model A
```

---

### 4. Regularization

**Definition:** Techniques to prevent overfitting by penalizing complexity.

**Common Types:**

**L1 Regularization (Lasso):**
```python
from sklearn.linear_model import Lasso

# Adds penalty: |weight|
model = Lasso(alpha=0.1)  # alpha controls strength
model.fit(X_train, y_train)

# Effect: Some weights become exactly 0
# → Feature selection
```

**L2 Regularization (Ridge):**
```python
from sklearn.linear_model import Ridge

# Adds penalty: weight²
model = Ridge(alpha=0.1)
model.fit(X_train, y_train)

# Effect: Weights become small but not 0
# → Reduces impact of all features
```

**Elastic Net (L1 + L2):**
```python
from sklearn.linear_model import ElasticNet

model = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

**Dropout (Neural Networks):**
```python
# Randomly drop neurons during training
model.add(Dropout(0.5))  # Drop 50% of neurons
```

---

### 5. Loss Function (Cost Function)

**Definition:** Measures how wrong the model's predictions are.

**For Regression:**
```python
# Mean Squared Error (MSE)
mse = (1/n) * Σ(y_true - y_pred)²

from sklearn.metrics import mean_squared_error
loss = mean_squared_error(y_true, y_pred)
```

**For Classification:**
```python
# Binary Cross-Entropy
loss = -Σ[y*log(ŷ) + (1-y)*log(1-ŷ)]

from sklearn.metrics import log_loss
loss = log_loss(y_true, y_pred_proba)
```

**Goal of Training:** Minimize the loss function

---

### 6. Gradient Descent

**Definition:** Algorithm to minimize the loss function.

**How it works:**
```
1. Start with random parameters
2. Calculate loss
3. Calculate gradient (direction of steepest increase)
4. Update parameters in opposite direction
5. Repeat until convergence
```

**Analogy:** Walking down a hill in the fog
- Can't see the bottom
- Feel the slope under your feet
- Take step in steepest downward direction

**Code Concept:**
```python
# Simplified gradient descent
learning_rate = 0.01

for epoch in range(1000):
    # Forward pass
    predictions = model(X_train)
    loss = calculate_loss(y_train, predictions)

    # Backward pass (compute gradients)
    gradients = calculate_gradients(loss)

    # Update parameters
    parameters = parameters - learning_rate * gradients
```

**Variants:**
- **Batch Gradient Descent**: Use all data
- **Stochastic Gradient Descent (SGD)**: Use one sample
- **Mini-batch Gradient Descent**: Use small batch (most common)

---

### 7. Learning Rate

**Definition:** How big steps to take during gradient descent.

```python
# Too small
learning_rate = 0.00001
→ Training too slow, may not converge

# Too large
learning_rate = 10
→ Overshoots minimum, diverges

# Just right
learning_rate = 0.01
→ Converges smoothly ✅
```

**Adaptive Learning Rates:**
```python
# Modern optimizers adjust learning rate automatically
from torch.optim import Adam

optimizer = Adam(model.parameters(), lr=0.001)
# Learning rate decreases over time
```

---

### 8. Epoch, Batch, Iteration

**Epoch:** One complete pass through entire training dataset
```python
for epoch in range(100):  # 100 epochs
    train_on_full_dataset()
```

**Batch:** Subset of training data processed together
```python
batch_size = 32
# Process 32 samples at a time
```

**Iteration:** One update of model parameters
```python
# If dataset has 1000 samples and batch_size=32:
iterations_per_epoch = 1000 / 32 = 31.25 ≈ 32 iterations
```

**Example:**
```
Dataset: 10,000 samples
Batch size: 100
Epochs: 50

Iterations per epoch: 10,000 / 100 = 100
Total iterations: 100 * 50 = 5,000
```

---

### 9. Baseline Model

**Definition:** Simple model to compare against.

**Why?** Know if your complex model is actually useful!

**Examples:**
```python
# Classification baseline: Most frequent class
from sklearn.dummy import DummyClassifier
baseline = DummyClassifier(strategy='most_frequent')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)

# Regression baseline: Mean of training targets
from sklearn.dummy import DummyRegressor
baseline = DummyRegressor(strategy='mean')
baseline.fit(X_train, y_train)
baseline_score = baseline.score(X_test, y_test)
```

**Your model should beat the baseline!**

---

### 10. Confusion Matrix (Classification)

**Shows all types of prediction outcomes:**

```
                Predicted
                Pos    Neg
Actual  Pos     TP     FN
        Neg     FP     TN

TP = True Positive (correctly predicted positive)
TN = True Negative (correctly predicted negative)
FP = False Positive (incorrectly predicted positive) - Type I error
FN = False Negative (incorrectly predicted negative) - Type II error
```

**Example - Email Spam:**
```
                Predicted
                Spam   Not Spam
Actual  Spam    45     5         (50 actual spam)
        Not     10     940       (950 actual not spam)

TP = 45  (correctly identified spam)
FN = 5   (missed spam)
FP = 10  (false alarms)
TN = 940 (correctly identified not spam)
```

**Derived Metrics:**
```python
Accuracy = (TP + TN) / Total = (45 + 940) / 1000 = 0.985

Precision = TP / (TP + FP) = 45 / (45 + 10) = 0.818
"Of predicted spam, how many were actually spam?"

Recall = TP / (TP + FN) = 45 / (45 + 5) = 0.900
"Of actual spam, how many did we catch?"

F1-Score = 2 * (Precision * Recall) / (Precision + Recall)
"Harmonic mean of precision and recall"
```

---

## Quick Reference Guide 📖

| Term | Definition | Example |
|------|------------|---------|
| **Feature** | Input variable | House size, # bedrooms |
| **Label** | Target to predict | House price, Is spam? |
| **Model** | Function: X → y | LinearRegression() |
| **Parameter** | Learned by model | Weights, biases |
| **Hyperparameter** | Set by you | learning_rate, max_depth |
| **Overfitting** | Memorizes training data | train_acc=99%, test_acc=65% |
| **Underfitting** | Too simple | train_acc=65%, test_acc=63% |
| **Bias** | Error from simplicity | Linear model for curves |
| **Variance** | Error from complexity | Different for each training set |
| **Regularization** | Prevent overfitting | L1, L2, Dropout |
| **Loss** | How wrong predictions are | MSE, Cross-Entropy |
| **Epoch** | Full pass through data | 100 epochs = see all data 100x |

---

## Key Takeaways 💡

1. **Features (X)** are inputs, **Labels (y)** are what you predict
2. **Always split** data: train/validation/test
3. **Overfitting** is the most common problem in ML
4. **Bias-variance tradeoff** is fundamental
5. **Regularization** helps prevent overfitting
6. **Baseline model** gives you a reference point
7. **Test error** is what actually matters, not training error

---

## Practice Quiz 🏋️

Test your understanding:

1. You have a model with train_acc=0.95 and test_acc=0.65. What's the problem?
2. What's the difference between a parameter and a hyperparameter?
3. If you have 1000 samples, batch_size=50, and 10 epochs, how many iterations?
4. When should you use L1 vs L2 regularization?
5. Your model always predicts "Not Spam". What might be wrong?

<details>
<summary>Answers</summary>

1. **Overfitting** - Large gap between train and test performance
2. **Parameter**: Learned from data (weights). **Hyperparameter**: Set before training (learning rate)
3. **200 iterations** - (1000/50) * 10 = 20 * 10
4. **L1**: Want feature selection (sparse model). **L2**: Want to reduce all features
5. **Class imbalance** - Probably way more "Not Spam" in training data

</details>

---

## Congratulations! 🎉

**You've completed Module 1: Foundations!**

You now understand:
✅ What Machine Learning is
✅ Different types of ML
✅ The complete ML workflow
✅ Essential concepts and terminology

**You're ready for:** [Module 2 - Classical Machine Learning →](../../Module%202%20-%20Classical%20Machine%20Learning/Lesson%201%20-%20Linear%20Models.md)

---

## Final Challenge Project 💪

**Build your first ML project!**

1. Pick a dataset from [Kaggle](https://www.kaggle.com/datasets)
   - Titanic (classification)
   - House Prices (regression)
   - Iris (multi-class)

2. Follow the complete workflow:
   - Load & explore data
   - Clean & prepare
   - Train baseline model
   - Train better model
   - Evaluate & compare
   - Document findings

3. Focus on:
   - Proper train/test split
   - Avoiding overfitting
   - Beating baseline
   - Understanding results

**Share your results** on Kaggle or GitHub!

---

## Further Reading 📚

- [Google's ML Glossary](https://developers.google.com/machine-learning/glossary)
- [Scikit-learn Glossary](https://scikit-learn.org/stable/glossary.html)
- [Deep Learning Book - Chapter 5 (ML Basics)](https://www.deeplearningbook.org/contents/ml.html)

---

**You've built a strong foundation!** Time to learn specific algorithms in Module 2. 🚀

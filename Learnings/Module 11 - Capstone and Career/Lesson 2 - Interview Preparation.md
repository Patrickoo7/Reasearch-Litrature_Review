# Lesson 2: ML Interview Preparation 🎯

**Module 11: Capstone & Career | Lesson 2 of 4**

Ace ML interviews with this comprehensive preparation guide!

---

## Interview Types 📝

1. **Coding** (LeetCode-style algorithms)
2. **ML Theory** (concepts, math, algorithms)
3. **ML System Design** (architecture, scalability)
4. **Take-home Project** (real-world problem)
5. **Behavioral** (STAR method)

---

## Part 1: Coding Interview 💻

### Essential Topics

```python
# 1. Arrays & Strings
def two_sum(nums, target):
    """Find two numbers that add up to target"""
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Test
print(two_sum([2, 7, 11, 15], 9))  # [0, 1]


# 2. Binary Search
def binary_search(arr, target):
    """Find target in sorted array"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1

# Test
print(binary_search([1, 3, 5, 7, 9], 5))  # 2


# 3. Sliding Window
def max_sum_subarray(arr, k):
    """Max sum of k consecutive elements"""
    window_sum = sum(arr[:k])
    max_sum = window_sum

    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, window_sum)

    return max_sum

# Test
print(max_sum_subarray([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))  # 39


# 4. Dynamic Programming
def fibonacci(n, memo={}):
    """Fibonacci with memoization"""
    if n in memo:
        return memo[n]
    if n <= 2:
        return 1

    memo[n] = fibonacci(n - 1, memo) + fibonacci(n - 2, memo)
    return memo[n]

# Test
print(fibonacci(10))  # 55


# 5. Trees
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def max_depth(root):
    """Find max depth of binary tree"""
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))
```

### Common Patterns

```python
# Pattern 1: Two Pointers
def reverse_string(s):
    """Reverse string in-place"""
    left, right = 0, len(s) - 1
    while left < right:
        s[left], s[right] = s[right], s[left]
        left += 1
        right -= 1
    return s


# Pattern 2: Hash Map
def first_unique_char(s):
    """Find first non-repeating character"""
    count = {}
    for char in s:
        count[char] = count.get(char, 0) + 1

    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1


# Pattern 3: Sorting
def merge_intervals(intervals):
    """Merge overlapping intervals"""
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], current[1])
        else:
            merged.append(current)

    return merged
```

---

## Part 2: ML Theory Questions 🧠

### Fundamental Concepts

**Q1: Explain bias-variance tradeoff**

```
Answer:
- Bias: Error from overly simplistic assumptions
  - High bias → underfitting
  - Example: Linear model for non-linear data

- Variance: Error from sensitivity to training data
  - High variance → overfitting
  - Example: Deep decision tree

- Tradeoff: Can't minimize both simultaneously
  - Sweet spot: Balance between the two
  - Regularization helps reduce variance
  - Feature engineering helps reduce bias
```

**Q2: Difference between L1 and L2 regularization**

```python
# L1 (Lasso): Adds sum of absolute values
# - Penalty: λ * Σ|w_i|
# - Effect: Sparse weights (some become 0)
# - Use: Feature selection

from sklearn.linear_model import Lasso
model_l1 = Lasso(alpha=0.1)

# L2 (Ridge): Adds sum of squares
# - Penalty: λ * Σw_i²
# - Effect: Small weights (none become 0)
# - Use: Prevent overfitting

from sklearn.linear_model import Ridge
model_l2 = Ridge(alpha=0.1)
```

**Q3: How does gradient descent work?**

```python
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    """
    1. Initialize weights randomly
    2. Calculate gradient of loss function
    3. Update weights: w = w - α * ∇L
    4. Repeat until convergence
    """
    m, n = X.shape
    weights = np.zeros(n)

    for _ in range(iterations):
        # Predictions
        predictions = X.dot(weights)

        # Gradient
        gradient = (1/m) * X.T.dot(predictions - y)

        # Update
        weights -= learning_rate * gradient

    return weights
```

**Q4: Explain cross-validation**

```
Purpose: Assess model generalization

K-Fold CV:
1. Split data into K folds
2. Train on K-1 folds, validate on 1
3. Repeat K times (each fold as validation once)
4. Average results

Advantages:
- Uses all data for training and validation
- Reduces variance in performance estimate
- Detects overfitting

When to use:
- Small datasets
- Model selection
- Hyperparameter tuning
```

**Q5: Classification metrics**

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# Confusion Matrix:
#              Predicted
#           |  0  |  1  |
# Actual 0  | TN  | FP  |
#        1  | FN  | TP  |

# Accuracy: (TP + TN) / Total
# - When to use: Balanced classes

# Precision: TP / (TP + FP)
# - When to use: Minimize false positives (spam detection)

# Recall: TP / (TP + FN)
# - When to use: Minimize false negatives (disease detection)

# F1-Score: 2 * (Precision * Recall) / (Precision + Recall)
# - When to use: Imbalanced classes, balance P and R

# ROC-AUC: Area under ROC curve
# - When to use: Compare models, threshold-independent
```

**Q6: Decision Trees vs Random Forests**

```
Decision Tree:
- Single tree splits on features
- Prone to overfitting
- Low bias, high variance
- Fast to train

Random Forest:
- Ensemble of many trees
- Bootstrap sampling + feature randomness
- Reduces variance through averaging
- More robust, less overfitting
- Slower but better performance
```

**Q7: Handling imbalanced data**

```python
# Method 1: Resampling
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

# Method 2: Class weights
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(class_weight='balanced')

# Method 3: Evaluation metrics
# Use F1, precision-recall, ROC-AUC instead of accuracy

# Method 4: Anomaly detection
# Treat minority class as anomaly
from sklearn.ensemble import IsolationForest
```

**Q8: Feature scaling - when and why?**

```python
# When needed:
# - Distance-based algorithms (KNN, SVM, K-Means)
# - Gradient descent (faster convergence)
# - Neural networks (stable training)

# Not needed:
# - Tree-based models (Random Forest, XGBoost)

# StandardScaler: (x - mean) / std
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# MinMaxScaler: (x - min) / (max - min)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

---

## Part 3: ML System Design 🏗️

### Example Question: "Design a recommendation system for an e-commerce platform"

**Answer Structure:**

```
1. CLARIFY REQUIREMENTS (5 min)
   - Scale: 1M users, 100K products
   - Latency: <100ms
   - Personalized vs non-personalized
   - Cold start problem

2. HIGH-LEVEL DESIGN (10 min)
   - Candidate generation (recall)
   - Ranking (precision)
   - Re-ranking (business rules)

3. DATA (5 min)
   - User features: demographics, history, preferences
   - Item features: category, price, ratings
   - Interaction data: clicks, purchases, ratings
   - Context: time, device, location

4. MODEL ARCHITECTURE (10 min)

   a) Candidate Generation:
      - Matrix factorization (ALS)
      - Two-tower neural network
      - Generate top 1000 candidates

   b) Ranking:
      - Gradient boosting (XGBoost)
      - Features: user-item interactions, popularity
      - Rank top 100

   c) Re-ranking:
      - Diversity
      - Business rules (inventory, margin)
      - Final top 20

5. TRAINING & SERVING (5 min)
   - Offline: Batch training daily
   - Online: Real-time feature updates
   - A/B testing for new models

6. EVALUATION (5 min)
   - Offline: AUC, precision@k, NDCG
   - Online: CTR, conversion rate, revenue
   - A/B tests with statistical significance

7. SCALABILITY (5 min)
   - Distributed training (Spark)
   - Model serving (TensorFlow Serving)
   - Feature store (Feast)
   - Caching (Redis)

8. MONITORING (3 min)
   - Data drift
   - Model performance
   - Latency/throughput
   - A/B test results
```

### Common System Design Patterns

```
1. Batch Prediction
   - Precompute predictions
   - Store in cache/database
   - Serve quickly

2. Online Prediction
   - Real-time feature extraction
   - Model serving infrastructure
   - Low latency (<100ms)

3. Two-Stage Models
   - Stage 1: Candidate generation (recall)
   - Stage 2: Ranking (precision)

4. Ensemble
   - Multiple models
   - Voting/averaging
   - Improved robustness
```

---

## Part 4: Behavioral Questions 🗣️

### STAR Method

```
Situation: Set the context
Task: Describe the challenge
Action: Explain what YOU did
Result: Share the outcome (quantify!)
```

### Common Questions

**Q: "Tell me about a challenging ML project"**

```
Situation:
"At Company X, we had low user engagement (15% DAU)"

Task:
"I was asked to build a recommendation system to increase engagement"

Action:
"I implemented a two-stage collaborative filtering system:
 - Analyzed user behavior data (10M interactions)
 - Built matrix factorization for candidate generation
 - Trained XGBoost for ranking
 - A/B tested with 10% of users"

Result:
"Increased DAU by 23% and session time by 18% over 2 months.
 Model served 50K recommendations/second with 80ms latency."
```

**Q: "How do you handle disagreements with teammates?"**

```
Situation:
"My team disagreed on model choice: deep learning vs gradient boosting"

Task:
"Need to decide objectively while maintaining team harmony"

Action:
"I proposed:
 1. Define success metrics (accuracy, latency, interpretability)
 2. Implement both approaches
 3. Compare on holdout set
 4. Present findings to team"

Result:
"Gradient boosting won (95% accuracy vs 94%, 10x faster inference).
 Team agreed based on data. Learned to let experiments decide."
```

---

## Part 5: Take-Home Project Tips 🏠

### Best Practices

```
1. UNDERSTAND THE PROBLEM
   - Read requirements 3 times
   - Clarify ambiguities via email
   - Define success metrics

2. STRUCTURE YOUR WORK
   - Clean code (PEP 8)
   - Notebooks for exploration
   - Scripts for production code
   - Tests (pytest)

3. DOCUMENTATION
   - README with setup instructions
   - Explain your approach
   - Show results clearly
   - Discuss limitations

4. GO BEYOND REQUIREMENTS
   - Try multiple models
   - Show feature importance
   - Discuss next steps
   - Production considerations

5. TIME MANAGEMENT
   - Don't overcomplicate
   - Focus on fundamentals
   - Submit on time
```

### Project Template

```
project/
├── README.md           # Problem, approach, results
├── requirements.txt    # Dependencies
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── data.py        # Data loading
│   ├── features.py    # Feature engineering
│   ├── models.py      # Model training
│   └── evaluate.py    # Evaluation
├── tests/
│   └── test_models.py
└── results/
    ├── metrics.txt
    └── plots/
```

---

## Quick Reference: 50 Must-Know Questions 📖

### Algorithms (10)
1. Explain gradient descent
2. Bias-variance tradeoff
3. L1 vs L2 regularization
4. Decision trees - how do they split?
5. Random forests - how do they work?
6. Gradient boosting intuition
7. SVM and kernel trick
8. K-means clustering
9. PCA - what and why?
10. Neural network backpropagation

### Evaluation (10)
11. Precision vs recall
12. When to use F1-score?
13. ROC-AUC interpretation
14. Cross-validation purpose
15. Overfitting detection
16. Class imbalance handling
17. A/B testing basics
18. Statistical significance
19. Confusion matrix
20. Regression metrics (RMSE, MAE, R²)

### Features (10)
21. Feature scaling - when?
22. One-hot encoding vs label encoding
23. Handling missing data
24. Feature selection methods
25. Curse of dimensionality
26. Feature engineering techniques
27. Outlier detection
28. Data normalization
29. Categorical encoding strategies
30. Time-based features

### Deep Learning (10)
31. Activation functions (ReLU, Sigmoid)
32. CNN - how convolution works
33. RNN vs LSTM
34. Dropout purpose
35. Batch normalization
36. Transfer learning
37. Vanishing gradient problem
38. Optimizers (SGD, Adam)
39. Loss functions
40. Embeddings

### Production (10)
41. Model deployment strategies
42. A/B testing in production
43. Data drift detection
44. Model monitoring
45. Batch vs online prediction
46. Model versioning
47. Handling model failures
48. Scalability considerations
49. Feature store
50. CI/CD for ML

---

## Key Takeaways 💡

1. **Practice coding** on LeetCode (50+ problems)
2. **Understand theory** deeply, not just memorize
3. **System design** requires practice - do mock interviews
4. **STAR method** for behavioral questions
5. **Take-home projects** - quality over speed
6. **Ask clarifying questions** in interviews
7. **Communicate your thought process** clearly

---

## Preparation Timeline ⏰

```
3 months before:
✅ Complete this course
✅ Build 2-3 portfolio projects
✅ Start LeetCode (easy/medium)

2 months before:
✅ Mock interviews (Pramp, Interviewing.io)
✅ Study ML theory (this lesson!)
✅ Practice system design

1 month before:
✅ Review weak areas
✅ More mock interviews
✅ Prepare behavioral stories

1 week before:
✅ Review notes
✅ Rest well
✅ Prepare questions for interviewer
```

---

**Next:** [Lesson 3 - Portfolio Building →](Lesson%203%20-%20Portfolio%20Building.md)

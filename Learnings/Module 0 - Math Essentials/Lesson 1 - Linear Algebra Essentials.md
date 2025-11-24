# Lesson 1: Linear Algebra Essentials ➗

**Module 0: Math Essentials | Lesson 1 of 4**

Learn just enough linear algebra to understand machine learning - no PhD required!

---

## Why Linear Algebra for ML?

ML is all about transforming data using **matrices and vectors**:

```python
# This is what happens inside ML models:
predictions = weights @ features + bias  # @ is matrix multiplication
```

**Everything in ML uses linear algebra:**
- Neural networks: Matrix multiplications
- Images: Matrices of pixels
- Datasets: Matrices of features
- Transformations: Matrix operations

---

## 1. Vectors 📐

**Definition:** Ordered list of numbers (1D array).

### Representation

```python
import numpy as np

# Vector as Python list
v = [1, 2, 3]

# Vector as NumPy array (preferred)
v = np.array([1, 2, 3])

print(v.shape)  # (3,) - 3 elements
```

**Geometric View:**
```
2D vector: [2, 3]
→ Point at (2, 3) or arrow from origin to (2, 3)

      |
    3 |     •(2,3)
    2 |    /
    1 |   /
    0 |__________
      0  1  2  3
```

### Vector Operations

**Addition:**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

c = a + b  # [5, 7, 9]
# Element-wise: [1+4, 2+5, 3+6]
```

**Scalar Multiplication:**
```python
v = np.array([1, 2, 3])
result = 2 * v  # [2, 4, 6]
```

**Dot Product (Inner Product):**
```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

dot = np.dot(a, b)  # 1*4 + 2*5 + 3*6 = 32
# Or
dot = a @ b  # Same result
```

**Why it matters in ML:**
```python
# Prediction = weighted sum
features = np.array([1.5, 2.0, 3.0])  # Input
weights = np.array([0.5, 0.3, 0.2])    # Learned parameters

prediction = np.dot(features, weights)  # Dot product!
# = 1.5*0.5 + 2.0*0.3 + 3.0*0.2 = 1.65
```

**Magnitude (Length):**
```python
v = np.array([3, 4])
magnitude = np.linalg.norm(v)  # √(3² + 4²) = 5.0
```

---

## 2. Matrices 🔲

**Definition:** 2D array of numbers (table).

### Representation

```python
# Matrix as 2D NumPy array
M = np.array([[1, 2, 3],
              [4, 5, 6]])

print(M.shape)  # (2, 3) - 2 rows, 3 columns
```

**Visualize:**
```
M = [1  2  3]  ← Row 0
    [4  5  6]  ← Row 1
     ↑  ↑  ↑
   Col0 1  2
```

**ML Example - Dataset:**
```python
# 3 samples (rows), 4 features (columns)
data = np.array([[5.1, 3.5, 1.4, 0.2],  # Sample 1
                 [4.9, 3.0, 1.4, 0.2],  # Sample 2
                 [6.2, 3.4, 5.4, 2.3]]) # Sample 3

# Shape: (3, 4) = 3 samples × 4 features
```

### Matrix Operations

**Addition:**
```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

C = A + B  # Element-wise
# [[6, 8], [10, 12]]
```

**Scalar Multiplication:**
```python
A = np.array([[1, 2], [3, 4]])
result = 2 * A  # [[2, 4], [6, 8]]
```

**Matrix Multiplication (Most Important!):**
```python
A = np.array([[1, 2],    # (2, 2)
              [3, 4]])

B = np.array([[5, 6],    # (2, 2)
              [7, 8]])

C = A @ B  # (2, 2)
# [[1*5+2*7, 1*6+2*8],
#  [3*5+4*7, 3*6+4*8]]
# = [[19, 22], [43, 50]]
```

**Rule:** (m × n) @ (n × p) = (m × p)
```
A: (2, 3) @ B: (3, 4) = C: (2, 4) ✅
A: (2, 3) @ B: (4, 5) = ERROR ❌ (3 ≠ 4)
```

**ML Example:**
```python
# Neural network layer
X = np.array([[1, 2, 3]])      # Input: (1, 3)
W = np.array([[0.1, 0.2],      # Weights: (3, 2)
              [0.3, 0.4],
              [0.5, 0.6]])

output = X @ W  # (1, 2)
# [[1*0.1+2*0.3+3*0.5, 1*0.2+2*0.4+3*0.6]]
# = [[2.2, 2.8]]
```

**Transpose:**
```python
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)

A_T = A.T  # (3, 2)
# [[1, 4],
#  [2, 5],
#  [3, 6]]
```

**Identity Matrix:**
```python
I = np.eye(3)  # 3×3 identity
# [[1, 0, 0],
#  [0, 1, 0],
#  [0, 0, 1]]

# Property: A @ I = I @ A = A
```

---

## 3. Matrix-Vector Operations in ML

### Dataset Representation

```python
# ML dataset
X = np.array([[1.0, 2.0, 3.0],   # Sample 1: 3 features
              [1.5, 2.5, 3.5],   # Sample 2
              [2.0, 3.0, 4.0]])  # Sample 3

y = np.array([10, 15, 20])       # Target values

# X shape: (3, 3) = 3 samples × 3 features
# y shape: (3,) = 3 labels
```

### Linear Regression Formula

```
y = Xw + b

where:
X: (n_samples, n_features) - data
w: (n_features,) - weights
b: scalar - bias
y: (n_samples,) - predictions
```

```python
# Example
X = np.array([[1, 2],
              [2, 3],
              [3, 4]])  # (3, 2)

w = np.array([0.5, 0.3])  # (2,)
b = 1.0

predictions = X @ w + b  # (3,)
# = [[1*0.5+2*0.3], [2*0.5+3*0.3], [3*0.5+4*0.3]] + 1.0
# = [1.1 + 1.0, 1.9 + 1.0, 2.7 + 1.0]
# = [2.1, 2.9, 3.7]
```

### Batch Processing

```python
# Process multiple samples at once
batch_size = 32
features = 10

X_batch = np.random.randn(batch_size, features)  # (32, 10)
W = np.random.randn(features, 5)                  # (10, 5)

output = X_batch @ W  # (32, 5)
# All 32 samples processed in one operation!
```

---

## 4. Special Matrices in ML

### Diagonal Matrix

```python
D = np.diag([1, 2, 3])
# [[1, 0, 0],
#  [0, 2, 0],
#  [0, 0, 3]]

# Used in: Scaling transformations
```

### Symmetric Matrix

```python
# A = A.T
S = np.array([[1, 2, 3],
              [2, 4, 5],
              [3, 5, 6]])

# Used in: Covariance matrices
```

---

## 5. Practical ML Operations

### Batch Normalization

```python
# Center data around 0
X = np.array([[1, 2], [3, 4], [5, 6]])

mean = np.mean(X, axis=0)  # Mean per feature
X_centered = X - mean
```

### Standardization

```python
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)

X_standardized = (X - mean) / std
```

### Euclidean Distance (KNN)

```python
def euclidean_distance(a, b):
    """Distance between two vectors"""
    diff = a - b
    return np.sqrt(np.sum(diff ** 2))

# Or using norm
distance = np.linalg.norm(a - b)
```

---

## Quick Reference 📖

| Operation | NumPy Code | Shape Rule |
|-----------|-----------|------------|
| Vector dot product | `a @ b` | (n,) @ (n,) → scalar |
| Matrix multiply | `A @ B` | (m,n) @ (n,p) → (m,p) |
| Element-wise multiply | `A * B` | Same shapes |
| Transpose | `A.T` | (m,n) → (n,m) |
| Identity | `np.eye(n)` | (n,n) |
| Zeros | `np.zeros((m,n))` | (m,n) |
| Ones | `np.ones((m,n))` | (m,n) |
| Random | `np.random.randn(m,n)` | (m,n) |

---

## Common Pitfalls ⚠️

**1. Shape Mismatch:**
```python
# ❌ Wrong
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[1, 2]])     # (1, 2)
C = A @ B  # Error! 3 ≠ 1

# ✅ Correct
A = np.array([[1, 2, 3]])  # (1, 3)
B = np.array([[1], [2], [3]])  # (3, 1)
C = A @ B  # (1, 1) ✅
```

**2. Row vs Column Vectors:**
```python
# Row vector
row = np.array([[1, 2, 3]])  # (1, 3)

# Column vector
col = np.array([[1], [2], [3]])  # (3, 1)

# Often need to reshape
v = np.array([1, 2, 3])  # (3,)
v_col = v.reshape(-1, 1)  # (3, 1)
```

**3. Broadcasting:**
```python
# NumPy auto-expands dimensions
A = np.array([[1, 2, 3],
              [4, 5, 6]])  # (2, 3)
b = np.array([10, 20, 30])  # (3,)

result = A + b  # Works! b broadcast to (2, 3)
# [[11, 22, 33],
#  [14, 25, 36]]
```

---

## Practice Exercises 🏋️

1. Calculate dot product of [1, 2, 3] and [4, 5, 6]
2. Multiply matrix [[1,2],[3,4]] with [[5,6],[7,8]]
3. What is shape of (100, 784) @ (784, 10)?
4. Implement standardization from scratch

<details>
<summary>Solutions</summary>

```python
# 1. Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.dot(a, b)  # 32

# 2. Matrix multiply
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
C = A @ B  # [[19, 22], [43, 50]]

# 3. Shape
# (100, 784) @ (784, 10) = (100, 10)

# 4. Standardization
X = np.array([[1, 2], [3, 4], [5, 6]])
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_std = (X - mean) / std
```
</details>

---

## Key Takeaways 💡

1. **Vectors**: 1D arrays, represent features or samples
2. **Matrices**: 2D arrays, represent datasets
3. **Matrix multiplication**: Core ML operation (X @ W)
4. **Shape matters**: Always check (m,n) @ (n,p) = (m,p)
5. **NumPy is your friend**: Use @ for matmul, * for element-wise

---

## What's Next?

You now know enough linear algebra for ML! Next, learn **probability and statistics** - equally important!

**Next:** [Lesson 2 - Probability & Statistics Basics →](Lesson%202%20-%20Probability%20and%20Statistics%20Basics.md)

---

**Congratulations!** You can now understand the math behind ML algorithms! 🎉

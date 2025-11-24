# Lesson 4: NumPy for ML 🔢

**Module 0: Math Essentials | Lesson 4 of 4**

Master NumPy - the foundation of all ML in Python!

---

## Why NumPy?

**NumPy makes Python fast for numerical computing:**
- 10-100x faster than pure Python
- Foundation for Pandas, scikit-learn, PyTorch, TensorFlow
- Efficient operations on arrays

---

## 1. Arrays - The Core Data Structure

```python
import numpy as np

# Create arrays
a = np.array([1, 2, 3, 4, 5])
print(a.shape)  # (5,)
print(a.dtype)  # int64

# 2D array (matrix)
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.shape)  # (2, 3)
```

### Array Creation Functions

```python
# Zeros
np.zeros((3, 4))  # 3×4 array of zeros

# Ones
np.ones((2, 3))

# Identity
np.eye(4)  # 4×4 identity matrix

# Range
np.arange(0, 10, 2)  # [0, 2, 4, 6, 8]

# Linspace
np.linspace(0, 1, 5)  # [0, 0.25, 0.5, 0.75, 1.0]

# Random
np.random.randn(3, 4)  # Normal distribution
np.random.rand(3, 4)   # Uniform [0, 1]
np.random.randint(0, 10, (3, 4))  # Random integers
```

---

## 2. Array Operations

### Element-wise Operations

```python
a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

# Arithmetic
print(a + b)   # [11, 22, 33, 44]
print(a * b)   # [10, 40, 90, 160]
print(a ** 2)  # [1, 4, 9, 16]
print(a > 2)   # [False, False, True, True]
```

### Matrix Operations

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Element-wise multiply
A * B  # [[5, 12], [21, 32]]

# Matrix multiply
A @ B  # [[19, 22], [43, 50]]

# Transpose
A.T
```

---

## 3. Indexing & Slicing

```python
a = np.array([10, 20, 30, 40, 50])

# Indexing
a[0]      # 10
a[-1]     # 50
a[1:4]    # [20, 30, 40]

# 2D indexing
m = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

m[0, 0]   # 1
m[1, :]   # [4, 5, 6] (row 1)
m[:, 1]   # [2, 5, 8] (column 1)
m[:2, :2] # [[1, 2], [4, 5]]

# Boolean indexing
a[a > 25]  # [30, 40, 50]
```

---

## 4. Broadcasting

**NumPy automatically expands smaller arrays!**

```python
# Add scalar
a = np.array([1, 2, 3])
a + 10  # [11, 12, 13]

# Add to matrix
m = np.array([[1, 2, 3],
              [4, 5, 6]])
m + np.array([10, 20, 30])
# [[11, 22, 33],
#  [14, 25, 36]]
```

---

## 5. Aggregations

```python
a = np.array([[1, 2, 3],
              [4, 5, 6]])

# Overall
np.sum(a)      # 21
np.mean(a)     # 3.5
np.std(a)      # 1.707
np.min(a)      # 1
np.max(a)      # 6

# Along axis
np.sum(a, axis=0)   # [5, 7, 9] (column sums)
np.sum(a, axis=1)   # [6, 15] (row sums)
np.mean(a, axis=0)  # [2.5, 3.5, 4.5]
```

---

## 6. Reshaping

```python
a = np.arange(12)  # [0, 1, 2, ..., 11]

# Reshape
a.reshape(3, 4)
# [[0, 1, 2, 3],
#  [4, 5, 6, 7],
#  [8, 9, 10, 11]]

# Flatten
m = np.array([[1, 2], [3, 4]])
m.flatten()  # [1, 2, 3, 4]

# Add dimension
a = np.array([1, 2, 3])
a[:, np.newaxis]  # [[1], [2], [3]]
```

---

## 7. ML-Specific Operations

### Standardization

```python
X = np.array([[1, 2], [3, 4], [5, 6]])

mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_standardized = (X - mean) / std
```

### Distance Calculation

```python
# Euclidean distance
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
```

### Softmax

```python
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / np.sum(exp_x)

logits = np.array([2.0, 1.0, 0.1])
probs = softmax(logits)  # [0.659, 0.242, 0.099]
```

---

## Quick Reference 📖

| Task | NumPy Code |
|------|-----------|
| Create array | `np.array([1,2,3])` |
| Zeros | `np.zeros((m,n))` |
| Random normal | `np.random.randn(m,n)` |
| Matrix multiply | `A @ B` |
| Element-wise | `A * B` |
| Mean | `np.mean(a)` |
| Reshape | `a.reshape(m,n)` |
| Transpose | `A.T` |

---

## Practice Exercises 🏋️

1. Create 5×5 matrix of random numbers, normalize to [0, 1]
2. Compute mean of each column
3. Select all elements > 0.5
4. Implement sigmoid function

<details>
<summary>Solutions</summary>

```python
# 1. Normalize
m = np.random.randn(5, 5)
m_norm = (m - m.min()) / (m.max() - m.min())

# 2. Column means
col_means = np.mean(m, axis=0)

# 3. Boolean indexing
m[m > 0.5]

# 4. Sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
</details>

---

## Key Takeaways 💡

1. NumPy is **fast** - use it instead of Python lists
2. **Broadcasting** saves code and memory
3. **Vectorization** >> loops
4. **axis=0** = columns, **axis=1** = rows
5. Learn NumPy = easier PyTorch/TensorFlow

---

**Module 0 Complete!** 🎉

**Next:** [Module 1 - Foundations →](../Module%201%20-%20Foundations/Lesson%201%20-%20Introduction%20to%20Machine%20Learning.md)

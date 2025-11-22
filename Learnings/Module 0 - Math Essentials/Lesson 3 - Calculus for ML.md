# Lesson 3: Calculus for ML 📐

**Module 0: Math Essentials | Lesson 3 of 4**

Just enough calculus to understand how neural networks learn!

---

## Why Calculus for ML?

**One core idea: How to minimize errors (loss) by adjusting parameters.**

Calculus tells us **which direction to move** and **how much** to change weights.

---

## 1. Derivatives - The Foundation

**Derivative**: Rate of change. How much does output change when input changes slightly?

```
f'(x) = lim[h→0] [f(x+h) - f(x)] / h
```

**Intuition:** Slope of the tangent line.

### Examples

```python
# f(x) = x²
# f'(x) = 2x

x = 3
slope = 2 * x  # 6
# "At x=3, function increases 6 units for each 1 unit increase in x"
```

### Common Derivatives

| Function | Derivative |
|----------|-----------|
| f(x) = c | f'(x) = 0 |
| f(x) = x | f'(x) = 1 |
| f(x) = x² | f'(x) = 2x |
| f(x) = xⁿ | f'(x) = nxⁿ⁻¹ |
| f(x) = eˣ | f'(x) = eˣ |
| f(x) = ln(x) | f'(x) = 1/x |
| f(x) = sin(x) | f'(x) = cos(x) |

### Derivative Rules

**Sum Rule:**
```
(f + g)' = f' + g'
```

**Product Rule:**
```
(f × g)' = f'g + fg'
```

**Chain Rule (Most Important for ML!):**
```
(f(g(x)))' = f'(g(x)) × g'(x)
```

Example:
```python
# f(x) = (3x + 2)²
# Let u = 3x + 2, then f = u²
# f'(x) = 2u × 3 = 2(3x + 2) × 3 = 6(3x + 2)
```

---

## 2. Partial Derivatives

**For functions with multiple variables.**

```
f(x, y) = x² + 2xy + y²

∂f/∂x = 2x + 2y  (treat y as constant)
∂f/∂y = 2x + 2y  (treat x as constant)
```

### Gradient

**Vector of all partial derivatives.**

```
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
```

**Points in direction of steepest ascent!**

```python
import numpy as np

def gradient(x, y):
    """Gradient of f(x,y) = x² + 2xy + y²"""
    df_dx = 2*x + 2*y
    df_dy = 2*x + 2*y
    return np.array([df_dx, df_dy])

# At point (1, 2)
grad = gradient(1, 2)  # [6, 6]
# Move opposite direction to minimize!
```

---

## 3. Gradient Descent - ML's Core Algorithm

**Goal:** Minimize loss function by following negative gradient.

```
θ_new = θ_old - α × ∇L(θ)

where:
θ = parameters (weights)
α = learning rate
∇L = gradient of loss
```

### Visual Example

```python
import numpy as np
import matplotlib.pyplot as plt

# Loss function: f(x) = (x - 3)²
def loss(x):
    return (x - 3) ** 2

def gradient_loss(x):
    return 2 * (x - 3)

# Gradient descent
x = 0  # Start point
alpha = 0.1  # Learning rate
steps = []

for i in range(20):
    steps.append(x)
    grad = gradient_loss(x)
    x = x - alpha * grad  # Update

print(f"Converged to: {x:.4f}")  # Should be ~3.0
```

---

## 4. Backpropagation = Chain Rule

**How neural networks learn: Chain rule applied to compute gradients.**

```python
# Simple network: x → w₁ → z₁ → σ(z₁) → a₁ → w₂ → z₂ → loss

# Forward pass
z1 = w1 * x
a1 = sigmoid(z1)
z2 = w2 * a1
loss = (z2 - y) ** 2

# Backward pass (chain rule)
dL/dz2 = 2 * (z2 - y)
dL/dw2 = dL/dz2 * a1
dL/da1 = dL/dz2 * w2
dL/dz1 = dL/da1 * sigmoid'(z1)
dL/dw1 = dL/dz1 * x

# Update weights
w1 = w1 - alpha * dL/dw1
w2 = w2 - alpha * dL/dw2
```

---

## 5. Common Activation Function Derivatives

```python
# Sigmoid: σ(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# ReLU: f(x) = max(0, x)
def relu_derivative(x):
    return (x > 0).astype(float)

# Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x)
def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2
```

---

## 6. Practical ML Example

```python
# Linear regression loss
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Gradient w.r.t weights
def mse_gradient(X, y_true, y_pred):
    # dL/dw = 2/n * X^T * (y_pred - y_true)
    n = len(y_true)
    return (2/n) * X.T @ (y_pred - y_true)

# Training step
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([3, 5, 7])
w = np.random.randn(2)

learning_rate = 0.01
for epoch in range(100):
    y_pred = X @ w
    loss = mse_loss(y, y_pred)
    
    # Compute gradient
    grad = mse_gradient(X, y, y_pred)
    
    # Update weights
    w = w - learning_rate * grad
```

---

## Quick Reference 📖

| Concept | Formula | Use in ML |
|---------|---------|-----------|
| Derivative | f'(x) | Rate of change |
| Gradient | ∇f | Direction to minimize loss |
| Chain Rule | (f∘g)' = f'(g) × g' | Backpropagation |
| GD Update | θ = θ - α∇L | Weight updates |

---

## Key Takeaways 💡

1. **Derivative** = rate of change = slope
2. **Gradient** = vector of partial derivatives
3. **Gradient Descent** = follow negative gradient to minimize
4. **Chain Rule** = backpropagation in neural networks
5. You don't need to compute derivatives manually - frameworks do it!

---

**Next:** [Lesson 4 - NumPy for ML →](Lesson%204%20-%20NumPy%20for%20ML.md)

---

**Congratulations!** You understand the calculus behind ML! 🎉

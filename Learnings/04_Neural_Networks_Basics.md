# Neural Networks Basics 🧠

The foundation of modern deep learning. Understanding these concepts is crucial!

## What is a Neural Network?

A neural network is a computational model inspired by biological neurons that learns to map inputs to outputs through interconnected layers of artificial neurons.

```
Input Layer → Hidden Layer(s) → Output Layer
```

---

## The Building Blocks

### 1. The Perceptron (Single Neuron) 🔘

The simplest neural network unit.

**Components:**
- **Inputs:** x₁, x₂, ..., xₙ
- **Weights:** w₁, w₂, ..., wₙ
- **Bias:** b
- **Activation Function:** f

**Computation:**
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
output = f(z)
```

**Simple Implementation:**
```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, x):
        # Weighted sum
        z = np.dot(self.weights, x) + self.bias
        # Step activation (0 or 1)
        return 1 if z > 0 else 0

# Example
perceptron = Perceptron(n_inputs=2)
output = perceptron.forward([0.5, 0.3])
print(output)  # 0 or 1
```

### 2. Activation Functions 📊

Transform the weighted sum into output. **Critical for learning non-linear patterns!**

#### Sigmoid (Logistic)
```python
σ(z) = 1 / (1 + e⁻ᶻ)

# Output: (0, 1)
# Use: Binary classification output layer
```

**Implementation:**
```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative (needed for backprop)
def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)
```

**Pros:** Smooth, bounded output
**Cons:** Vanishing gradients, not zero-centered

#### Tanh (Hyperbolic Tangent)
```python
tanh(z) = (e^z - e^-z) / (e^z + e^-z)

# Output: (-1, 1)
# Use: Hidden layers (better than sigmoid)
```

**Pros:** Zero-centered, stronger gradients
**Cons:** Still vanishing gradients

#### ReLU (Rectified Linear Unit) ⭐ Most Popular!
```python
ReLU(z) = max(0, z)

# Output: [0, ∞)
# Use: Hidden layers in deep networks
```

**Implementation:**
```python
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)
```

**Pros:**
- No vanishing gradient
- Computationally efficient
- Sparse activation

**Cons:**
- "Dying ReLU" problem (neurons can die)

#### Leaky ReLU
```python
Leaky ReLU(z) = max(αz, z)  where α = 0.01

# Fixes dying ReLU problem
```

#### Softmax (Output layer for multiclass)
```python
softmax(zᵢ) = e^zᵢ / Σ(e^zⱼ)

# Output: Probabilities summing to 1
# Use: Multiclass classification output
```

**Implementation:**
```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / np.sum(exp_z)

# Example
z = [2.0, 1.0, 0.1]
probs = softmax(z)
print(probs)  # [0.659, 0.242, 0.099]
print(sum(probs))  # 1.0
```

**Activation Function Cheat Sheet:**
| Function | Range | Use Case |
|----------|-------|----------|
| Sigmoid | (0, 1) | Binary classification output |
| Tanh | (-1, 1) | Hidden layers (rarely used now) |
| ReLU | [0, ∞) | Hidden layers (default choice) |
| Leaky ReLU | (-∞, ∞) | When ReLU causes dying neurons |
| Softmax | (0, 1) | Multiclass output (probabilities) |

---

## Multi-Layer Perceptron (MLP) 🏗️

Also called **Feedforward Neural Network** or **Fully Connected Network**.

### Architecture
```
Input Layer (features)
    ↓
Hidden Layer 1 (neurons with weights)
    ↓
Hidden Layer 2 (optional, more layers)
    ↓
Output Layer (predictions)
```

### Example: 3-Layer Network
```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Layer 1
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        # Layer 2 (output)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)  # or softmax for multiclass

        return self.a2

# Create network: 4 inputs → 8 hidden → 1 output
nn = NeuralNetwork(input_size=4, hidden_size=8, output_size=1)

# Forward pass
X = np.array([[1, 2, 3, 4]])  # 1 sample, 4 features
output = nn.forward(X)
print(output)  # Prediction
```

---

## Forward Propagation ➡️

Process of computing output from input by passing through layers.

**Steps:**
1. Multiply inputs by weights
2. Add bias
3. Apply activation function
4. Repeat for each layer

**Mathematical Notation:**
```
Layer 1:
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
a⁽¹⁾ = σ(z⁽¹⁾)

Layer 2:
z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
a⁽²⁾ = σ(z⁽²⁾)

...and so on
```

**Visual Example:**
```
Input: [x₁=1, x₂=2]

Hidden Layer:
neuron₁: z₁ = (w₁₁×1 + w₁₂×2) + b₁ = 5
         a₁ = ReLU(5) = 5

neuron₂: z₂ = (w₂₁×1 + w₂₂×2) + b₂ = -2
         a₂ = ReLU(-2) = 0

Output Layer:
output = (w₃×5 + w₄×0) + b₃ = 8
       = sigmoid(8) = 0.9997
```

---

## Loss Functions 📉

Measure how wrong the network's predictions are.

### 1. Mean Squared Error (MSE) - Regression
```python
MSE = (1/n) Σ(y_true - y_pred)²
```

```python
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
```

### 2. Binary Cross-Entropy - Binary Classification
```python
BCE = -(1/n) Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

```python
def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))
```

### 3. Categorical Cross-Entropy - Multiclass
```python
CCE = -(1/n) Σ Σ y_ij · log(ŷ_ij)
```

```python
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / len(y_true)
```

---

## Backpropagation 🔙

The **most important algorithm** in neural networks! It computes gradients efficiently.

### Goal
Find how to adjust weights to minimize loss.

### Chain Rule
```
∂Loss/∂W = ∂Loss/∂output × ∂output/∂z × ∂z/∂W
```

### Backpropagation Steps

**1. Forward Pass:** Compute outputs and cache intermediate values

**2. Compute Output Gradient:**
```python
dL/da = a - y  # For sigmoid + binary cross-entropy
```

**3. Backpropagate Through Layers:**
```python
# Output layer
dL/dz2 = dL/da2 * da2/dz2
dL/dW2 = a1.T @ dL/dz2
dL/db2 = sum(dL/dz2)

# Hidden layer
dL/da1 = dL/dz2 @ W2.T
dL/dz1 = dL/da1 * da1/dz1  # Element-wise
dL/dW1 = X.T @ dL/dz1
dL/db1 = sum(dL/dz1)
```

**4. Update Weights:**
```python
W = W - learning_rate * dL/dW
b = b - learning_rate * dL/db
```

### Complete Implementation
```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.lr = learning_rate

    def forward(self, X):
        # Cache values for backprop
        self.X = X

        # Hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)

        # Output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2

    def backward(self, y):
        m = y.shape[0]  # Number of samples

        # Output layer gradients
        dz2 = self.a2 - y  # For sigmoid + BCE
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Hidden layer gradients
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * relu_derivative(self.z1)
        dW1 = np.dot(self.X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def train(self, X, y, epochs=1000):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = binary_crossentropy(y, output)
            losses.append(loss)

            # Backward pass
            self.backward(y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

# Example usage
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])  # XOR problem

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
losses = nn.train(X_train, y_train, epochs=5000)

# Test
predictions = nn.forward(X_train)
print("\nPredictions:")
print(predictions)
```

---

## Gradient Descent Variants 📉

### 1. Batch Gradient Descent
```python
# Use ALL training data for each update
for epoch in range(epochs):
    gradients = compute_gradients(X_train, y_train)
    weights = weights - learning_rate * gradients
```
✅ Smooth convergence
❌ Slow for large datasets

### 2. Stochastic Gradient Descent (SGD)
```python
# Use ONE sample at a time
for epoch in range(epochs):
    for x, y in zip(X_train, y_train):
        gradients = compute_gradients(x, y)
        weights = weights - learning_rate * gradients
```
✅ Fast, can escape local minima
❌ Noisy, erratic convergence

### 3. Mini-Batch Gradient Descent ⭐ Most Common
```python
# Use small batches (e.g., 32, 64, 128 samples)
batch_size = 32
for epoch in range(epochs):
    for batch_X, batch_y in get_batches(X_train, y_train, batch_size):
        gradients = compute_gradients(batch_X, batch_y)
        weights = weights - learning_rate * gradients
```
✅ Best of both worlds
✅ Efficient on GPUs

---

## Advanced Optimizers 🚀

### 1. Momentum
Accelerates in relevant direction, dampens oscillations.

```python
velocity = 0
for each iteration:
    gradient = compute_gradient()
    velocity = β * velocity + (1 - β) * gradient
    weights = weights - learning_rate * velocity
```
β = 0.9 typically

### 2. RMSprop
Adapts learning rate per parameter.

```python
cache = 0
for each iteration:
    gradient = compute_gradient()
    cache = β * cache + (1 - β) * gradient²
    weights = weights - learning_rate * gradient / (√cache + ε)
```

### 3. Adam (Adaptive Moment Estimation) ⭐ Most Popular!
Combines momentum + RMSprop.

```python
m = 0  # First moment
v = 0  # Second moment
t = 0  # Time step

for each iteration:
    t += 1
    gradient = compute_gradient()

    # Update moments
    m = β1 * m + (1 - β1) * gradient
    v = β2 * v + (1 - β2) * gradient²

    # Bias correction
    m_hat = m / (1 - β1^t)
    v_hat = v / (1 - β2^t)

    # Update weights
    weights = weights - learning_rate * m_hat / (√v_hat + ε)
```

**Default hyperparameters:**
- β1 = 0.9
- β2 = 0.999
- ε = 1e-8
- learning_rate = 0.001

---

## Weight Initialization 🎲

**Critical for training!** Bad initialization → slow/failed training.

### 1. Zero Initialization ❌ Bad!
```python
W = np.zeros((n_in, n_out))
```
Problem: All neurons learn the same features

### 2. Random Initialization
```python
W = np.random.randn(n_in, n_out) * 0.01
```
Small random values, but can be too small for deep networks

### 3. Xavier/Glorot Initialization ⭐
For sigmoid/tanh activations.

```python
W = np.random.randn(n_in, n_out) * np.sqrt(1 / n_in)
# or
W = np.random.randn(n_in, n_out) * np.sqrt(2 / (n_in + n_out))
```

### 4. He Initialization ⭐
For ReLU activations.

```python
W = np.random.randn(n_in, n_out) * np.sqrt(2 / n_in)
```

**Rule of Thumb:**
- ReLU → He initialization
- Sigmoid/Tanh → Xavier initialization

---

## Common Problems & Solutions 🔧

### 1. Vanishing Gradients 📉
**Problem:** Gradients become very small in deep networks
**Symptoms:** Early layers learn slowly or not at all
**Solutions:**
- Use ReLU instead of sigmoid/tanh
- Batch normalization
- Residual connections (ResNet)
- Better initialization

### 2. Exploding Gradients 📈
**Problem:** Gradients become very large
**Symptoms:** NaN values, unstable training
**Solutions:**
- Gradient clipping
- Lower learning rate
- Better initialization
- Batch normalization

### 3. Overfitting 🎯
**Problem:** Model memorizes training data
**Symptoms:** High train accuracy, low test accuracy
**Solutions:**
- More training data
- Regularization (L1, L2, dropout)
- Simpler model
- Early stopping

### 4. Slow Training 🐌
**Solutions:**
- Better optimizer (Adam)
- Batch normalization
- Better learning rate
- GPU acceleration
- Smaller model

---

## PyTorch Implementation 🔥

Modern neural network with PyTorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        return out

# Create model
model = NeuralNet(input_size=2, hidden_size=4, output_size=1)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training data
X_train = torch.FloatTensor([[0,0], [0,1], [1,0], [1,1]])
y_train = torch.FloatTensor([[0], [1], [1], [0]])

# Training loop
for epoch in range(5000):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/5000], Loss: {loss.item():.4f}')

# Test
with torch.no_grad():
    predictions = model(X_train)
    print('\nPredictions:')
    print(predictions)
```

---

## Key Takeaways 💡

1. **Neurons** compute weighted sums + activation functions
2. **Activation functions** enable non-linearity (ReLU is default)
3. **Forward prop** computes predictions layer by layer
4. **Backprop** computes gradients using chain rule
5. **Optimizers** update weights (Adam is default)
6. **Initialization** matters (He for ReLU, Xavier for sigmoid)
7. **Mini-batch** gradient descent is standard
8. **PyTorch/TensorFlow** handle backprop automatically!

## Practice Exercises 🏋️

1. Implement XOR function from scratch using NumPy
2. Visualize how weights change during training
3. Experiment with different activation functions
4. Compare optimizers (SGD vs Adam)
5. Implement batch normalization

## Next Steps 📚

Ready for deeper topics?
- **[05_Deep_Learning_Fundamentals.md](05_Deep_Learning_Fundamentals.md)** - CNNs, RNNs, advanced techniques
- **[07_Modern_Architectures.md](07_Modern_Architectures.md)** - Transformers, ResNets, modern models

---

**You now understand how neural networks learn!** 🎉 This is the foundation for all of deep learning.

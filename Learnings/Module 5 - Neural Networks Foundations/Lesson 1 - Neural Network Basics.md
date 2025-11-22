# Lesson 1: Neural Network Basics 🧠

**Module 5: Neural Networks Foundations | Lesson 1 of 4**

Master the fundamentals of neural networks - the foundation of modern deep learning!

---

## Why Neural Networks?

**Universal Function Approximators:**
- Can learn any complex function
- Automatically extract features
- Excel at unstructured data (images, text, audio)
- Power modern AI (ChatGPT, DALL-E, AlphaGo)

---

## 1. The Perceptron (Single Neuron) ⚡

### Mathematical Model

```
Output = Activation(Σ(weights × inputs) + bias)

y = f(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)
```

### Python Implementation

```python
import numpy as np

class Perceptron:
    def __init__(self, n_inputs):
        # Random initialization
        self.weights = np.random.randn(n_inputs)
        self.bias = np.random.randn()

    def forward(self, x):
        """Forward pass"""
        # Weighted sum
        z = np.dot(self.weights, x) + self.bias

        # Activation (step function)
        return 1 if z > 0 else 0

    def train(self, X, y, epochs=100, lr=0.1):
        """Train perceptron"""
        for epoch in range(epochs):
            errors = 0
            for xi, target in zip(X, y):
                # Predict
                prediction = self.forward(xi)

                # Update if error
                error = target - prediction
                if error != 0:
                    self.weights += lr * error * xi
                    self.bias += lr * error
                    errors += 1

            if errors == 0:
                print(f"Converged at epoch {epoch}")
                break

# Example: OR gate
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 1, 1, 1])

perceptron = Perceptron(n_inputs=2)
perceptron.train(X, y)

# Test
for xi in X:
    print(f"{xi} → {perceptron.forward(xi)}")
```

---

## 2. Activation Functions 📊

### Why Activation Functions?

**Without activation:** Neural network = linear regression (can't learn complex patterns)
**With activation:** Can learn non-linear patterns!

### Common Activations

```python
import matplotlib.pyplot as plt

# Sigmoid: σ(x) = 1 / (1 + e^(-x))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Tanh: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
def tanh(x):
    return np.tanh(x)

# ReLU: f(x) = max(0, x)
def relu(x):
    return np.maximum(0, x)

# Leaky ReLU
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Visualize
x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x))
plt.title('Sigmoid')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(x, tanh(x))
plt.title('Tanh')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(x, relu(x))
plt.title('ReLU')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(x, leaky_relu(x))
plt.title('Leaky ReLU')
plt.grid(True)

plt.tight_layout()
plt.show()
```

### When to Use Each

| Activation | Use Case | Pros | Cons |
|------------|----------|------|------|
| **Sigmoid** | Binary classification (output layer) | Output [0,1] | Vanishing gradient |
| **Tanh** | Hidden layers (RNN) | Output [-1,1] | Vanishing gradient |
| **ReLU** | **Hidden layers (default)** | **Fast, no vanishing gradient** | Dead neurons |
| **Leaky ReLU** | Hidden layers (fix dead ReLU) | No dead neurons | Hyperparameter |
| **Softmax** | Multi-class classification (output) | Probability distribution | N/A |

---

## 3. Multi-Layer Neural Network 🏗️

### Architecture

```
Input Layer → Hidden Layer(s) → Output Layer

Example:
[x₁, x₂] → [h₁, h₂, h₃] → [y]
   2           3              1
```

### NumPy Implementation

```python
class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        layer_sizes: [input_size, hidden_size, ..., output_size]
        """
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.1
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]  # Store for backprop

        # Pass through each layer
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(self.activations[-1], w) + b
            a = relu(z)  # Hidden layer activation
            self.activations.append(a)

        # Output layer (linear for regression, sigmoid for binary classification)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        output = sigmoid(z)  # For binary classification
        self.activations.append(output)

        return output

    def predict(self, X):
        """Make predictions"""
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Example: XOR problem (non-linearly separable)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Create network: 2 inputs → 4 hidden → 1 output
nn = NeuralNetwork([2, 4, 1])

# Forward pass
output = nn.forward(X)
print(f"Initial predictions:\n{output}")
```

---

## 4. Loss Functions 📉

### Binary Classification

```python
# Binary Cross-Entropy
def binary_cross_entropy(y_true, y_pred):
    """
    BCE = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
    """
    epsilon = 1e-15  # Avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.mean(y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred))
    return loss

# Example
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])

loss = binary_cross_entropy(y_true, y_pred)
print(f"BCE Loss: {loss:.4f}")
```

### Multi-Class Classification

```python
# Categorical Cross-Entropy
def categorical_cross_entropy(y_true, y_pred):
    """
    CCE = -Σ y*log(ŷ)
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    return loss

# Softmax activation
def softmax(x):
    """Convert logits to probabilities"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

# Example
logits = np.array([[2.0, 1.0, 0.1]])
y_true = np.array([[1, 0, 0]])  # One-hot encoded

y_pred = softmax(logits)
loss = categorical_cross_entropy(y_true, y_pred)

print(f"Predicted probabilities: {y_pred}")
print(f"CCE Loss: {loss:.4f}")
```

### Regression

```python
# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Mean Absolute Error
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
```

---

## 5. Complete Example: MNIST Digits 🔢

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
digits = load_digits()
X, y = digits.data, digits.target

# Binary classification: 0 vs not-0
y_binary = (y == 0).astype(int)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42
)

# Normalize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create network: 64 inputs → 32 hidden → 1 output
nn = NeuralNetwork([64, 32, 1])

# Forward pass
y_pred = nn.forward(X_test)
predictions = (y_pred > 0.5).astype(int)

# Accuracy (without training - random!)
accuracy = np.mean(predictions.flatten() == y_test)
print(f"Accuracy (untrained): {accuracy:.3f}")
```

---

## 6. Network Architecture Design 🎯

### Rule of Thumb

```python
# Input layer: Number of features
input_size = X.shape[1]

# Hidden layers:
# - Start with 1-2 hidden layers
# - Size: between input and output size
# - Common: same size as input, or powers of 2 (64, 128, 256)

# Output layer:
# - Binary classification: 1 neuron (sigmoid)
# - Multi-class: num_classes neurons (softmax)
# - Regression: 1 neuron (linear)

# Example architectures
architectures = {
    'Small': [input_size, 64, output_size],
    'Medium': [input_size, 128, 64, output_size],
    'Large': [input_size, 256, 128, 64, output_size]
}
```

### Capacity vs Complexity

```python
def count_parameters(layer_sizes):
    """Count trainable parameters"""
    total = 0
    for i in range(len(layer_sizes) - 1):
        # Weights + biases
        params = layer_sizes[i] * layer_sizes[i+1] + layer_sizes[i+1]
        total += params
        print(f"Layer {i+1}: {params:,} parameters")

    print(f"Total: {total:,} parameters")
    return total

# Example
count_parameters([784, 128, 64, 10])  # MNIST classifier
```

---

## Quick Reference 📖

**Basic Neural Network Structure:**

```python
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # Hidden layer
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        # Output layer
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)

        return self.a2
```

**Activation Function Cheat Sheet:**

```python
# Hidden layers: ReLU (default)
hidden = relu(z)

# Output layer (classification): Sigmoid or Softmax
output_binary = sigmoid(z)
output_multiclass = softmax(z)

# Output layer (regression): Linear (no activation)
output_regression = z
```

---

## Practice Exercises 🏋️

1. Implement AND, OR, XOR gates with perceptron/network
2. Create 3-layer network from scratch for binary classification
3. Visualize decision boundaries of 2D classification
4. Compare different activation functions on same problem

<details>
<summary>Solutions</summary>

```python
# 1. XOR with network (can't be solved with single perceptron!)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

nn_xor = NeuralNetwork([2, 4, 1])
# Would need training (next lesson!)

# 2. 3-layer network
class ThreeLayerNN:
    def __init__(self):
        self.W1 = np.random.randn(2, 4) * 0.1
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 3) * 0.1
        self.b2 = np.zeros((1, 3))
        self.W3 = np.random.randn(3, 1) * 0.1
        self.b3 = np.zeros((1, 1))

    def forward(self, X):
        h1 = relu(X @ self.W1 + self.b1)
        h2 = relu(h1 @ self.W2 + self.b2)
        out = sigmoid(h2 @ self.W3 + self.b3)
        return out

# 3. Decision boundary visualization
def plot_decision_boundary(model, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdYlBu')
    plt.show()

# 4. Compare activations
activations = {
    'sigmoid': sigmoid,
    'tanh': tanh,
    'relu': relu,
    'leaky_relu': leaky_relu
}

for name, act in activations.items():
    # Create network with different activation
    # Compare performance
    pass
```
</details>

---

## Key Takeaways 💡

1. **Perceptron** = single neuron, linear classifier
2. **Activation functions** enable non-linearity
3. **ReLU** is default for hidden layers
4. **Multi-layer networks** can learn complex patterns
5. **Binary classification** → Sigmoid output + BCE loss
6. **Multi-class** → Softmax output + CCE loss
7. **Architecture design** matters (but start simple!)

---

**Next:** [Lesson 2 - Backpropagation & Training →](Lesson%202%20-%20Backpropagation%20and%20Training.md)

---

**You now understand neural network fundamentals!** 🎉

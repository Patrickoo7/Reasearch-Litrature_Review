# Lesson 3: Regularization & Best Practices 🛡️

**Module 5: Neural Networks Foundations | Lesson 3 of 4**

Master techniques to prevent overfitting and train better neural networks!

---

## Why Regularization?

**Problem:** Neural networks easily overfit
- Memorize training data
- Poor generalization to new data

**Solution:** Regularization techniques

---

## 1. Dropout 🎲

**Idea:** Randomly drop neurons during training

```python
def dropout(x, drop_rate=0.5, training=True):
    """
    Dropout regularization

    During training: randomly set neurons to 0
    During inference: use all neurons (scaled)
    """
    if not training:
        return x

    # Create mask
    mask = np.random.binomial(1, 1-drop_rate, size=x.shape)

    # Apply mask and scale
    return x * mask / (1 - drop_rate)

# Example
x = np.array([[1, 2, 3, 4, 5]])

print("Original:", x)
print("Dropout 0.5:", dropout(x, 0.5, training=True))
print("Dropout 0.5:", dropout(x, 0.5, training=True))  # Different!
print("Inference:", dropout(x, 0.5, training=False))   # No dropout
```

**In Neural Network:**

```python
class NNWithDropout:
    def forward(self, X, training=True):
        # Hidden layer
        z1 = X @ self.W1 + self.b1
        a1 = relu(z1)
        a1 = dropout(a1, drop_rate=0.5, training=training)  # Dropout!

        # Output layer
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)

        return a2
```

---

## 2. Batch Normalization 📊

**Idea:** Normalize inputs to each layer

```python
class BatchNorm:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.1):
        self.epsilon = epsilon
        self.momentum = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features)   # Scale
        self.beta = np.zeros(num_features)   # Shift

        # Running statistics (for inference)
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, x, training=True):
        if training:
            # Compute batch statistics
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)

            # Normalize
            x_norm = (x - batch_mean) / np.sqrt(batch_var + self.epsilon)

            # Update running statistics
            self.running_mean = (1 - self.momentum) * self.running_mean + \
                               self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + \
                              self.momentum * batch_var

        else:
            # Use running statistics
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        # Scale and shift
        out = self.gamma * x_norm + self.beta

        return out

# Usage
bn = BatchNorm(num_features=128)
x_batch = np.random.randn(32, 128)  # Batch of 32 samples

# Training
x_normalized = bn.forward(x_batch, training=True)

# Inference
x_test = np.random.randn(1, 128)
x_test_normalized = bn.forward(x_test, training=False)
```

---

## 3. Weight Initialization 🎯

**Problem:** Bad initialization → vanishing/exploding gradients

### Xavier/Glorot Initialization

```python
def xavier_init(n_in, n_out):
    """Xavier initialization (for sigmoid/tanh)"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

# Or Gaussian version
def xavier_normal(n_in, n_out):
    std = np.sqrt(2 / (n_in + n_out))
    return np.random.randn(n_in, n_out) * std
```

### He Initialization

```python
def he_init(n_in, n_out):
    """He initialization (for ReLU)"""
    std = np.sqrt(2 / n_in)
    return np.random.randn(n_in, n_out) * std
```

**When to use:**
- **Xavier:** Sigmoid, Tanh activations
- **He:** ReLU, Leaky ReLU activations

---

## 4. L2 Regularization (Weight Decay) 🏋️

```python
def l2_regularization(weights, lambda_reg):
    """L2 penalty"""
    l2_penalty = 0
    for w in weights:
        l2_penalty += np.sum(w ** 2)
    return 0.5 * lambda_reg * l2_penalty

# In training
loss = data_loss + l2_regularization(model.weights, lambda_reg=0.01)

# Update with L2
# gradient = data_gradient + lambda_reg * weight
self.W1 -= lr * (dW1 + lambda_reg * self.W1)
```

---

## 5. Early Stopping ⏹️

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop

# Usage
early_stopping = EarlyStopping(patience=10)

for epoch in range(1000):
    # Train...
    val_loss = evaluate(model, val_data)

    if early_stopping(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break
```

---

## 6. Data Augmentation 🔄

```python
def augment_data(X, y):
    """Simple augmentation for images"""
    augmented_X = []
    augmented_y = []

    for x, label in zip(X, y):
        # Original
        augmented_X.append(x)
        augmented_y.append(label)

        # Flipped
        augmented_X.append(np.fliplr(x))
        augmented_y.append(label)

        # Rotated (small angle)
        # angle = np.random.uniform(-15, 15)
        # augmented_X.append(rotate(x, angle))
        # augmented_y.append(label)

    return np.array(augmented_X), np.array(augmented_y)
```

---

## 7. Gradient Clipping 📏

```python
def clip_gradients(gradients, max_norm=5.0):
    """Clip gradients by norm"""
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for key in gradients:
            gradients[key] *= clip_coef

    return gradients

# Usage
gradients = model.compute_gradients(X, y)
gradients = clip_gradients(gradients, max_norm=5.0)
model.update_weights(gradients)
```

---

## Complete Best Practices Checklist ✅

```python
class BestPracticesNN:
    def __init__(self, input_size, hidden_size, output_size):
        # ✅ 1. He initialization for ReLU
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2/hidden_size)
        self.b2 = np.zeros((1, output_size))

        # ✅ 2. Batch normalization
        self.bn1 = BatchNorm(hidden_size)

        # ✅ 3. Dropout rate
        self.dropout_rate = 0.5

        # ✅ 4. L2 regularization
        self.lambda_reg = 0.01

    def forward(self, X, training=True):
        # Hidden layer
        z1 = X @ self.W1 + self.b1

        # ✅ Batch norm
        z1 = self.bn1.forward(z1, training)

        # ✅ ReLU activation
        a1 = np.maximum(0, z1)

        # ✅ Dropout
        if training:
            a1 = dropout(a1, self.dropout_rate, training)

        # Output layer
        z2 = a1 @ self.W2 + self.b2
        a2 = sigmoid(z2)

        return a2

    def loss(self, y_true, y_pred):
        # Data loss
        data_loss = binary_cross_entropy(y_true, y_pred)

        # ✅ L2 regularization
        l2_loss = 0.5 * self.lambda_reg * (np.sum(self.W1**2) + np.sum(self.W2**2))

        return data_loss + l2_loss

# ✅ Training with early stopping
early_stop = EarlyStopping(patience=20)

for epoch in range(1000):
    # Train
    loss = train_epoch(model, X_train, y_train)

    # Validate
    val_loss = evaluate(model, X_val, y_val)

    if early_stop(val_loss):
        break
```

---

## Quick Reference 📖

**Regularization Techniques:**

| Technique | When to Use | Typical Values |
|-----------|-------------|----------------|
| Dropout | Large networks | 0.2-0.5 |
| Batch Norm | Deep networks | Always use |
| L2 Regularization | Prevent overfitting | 0.001-0.01 |
| Early Stopping | Limited time | Patience: 10-20 |
| Data Augmentation | Small datasets | Always for images |
| Gradient Clipping | RNNs, unstable training | max_norm: 1-5 |

**Initialization:**
- **ReLU:** He initialization
- **Sigmoid/Tanh:** Xavier initialization

**Training Checklist:**
```
✅ He/Xavier initialization
✅ Batch normalization
✅ Dropout (0.5 for hidden layers)
✅ Adam optimizer (lr=0.001)
✅ Early stopping
✅ Learning rate scheduling
✅ Data augmentation (if applicable)
✅ Gradient clipping (if needed)
```

---

## Key Takeaways 💡

1. **Dropout** prevents co-adaptation of neurons
2. **Batch norm** accelerates training & regularizes
3. **Proper initialization** prevents gradient issues
4. **Early stopping** prevents overfitting
5. **L2 regularization** keeps weights small
6. **Combine multiple techniques** for best results
7. **Start with defaults**, tune if needed

---

**Next:** [Lesson 4 - PyTorch Fundamentals →](Lesson%204%20-%20PyTorch%20Fundamentals.md)

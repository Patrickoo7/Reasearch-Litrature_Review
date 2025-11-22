# Lesson 4: PyTorch Fundamentals 🔥

**Module 5: Neural Networks Foundations | Lesson 4 of 4**

Master PyTorch - the industry-standard deep learning framework!

---

## Why PyTorch?

**Most popular framework in research & industry:**
- Pythonic and intuitive
- Dynamic computation graphs
- Excellent debugging
- Strong ecosystem
- GPU support

---

## 1. Tensors - The Building Blocks 🧱

```python
import torch
import numpy as np

# Create tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([[1, 2], [3, 4]])
z = torch.randn(3, 4)  # Random tensor

print(f"x: {x}")
print(f"y:\n{y}")
print(f"z shape: {z.shape}")

# From NumPy
np_array = np.array([1, 2, 3])
torch_tensor = torch.from_numpy(np_array)

# To NumPy
back_to_numpy = torch_tensor.numpy()

# Common operations
a = torch.ones(2, 3)
b = torch.zeros(2, 3)
c = torch.eye(3)  # Identity matrix

# Tensor operations
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])

print(f"Add: {x + y}")
print(f"Multiply: {x * y}")
print(f"Dot product: {torch.dot(x, y)}")
print(f"Matrix multiplication: {torch.mm(a, c)}")

# Reshaping
x = torch.randn(4, 4)
y = x.view(16)  # Reshape to 1D
z = x.view(-1, 8)  # -1 infers dimension

print(f"Original: {x.shape}")
print(f"View 16: {y.shape}")
print(f"View -1,8: {z.shape}")
```

---

## 2. Autograd - Automatic Differentiation 🎓

```python
# requires_grad=True tracks operations
x = torch.tensor([2.0], requires_grad=True)
y = x ** 2 + 3 * x + 1

# Backward pass
y.backward()

# Gradient
print(f"dy/dx = {x.grad}")  # 2*x + 3 = 2*2 + 3 = 7

# Example: Linear regression gradient
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)

x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_true = torch.tensor([[2.0], [4.0], [6.0]])

# Forward
y_pred = x_data * w + b

# Loss
loss = ((y_pred - y_true) ** 2).mean()

# Backward
loss.backward()

print(f"Loss: {loss.item():.4f}")
print(f"dL/dw: {w.grad.item():.4f}")
print(f"dL/db: {b.grad.item():.4f}")

# Zero gradients for next iteration
w.grad.zero_()
b.grad.zero_()
```

---

## 3. Building Neural Networks 🏗️

```python
import torch.nn as nn
import torch.nn.functional as F

# Method 1: Using nn.Sequential
model = nn.Sequential(
    nn.Linear(784, 128),   # Input layer
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(128, 64),    # Hidden layer
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(64, 10)      # Output layer
)

# Method 2: Custom nn.Module (preferred)
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc3(x)

        return x

# Create model
model = NeuralNetwork(784, 128, 10)
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
```

---

## 4. Complete Training Loop 🔄

```python
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Hyperparameters
input_size = 784
hidden_size = 128
output_size = 10
learning_rate = 0.001
batch_size = 64
epochs = 10

# Create model
model = NeuralNetwork(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Dummy data
X_train = torch.randn(1000, 784)
y_train = torch.randint(0, 10, (1000,))

X_test = torch.randn(200, 784)
y_test = torch.randint(0, 10, (200,))

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
for epoch in range(epochs):
    model.train()  # Set to training mode
    train_loss = 0

    for batch_X, batch_y in train_loader:
        # Zero gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        loss.backward()

        # Update weights
        optimizer.step()

        train_loss += loss.item()

    # Validation
    model.eval()  # Set to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():  # No gradient computation
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    avg_loss = train_loss / len(train_loader)

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
```

---

## 5. GPU Support 🚀

```python
# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Move model to GPU
model = model.to(device)

# Move data to GPU
X_train = X_train.to(device)
y_train = y_train.to(device)

# Training loop with GPU
for batch_X, batch_y in train_loader:
    # Move batch to GPU
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)

    # Forward
    outputs = model(batch_X)
    loss = criterion(outputs, batch_y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Move back to CPU for numpy
predictions = model(X_test.to(device)).cpu().numpy()
```

---

## 6. Save & Load Models 💾

```python
# Save entire model
torch.save(model, 'model.pth')

# Load entire model
model = torch.load('model.pth')

# Save only state dict (preferred)
torch.save(model.state_dict(), 'model_state.pth')

# Load state dict
model = NeuralNetwork(784, 128, 10)
model.load_state_dict(torch.load('model_state.pth'))

# Save with optimizer state
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

---

## 7. Real Example: MNIST 🔢

```python
from torchvision import datasets, transforms

# Data transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Model
class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = MNISTNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')

# Test
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# Run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(1, 6):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
```

---

## Quick Reference 📖

**Essential Imports:**
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
```

**Model Template:**
```python
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**Training Template:**
```python
model = Model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(epochs):
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
```

**Common Layers:**
- `nn.Linear(in, out)` - Fully connected
- `nn.Conv2d(in, out, kernel)` - Convolution
- `nn.BatchNorm1d(size)` - Batch norm
- `nn.Dropout(p)` - Dropout
- `nn.ReLU()` - ReLU activation

**Common Loss Functions:**
- `nn.CrossEntropyLoss()` - Classification
- `nn.MSELoss()` - Regression
- `nn.BCELoss()` - Binary classification

---

## Key Takeaways 💡

1. **Tensors** are like NumPy arrays + GPU + autograd
2. **Autograd** automatically computes gradients
3. **nn.Module** for custom models
4. **DataLoader** for batching & shuffling
5. **model.train()** / **model.eval()** toggle modes
6. **optimizer.zero_grad()** before backward()
7. **.to(device)** for GPU support

---

**Module 5 Complete!** 🎉

**Next Module:** [Module 6 - Deep Learning for Computer Vision →](../Module%206%20-%20Deep%20Learning%20for%20Computer%20Vision/Lesson%201%20-%20CNNs%20Architecture.md)

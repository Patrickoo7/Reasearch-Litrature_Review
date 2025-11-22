# Lesson 3: Transfer Learning 🎯

**Module 6: Deep Learning for Computer Vision | Lesson 3 of 4**

Master transfer learning - train SOTA models with limited data!

---

## Why Transfer Learning?

**Problem:** Training from scratch needs:
- Millions of images
- Days/weeks of GPU time
- Lots of expertise

**Solution:** Use pre-trained weights from ImageNet!
- Train in hours, not days
- Works with small datasets
- Better accuracy

---

## 1. Transfer Learning Strategies 📚

### Strategy 1: Feature Extractor (Frozen)

```python
import torch
import torch.nn as nn
import torchvision.models as models

# Load pre-trained model
model = models.resnet50(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 10)  # 10 classes

# Only classifier trains
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

### Strategy 2: Fine-tuning (Full Network)

```python
model = models.resnet50(pretrained=True)

# All layers trainable
for param in model.parameters():
    param.requires_grad = True

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 10)

# Different learning rates
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
    {'params': list(model.layer1.parameters()) +
               list(model.layer2.parameters()) +
               list(model.layer3.parameters()), 'lr': 0.00001}
])
```

---

## 2. Complete Example: Custom Dataset 📸

```python
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Data transforms
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Load dataset
train_dataset = datasets.ImageFolder('data/train', transform=train_transform)
val_dataset = datasets.ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Validate
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total

# Training loop
for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate(model, val_loader, criterion)

    print(f'Epoch {epoch+1}:')
    print(f'  Train Loss: {train_loss:.3f}, Acc: {train_acc:.2f}%')
    print(f'  Val Loss: {val_loss:.3f}, Acc: {val_acc:.2f}%')
```

---

## Quick Reference 📖

**Decision Tree:**
```
Dataset Size?
├─ Very small (<1000 images)
│  └─ Feature extraction only (freeze all)
├─ Small (1000-10k images)
│  └─ Fine-tune last few layers
└─ Large (>10k images)
   └─ Fine-tune entire network
```

**Best Practices:**
1. Always use data augmentation
2. Start with frozen, then fine-tune
3. Use smaller LR for pre-trained layers
4. Monitor validation accuracy
5. Save best model (not last!)

---

## Key Takeaways 💡

1. **Transfer learning** = use pre-trained weights
2. **Feature extraction** = freeze everything
3. **Fine-tuning** = train with small LR
4. **Works well** even with small datasets
5. **ImageNet** weights transfer to most vision tasks

---

**Next:** [Lesson 4 - Object Detection Basics →](Lesson%204%20-%20Object%20Detection%20Basics.md)

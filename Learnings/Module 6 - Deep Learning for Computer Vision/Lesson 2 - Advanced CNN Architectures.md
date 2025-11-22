# Lesson 2: Advanced CNN Architectures 🏆

**Module 6: Deep Learning for Computer Vision | Lesson 2 of 4**

Learn from the best - VGG, ResNet, EfficientNet and more!

---

## Evolution of CNN Architectures

**Timeline:**
- 2012: AlexNet (ImageNet winner)
- 2014: VGG (very deep networks)
- 2015: **ResNet** (skip connections, 152 layers)
- 2016: Inception (multi-scale features)
- 2017: MobileNet (efficient for mobile)
- 2019: **EfficientNet** (SOTA efficiency)

---

## 1. VGG (Very Deep) 🏗️

**Key ideas:**
- Small 3×3 filters throughout
- Double channels after pooling
- Very deep (16-19 layers)

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Blocks 4 & 5 (similar pattern, 512 channels)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
```

---

## 2. ResNet (Skip Connections) ⭐

**Problem:** Very deep networks degrade (vanishing gradients)
**Solution:** Skip connections (residual connections)

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection (identity shortcut)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += identity  # Skip connection!
        out = F.relu(out)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

---

## 3. MobileNet (Efficient) 📱

**Key idea:** Depthwise separable convolutions

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Depthwise convolution (each channel separately)
        self.depthwise = nn.Conv2d(in_channels, in_channels,
                                  kernel_size=3, stride=stride,
                                  padding=1, groups=in_channels)

        # Pointwise convolution (1×1)
        self.pointwise = nn.Conv2d(in_channels, out_channels,
                                  kernel_size=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x

# Standard conv: k²×c_in×c_out params
# Depthwise separable: k²×c_in + c_in×c_out params
# Reduction: ~8-9x fewer parameters!
```

---

## 4. Pre-trained Models (Transfer Learning) 🎯

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet50(pretrained=True)
print(resnet)

# Freeze all layers
for param in resnet.parameters():
    param.requires_grad = False

# Replace final layer for your task
num_features = resnet.fc.in_features
resnet.fc = nn.Linear(num_features, 10)  # 10 classes

# Only train final layer
optimizer = torch.optim.Adam(resnet.fc.parameters(), lr=0.001)

# Or fine-tune all layers with smaller LR
for param in resnet.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': resnet.fc.parameters(), 'lr': 0.001},
    {'params': list(resnet.parameters())[:-2], 'lr': 0.0001}
])
```

---

## Quick Reference 📖

**Available Pre-trained Models:**
```python
# torchvision.models
models.resnet18(pretrained=True)
models.resnet50(pretrained=True)
models.vgg16(pretrained=True)
models.mobilenet_v2(pretrained=True)
models.efficientnet_b0(pretrained=True)
```

**When to use which:**
- **VGG:** Interpretability, simple architecture
- **ResNet:** Default choice, very deep
- **MobileNet:** Mobile deployment, speed
- **EfficientNet:** Best accuracy/efficiency

---

## Key Takeaways 💡

1. **ResNet** = skip connections enable very deep networks
2. **Pre-trained models** save time and improve accuracy
3. **Transfer learning** works even with small datasets
4. **MobileNet** for efficient deployment
5. **Always start with pre-trained** when possible

---

**Next:** [Lesson 3 - Transfer Learning →](Lesson%203%20-%20Transfer%20Learning.md)

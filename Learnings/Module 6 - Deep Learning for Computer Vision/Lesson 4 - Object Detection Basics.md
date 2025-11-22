# Lesson 4: Object Detection Basics 🎯

**Module 6: Deep Learning for Computer Vision | Lesson 4 of 4**

Learn object detection - finding and classifying multiple objects!

---

## Classification vs Detection 📍

**Classification:** What is in the image?
**Detection:** What + Where in the image?

Output: Bounding boxes + class labels

---

## 1. Bounding Boxes 📦

```python
# Bounding box format: [x_min, y_min, x_max, y_max]
bbox = [100, 150, 300, 400]

# Or: [x_center, y_center, width, height]
bbox_cxcywh = [200, 275, 200, 250]

def draw_bbox(image, bbox, label='Object'):
    import cv2
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image
```

---

## 2. IoU (Intersection over Union) 📊

```python
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes
    box format: [x1, y1, x2, y2]
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    iou = intersection / union if union > 0 else 0

    return iou

# Example
box1 = [100, 100, 200, 200]
box2 = [150, 150, 250, 250]

iou = compute_iou(box1, box2)
print(f"IoU: {iou:.3f}")  # ~0.14
```

---

## 3. Non-Maximum Suppression (NMS) 🎯

```python
def nms(boxes, scores, iou_threshold=0.5):
    """
    Non-maximum suppression
    boxes: list of [x1, y1, x2, y2]
    scores: confidence scores
    """
    # Sort by score
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    keep = []
    while len(indices) > 0:
        # Keep highest score
        current = indices[0]
        keep.append(current)

        # Remove boxes with high IoU
        indices = [i for i in indices[1:]
                  if compute_iou(boxes[current], boxes[i]) < iou_threshold]

    return keep

# Example
boxes = [
    [100, 100, 200, 200],
    [110, 110, 210, 210],  # Overlaps with first
    [300, 300, 400, 400]
]
scores = [0.9, 0.8, 0.95]

keep = nms(boxes, scores, iou_threshold=0.5)
print(f"Keep boxes: {keep}")  # [2, 0] - highest score, then non-overlapping
```

---

## 4. YOLO (You Only Look Once) Concept ⚡

**Idea:** Single pass detection (very fast!)

```python
# Simplified YOLO output
# Grid: 7×7
# Each cell predicts: [x, y, w, h, confidence, class_probs...]

class SimpleYOLO(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()

        # Backbone (feature extractor)
        self.features = models.resnet18(pretrained=True)
        self.features.fc = nn.Identity()  # Remove classifier

        # Detection head
        # Each grid cell: 5 values (x, y, w, h, conf) + num_classes
        self.detector = nn.Linear(512, 7 * 7 * (5 + num_classes))

        self.num_classes = num_classes

    def forward(self, x):
        # Extract features
        features = self.features(x)  # (batch, 512)

        # Predict
        output = self.detector(features)  # (batch, 7*7*(5+classes))

        # Reshape: (batch, 7, 7, 5+classes)
        output = output.view(-1, 7, 7, 5 + self.num_classes)

        return output
```

---

## 5. Using torchvision Detection Models 🚀

```python
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained Faster R-CNN
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Prepare image
from PIL import Image
import torchvision.transforms as T

image = Image.open('image.jpg')
transform = T.Compose([T.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

# Detect
with torch.no_grad():
    predictions = model(image_tensor)

# Parse predictions
boxes = predictions[0]['boxes'].cpu().numpy()
labels = predictions[0]['labels'].cpu().numpy()
scores = predictions[0]['scores'].cpu().numpy()

# Filter by confidence
threshold = 0.5
keep = scores > threshold

print(f"Detected {keep.sum()} objects")
for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
    print(f"  Class {label}: {score:.3f} at {box}")
```

---

## Quick Reference 📖

**Popular Detection Models:**
- **YOLO:** Very fast, real-time
- **Faster R-CNN:** High accuracy
- **SSD:** Balance speed/accuracy
- **RetinaNet:** Single-stage, accurate

**Metrics:**
- **IoU:** Overlap measure (>0.5 = match)
- **mAP:** Mean Average Precision
- **FPS:** Frames per second (speed)

---

## Key Takeaways 💡

1. **Object detection** = classification + localization
2. **IoU** measures bounding box overlap
3. **NMS** removes duplicate detections
4. **YOLO** = fast single-stage detector
5. **Faster R-CNN** = accurate two-stage detector
6. Use **pre-trained models** from torchvision

---

**Module 6 Complete!** 🎉

**Next Module:** [Module 7 - Time Series & Forecasting →](../Module%207%20-%20Time%20Series%20and%20Forecasting/Lesson%201%20-%20Time%20Series%20Fundamentals.md)

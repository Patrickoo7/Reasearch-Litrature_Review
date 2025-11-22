# Production Machine Learning & MLOps 🚀

Moving from Jupyter notebooks to production systems. The gap between research and deployment!

## Table of Contents
1. [Model Deployment](#model-deployment)
2. [MLOps Fundamentals](#mlops-fundamentals)
3. [Model Serving](#model-serving)
4. [Monitoring & Observability](#monitoring--observability)
5. [Model Versioning](#model-versioning)
6. [Data Pipelines](#data-pipelines)
7. [Scaling ML Systems](#scaling-ml-systems)

---

## The Research-Production Gap 🌉

### Research/Development
```
Jupyter Notebook
↓
Train on laptop/single GPU
↓
Evaluate on test set
↓
"It works!" 🎉
```

### Production Reality
```
- Millions of users
- Real-time predictions
- 24/7 availability
- Data drift over time
- Model retraining
- A/B testing
- Compliance & privacy
- Cost optimization
```

**Key Difference:** Research optimizes model metrics, production optimizes business value!

---

## Model Deployment 📦

### 1. Save Your Model

**PyTorch:**
```python
import torch

# Save entire model
torch.save(model, 'model.pth')

# Load
model = torch.load('model.pth')

# Better: Save only state dict (recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# Load
model = MyModel()
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

**TensorFlow/Keras:**
```python
# Save
model.save('model.h5')

# Load
model = tf.keras.models.load_model('model.h5')
```

**Scikit-learn:**
```python
import joblib

# Save
joblib.dump(model, 'model.pkl')

# Load
model = joblib.load('model.pkl')
```

### 2. Model Export Formats

**ONNX (Open Neural Network Exchange)**
- Cross-framework compatibility
- Optimized for inference

```python
import torch.onnx

# Export PyTorch to ONNX
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    export_params=True,
    opset_version=11,
    input_names=['input'],
    output_names=['output']
)

# Load with ONNX Runtime (faster inference)
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
output = session.run(['output'], {'input': input_data})
```

**TorchScript**
- PyTorch's serialization format
- Faster, production-ready

```python
# Tracing
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save('model_traced.pt')

# Scripting (handles control flow better)
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# Load
loaded_model = torch.jit.load('model_traced.pt')
```

**TensorFlow SavedModel**
```python
# Save
tf.saved_model.save(model, 'saved_model/')

# Load
model = tf.saved_model.load('saved_model/')
```

### 3. Model Optimization

**Quantization:** Reduce precision (32-bit → 8-bit)
```python
# PyTorch Dynamic Quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Results: 4x smaller, 2-4x faster
```

**Pruning:** Remove unnecessary weights
```python
import torch.nn.utils.prune as prune

# Prune 30% of weights
prune.l1_unstructured(module, name='weight', amount=0.3)
```

**Knowledge Distillation:** Train small model to mimic large model
```python
# Teacher (large model)
teacher_output = teacher_model(x)

# Student (small model)
student_output = student_model(x)

# Distillation loss
loss = F.kl_div(
    F.log_softmax(student_output / temperature, dim=1),
    F.softmax(teacher_output / temperature, dim=1)
)
```

---

## Model Serving 🌐

### REST API with Flask

```python
from flask import Flask, request, jsonify
import torch
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

# Load model
model = torch.jit.load('model_traced.pt')
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Load and preprocess image
    image = Image.open(request.files['image']).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)

    results = {
        'predictions': [
            {'class': idx.item(), 'probability': prob.item()}
            for idx, prob in zip(top5_indices, top5_prob)
        ]
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

**Client Example:**
```python
import requests

url = 'http://localhost:5000/predict'
files = {'image': open('cat.jpg', 'rb')}
response = requests.post(url, files=files)
print(response.json())
```

### FastAPI (Modern, Async)

```python
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import torch
import uvicorn

app = FastAPI()

class PredictionResponse(BaseModel):
    class_name: str
    confidence: float

@app.post('/predict', response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess and predict
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image_tensor)
        prediction = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1).max().item()

    return PredictionResponse(
        class_name=class_names[prediction],
        confidence=confidence
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
```

### TensorFlow Serving

```bash
# Start TF Serving with Docker
docker run -p 8501:8501 \
  --mount type=bind,source=/path/to/saved_model,target=/models/my_model \
  -e MODEL_NAME=my_model \
  -t tensorflow/serving

# Make prediction
curl -X POST http://localhost:8501/v1/models/my_model:predict \
  -d '{"instances": [[1.0, 2.0, 3.0, 4.0]]}'
```

### BentoML (Production-grade)

```python
import bentoml
from bentoml.io import Image, JSON

# Save model
bentoml.pytorch.save_model('image_classifier', model)

# Create service
@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 20},
)
class ImageClassifier:
    def __init__(self):
        self.model = bentoml.pytorch.get('image_classifier:latest').to_runner()

    @bentoml.api
    def predict(self, image: Image) -> JSON:
        # Preprocessing
        tensor = transform(image).unsqueeze(0)

        # Inference
        output = self.model(tensor)

        return {"prediction": output.argmax().item()}

# Build and deploy
# bentoml build
# bentoml containerize image_classifier:latest
# docker run -p 3000:3000 image_classifier:latest
```

---

## Monitoring & Observability 📊

### What to Monitor

**1. Model Performance Metrics**
```python
import prometheus_client as prom

# Define metrics
prediction_latency = prom.Histogram(
    'prediction_latency_seconds',
    'Time spent processing prediction'
)

prediction_counter = prom.Counter(
    'predictions_total',
    'Total predictions made',
    ['model_version', 'outcome']
)

model_accuracy = prom.Gauge(
    'model_accuracy',
    'Current model accuracy'
)

# Track metrics
@prediction_latency.time()
def make_prediction(input_data):
    result = model.predict(input_data)
    prediction_counter.labels(
        model_version='v1.2',
        outcome=result
    ).inc()
    return result
```

**2. Data Drift Detection**

Monitor if input distribution changes over time.

```python
from scipy import stats
import numpy as np

class DataDriftDetector:
    def __init__(self, reference_data):
        self.reference_data = reference_data
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)

    def detect_drift(self, new_data, threshold=0.05):
        """KS test for distribution shift"""
        drift_detected = []

        for feature_idx in range(new_data.shape[1]):
            ref_feature = self.reference_data[:, feature_idx]
            new_feature = new_data[:, feature_idx]

            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(ref_feature, new_feature)

            if p_value < threshold:
                drift_detected.append({
                    'feature': feature_idx,
                    'p_value': p_value,
                    'drift': True
                })

        return drift_detected

# Usage
detector = DataDriftDetector(training_data)
drift_report = detector.detect_drift(production_data)
if drift_report:
    print("⚠️ Data drift detected! Consider retraining.")
```

**3. Model Degradation**

Track performance over time.

```python
class ModelPerformanceTracker:
    def __init__(self, window_size=1000):
        self.predictions = []
        self.actuals = []
        self.window_size = window_size

    def log_prediction(self, prediction, actual=None):
        self.predictions.append(prediction)
        if actual is not None:
            self.actuals.append(actual)

        # Keep only recent predictions
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            if self.actuals:
                self.actuals.pop(0)

    def compute_metrics(self):
        if not self.actuals:
            return None

        accuracy = sum([p == a for p, a in zip(self.predictions, self.actuals)]) / len(self.actuals)
        return {
            'accuracy': accuracy,
            'num_samples': len(self.actuals)
        }

    def should_retrain(self, threshold=0.85):
        metrics = self.compute_metrics()
        if metrics and metrics['accuracy'] < threshold:
            return True
        return False
```

**4. System Metrics**
- Latency (p50, p95, p99)
- Throughput (requests/second)
- Error rate
- Resource utilization (CPU, GPU, memory)

### Logging

```python
import logging
import json
from datetime import datetime

# Structured logging
class StructuredLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)

    def log_prediction(self, input_data, prediction, latency, model_version):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'prediction',
            'model_version': model_version,
            'prediction': prediction,
            'latency_ms': latency * 1000,
            'input_shape': input_data.shape,
        }
        self.logger.info(json.dumps(log_entry))

# Usage
logger = StructuredLogger('ml_service')
logger.log_prediction(
    input_data=x,
    prediction=result,
    latency=0.123,
    model_version='v1.2.3'
)
```

---

## MLOps Fundamentals 🛠️

### CI/CD for ML

**Continuous Integration:**
```yaml
# .github/workflows/ml-ci.yml
name: ML Pipeline CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Setup Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest

    - name: Run tests
      run: pytest tests/

    - name: Train model
      run: python train.py --config config/test.yaml

    - name: Evaluate model
      run: python evaluate.py

    - name: Check model performance
      run: |
        python -c "
        import json
        metrics = json.load(open('metrics.json'))
        assert metrics['accuracy'] > 0.85, 'Model accuracy too low'
        "
```

**Continuous Deployment:**
```yaml
# .github/workflows/ml-cd.yml
name: Deploy Model

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Build Docker image
      run: docker build -t my-ml-model:${{ github.sha }} .

    - name: Push to registry
      run: |
        docker tag my-ml-model:${{ github.sha }} registry.com/my-ml-model:latest
        docker push registry.com/my-ml-model:latest

    - name: Deploy to production
      run: kubectl set image deployment/ml-model ml-model=registry.com/my-ml-model:latest
```

### Experiment Tracking with MLflow

```python
import mlflow
import mlflow.pytorch

# Start experiment
mlflow.set_experiment('image-classification')

with mlflow.start_run():
    # Log parameters
    mlflow.log_param('learning_rate', 0.001)
    mlflow.log_param('batch_size', 32)
    mlflow.log_param('epochs', 10)

    # Training loop
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_loss, val_acc = validate(model, val_loader)

        # Log metrics
        mlflow.log_metric('train_loss', train_loss, step=epoch)
        mlflow.log_metric('val_loss', val_loss, step=epoch)
        mlflow.log_metric('val_accuracy', val_acc, step=epoch)

    # Log model
    mlflow.pytorch.log_model(model, 'model')

    # Log artifacts
    mlflow.log_artifact('config.yaml')
    mlflow.log_artifact('plots/confusion_matrix.png')

# Later: Load best model
best_run = mlflow.search_runs(order_by=['metrics.val_accuracy DESC']).iloc[0]
model_uri = f"runs:/{best_run.run_id}/model"
loaded_model = mlflow.pytorch.load_model(model_uri)
```

### Feature Store

Centralize feature engineering and serving.

```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64
from datetime import timedelta

# Define entity
user = Entity(name='user', join_keys=['user_id'])

# Define feature view
user_features = FeatureView(
    name='user_features',
    entities=[user],
    ttl=timedelta(days=1),
    schema=[
        Field(name='age', dtype=Int64),
        Field(name='avg_purchase', dtype=Float32),
        Field(name='num_purchases', dtype=Int64),
    ]
)

# Initialize store
store = FeatureStore(repo_path='.')

# Get features for online serving
entity_rows = [{'user_id': 1001}, {'user_id': 1002}]
features = store.get_online_features(
    features=['user_features:age', 'user_features:avg_purchase'],
    entity_rows=entity_rows
).to_dict()
```

---

## Scaling ML Systems ⚡

### Batch Prediction

Process large volumes offline.

```python
import torch
from torch.utils.data import DataLoader

def batch_predict(model, dataloader, device='cuda'):
    model.eval()
    model.to(device)

    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input'].to(device)

            # Predict
            outputs = model(inputs)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(predictions.cpu().numpy())

    return all_predictions

# Usage
dataset = MyDataset(data_path)
dataloader = DataLoader(dataset, batch_size=256, num_workers=4)
predictions = batch_predict(model, dataloader)
```

### Model Parallelism (Large Models)

```python
# Split model across multiple GPUs
class ParallelModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
        self.layer2 = nn.Linear(1000, 1000).to('cuda:1')
        self.layer3 = nn.Linear(1000, 10).to('cuda:1')

    def forward(self, x):
        x = x.to('cuda:0')
        x = F.relu(self.layer1(x))

        x = x.to('cuda:1')
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        return x
```

### Data Parallelism

```python
# Replicate model across GPUs
model = MyModel()
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to('cuda')

# Training automatically uses all GPUs
for inputs, labels in dataloader:
    outputs = model(inputs)  # Distributed automatically
    loss = criterion(outputs, labels)
```

### Distributed Training (PyTorch)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

def train(rank, world_size):
    setup(rank, world_size)

    # Create model and wrap with DDP
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # Training loop
    for data, labels in dataloader:
        data, labels = data.to(rank), labels.to(rank)

        outputs = ddp_model(data)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Launch with torchrun
# torchrun --nproc_per_node=4 train.py
```

### Caching

```python
from functools import lru_cache
import hashlib
import pickle

class PredictionCache:
    def __init__(self, cache_dir='cache/'):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_key(self, input_data):
        # Hash input to create cache key
        data_bytes = pickle.dumps(input_data)
        return hashlib.md5(data_bytes).hexdigest()

    def get(self, input_data):
        key = self._get_cache_key(input_data)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, input_data, prediction):
        key = self._get_cache_key(input_data)
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")

        with open(cache_file, 'wb') as f:
            pickle.dump(prediction, f)

# Usage
cache = PredictionCache()

def predict_with_cache(input_data):
    # Check cache
    cached_result = cache.get(input_data)
    if cached_result is not None:
        return cached_result

    # Compute prediction
    prediction = model.predict(input_data)

    # Cache result
    cache.set(input_data, prediction)

    return prediction
```

---

## A/B Testing 🧪

```python
import random
from dataclasses import dataclass

@dataclass
class ABTest:
    name: str
    control_model: Any
    treatment_model: Any
    treatment_percentage: float = 0.5

    def get_model(self, user_id: str):
        # Consistent assignment based on user_id
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        if (hash_value % 100) < (self.treatment_percentage * 100):
            return self.treatment_model, 'treatment'
        else:
            return self.control_model, 'control'

# Setup A/B test
ab_test = ABTest(
    name='model_v2_test',
    control_model=model_v1,
    treatment_model=model_v2,
    treatment_percentage=0.2  # 20% get new model
)

# Serve predictions
def serve_prediction(user_id, input_data):
    model, variant = ab_test.get_model(user_id)
    prediction = model.predict(input_data)

    # Log for analysis
    log_ab_test_event(user_id, variant, prediction)

    return prediction
```

---

## Best Practices 📋

### 1. Reproducibility
```python
# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Log everything
mlflow.log_params({
    'random_seed': 42,
    'torch_version': torch.__version__,
    'cuda_version': torch.version.cuda
})
```

### 2. Model Versioning
```
models/
  v1.0/
    model.pt
    metadata.json
  v1.1/
    model.pt
    metadata.json
  v2.0/
    model.pt
    metadata.json
```

### 3. Gradual Rollout
- Deploy to 1% of traffic
- Monitor metrics
- Increase to 10%, 50%, 100%
- Rollback quickly if issues

### 4. Shadow Mode
- Run new model alongside old
- Log predictions from both
- Compare offline before switching

### 5. Fallbacks
```python
def predict_with_fallback(input_data):
    try:
        return new_model.predict(input_data)
    except Exception as e:
        logger.error(f"New model failed: {e}")
        return old_model.predict(input_data)
```

---

## Key Takeaways 💡

1. **Production != Research:** Different constraints and priorities
2. **Monitor everything:** Model performance, data drift, system metrics
3. **Automate:** CI/CD, retraining, deployment
4. **Version control:** Models, data, code, experiments
5. **Start simple:** Deploy basic version, iterate
6. **Optimize for latency:** Quantization, caching, batching
7. **Plan for failure:** Fallbacks, graceful degradation
8. **A/B test:** Measure real impact on business metrics

## MLOps Stack 🔧

**Experiment Tracking:** MLflow, Weights & Biases, Neptune.ai
**Model Serving:** BentoML, TF Serving, Triton
**Monitoring:** Prometheus, Grafana, WhyLabs
**Orchestration:** Kubeflow, MLflow, Airflow
**Feature Store:** Feast, Tecton
**Version Control:** DVC, Git-LFS
**Infrastructure:** Kubernetes, Docker, AWS SageMaker

---

**Production ML is hard, but systematic approaches make it manageable!** 🚀

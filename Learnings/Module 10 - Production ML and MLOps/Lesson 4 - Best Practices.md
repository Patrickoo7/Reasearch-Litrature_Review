# Lesson 4: Production ML Best Practices ⭐

**Module 10: Production ML & MLOps | Lesson 4 of 4**

Master best practices for reliable production ML systems!

---

## Complete Production Checklist ✅

### 1. Development

```
✅ Version control (Git)
✅ Data versioning (DVC)
✅ Experiment tracking (MLflow)
✅ Reproducible environments (requirements.txt, Docker)
✅ Unit tests for data processing
✅ Unit tests for model code
✅ Integration tests
✅ Code reviews
```

### 2. Model Training

```
✅ Train/validation/test splits
✅ Cross-validation
✅ Hyperparameter tuning
✅ Feature importance analysis
✅ Model explainability (SHAP)
✅ Bias/fairness checks
✅ Document model card
```

### 3. Deployment

```
✅ Model serialization
✅ API endpoint (FastAPI)
✅ Input validation
✅ Error handling
✅ Health checks
✅ Docker containerization
✅ Load testing
✅ Security (authentication)
```

### 4. Monitoring

```
✅ Performance metrics
✅ Data drift detection
✅ Logging
✅ Alerting
✅ A/B testing
✅ Automated retraining
✅ Dashboard
```

---

## Production-Ready API Template 🚀

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
import joblib
import numpy as np
import logging
from prometheus_client import Counter, Histogram
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')

# Load model
model = joblib.load('model.joblib')

app = FastAPI(title="Production ML API", version="1.0.0")

# Request schema with validation
class PredictionRequest(BaseModel):
    features: List[float]

    @validator('features')
    def validate_features(cls, v):
        if len(v) != 30:  # Expected number of features
            raise ValueError('Expected 30 features')
        if any(np.isnan(v)):
            raise ValueError('Features contain NaN')
        return v

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: float
    version: str

@app.post('/predict', response_model=PredictionResponse)
@prediction_latency.time()
async def predict(request: PredictionRequest):
    """Make prediction"""
    try:
        prediction_counter.inc()

        # Convert to array
        features = np.array(request.features).reshape(1, -1)

        # Predict
        prediction = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        probability = float(probabilities[prediction])
        confidence = float(probabilities.max())

        # Log
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.3f}")

        # Alert on low confidence
        if confidence < 0.7:
            logger.warning(f"Low confidence: {confidence:.3f}")

        return PredictionResponse(
            prediction=prediction,
            probability=probability,
            confidence=confidence,
            version="1.0.0"
        )

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/health')
async def health():
    """Health check"""
    return {'status': 'healthy', 'model_loaded': model is not None}

@app.get('/metrics')
async def metrics():
    """Prometheus metrics"""
    from prometheus_client import generate_latest
    return generate_latest()

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## Testing Strategy 🧪

```python
import pytest
import numpy as np
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health():
    """Test health endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'healthy'

def test_predict_valid():
    """Test valid prediction"""
    data = {
        'features': list(np.random.randn(30))
    }

    response = client.post('/predict', json=data)
    assert response.status_code == 200
    assert 'prediction' in response.json()
    assert 0 <= response.json()['confidence'] <= 1

def test_predict_invalid_length():
    """Test invalid feature length"""
    data = {
        'features': [1, 2, 3]  # Wrong length
    }

    response = client.post('/predict', json=data)
    assert response.status_code == 422

def test_predict_nan():
    """Test NaN handling"""
    data = {
        'features': [np.nan] * 30
    }

    response = client.post('/predict', json=data)
    assert response.status_code == 422

# Run: pytest test_app.py
```

---

## Security Best Practices 🔒

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    expected_token = os.getenv('API_TOKEN')

    if token != expected_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

    return token

@app.post('/predict', dependencies=[Depends(verify_token)])
async def predict(request: PredictionRequest):
    """Protected prediction endpoint"""
    # ... prediction code
    pass

# Usage:
# curl -H "Authorization: Bearer YOUR_TOKEN" \
#      -X POST http://localhost:8000/predict \
#      -d '{"features": [...]}'
```

---

## Configuration Management 🎛️

```python
from pydantic import BaseSettings

class Settings(BaseSettings):
    """Application settings"""
    app_name: str = "ML API"
    model_path: str = "model.joblib"
    confidence_threshold: float = 0.7
    max_requests_per_minute: int = 100
    log_level: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

# .env file:
# MODEL_PATH=models/latest.joblib
# CONFIDENCE_THRESHOLD=0.75
# LOG_LEVEL=DEBUG
```

---

## Error Handling & Recovery 🚨

```python
class ModelLoadError(Exception):
    """Model loading failed"""
    pass

def load_model_with_fallback():
    """Load model with fallback"""
    try:
        model = joblib.load(settings.model_path)
        logger.info(f"Loaded model from {settings.model_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")

        # Try backup
        try:
            model = joblib.load('models/backup.joblib')
            logger.warning("Loaded backup model")
            return model

        except:
            raise ModelLoadError("All model loading attempts failed")

# Graceful degradation
@app.post('/predict')
async def predict(request: PredictionRequest):
    if model is None:
        return {
            'prediction': None,
            'error': 'Model unavailable',
            'fallback': True
        }

    # Normal prediction...
```

---

## Documentation 📚

```python
# Model Card Template
MODEL_CARD = """
# Model Card: Breast Cancer Classifier

## Model Details
- **Type:** Random Forest Classifier
- **Version:** 1.0.0
- **Date:** 2024-01-15
- **Author:** ML Team

## Intended Use
- **Primary use:** Breast cancer prediction from diagnostic features
- **Out-of-scope:** Not for clinical diagnosis without expert review

## Training Data
- **Dataset:** Wisconsin Breast Cancer Dataset
- **Size:** 569 samples, 30 features
- **Split:** 80% train, 20% test

## Performance
- **Accuracy:** 95.6%
- **Precision:** 96.2%
- **Recall:** 94.8%
- **F1-Score:** 95.5%

## Ethical Considerations
- Model trained on limited demographic data
- Not validated across diverse populations
- Requires clinical expert validation

## Maintenance
- Monthly retraining on new data
- Continuous monitoring for data drift
- Performance reviewed quarterly
"""
```

---

## Key Takeaways 💡

1. **Production ≠ notebook code**
2. **Testing** is critical
3. **Monitoring** from day 1
4. **Security** not optional
5. **Documentation** enables maintenance
6. **Automate** everything possible
7. **Plan for failure** (graceful degradation)

---

**Module 10 Complete!** 🎉

**Next Module:** [Module 11 - Capstone & Career →](../Module%2011%20-%20Capstone%20and%20Career/Lesson%201%20-%20End-to-End%20Project.md)

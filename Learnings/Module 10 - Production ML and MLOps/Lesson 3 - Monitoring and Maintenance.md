# Lesson 3: Monitoring & Maintenance 📡

**Module 10: Production ML & MLOps | Lesson 3 of 4**

Monitor and maintain ML systems in production!

---

## Why Monitor?

**ML models degrade over time:**
- Data drift (input distribution changes)
- Concept drift (relationship changes)
- Performance degradation
- System issues

---

## 1. Model Performance Monitoring 📊

```python
import logging
from datetime import datetime

class ModelMonitor:
    def __init__(self, model, threshold=0.8):
        self.model = model
        self.threshold = threshold
        self.predictions = []
        self.actuals = []

    def predict(self, X):
        """Predict with logging"""
        prediction = self.model.predict(X)

        # Log prediction
        self.predictions.append({
            'timestamp': datetime.now(),
            'input': X.tolist(),
            'prediction': prediction.tolist()
        })

        return prediction

    def log_actual(self, y_true):
        """Log ground truth when available"""
        self.actuals.append({
            'timestamp': datetime.now(),
            'actual': y_true
        })

    def check_performance(self):
        """Calculate current accuracy"""
        if len(self.actuals) == 0:
            return None

        from sklearn.metrics import accuracy_score

        y_pred = [p['prediction'] for p in self.predictions[-len(self.actuals):]]
        y_true = [a['actual'] for a in self.actuals]

        accuracy = accuracy_score(y_true, y_pred)

        # Alert if below threshold
        if accuracy < self.threshold:
            logging.warning(f"Model accuracy ({accuracy:.3f}) below threshold ({self.threshold})")

        return accuracy

# Usage
monitor = ModelMonitor(model, threshold=0.8)

# Predict
prediction = monitor.predict(X_new)

# Later, when ground truth available
monitor.log_actual(y_true)

# Check performance
current_accuracy = monitor.check_performance()
```

---

## 2. Data Drift Detection 🌊

```python
from scipy.stats import ks_2samp
import numpy as np

class DataDriftDetector:
    def __init__(self, reference_data, threshold=0.05):
        """
        reference_data: Training data distribution
        threshold: p-value threshold
        """
        self.reference = reference_data
        self.threshold = threshold

    def detect_drift(self, new_data):
        """Detect drift using Kolmogorov-Smirnov test"""
        drifted_features = []

        for i in range(self.reference.shape[1]):
            # KS test
            statistic, p_value = ks_2samp(
                self.reference[:, i],
                new_data[:, i]
            )

            if p_value < self.threshold:
                drifted_features.append(i)

        return drifted_features

# Usage
detector = DataDriftDetector(X_train, threshold=0.05)

# Check new data
drifted = detector.detect_drift(X_new)

if len(drifted) > 0:
    print(f"Drift detected in features: {drifted}")
    # Trigger retraining
```

---

## 3. Logging & Alerting 📝

```python
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('ml_model.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ML_Model')

def predict_with_logging(model, X):
    """Predict with comprehensive logging"""
    try:
        # Log input
        logger.info(f"Received prediction request: {X.shape}")

        # Validate input
        if np.isnan(X).any():
            logger.warning("Input contains NaN values")
            X = np.nan_to_num(X)

        # Predict
        prediction = model.predict(X)
        confidence = model.predict_proba(X).max(axis=1)

        # Log prediction
        logger.info(f"Prediction: {prediction}, Confidence: {confidence}")

        # Alert on low confidence
        if confidence.min() < 0.5:
            logger.warning(f"Low confidence prediction: {confidence.min():.3f}")

        return prediction

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}", exc_info=True)
        raise
```

---

## 4. Metrics Dashboard 📈

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions')
prediction_duration = Histogram('model_prediction_duration_seconds', 'Prediction duration')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
prediction_confidence = Histogram('model_prediction_confidence', 'Prediction confidence')

@prediction_duration.time()
def monitored_predict(model, X):
    """Predict with Prometheus metrics"""
    prediction_counter.inc()

    prediction = model.predict(X)
    confidence = model.predict_proba(X).max()

    prediction_confidence.observe(confidence)

    return prediction

# Start metrics server
start_http_server(8000)

# Update accuracy periodically
def update_accuracy():
    current_acc = monitor.check_performance()
    if current_acc is not None:
        model_accuracy.set(current_acc)

# Grafana can visualize these metrics
```

---

## 5. A/B Testing in Production 🧪

```python
import random

class ABTestingDeployer:
    def __init__(self, model_a, model_b, split=0.5):
        self.model_a = model_a  # Current model
        self.model_b = model_b  # New model
        self.split = split
        self.results_a = []
        self.results_b = []

    def predict(self, X, user_id):
        """Route to A or B based on user ID"""
        # Deterministic assignment
        if hash(user_id) % 100 < self.split * 100:
            model = self.model_a
            variant = 'A'
        else:
            model = self.model_b
            variant = 'B'

        prediction = model.predict(X)

        # Log for analysis
        logging.info(f"User {user_id} -> Variant {variant}")

        return prediction, variant

    def log_outcome(self, user_id, variant, actual):
        """Log actual outcome"""
        if variant == 'A':
            self.results_a.append(actual)
        else:
            self.results_b.append(actual)

    def compare_models(self):
        """Statistical comparison"""
        from scipy.stats import ttest_ind

        if len(self.results_a) < 30 or len(self.results_b) < 30:
            return "Not enough data"

        stat, p_value = ttest_ind(self.results_a, self.results_b)

        return {
            'mean_a': np.mean(self.results_a),
            'mean_b': np.mean(self.results_b),
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

---

## 6. Automated Retraining 🔄

```python
def should_retrain(monitor, drift_detector, X_new):
    """Decide if retraining needed"""
    # Check 1: Performance degradation
    current_accuracy = monitor.check_performance()
    if current_accuracy and current_accuracy < 0.8:
        return True, "Performance below threshold"

    # Check 2: Data drift
    drifted_features = drift_detector.detect_drift(X_new)
    if len(drifted_features) > 3:
        return True, f"Drift in {len(drifted_features)} features"

    # Check 3: Time since last training
    from datetime import datetime, timedelta
    if datetime.now() - last_training_date > timedelta(days=30):
        return True, "30 days since last training"

    return False, None

def automated_retraining_pipeline():
    """Automated retraining workflow"""
    retrain, reason = should_retrain(monitor, drift_detector, X_new)

    if retrain:
        logging.info(f"Triggering retraining: {reason}")

        # 1. Fetch new data
        X_new, y_new = fetch_new_data()

        # 2. Retrain
        new_model = train_model(X_new, y_new)

        # 3. Evaluate
        new_accuracy = new_model.score(X_test, y_test)
        old_accuracy = model.score(X_test, y_test)

        # 4. Deploy if better
        if new_accuracy > old_accuracy:
            logging.info(f"New model better: {new_accuracy:.3f} > {old_accuracy:.3f}")
            deploy_model(new_model)
        else:
            logging.warning(f"New model worse: {new_accuracy:.3f} < {old_accuracy:.3f}")
```

---

## Quick Reference 📖

**Monitoring Checklist:**
```
✅ Log all predictions
✅ Track performance metrics
✅ Detect data drift
✅ Alert on anomalies
✅ A/B test new models
✅ Automated retraining
✅ Dashboard visualization
```

**Key Metrics:**
- **Accuracy/F1** over time
- **Latency** (p50, p95, p99)
- **Throughput** (requests/sec)
- **Error rate**
- **Data drift** score
- **Confidence** distribution

---

## Key Takeaways 💡

1. **Monitor everything** in production
2. **Data drift** causes degradation
3. **Automated retraining** essential
4. **A/B testing** for safe deployment
5. **Logging** critical for debugging
6. **Dashboards** for visibility
7. **Alert** on anomalies

---

**Next:** [Lesson 4 - Best Practices →](Lesson%204%20-%20Best%20Practices.md)

# Lesson 2: MLOps Basics 🔧

**Module 10: Production ML & MLOps | Lesson 2 of 4**

Master MLOps - DevOps for Machine Learning!

---

## What is MLOps?

**ML + DevOps + Data Engineering:**
- Automate ML pipeline
- Version control (code + data + models)
- CI/CD for ML
- Monitoring & retraining

---

## 1. Version Control 📝

### Git for Code

```bash
# .gitignore for ML projects
*.pyc
__pycache__/
*.ipynb_checkpoints
*.pkl
*.joblib
*.h5
*.pth
data/
models/
.env
```

### DVC for Data & Models

```bash
# Install DVC
pip install dvc

# Initialize
git init
dvc init

# Track data
dvc add data/train.csv
git add data/train.csv.dvc .gitignore
git commit -m "Add training data"

# Track model
dvc add models/model.joblib
git add models/model.joblib.dvc
git commit -m "Add model v1"

# Remote storage
dvc remote add -d storage s3://my-bucket/dvc-storage
dvc push
```

---

## 2. Experiment Tracking 📊

### MLflow

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Start experiment
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

# View experiments
# mlflow ui
```

### Weights & Biases

```python
import wandb

# Initialize
wandb.init(project="my-project", config={
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": 32
})

# Log metrics
for epoch in range(10):
    # Train...
    wandb.log({
        "epoch": epoch,
        "loss": loss,
        "accuracy": accuracy
    })

# Finish
wandb.finish()
```

---

## 3. Pipeline Automation 🔄

### Airflow DAG

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def extract_data():
    """Extract data from source"""
    pass

def train_model():
    """Train ML model"""
    pass

def evaluate_model():
    """Evaluate model"""
    pass

def deploy_model():
    """Deploy if performance good"""
    pass

# Define DAG
with DAG(
    'ml_pipeline',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@daily',
    catchup=False
) as dag:

    extract = PythonOperator(
        task_id='extract',
        python_callable=extract_data
    )

    train = PythonOperator(
        task_id='train',
        python_callable=train_model
    )

    evaluate = PythonOperator(
        task_id='evaluate',
        python_callable=evaluate_model
    )

    deploy = PythonOperator(
        task_id='deploy',
        python_callable=deploy_model
    )

    # Dependencies
    extract >> train >> evaluate >> deploy
```

---

## 4. CI/CD for ML 🚀

### GitHub Actions

```yaml
# .github/workflows/ml-ci.yml
name: ML CI/CD

on: [push]

jobs:
  train-and-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.9

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run tests
      run: |
        pytest tests/

    - name: Train model
      run: |
        python train.py

    - name: Evaluate model
      run: |
        python evaluate.py

    - name: Deploy
      if: github.ref == 'refs/heads/main'
      run: |
        python deploy.py
```

---

## 5. Model Registry 📚

```python
import mlflow

# Register model
mlflow.set_tracking_uri("http://localhost:5000")

# Log and register
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="MyModel"
    )

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.sklearn.load_model(
    model_uri="models:/MyModel/Production"
)
```

---

## 6. Complete MLOps Project Structure 📁

```
ml-project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── .gitkeep
├── models/
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __init__.py
│   ├── data/
│   │   └── make_dataset.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       └── visualize.py
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_api.py
├── api/
│   ├── app.py
│   └── Dockerfile
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── requirements.txt
├── setup.py
├── README.md
├── .gitignore
└── .dvcignore
```

---

## Quick Reference 📖

**MLOps Stack:**

| Component | Tools |
|-----------|-------|
| Version Control | Git, DVC |
| Experiment Tracking | MLflow, W&B |
| Orchestration | Airflow, Prefect |
| CI/CD | GitHub Actions, Jenkins |
| Model Registry | MLflow, DVC |
| Monitoring | Prometheus, Grafana |
| Deployment | Docker, Kubernetes |

**Best Practices:**
1. Version everything (code, data, models)
2. Automate pipelines
3. Track all experiments
4. Test before deploying
5. Monitor in production
6. Reproducibility first

---

## Key Takeaways 💡

1. **MLOps** = automated ML lifecycle
2. **DVC** for data/model versioning
3. **MLflow** for experiment tracking
4. **Airflow** for orchestration
5. **CI/CD** essential for production
6. **Model registry** manages versions
7. **Reproducibility** critical

---

**Next:** [Lesson 3 - Monitoring & Maintenance →](Lesson%203%20-%20Monitoring%20and%20Maintenance.md)

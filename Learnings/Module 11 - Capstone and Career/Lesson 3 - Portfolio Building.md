# Lesson 3: Portfolio Building 🎨

**Module 11: Capstone & Career | Lesson 3 of 4**

Build a portfolio that gets you hired!

---

## Why Portfolio Matters 🎯

**Hiring managers look for:**
1. **Real projects** (not just tutorials)
2. **Clean code** (readable, documented)
3. **Complete solutions** (end-to-end)
4. **Communication** (explain your work)

**Portfolio > Certifications**

---

## Portfolio Structure 📁

### 1. GitHub Profile

```
Your GitHub is your resume!

Profile Setup:
✅ Professional photo
✅ Bio: "ML Engineer | Python | PyTorch"
✅ Location & email
✅ Pin 4-6 best projects
✅ Consistent commit activity (green squares!)
✅ README.md on profile
```

### Sample Profile README

```markdown
# Hi, I'm [Your Name] 👋

## About Me
ML Engineer passionate about solving real-world problems with AI.
Currently learning advanced NLP and MLOps.

## Skills
- **Languages:** Python, SQL, R
- **ML/DL:** scikit-learn, PyTorch, TensorFlow, XGBoost
- **MLOps:** Docker, MLflow, Airflow, FastAPI
- **Cloud:** AWS (SageMaker, S3, EC2)

## Featured Projects
- 🔍 [Customer Churn Prediction](link) - XGBoost model, 87% AUC
- 🎬 [Movie Recommender](link) - Collaborative filtering, 100K users
- 📈 [Stock Price Forecasting](link) - LSTM, Prophet, 5-year data
- 🏥 [Medical Image Classification](link) - CNN, 95% accuracy

## GitHub Stats
![Stats](https://github-readme-stats.vercel.app/api?username=yourusername)

## Contact
- LinkedIn: [profile-link]
- Email: your.email@example.com
- Blog: [your-blog.com]
```

---

## Project Ideas by Level 🎓

### Beginner (Choose 2)

**1. House Price Prediction**
```
Dataset: Kaggle House Prices
Goal: Predict house prices
Techniques: Linear regression, feature engineering
Showcase: Data cleaning, EDA, model comparison
```

**2. Customer Segmentation**
```
Dataset: Online Retail (UCI)
Goal: Segment customers
Techniques: K-means, RFM analysis
Showcase: Clustering, visualization
```

**3. Sentiment Analysis**
```
Dataset: IMDB reviews
Goal: Classify positive/negative
Techniques: TF-IDF, Logistic Regression
Showcase: Text preprocessing, classification
```

### Intermediate (Choose 2)

**4. Customer Churn Prediction**
```
Dataset: Telco Churn
Goal: Predict customer churn
Techniques: XGBoost, SMOTE, SHAP
Showcase: Imbalanced data, interpretability, deployment
```

**5. Image Classifier**
```
Dataset: CIFAR-10 or custom
Goal: Classify images
Techniques: CNN, transfer learning (ResNet)
Showcase: Deep learning, data augmentation
```

**6. Recommender System**
```
Dataset: MovieLens
Goal: Movie recommendations
Techniques: Collaborative filtering, matrix factorization
Showcase: Scalability, evaluation metrics
```

### Advanced (Choose 1-2)

**7. End-to-End ML Pipeline**
```
Goal: Production-ready system
Components:
- Data ingestion (API/database)
- Feature store
- Model training (MLflow)
- Deployment (FastAPI + Docker)
- Monitoring (Prometheus)
Showcase: MLOps, system design
```

**8. Real-time Object Detection**
```
Goal: Detect objects in video
Techniques: YOLO, OpenCV
Showcase: Real-time inference, edge deployment
```

**9. Time Series Forecasting**
```
Dataset: Stock prices, weather, sales
Goal: Multi-step forecasting
Techniques: ARIMA, Prophet, LSTM
Showcase: Time series, ensemble methods
```

---

## Project Template 📋

### Minimal Viable Portfolio Project

```
project-name/
├── README.md               # Main documentation
├── requirements.txt        # Dependencies
├── .gitignore             # Ignore data, models
├── data/
│   ├── raw/               # Original data (gitignored)
│   ├── processed/         # Cleaned data
│   └── README.md          # Data description
├── notebooks/
│   ├── 01_eda.ipynb       # Exploration
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration
│   ├── data/
│   │   ├── __init__.py
│   │   └── load_data.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── build_features.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py
│   │   └── predict.py
│   └── visualization/
│       ├── __init__.py
│       └── visualize.py
├── tests/
│   ├── test_data.py
│   └── test_model.py
├── models/
│   └── .gitkeep           # Model files gitignored
├── results/
│   ├── figures/           # Plots
│   └── metrics.json       # Performance
├── api/                   # (Optional) Deployment
│   ├── app.py
│   ├── Dockerfile
│   └── requirements.txt
└── docs/
    └── methodology.md     # Detailed explanation
```

---

## README Best Practices 📝

### Excellent README Template

```markdown
# Project Title

One-line description of what the project does.

![Demo](results/demo.gif)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)

## Overview

**Problem Statement:**
Clear description of the problem you're solving.

**Goal:**
What you're trying to achieve.

**Why this matters:**
Real-world impact or learning objectives.

## Dataset

- **Source:** [Link to dataset]
- **Size:** X samples, Y features
- **Target:** Description
- **Split:** 80% train, 20% test

**Features:**
- Feature 1: Description
- Feature 2: Description
- ...

## Methodology

### 1. Data Preprocessing
- Handled missing values (median imputation)
- Encoded categorical variables (one-hot)
- Scaled features (StandardScaler)

### 2. Feature Engineering
- Created X new features
- Feature selection (reduced from Y to Z features)

### 3. Models Tested
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Logistic Regression | 0.82 | 0.80 | 0.75 | 0.77 |
| Random Forest | 0.89 | 0.87 | 0.85 | 0.86 |
| **XGBoost** | **0.93** | **0.91** | **0.89** | **0.90** |

### 4. Hyperparameter Tuning
Used RandomizedSearchCV with 5-fold CV.

## Results

### Model Performance
- **Best Model:** XGBoost
- **ROC-AUC:** 0.93
- **Key Finding:** Feature X is the strongest predictor

### Visualizations

![Confusion Matrix](results/figures/confusion_matrix.png)
![Feature Importance](results/figures/feature_importance.png)
![ROC Curve](results/figures/roc_curve.png)

### Business Impact
- Predicts churn with 93% accuracy
- Can save $X per year in retention costs
- Identifies top 3 risk factors

## Installation

```bash
# Clone repository
git clone https://github.com/username/project.git
cd project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training
```bash
python src/models/train.py --data data/processed/data.csv
```

### Prediction
```bash
python src/models/predict.py --model models/best_model.joblib --input data.csv
```

### API (Optional)
```bash
uvicorn api.app:app --reload

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, ...]}'
```

## Key Learnings
- Learned how to handle imbalanced data
- XGBoost significantly outperformed other models
- Feature engineering increased performance by 12%

## Future Work
- [ ] Implement deep learning model
- [ ] Add real-time prediction API
- [ ] Deploy on AWS
- [ ] A/B test in production

## References
- [Paper 1](link)
- [Article 2](link)

## License
MIT

## Contact
- Email: your.email@example.com
- LinkedIn: [profile-link]
```

---

## Code Quality Checklist ✅

### Clean Code Principles

```python
# ❌ BAD
def f(x):
    return x*2+5

y = f(10)


# ✅ GOOD
def calculate_final_price(base_price: float) -> float:
    """
    Calculate final price with markup.

    Args:
        base_price: Original price in dollars

    Returns:
        Final price with 2x markup and $5 fee
    """
    markup_multiplier = 2
    processing_fee = 5
    return base_price * markup_multiplier + processing_fee

final_price = calculate_final_price(base_price=10)
```

### Style Guide

```python
# Follow PEP 8
# Use meaningful names
# Add docstrings
# Type hints (Python 3.6+)

from typing import List, Tuple
import numpy as np

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_type: str = 'random_forest'
) -> Tuple[object, dict]:
    """
    Train machine learning model.

    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        model_type: Type of model ('random_forest', 'xgboost')

    Returns:
        Tuple of (trained_model, metrics_dict)

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Train
    model.fit(X_train, y_train)

    # Metrics
    train_score = model.score(X_train, y_train)
    metrics = {'train_accuracy': train_score}

    return model, metrics
```

### Testing

```python
# tests/test_preprocessing.py
import pytest
import numpy as np
from src.features.build_features import create_features

def test_create_features_shape():
    """Test output shape matches input"""
    X = np.random.rand(100, 5)
    X_transformed = create_features(X)
    assert X_transformed.shape[0] == 100

def test_create_features_no_nan():
    """Test no NaN values in output"""
    X = np.random.rand(100, 5)
    X_transformed = create_features(X)
    assert not np.isnan(X_transformed).any()

def test_create_features_invalid_input():
    """Test error handling"""
    with pytest.raises(ValueError):
        create_features(None)
```

---

## Deployment & Demo 🚀

### Option 1: Streamlit App

```python
# app.py
import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('models/model.joblib')

st.title('Customer Churn Predictor')

# Input fields
tenure = st.slider('Tenure (months)', 0, 72, 12)
monthly_charges = st.number_input('Monthly Charges', 0, 200, 50)
total_charges = st.number_input('Total Charges', 0, 10000, 500)

# Predict
if st.button('Predict'):
    features = np.array([[tenure, monthly_charges, total_charges]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.error(f'⚠️ High churn risk: {probability:.1%}')
    else:
        st.success(f'✅ Low churn risk: {1-probability:.1%}')

# Run: streamlit run app.py
```

### Option 2: FastAPI

```python
# api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = joblib.load('../models/model.joblib')

class PredictionRequest(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float

@app.post('/predict')
def predict(request: PredictionRequest):
    features = np.array([[
        request.tenure,
        request.monthly_charges,
        request.total_charges
    ]])

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        'churn': bool(prediction),
        'probability': probability
    }

# Run: uvicorn app:app --reload
```

### Option 3: GitHub Pages

```markdown
Host your portfolio website for free!

1. Create repository: username.github.io
2. Add index.html
3. Push to GitHub
4. Visit: https://username.github.io

Use templates:
- Jekyll
- Hugo
- Simple HTML/CSS
```

---

## Resume Tips 📄

### ML Resume Structure

```
[NAME]
[Title: Machine Learning Engineer]
[Email | LinkedIn | GitHub | Portfolio]

SUMMARY
---------
ML Engineer with X projects in [domains]. Proficient in
Python, scikit-learn, PyTorch, and AWS. Passionate about
[specific area].

SKILLS
---------
• Languages: Python, SQL, R
• ML/DL: scikit-learn, PyTorch, TensorFlow, XGBoost
• MLOps: Docker, MLflow, Airflow, FastAPI
• Cloud: AWS (SageMaker, S3), GCP
• Tools: Git, Jupyter, Linux

PROJECTS
---------
Customer Churn Prediction | Python, XGBoost, SHAP, FastAPI
• Developed model predicting customer churn with 93% ROC-AUC
• Implemented SMOTE for class imbalance (26% minority class)
• Deployed FastAPI endpoint serving 1000 req/sec
• Tech: XGBoost, SHAP for interpretability, Docker
[GitHub Link]

Movie Recommender System | Python, PyTorch, Collaborative Filtering
• Built recommender for 10M ratings, 100K users
• Implemented matrix factorization with ALS
• Achieved NDCG@10 of 0.84
[GitHub Link] [Demo Link]

EXPERIENCE (if any)
---------
[Previous roles, even non-ML, highlighting transferable skills]

EDUCATION
---------
[Degree] in [Field] - [University] (Year)
Relevant coursework: ML, Statistics, Algorithms
```

### Quantify Everything

```
❌ "Built a machine learning model"
✅ "Built XGBoost model achieving 93% ROC-AUC on 50K samples"

❌ "Deployed API"
✅ "Deployed FastAPI endpoint serving 1000 requests/sec with <50ms latency"

❌ "Improved model performance"
✅ "Increased F1-score from 0.82 to 0.91 through feature engineering"
```

---

## LinkedIn Optimization 💼

```
Profile Photo: Professional headshot
Headline: "ML Engineer | Python | PyTorch | Building AI solutions"

About:
---------
Passionate ML Engineer specializing in [domain].

🔹 Expertise: Predictive modeling, deep learning, MLOps
🔹 Tech Stack: Python, PyTorch, AWS, Docker
🔹 Recent: Built churn prediction system (93% AUC)

Open to opportunities in [roles/industries].

📧 email@example.com
💻 github.com/username

Featured:
---------
• Pin your best projects
• Write articles about learnings
• Share your code snippets

Endorsements:
---------
Ask connections to endorse your skills
```

---

## Portfolio Checklist 🎯

```
GitHub:
✅ 4-6 diverse projects
✅ Clean, documented code
✅ Comprehensive READMEs
✅ Regular commits (green squares)
✅ Profile README

Projects:
✅ End-to-end solutions
✅ Real-world problems
✅ Visualizations
✅ Deployed demos (Streamlit/API)

Documentation:
✅ Clear problem statements
✅ Methodology explained
✅ Results quantified
✅ Code commented

Resume:
✅ Quantified achievements
✅ GitHub links
✅ Tech stack highlighted

LinkedIn:
✅ Professional photo
✅ ML-focused headline
✅ Projects featured
✅ Active engagement
```

---

## Key Takeaways 💡

1. **Quality > Quantity** (4 great projects > 20 tutorials)
2. **End-to-end solutions** (problem → deployment)
3. **Document everything** (README, code comments)
4. **Deploy demos** (Streamlit, API)
5. **Clean code** (PEP 8, tests, type hints)
6. **Tell a story** (problem, approach, impact)
7. **Keep learning** (update portfolio regularly)

---

## 30-Day Portfolio Plan 📅

```
Week 1: Setup
• GitHub profile
• Choose 4 project ideas
• Set up project templates

Week 2-3: Build Projects
• Project 1: Beginner (3 days)
• Project 2: Intermediate (5 days)
• Project 3: Intermediate (5 days)

Week 4: Polish
• READMEs for all projects
• Deploy 2 projects (Streamlit)
• Resume update
• LinkedIn optimization

Ongoing:
• 1 commit per day (even small)
• Share learnings on LinkedIn
• Contribute to open source
```

---

**Next:** [Lesson 4 - Career Paths in ML →](Lesson%204%20-%20Career%20Paths.md)

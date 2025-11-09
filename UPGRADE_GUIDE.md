# Upgrade Guide for Research Reproducer

This guide outlines how to upgrade and enhance the Research Reproducer tool, organized by priority and difficulty.

## 🚀 Quick Wins (1-2 days each)

### 1. Add Comprehensive Testing ✅ STARTED
**Impact**: High | **Difficulty**: Low

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/research_reproducer --cov-report=html

# Run only integration tests
pytest tests/ -m integration
```

**TODO**:
- [ ] Add unit tests for all modules
- [ ] Add integration tests
- [ ] Setup CI/CD with GitHub Actions
- [ ] Achieve >80% code coverage

### 2. Add Caching ✅ IMPLEMENTED
**Impact**: High | **Difficulty**: Low

Already implemented in `cache.py`. To use:

```python
from research_reproducer.cache import ReproducerCache

cache = ReproducerCache()

# Cache paper metadata
cache.set_paper_metadata(arxiv_id, metadata)
cached = cache.get_paper_metadata(arxiv_id)

# Clear cache
cache.clear_cache('papers')  # or 'repositories' or 'analysis'
```

**TODO**:
- [ ] Integrate caching into main orchestrator
- [ ] Add cache CLI commands
- [ ] Add cache statistics to reports

### 3. Better Error Recovery ✅ IMPLEMENTED
**Impact**: High | **Difficulty**: Low

Already implemented in `retry_utils.py`. To use:

```python
from research_reproducer.retry_utils import retry_with_backoff

@retry_with_backoff(max_attempts=3, base_delay=2.0)
def fetch_from_api():
    # Your API call here
    pass
```

**TODO**:
- [ ] Apply retry logic to all network calls
- [ ] Add exponential backoff to GitHub API calls
- [ ] Better error messages with recovery suggestions

### 4. Add Progress Persistence
**Impact**: Medium | **Difficulty**: Low

Save progress so users can resume interrupted reproductions:

```python
# src/research_reproducer/checkpoint.py
import json
from pathlib import Path

class ReproductionCheckpoint:
    def __init__(self, session_dir):
        self.checkpoint_file = Path(session_dir) / '.checkpoint.json'

    def save(self, stage, data):
        checkpoint = self.load()
        checkpoint[stage] = data
        checkpoint['last_stage'] = stage

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load(self):
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                return json.load(f)
        return {}

    def get_last_stage(self):
        checkpoint = self.load()
        return checkpoint.get('last_stage')
```

---

## 💪 Medium Priority (3-7 days each)

### 5. Add More Paper Sources
**Impact**: High | **Difficulty**: Medium

**OpenReview Integration**:
```python
# src/research_reproducer/sources/openreview.py
import requests

class OpenReviewSource:
    def __init__(self):
        self.base_url = 'https://api.openreview.net'

    def get_paper(self, paper_id):
        url = f'{self.base_url}/notes?id={paper_id}'
        response = requests.get(url)
        return response.json()

    def search_papers(self, query):
        # Implement search
        pass
```

**Semantic Scholar API**:
```python
# Better citation tracking and paper recommendations
import requests

def get_paper_citations(arxiv_id):
    url = f'https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}'
    response = requests.get(url)
    data = response.json()
    return data.get('citations', [])
```

### 6. Improve Dependency Resolution
**Impact**: High | **Difficulty**: Medium

Use Poetry or pip-tools for better dependency management:

```python
# src/research_reproducer/dependency_resolver.py
import subprocess

class DependencyResolver:
    def resolve_conflicts(self, requirements):
        """Use pip-compile to resolve dependency conflicts"""
        # Write to temp requirements.txt
        # Run pip-compile
        # Parse resolved dependencies
        pass

    def suggest_alternatives(self, package, version):
        """Suggest alternative versions if current fails"""
        pass

    def create_lock_file(self, requirements):
        """Generate lock file for reproducible installs"""
        pass
```

### 7. Add GPU Detection & Management
**Impact**: Medium | **Difficulty**: Medium

```python
# src/research_reproducer/gpu_utils.py
import subprocess

def detect_gpu():
    """Detect available GPUs"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                name, memory = line.split(',')
                gpus.append({
                    'name': name.strip(),
                    'memory': memory.strip()
                })
            return gpus
    except FileNotFoundError:
        pass
    return []

def estimate_gpu_memory_needed(model_size_gb):
    """Estimate GPU memory needed"""
    # Add ~20% overhead for activations
    return model_size_gb * 1.2

def recommend_gpu(requirements):
    """Recommend GPU based on requirements"""
    available = detect_gpu()
    # Match requirements to available GPUs
    pass
```

### 8. Add Result Validation
**Impact**: High | **Difficulty**: Medium

```python
# src/research_reproducer/validator.py
class ResultValidator:
    def __init__(self, paper_metadata):
        self.paper_metadata = paper_metadata

    def extract_claims(self):
        """Extract numerical claims from paper abstract"""
        # Use regex or NLP to find claims like "achieves 95% accuracy"
        pass

    def validate_results(self, execution_output):
        """Compare execution results to paper claims"""
        claims = self.extract_claims()
        results = self.parse_output(execution_output)

        validation = {
            'matches': [],
            'mismatches': [],
            'missing': []
        }

        for claim in claims:
            if claim in results:
                if self.values_match(claim, results[claim]):
                    validation['matches'].append(claim)
                else:
                    validation['mismatches'].append(claim)
            else:
                validation['missing'].append(claim)

        return validation
```

---

## 🔥 Advanced Features (1-2 weeks each)

### 9. Web Interface
**Impact**: High | **Difficulty**: High

Use FastAPI + React or Gradio for quick prototype:

```python
# src/research_reproducer/web/app.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import StreamingResponse
import gradio as gr

app = FastAPI()

@app.post("/reproduce")
async def reproduce_paper(paper_source: str, background_tasks: BackgroundTasks):
    # Start reproduction in background
    background_tasks.add_task(run_reproduction, paper_source)
    return {"status": "started", "task_id": "..."}

# Or use Gradio for simplicity
def gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Research Reproducer")

        with gr.Tab("Reproduce"):
            paper_input = gr.Textbox(label="Paper (arXiv ID, PDF, or URL)")
            reproduce_btn = gr.Button("Reproduce")
            output = gr.JSON(label="Results")

            reproduce_btn.click(
                fn=reproduce_wrapper,
                inputs=paper_input,
                outputs=output
            )

        with gr.Tab("Analyze"):
            # Analysis interface
            pass

    return demo

if __name__ == "__main__":
    gradio_interface().launch()
```

### 10. Cloud Execution Integration
**Impact**: High | **Difficulty**: High

**Google Colab Integration**:
```python
# src/research_reproducer/cloud/colab.py
class ColabExecutor:
    def create_notebook(self, analysis):
        """Generate Colab notebook from analysis"""
        notebook = {
            'cells': [
                {
                    'cell_type': 'code',
                    'source': [
                        '# Clone repository\n',
                        f'!git clone {analysis["repo_url"]}\n'
                    ]
                },
                {
                    'cell_type': 'code',
                    'source': [
                        '# Install dependencies\n',
                        f'!pip install -r requirements.txt\n'
                    ]
                },
                # Add more cells
            ]
        }
        return notebook

    def upload_to_drive(self, notebook):
        """Upload notebook to Google Drive"""
        pass
```

**Paperspace Integration**:
```python
# Use Paperspace Gradient API
import requests

def create_paperspace_job(repo_url, command):
    api_key = os.getenv('PAPERSPACE_API_KEY')

    payload = {
        'container': 'paperspace/tensorflow:latest',
        'machineType': 'P4000',
        'command': f'git clone {repo_url} && cd repo && {command}'
    }

    response = requests.post(
        'https://api.paperspace.io/jobs/createJob',
        headers={'X-API-Key': api_key},
        json=payload
    )
    return response.json()
```

### 11. Automated Benchmarking
**Impact**: Medium | **Difficulty**: High

```python
# src/research_reproducer/benchmark.py
class ReproductionBenchmark:
    def __init__(self):
        self.results = []

    def benchmark_paper(self, paper_id):
        """Run full reproduction benchmark"""
        start_time = time.time()

        metrics = {
            'paper_id': paper_id,
            'repo_found': False,
            'env_setup_success': False,
            'execution_success': False,
            'time_to_setup': 0,
            'time_to_execute': 0,
            'total_time': 0,
            'errors': []
        }

        # Run reproduction with detailed timing
        # Record all metrics

        metrics['total_time'] = time.time() - start_time
        self.results.append(metrics)

        return metrics

    def generate_report(self):
        """Generate benchmark report"""
        success_rate = sum(1 for r in self.results if r['execution_success']) / len(self.results)
        avg_time = sum(r['total_time'] for r in self.results) / len(self.results)

        return {
            'total_papers': len(self.results),
            'success_rate': success_rate,
            'avg_setup_time': avg_time,
            'common_errors': self._analyze_errors()
        }
```

### 12. Smart Error Diagnosis
**Impact**: High | **Difficulty**: High

Use LLM to analyze errors and suggest fixes:

```python
# src/research_reproducer/error_analyzer.py
import openai

class ErrorAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

    def analyze_error(self, error_message, context):
        """Use LLM to analyze error and suggest fix"""
        prompt = f"""
        Context: Trying to reproduce research paper
        Error: {error_message}

        Environment: {context['environment']}
        Dependencies: {context['dependencies']}

        Analyze this error and suggest:
        1. Root cause
        2. Specific fix
        3. Alternative approaches
        """

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content

    def common_fixes(self, error_type):
        """Database of common fixes"""
        fixes = {
            'ModuleNotFoundError': [
                'pip install {module}',
                'conda install {module}',
                'Check if package name has changed'
            ],
            'CUDA error': [
                'Install CUDA toolkit',
                'Use CPU version instead',
                'Check CUDA version compatibility'
            ],
            # More patterns...
        }
        return fixes.get(error_type, [])
```

---

## 📊 Performance Optimizations

### 13. Parallel Processing
**Impact**: Medium | **Difficulty**: Medium

```python
# src/research_reproducer/parallel.py
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio

class ParallelReproducer:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    async def reproduce_multiple(self, paper_ids):
        """Reproduce multiple papers in parallel"""
        tasks = [self.reproduce_one(paper_id) for paper_id in paper_ids]
        results = await asyncio.gather(*tasks)
        return results

    def batch_analyze(self, paper_ids):
        """Analyze multiple papers in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.analyze_paper, paper_ids))
        return results
```

### 14. Database Backend
**Impact**: Medium | **Difficulty**: Medium

Store results in SQLite/PostgreSQL:

```python
# src/research_reproducer/database.py
import sqlite3
from datetime import datetime

class ReproductionDatabase:
    def __init__(self, db_path='reproductions.db'):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS reproductions (
                id INTEGER PRIMARY KEY,
                paper_id TEXT,
                paper_title TEXT,
                repo_url TEXT,
                success BOOLEAN,
                execution_time REAL,
                created_at TIMESTAMP
            )
        ''')

    def save_reproduction(self, report):
        self.conn.execute('''
            INSERT INTO reproductions (paper_id, paper_title, repo_url, success, execution_time, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            report['paper'].get('arxiv_id'),
            report['paper'].get('title'),
            report.get('repositories', [{}])[0].get('url'),
            report['success'],
            report.get('execution', {}).get('execution_time'),
            datetime.now()
        ))
        self.conn.commit()

    def get_stats(self):
        cursor = self.conn.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN success THEN 1 ELSE 0 END) as successful,
                AVG(execution_time) as avg_time
            FROM reproductions
        ''')
        return cursor.fetchone()
```

---

## 🎯 Specialized Features

### 15. Dataset Management
**Impact**: Medium | **Difficulty**: Medium

```python
# src/research_reproducer/dataset_manager.py
class DatasetManager:
    def detect_dataset(self, paper_metadata):
        """Detect which dataset paper uses"""
        # Parse abstract and methods for dataset mentions
        common_datasets = {
            'ImageNet', 'COCO', 'MNIST', 'CIFAR-10',
            'WikiText', 'SQuAD', 'GLUE', 'Common Crawl'
        }
        # Return detected datasets
        pass

    def download_dataset(self, dataset_name, target_dir):
        """Download common research datasets"""
        downloaders = {
            'MNIST': self.download_mnist,
            'CIFAR-10': self.download_cifar10,
            # More datasets...
        }

        if dataset_name in downloaders:
            downloaders[dataset_name](target_dir)

    def estimate_dataset_size(self, dataset_name):
        """Estimate disk space needed"""
        sizes = {
            'MNIST': '50 MB',
            'ImageNet': '150 GB',
            'COCO': '25 GB'
        }
        return sizes.get(dataset_name, 'Unknown')
```

### 16. Experiment Tracking
**Impact**: Medium | **Difficulty**: Medium

Integrate with MLflow or Weights & Biases:

```python
# src/research_reproducer/experiment_tracking.py
import mlflow

class ExperimentTracker:
    def __init__(self, tracking_uri='./mlruns'):
        mlflow.set_tracking_uri(tracking_uri)

    def track_reproduction(self, paper_id, orchestrator):
        """Track reproduction as MLflow experiment"""
        with mlflow.start_run(run_name=paper_id):
            # Log parameters
            mlflow.log_param('paper_id', paper_id)
            mlflow.log_param('env_type', orchestrator.env_info['type'])

            # Log metrics
            mlflow.log_metric('success', 1 if orchestrator.report['success'] else 0)
            mlflow.log_metric('execution_time', orchestrator.report.get('execution', {}).get('execution_time', 0))

            # Log artifacts
            mlflow.log_artifact(orchestrator.session_dir / 'report.json')
```

---

## 🚢 Deployment & Distribution

### 17. Docker Image
**Impact**: High | **Difficulty**: Low

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ /app/src/
COPY setup.py .

RUN pip install -e .

ENTRYPOINT ["research-reproduce"]
```

```bash
# Build and run
docker build -t research-reproducer .
docker run -v $(pwd)/reproductions:/app/reproductions research-reproducer reproduce 2301.12345
```

### 18. PyPI Package
**Impact**: High | **Difficulty**: Low

```bash
# setup.py already configured
# To publish:
python setup.py sdist bdist_wheel
twine upload dist/*

# Users can then:
pip install research-reproducer
```

---

## 📝 Documentation Improvements

- [ ] Add video tutorials
- [ ] Create interactive Jupyter notebook examples
- [ ] Add troubleshooting flowcharts
- [ ] Create comparison benchmarks vs manual reproduction
- [ ] Add case studies of successful reproductions

---

## Priority Recommendation

Start with this order for maximum impact:

1. **Testing** (improves reliability)
2. **Caching + Retry** (improves user experience)
3. **More paper sources** (expands usefulness)
4. **Web interface** (lowers barrier to entry)
5. **Cloud execution** (solves resource constraints)
6. **Result validation** (adds scientific value)

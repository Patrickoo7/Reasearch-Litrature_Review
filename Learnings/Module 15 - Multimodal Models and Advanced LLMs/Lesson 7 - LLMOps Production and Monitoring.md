# Lesson 7: LLMOps - Production Deployment and Monitoring

## Learning Objectives

By the end of this lesson, you will be able to:

✅ Understand the differences between traditional MLOps and LLMOps
✅ Implement prompt versioning and management systems
✅ Track token usage and optimize LLM costs
✅ Monitor latency, throughput, and quality metrics
✅ Detect hallucinations, toxicity, and PII leakage
✅ Implement A/B testing for prompts and models
✅ Deploy caching strategies for cost reduction
✅ Build observability systems with LangSmith, W&B, Helicone
✅ Apply deployment patterns: blue-green, canary, feature flags
✅ Create production-ready LLM applications with monitoring

---

## Prerequisites

- **Module 7B**: NLP fundamentals, transformers, prompt engineering
- **Module 10**: Production ML & MLOps basics
- **Module 15 Lessons 1-6**: LLM architectures, inference optimization
- **Module 16**: RAG systems (for production RAG monitoring)
- Python experience with FastAPI, databases, monitoring tools

---

## 1. Introduction: LLMOps vs Traditional MLOps

### What is LLMOps?

**LLMOps** (Large Language Model Operations) is the practice of deploying, monitoring, and maintaining LLM-based applications in production. It extends traditional MLOps with LLM-specific challenges.

### Key Differences: MLOps vs LLMOps

| Aspect | Traditional MLOps | LLMOps |
|--------|------------------|---------|
| **Model Updates** | Retrain entire model | Prompt updates (no retraining) |
| **Versioning** | Model weights + code | Prompts + model versions + RAG data |
| **Cost Drivers** | Compute for training | Token usage (API calls) |
| **Latency** | Milliseconds (inference) | Seconds (generation) |
| **Quality Metrics** | Accuracy, F1, AUC | Hallucination rate, toxicity, relevance |
| **Debugging** | Feature importance, SHAP | Prompt inspection, trace debugging |
| **A/B Testing** | Model versions | Prompts, models, temperature settings |
| **Data Drift** | Feature distribution shift | Semantic drift, topic shift |

### Unique LLMOps Challenges

1. **Non-deterministic outputs**: Same prompt → different responses
2. **Token costs**: Every inference costs money (unlike traditional ML)
3. **Prompt engineering**: Business logic in natural language
4. **Hallucinations**: Model generates false information
5. **Context windows**: Limited input size (2K-128K tokens)
6. **Latency**: Slow generation (1-100 tokens/sec)
7. **Privacy**: Risk of leaking training data or user PII

---

## 2. Prompt Versioning and Management

### Why Prompt Versioning Matters

Prompts are **code** in LLM systems. Changes to prompts can:
- Improve accuracy dramatically
- Break existing functionality
- Increase costs (longer prompts)
- Change user experience

### Example: Prompt Evolution

```python
# Version 1.0 - Basic (poor quality)
prompt_v1 = "Summarize this text: {text}"

# Version 2.0 - Better instructions (improved)
prompt_v2 = """
Summarize the following text in 2-3 sentences.
Focus on the main points and key takeaways.

Text: {text}

Summary:
"""

# Version 3.0 - Production-ready (best)
prompt_v3 = """
You are a professional summarizer. Create a concise 2-3 sentence summary of the text below.

Requirements:
- Focus on main ideas and key facts
- Use clear, simple language
- Do not add opinions or interpretations
- If the text is unclear, say "Unable to summarize"

Text: {text}

Summary:
"""
```

### Prompt Versioning System

```python
from datetime import datetime
from typing import Dict, List
import json

class PromptVersion:
    """Versioned prompt with metadata."""

    def __init__(self, template: str, version: str, metadata: Dict = None):
        self.template = template
        self.version = version
        self.created_at = datetime.now()
        self.metadata = metadata or {}
        self.performance_metrics = {}

    def render(self, **kwargs) -> str:
        """Render prompt with variables."""
        return self.template.format(**kwargs)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'template': self.template,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics
        }

class PromptRegistry:
    """Central registry for managing prompts."""

    def __init__(self, storage_path: str = 'prompts.json'):
        self.storage_path = storage_path
        self.prompts: Dict[str, List[PromptVersion]] = {}
        self.load()

    def register(self, name: str, prompt: PromptVersion):
        """Register a new prompt version."""
        if name not in self.prompts:
            self.prompts[name] = []
        self.prompts[name].append(prompt)
        self.save()
        print(f"✅ Registered {name} v{prompt.version}")

    def get(self, name: str, version: str = None) -> PromptVersion:
        """Get specific version or latest."""
        if name not in self.prompts:
            raise ValueError(f"Prompt '{name}' not found")

        versions = self.prompts[name]
        if version is None:
            return versions[-1]  # Latest version

        for p in versions:
            if p.version == version:
                return p
        raise ValueError(f"Version '{version}' not found for '{name}'")

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a prompt."""
        if name not in self.prompts:
            return []
        return [p.version for p in self.prompts[name]]

    def save(self):
        """Save to disk."""
        data = {
            name: [p.to_dict() for p in versions]
            for name, versions in self.prompts.items()
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            # Reconstruct PromptVersion objects
            # (simplified - would need full deserialization)
        except FileNotFoundError:
            pass

# Usage Example
registry = PromptRegistry()

# Register prompts
summarize_v1 = PromptVersion(
    template=prompt_v1,
    version="1.0.0",
    metadata={"author": "alice", "purpose": "initial"}
)
registry.register("summarize", summarize_v1)

summarize_v3 = PromptVersion(
    template=prompt_v3,
    version="3.0.0",
    metadata={"author": "bob", "purpose": "production-ready"}
)
registry.register("summarize", summarize_v3)

# Get latest version
latest = registry.get("summarize")
print(f"Using version: {latest.version}")

# Get specific version (for rollback)
old_version = registry.get("summarize", version="1.0.0")
```

### Prompt Templates with Jinja2

```python
from jinja2 import Template

class PromptTemplate:
    """Advanced prompt templating with Jinja2."""

    def __init__(self, template: str):
        self.template = Template(template)

    def render(self, **kwargs) -> str:
        return self.template.render(**kwargs)

# Conditional logic in prompts
template_text = """
You are a {{ role }}.

{% if context %}
Context: {{ context }}
{% endif %}

Task: {{ task }}

{% if examples %}
Examples:
{% for example in examples %}
- Input: {{ example.input }}
  Output: {{ example.output }}
{% endfor %}
{% endif %}

Now complete the task:
Input: {{ input }}
Output:
"""

prompt_template = PromptTemplate(template_text)

# Render with different contexts
output = prompt_template.render(
    role="helpful assistant",
    task="Translate to French",
    input="Hello, how are you?",
    examples=[
        {"input": "Good morning", "output": "Bonjour"},
        {"input": "Thank you", "output": "Merci"}
    ]
)
print(output)
```

---

## 3. Token Usage Tracking and Cost Monitoring

### Understanding Token Costs

```python
# Token pricing (as of 2024 - example)
PRICING = {
    "gpt-4": {
        "input": 0.03 / 1000,   # $0.03 per 1K tokens
        "output": 0.06 / 1000   # $0.06 per 1K tokens
    },
    "gpt-3.5-turbo": {
        "input": 0.0015 / 1000,
        "output": 0.002 / 1000
    },
    "claude-3-opus": {
        "input": 0.015 / 1000,
        "output": 0.075 / 1000
    }
}

def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for an LLM call."""
    pricing = PRICING.get(model, PRICING["gpt-3.5-turbo"])
    input_cost = input_tokens * pricing["input"]
    output_cost = output_tokens * pricing["output"]
    return input_cost + output_cost

# Example
cost = calculate_cost("gpt-4", input_tokens=1500, output_tokens=500)
print(f"Cost: ${cost:.4f}")  # $0.0750
```

### Token Usage Tracker

```python
import time
from collections import defaultdict
from datetime import datetime, timedelta
import tiktoken

class TokenTracker:
    """Track token usage and costs across LLM calls."""

    def __init__(self):
        self.usage_history = []
        self.daily_usage = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0})
        self.user_usage = defaultdict(lambda: {"input": 0, "output": 0, "cost": 0})

    def log_usage(self,
                  model: str,
                  input_tokens: int,
                  output_tokens: int,
                  user_id: str = None,
                  metadata: dict = None):
        """Log a single LLM call."""
        cost = calculate_cost(model, input_tokens, output_tokens)

        record = {
            "timestamp": datetime.now(),
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
            "user_id": user_id,
            "metadata": metadata or {}
        }

        self.usage_history.append(record)

        # Update daily totals
        date = datetime.now().date()
        self.daily_usage[date]["input"] += input_tokens
        self.daily_usage[date]["output"] += output_tokens
        self.daily_usage[date]["cost"] += cost

        # Update user totals
        if user_id:
            self.user_usage[user_id]["input"] += input_tokens
            self.user_usage[user_id]["output"] += output_tokens
            self.user_usage[user_id]["cost"] += cost

    def get_daily_cost(self, days: int = 7) -> dict:
        """Get costs for last N days."""
        cutoff = datetime.now().date() - timedelta(days=days)
        return {
            date: stats["cost"]
            for date, stats in self.daily_usage.items()
            if date >= cutoff
        }

    def get_top_users(self, top_n: int = 10) -> list:
        """Get top users by cost."""
        users = [
            (user_id, stats["cost"])
            for user_id, stats in self.user_usage.items()
        ]
        return sorted(users, key=lambda x: x[1], reverse=True)[:top_n]

    def estimate_monthly_cost(self) -> float:
        """Estimate monthly cost based on recent usage."""
        recent_days = 7
        recent_cost = sum(self.get_daily_cost(recent_days).values())
        daily_avg = recent_cost / recent_days
        return daily_avg * 30

    def alert_if_over_budget(self, daily_budget: float):
        """Alert if today's cost exceeds budget."""
        today = datetime.now().date()
        today_cost = self.daily_usage[today]["cost"]

        if today_cost > daily_budget:
            print(f"⚠️ ALERT: Daily cost ${today_cost:.2f} exceeds budget ${daily_budget:.2f}")
            return True
        return False

# Usage example
tracker = TokenTracker()

# Simulate LLM calls
tracker.log_usage("gpt-4", 1000, 500, user_id="user_123")
tracker.log_usage("gpt-3.5-turbo", 2000, 300, user_id="user_456")
tracker.log_usage("gpt-4", 1500, 700, user_id="user_123")

# Check costs
print(f"Estimated monthly cost: ${tracker.estimate_monthly_cost():.2f}")
print("Top users:", tracker.get_top_users(top_n=5))

# Budget alert
tracker.alert_if_over_budget(daily_budget=10.0)
```

### Real-time Cost Dashboard

```python
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def plot_daily_costs(tracker: TokenTracker, days: int = 30):
    """Visualize daily costs."""
    daily_costs = tracker.get_daily_cost(days=days)

    dates = list(daily_costs.keys())
    costs = list(daily_costs.values())

    plt.figure(figsize=(12, 6))
    plt.bar(dates, costs, color='steelblue')
    plt.xlabel('Date')
    plt.ylabel('Cost ($)')
    plt.title('Daily LLM Costs')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('daily_llm_costs.png', dpi=150)
    plt.close()

    print("✅ Cost dashboard saved to daily_llm_costs.png")
```

---

## 4. Latency and Throughput Monitoring

### Latency Tracker

```python
import time
from statistics import mean, median, stdev

class LatencyTracker:
    """Track LLM response latencies."""

    def __init__(self):
        self.latencies = []
        self.slow_requests = []  # Requests over threshold

    def track(self, start_time: float, end_time: float,
              metadata: dict = None, slow_threshold: float = 5.0):
        """Track a single request latency."""
        latency = end_time - start_time

        record = {
            "latency": latency,
            "timestamp": datetime.now(),
            "metadata": metadata or {}
        }

        self.latencies.append(latency)

        if latency > slow_threshold:
            self.slow_requests.append(record)
            print(f"⚠️ Slow request: {latency:.2f}s (threshold: {slow_threshold}s)")

    def get_stats(self) -> dict:
        """Get latency statistics."""
        if not self.latencies:
            return {}

        return {
            "mean": mean(self.latencies),
            "median": median(self.latencies),
            "std": stdev(self.latencies) if len(self.latencies) > 1 else 0,
            "min": min(self.latencies),
            "max": max(self.latencies),
            "p95": sorted(self.latencies)[int(len(self.latencies) * 0.95)],
            "p99": sorted(self.latencies)[int(len(self.latencies) * 0.99)],
            "count": len(self.latencies),
            "slow_count": len(self.slow_requests)
        }

    def print_stats(self):
        """Print latency statistics."""
        stats = self.get_stats()
        print("\n📊 Latency Statistics:")
        print(f"  Mean: {stats['mean']:.2f}s")
        print(f"  Median: {stats['median']:.2f}s")
        print(f"  P95: {stats['p95']:.2f}s")
        print(f"  P99: {stats['p99']:.2f}s")
        print(f"  Min/Max: {stats['min']:.2f}s / {stats['max']:.2f}s")
        print(f"  Slow requests: {stats['slow_count']}/{stats['count']}")

# Usage with context manager
class TimedLLMCall:
    """Context manager for timing LLM calls."""

    def __init__(self, tracker: LatencyTracker, metadata: dict = None):
        self.tracker = tracker
        self.metadata = metadata
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        self.tracker.track(self.start_time, end_time, self.metadata)

# Example usage
latency_tracker = LatencyTracker()

# Simulate LLM calls
for i in range(100):
    with TimedLLMCall(latency_tracker, metadata={"request_id": i}):
        time.sleep(0.5 + (i % 10) * 0.1)  # Simulate varying latencies

latency_tracker.print_stats()
```

### Throughput Monitoring

```python
from collections import deque
from threading import Lock

class ThroughputMonitor:
    """Monitor requests per second/minute."""

    def __init__(self, window_size: int = 60):
        self.window_size = window_size  # seconds
        self.timestamps = deque()
        self.lock = Lock()

    def record_request(self):
        """Record a new request."""
        with self.lock:
            now = time.time()
            self.timestamps.append(now)

            # Remove old timestamps outside window
            cutoff = now - self.window_size
            while self.timestamps and self.timestamps[0] < cutoff:
                self.timestamps.popleft()

    def get_rps(self) -> float:
        """Get current requests per second."""
        with self.lock:
            if not self.timestamps:
                return 0.0

            time_span = time.time() - self.timestamps[0]
            if time_span == 0:
                return 0.0

            return len(self.timestamps) / time_span

    def get_rpm(self) -> float:
        """Get requests per minute."""
        return self.get_rps() * 60

# Usage
throughput = ThroughputMonitor(window_size=60)

# Simulate requests
for _ in range(100):
    throughput.record_request()
    time.sleep(0.1)

print(f"Current RPS: {throughput.get_rps():.2f}")
print(f"Current RPM: {throughput.get_rpm():.2f}")
```

---

## 5. Quality Metrics: Hallucination, Toxicity, PII Detection

### Hallucination Detection

```python
from typing import List

class HallucinationDetector:
    """Detect potential hallucinations in LLM outputs."""

    def __init__(self):
        self.hallucination_patterns = [
            "I don't have information",
            "I cannot verify",
            "I'm not sure",
            "I don't know",
            "I cannot confirm"
        ]

    def check_uncertainty(self, text: str) -> bool:
        """Check if LLM expresses uncertainty."""
        text_lower = text.lower()
        return any(pattern.lower() in text_lower
                  for pattern in self.hallucination_patterns)

    def check_factual_consistency(self,
                                  answer: str,
                                  context: str) -> float:
        """
        Check if answer is consistent with context.
        Returns confidence score 0-1.
        (In production, use NLI models like BART-MNLI)
        """
        # Simplified: Check if key entities in answer appear in context
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        overlap = len(answer_words & context_words)
        score = overlap / len(answer_words) if answer_words else 0

        return min(score, 1.0)

    def detect(self, answer: str, context: str = None) -> dict:
        """Comprehensive hallucination check."""
        results = {
            "expresses_uncertainty": self.check_uncertainty(answer),
            "factual_consistency": None,
            "likely_hallucination": False
        }

        if context:
            consistency = self.check_factual_consistency(answer, context)
            results["factual_consistency"] = consistency

            # Flag if low consistency and no uncertainty expressed
            if consistency < 0.3 and not results["expresses_uncertainty"]:
                results["likely_hallucination"] = True

        return results

# Usage
detector = HallucinationDetector()

context = "The Eiffel Tower is located in Paris, France. It was built in 1889."
answer1 = "The Eiffel Tower is in Paris and was completed in 1889."
answer2 = "The Eiffel Tower is in London and was built in 1850."

print("Answer 1:", detector.detect(answer1, context))
print("Answer 2:", detector.detect(answer2, context))
```

### Toxicity Detection

```python
# Using Perspective API or local models
class ToxicityDetector:
    """Detect toxic content in LLM outputs."""

    def __init__(self):
        # In production, use Perspective API or Detoxify library
        self.toxic_keywords = [
            "hate", "violent", "offensive", "discriminatory"
            # ... much longer list in production
        ]

    def detect(self, text: str) -> dict:
        """
        Detect toxicity.
        Returns: {
            'is_toxic': bool,
            'toxicity_score': float 0-1,
            'categories': list
        }
        """
        text_lower = text.lower()

        # Simplified keyword matching (use ML models in production)
        toxic_found = [
            word for word in self.toxic_keywords
            if word in text_lower
        ]

        is_toxic = len(toxic_found) > 0
        toxicity_score = min(len(toxic_found) * 0.2, 1.0)

        return {
            "is_toxic": is_toxic,
            "toxicity_score": toxicity_score,
            "toxic_keywords": toxic_found
        }

# Better: Use Detoxify library
# pip install detoxify
try:
    from detoxify import Detoxify

    class ProductionToxicityDetector:
        def __init__(self):
            self.model = Detoxify('original')

        def detect(self, text: str) -> dict:
            results = self.model.predict(text)
            return {
                "toxicity": results['toxicity'],
                "severe_toxicity": results['severe_toxicity'],
                "obscene": results['obscene'],
                "threat": results['threat'],
                "insult": results['insult'],
                "identity_attack": results['identity_attack']
            }
except ImportError:
    print("Install detoxify: pip install detoxify")
```

### PII Detection

```python
import re

class PIIDetector:
    """Detect Personally Identifiable Information."""

    def __init__(self):
        # Regex patterns for common PII
        self.patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }

    def detect(self, text: str) -> dict:
        """Detect PII in text."""
        findings = {}

        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                findings[pii_type] = matches

        return {
            "contains_pii": bool(findings),
            "pii_types": list(findings.keys()),
            "findings": findings
        }

    def redact(self, text: str) -> str:
        """Redact PII from text."""
        redacted = text

        for pii_type, pattern in self.patterns.items():
            redacted = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', redacted)

        return redacted

# Usage
pii_detector = PIIDetector()

text = "Contact me at john.doe@email.com or call 555-123-4567"
result = pii_detector.detect(text)
print("PII detected:", result)
print("Redacted:", pii_detector.redact(text))
```

---

## 6. Caching Strategies for Cost Reduction

### Semantic Caching

```python
import hashlib
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """
    Cache LLM responses based on semantic similarity.
    Saves costs by returning cached responses for similar queries.
    """

    def __init__(self, similarity_threshold: float = 0.95):
        self.cache = {}  # query_embedding -> response
        self.embeddings = []
        self.responses = []
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> np.ndarray:
        """Get embedding for text."""
        return self.model.encode(text, convert_to_numpy=True)

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str) -> Optional[str]:
        """Get cached response if similar query exists."""
        if not self.embeddings:
            self.misses += 1
            return None

        query_emb = self._embed(query)

        # Find most similar cached query
        similarities = [
            self._cosine_similarity(query_emb, cached_emb)
            for cached_emb in self.embeddings
        ]

        max_sim = max(similarities)

        if max_sim >= self.similarity_threshold:
            # Cache hit!
            idx = similarities.index(max_sim)
            self.hits += 1
            print(f"✅ Cache HIT (similarity: {max_sim:.3f})")
            return self.responses[idx]

        self.misses += 1
        return None

    def set(self, query: str, response: str):
        """Cache a query-response pair."""
        query_emb = self._embed(query)
        self.embeddings.append(query_emb)
        self.responses.append(response)

    def get_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

# Usage
cache = SemanticCache(similarity_threshold=0.90)

# First query - cache miss
query1 = "What is the capital of France?"
response1 = cache.get(query1)
if response1 is None:
    response1 = "The capital of France is Paris."  # Call LLM
    cache.set(query1, response1)

# Similar query - cache hit!
query2 = "What's the capital city of France?"
response2 = cache.get(query2)
if response2 is None:
    response2 = "..."  # Would call LLM, but we get cache hit

print(f"Cache hit rate: {cache.get_hit_rate():.1%}")
```

### Exact Match Caching

```python
from functools import lru_cache
import hashlib

class ExactMatchCache:
    """Simple exact-match cache for LLM responses."""

    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}

    def _hash(self, query: str, model: str, temperature: float) -> str:
        """Create cache key."""
        key = f"{query}|{model}|{temperature}"
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, query: str, model: str, temperature: float) -> Optional[str]:
        """Get cached response."""
        key = self._hash(query, model, temperature)

        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]

        return None

    def set(self, query: str, model: str, temperature: float, response: str):
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = min(self.access_count, key=self.access_count.get)
            del self.cache[lru_key]
            del self.access_count[lru_key]

        key = self._hash(query, model, temperature)
        self.cache[key] = response
        self.access_count[key] = 1
```

---

## 7. A/B Testing for Prompts and Models

### Prompt A/B Testing Framework

```python
import random
from typing import Callable, Dict, List
from dataclasses import dataclass

@dataclass
class Variant:
    """A/B test variant."""
    name: str
    prompt_template: str
    model: str
    temperature: float
    weight: float = 1.0  # Traffic allocation weight

class ABTestFramework:
    """A/B testing for prompts and models."""

    def __init__(self):
        self.variants: Dict[str, List[Variant]] = {}
        self.results = []

    def create_test(self, test_name: str, variants: List[Variant]):
        """Create a new A/B test."""
        # Normalize weights
        total_weight = sum(v.weight for v in variants)
        for v in variants:
            v.weight = v.weight / total_weight

        self.variants[test_name] = variants
        print(f"✅ Created test '{test_name}' with {len(variants)} variants")

    def select_variant(self, test_name: str) -> Variant:
        """Select a variant based on weights."""
        variants = self.variants[test_name]

        # Weighted random selection
        rand = random.random()
        cumulative = 0.0

        for variant in variants:
            cumulative += variant.weight
            if rand <= cumulative:
                return variant

        return variants[-1]  # Fallback

    def record_result(self, test_name: str, variant_name: str,
                     user_feedback: float, latency: float, cost: float):
        """Record test result."""
        self.results.append({
            "test": test_name,
            "variant": variant_name,
            "feedback": user_feedback,  # 0-1 score
            "latency": latency,
            "cost": cost
        })

    def analyze(self, test_name: str) -> Dict:
        """Analyze A/B test results."""
        test_results = [r for r in self.results if r["test"] == test_name]

        if not test_results:
            return {}

        # Group by variant
        variants = {}
        for r in test_results:
            v = r["variant"]
            if v not in variants:
                variants[v] = {"feedback": [], "latency": [], "cost": []}

            variants[v]["feedback"].append(r["feedback"])
            variants[v]["latency"].append(r["latency"])
            variants[v]["cost"].append(r["cost"])

        # Calculate metrics
        analysis = {}
        for v, data in variants.items():
            analysis[v] = {
                "avg_feedback": mean(data["feedback"]),
                "avg_latency": mean(data["latency"]),
                "avg_cost": mean(data["cost"]),
                "count": len(data["feedback"])
            }

        return analysis

    def get_winner(self, test_name: str, metric: str = "feedback") -> str:
        """Determine winning variant."""
        analysis = self.analyze(test_name)

        if not analysis:
            return None

        key_func = lambda x: x[1][f"avg_{metric}"]
        winner = max(analysis.items(), key=key_func)

        return winner[0]

# Usage Example
ab_test = ABTestFramework()

# Create test with different prompts
variants = [
    Variant(
        name="concise",
        prompt_template="Summarize in 1 sentence: {text}",
        model="gpt-3.5-turbo",
        temperature=0.3,
        weight=0.5  # 50% traffic
    ),
    Variant(
        name="detailed",
        prompt_template="Provide a detailed summary with key points: {text}",
        model="gpt-4",
        temperature=0.5,
        weight=0.5  # 50% traffic
    )
]

ab_test.create_test("summary_test", variants)

# Simulate user interactions
for i in range(100):
    variant = ab_test.select_variant("summary_test")

    # Simulate metrics
    feedback = random.uniform(0.6, 0.9)
    latency = random.uniform(1.0, 3.0)
    cost = random.uniform(0.001, 0.01)

    ab_test.record_result("summary_test", variant.name, feedback, latency, cost)

# Analyze results
results = ab_test.analyze("summary_test")
for variant, metrics in results.items():
    print(f"\n{variant}:")
    print(f"  Avg Feedback: {metrics['avg_feedback']:.2f}")
    print(f"  Avg Latency: {metrics['avg_latency']:.2f}s")
    print(f"  Avg Cost: ${metrics['avg_cost']:.4f}")
    print(f"  Samples: {metrics['count']}")

winner = ab_test.get_winner("summary_test", metric="feedback")
print(f"\n🏆 Winner: {winner}")
```

---

## 8. Observability: LangSmith, Weights & Biases, Helicone

### LangSmith Integration

```python
# LangSmith: Official LangChain monitoring tool
# pip install langsmith langchain

from langsmith import Client
from langchain.callbacks.tracers import LangChainTracer
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

# Initialize LangSmith
client = Client(api_key="your-langsmith-api-key")
tracer = LangChainTracer(project_name="production-llm-app")

# Use with LangChain
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    callbacks=[tracer]  # Enable tracing
)

# Make calls - automatically logged to LangSmith
response = llm([HumanMessage(content="What is machine learning?")])

# View in LangSmith dashboard:
# - Full conversation traces
# - Token usage per call
# - Latency breakdown
# - Cost tracking
# - Error rates
```

### Weights & Biases Integration

```python
# W&B for LLM monitoring
# pip install wandb

import wandb

# Initialize W&B
wandb.init(project="llm-production", name="gpt-4-app")

class WandBLogger:
    """Log LLM metrics to Weights & Biases."""

    def __init__(self, project: str):
        self.run = wandb.init(project=project)

    def log_llm_call(self,
                     prompt: str,
                     response: str,
                     model: str,
                     tokens: int,
                     latency: float,
                     cost: float,
                     quality_score: float = None):
        """Log a single LLM call."""

        wandb.log({
            "model": model,
            "tokens": tokens,
            "latency": latency,
            "cost": cost,
            "quality_score": quality_score or 0,
            "timestamp": time.time()
        })

        # Log example conversations as tables
        wandb.log({
            "examples": wandb.Table(
                columns=["prompt", "response", "model", "tokens"],
                data=[[prompt, response, model, tokens]]
            )
        })

    def log_daily_summary(self,
                         total_cost: float,
                         total_tokens: int,
                         avg_latency: float):
        """Log daily aggregated metrics."""
        wandb.log({
            "daily/total_cost": total_cost,
            "daily/total_tokens": total_tokens,
            "daily/avg_latency": avg_latency
        })

# Usage
logger = WandBLogger(project="llm-app")
logger.log_llm_call(
    prompt="Explain quantum computing",
    response="Quantum computing uses...",
    model="gpt-4",
    tokens=500,
    latency=2.3,
    cost=0.015,
    quality_score=0.9
)
```

### Helicone Integration

```python
# Helicone: LLM observability platform
# Just modify your OpenAI base URL!

import openai

# Add Helicone proxy
openai.api_base = "https://oai.hel.icone.ai/v1"
openai.api_key = "your-openai-key"

# Add Helicone API key in headers
headers = {
    "Helicone-Auth": "Bearer your-helicone-key",
    "Helicone-Property-Environment": "production",
    "Helicone-Property-User": "user_123"
}

# Make calls - automatically logged to Helicone
response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Hello"}],
    headers=headers
)

# Helicone dashboard shows:
# - Request/response logs
# - Cost breakdown
# - Latency percentiles
# - Error tracking
# - User segmentation
```

---

## 9. Deployment Patterns: Blue-Green, Canary, Feature Flags

### Blue-Green Deployment

```python
class BlueGreenDeployment:
    """
    Blue-Green deployment for LLM applications.
    Two identical environments (blue/green).
    Switch traffic instantly between them.
    """

    def __init__(self):
        self.environments = {
            "blue": {
                "model": "gpt-3.5-turbo",
                "prompt_version": "v1.0",
                "active": True
            },
            "green": {
                "model": "gpt-4",
                "prompt_version": "v2.0",
                "active": False
            }
        }
        self.current = "blue"

    def get_active_config(self) -> dict:
        """Get configuration for active environment."""
        return self.environments[self.current]

    def switch_traffic(self):
        """Switch traffic to other environment."""
        old = self.current
        self.current = "green" if self.current == "blue" else "blue"

        # Update active flags
        self.environments[old]["active"] = False
        self.environments[self.current]["active"] = True

        print(f"✅ Switched traffic: {old} → {self.current}")
        print(f"   Now using: {self.environments[self.current]}")

    def rollback(self):
        """Instant rollback to previous environment."""
        print("⚠️ Rolling back!")
        self.switch_traffic()

# Usage
deployment = BlueGreenDeployment()

# Current traffic goes to blue (GPT-3.5)
config = deployment.get_active_config()
print("Active:", config)

# Deploy new version to green (GPT-4)
# Test green environment
# When ready, switch traffic
deployment.switch_traffic()

# If issues detected, rollback instantly
deployment.rollback()
```

### Canary Deployment

```python
import random

class CanaryDeployment:
    """
    Canary deployment: Gradually shift traffic to new version.
    Monitor metrics, auto-rollback if issues detected.
    """

    def __init__(self):
        self.versions = {
            "stable": {
                "model": "gpt-3.5-turbo",
                "prompt_version": "v1.0",
                "traffic_pct": 100
            },
            "canary": {
                "model": "gpt-4",
                "prompt_version": "v2.0",
                "traffic_pct": 0
            }
        }
        self.canary_metrics = {"errors": 0, "success": 0}

    def select_version(self) -> str:
        """Select version based on traffic percentage."""
        canary_pct = self.versions["canary"]["traffic_pct"]

        if random.random() < (canary_pct / 100):
            return "canary"
        return "stable"

    def increase_canary_traffic(self, increment: int = 10):
        """Gradually increase canary traffic."""
        current = self.versions["canary"]["traffic_pct"]
        new_pct = min(current + increment, 100)

        self.versions["canary"]["traffic_pct"] = new_pct
        self.versions["stable"]["traffic_pct"] = 100 - new_pct

        print(f"📈 Canary traffic: {current}% → {new_pct}%")

    def record_result(self, version: str, success: bool):
        """Record canary result."""
        if version == "canary":
            if success:
                self.canary_metrics["success"] += 1
            else:
                self.canary_metrics["errors"] += 1

    def check_canary_health(self) -> bool:
        """Check if canary is healthy."""
        total = (self.canary_metrics["success"] +
                self.canary_metrics["errors"])

        if total < 100:  # Need enough samples
            return True

        error_rate = self.canary_metrics["errors"] / total

        if error_rate > 0.05:  # 5% error threshold
            print(f"⚠️ Canary error rate too high: {error_rate:.1%}")
            return False

        return True

    def auto_rollback(self):
        """Rollback canary if unhealthy."""
        if not self.check_canary_health():
            print("🔴 Auto-rollback triggered!")
            self.versions["canary"]["traffic_pct"] = 0
            self.versions["stable"]["traffic_pct"] = 100
            return True
        return False

# Usage: Gradual rollout
canary = CanaryDeployment()

# Stage 1: 10% traffic to canary
canary.increase_canary_traffic(10)

# Simulate requests and monitor
for i in range(200):
    version = canary.select_version()
    success = random.random() > 0.02  # 2% error rate
    canary.record_result(version, success)

# Check health before increasing
if canary.check_canary_health():
    # Stage 2: Increase to 25%
    canary.increase_canary_traffic(15)
else:
    canary.auto_rollback()
```

### Feature Flags

```python
class FeatureFlagManager:
    """
    Feature flags for LLM applications.
    Enable/disable features without deployment.
    """

    def __init__(self):
        self.flags = {
            "use_gpt4": False,
            "enable_rag": True,
            "use_semantic_cache": True,
            "enable_pii_detection": True,
            "use_streaming": False
        }
        self.user_overrides = {}  # user_id -> flags

    def is_enabled(self, flag: str, user_id: str = None) -> bool:
        """Check if feature is enabled."""
        # Check user-specific override
        if user_id and user_id in self.user_overrides:
            if flag in self.user_overrides[user_id]:
                return self.user_overrides[user_id][flag]

        # Fall back to global flag
        return self.flags.get(flag, False)

    def enable(self, flag: str):
        """Enable feature globally."""
        self.flags[flag] = True
        print(f"✅ Enabled: {flag}")

    def disable(self, flag: str):
        """Disable feature globally."""
        self.flags[flag] = False
        print(f"❌ Disabled: {flag}")

    def enable_for_user(self, flag: str, user_id: str):
        """Enable feature for specific user (beta testing)."""
        if user_id not in self.user_overrides:
            self.user_overrides[user_id] = {}

        self.user_overrides[user_id][flag] = True
        print(f"✅ Enabled {flag} for user {user_id}")

    def percentage_rollout(self, flag: str, percentage: int):
        """Enable feature for percentage of users."""
        # In production, use hash of user_id for deterministic assignment
        import hashlib

        def should_enable(user_id: str) -> bool:
            hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
            return (hash_val % 100) < percentage

        # Store rollout config
        self.flags[f"{flag}_rollout_pct"] = percentage

# Usage
flags = FeatureFlagManager()

# Gradually enable GPT-4
flags.enable_for_user("use_gpt4", "beta_user_1")
flags.enable_for_user("use_gpt4", "beta_user_2")

# Check flags in application
def generate_response(user_id: str, prompt: str):
    if flags.is_enabled("use_gpt4", user_id):
        model = "gpt-4"
    else:
        model = "gpt-3.5-turbo"

    # Use RAG if enabled
    if flags.is_enabled("enable_rag", user_id):
        # Retrieve context
        pass

    # Generate response
    return f"Using {model} for {user_id}"
```

---

## 10. Production Example: Complete LLMOps System

```python
# Complete production-ready LLM application with monitoring

class ProductionLLMApp:
    """
    Production LLM application with full observability.
    Includes: versioning, monitoring, caching, quality checks, deployment.
    """

    def __init__(self, config: dict):
        self.config = config

        # Initialize components
        self.prompt_registry = PromptRegistry()
        self.token_tracker = TokenTracker()
        self.latency_tracker = LatencyTracker()
        self.semantic_cache = SemanticCache()
        self.hallucination_detector = HallucinationDetector()
        self.pii_detector = PIIDetector()
        self.feature_flags = FeatureFlagManager()
        self.ab_test = ABTestFramework()

        # Monitoring
        self.alert_thresholds = {
            "daily_cost": 100.0,
            "error_rate": 0.05,
            "p95_latency": 10.0
        }

    def generate(self,
                prompt_name: str,
                user_id: str,
                variables: dict,
                context: str = None) -> dict:
        """
        Generate LLM response with full monitoring.
        """
        start_time = time.time()

        try:
            # 1. Get prompt version
            prompt = self.prompt_registry.get(prompt_name)
            rendered_prompt = prompt.render(**variables)

            # 2. Check semantic cache
            cached_response = self.semantic_cache.get(rendered_prompt)
            if cached_response:
                return {
                    "response": cached_response,
                    "cached": True,
                    "latency": time.time() - start_time
                }

            # 3. Select model (A/B test or feature flag)
            if self.feature_flags.is_enabled("use_gpt4", user_id):
                model = "gpt-4"
            else:
                model = "gpt-3.5-turbo"

            # 4. Call LLM (simulated)
            response_text = f"Simulated response from {model}"
            input_tokens = len(rendered_prompt.split())
            output_tokens = len(response_text.split())

            # 5. Quality checks
            hallucination = self.hallucination_detector.detect(
                response_text, context
            )

            pii_check = self.pii_detector.detect(response_text)
            if pii_check["contains_pii"]:
                print("⚠️ PII detected! Redacting...")
                response_text = self.pii_detector.redact(response_text)

            # 6. Track metrics
            end_time = time.time()
            self.token_tracker.log_usage(
                model, input_tokens, output_tokens, user_id
            )
            self.latency_tracker.track(start_time, end_time)

            # 7. Cache response
            self.semantic_cache.set(rendered_prompt, response_text)

            # 8. Check alerts
            self._check_alerts()

            return {
                "response": response_text,
                "cached": False,
                "model": model,
                "latency": end_time - start_time,
                "tokens": input_tokens + output_tokens,
                "quality_checks": {
                    "hallucination": hallucination,
                    "pii_detected": pii_check["contains_pii"]
                }
            }

        except Exception as e:
            print(f"❌ Error: {e}")
            # Log error, trigger alert
            return {"error": str(e)}

    def _check_alerts(self):
        """Check if any alerts should be triggered."""
        # Daily cost check
        self.token_tracker.alert_if_over_budget(
            self.alert_thresholds["daily_cost"]
        )

        # Latency check
        stats = self.latency_tracker.get_stats()
        if stats and stats["p95"] > self.alert_thresholds["p95_latency"]:
            print(f"⚠️ High latency: P95 = {stats['p95']:.2f}s")

    def get_dashboard_metrics(self) -> dict:
        """Get metrics for monitoring dashboard."""
        return {
            "cost": {
                "daily": self.token_tracker.get_daily_cost(days=1),
                "estimated_monthly": self.token_tracker.estimate_monthly_cost()
            },
            "latency": self.latency_tracker.get_stats(),
            "cache": {
                "hit_rate": self.semantic_cache.get_hit_rate()
            },
            "top_users": self.token_tracker.get_top_users(top_n=10)
        }

# Usage
app = ProductionLLMApp(config={})

# Register prompts
app.prompt_registry.register(
    "summarize",
    PromptVersion(
        template="Summarize this text: {text}",
        version="1.0.0"
    )
)

# Generate with monitoring
result = app.generate(
    prompt_name="summarize",
    user_id="user_123",
    variables={"text": "Long article..."},
    context="Article about AI"
)

print(result)

# View dashboard metrics
metrics = app.get_dashboard_metrics()
print("\n📊 Dashboard Metrics:")
print(f"Estimated monthly cost: ${metrics['cost']['estimated_monthly']:.2f}")
print(f"Cache hit rate: {metrics['cache']['hit_rate']:.1%}")
```

---

## Practice Exercises

### Exercise 1: Build a Cost Alert System
Create a system that:
- Tracks hourly LLM costs
- Sends alerts when hourly cost > $5
- Auto-disables expensive features if budget exceeded
- Generates daily cost reports

### Exercise 2: Implement Prompt A/B Testing
Set up an A/B test comparing:
- Short vs detailed prompts
- Different temperature settings
- GPT-3.5 vs GPT-4
- Measure: quality, latency, cost

### Exercise 3: Build a Quality Monitoring Dashboard
Create a dashboard showing:
- Hallucination rate over time
- Toxicity detection metrics
- PII leakage incidents
- Real-time alerts

---

## Key Takeaways

✅ **LLMOps ≠ MLOps**: Different challenges (prompts, tokens, latency)
✅ **Version prompts**: Treat prompts as code, use version control
✅ **Track costs**: Token usage drives costs, monitor closely
✅ **Monitor quality**: Hallucinations, toxicity, PII are unique risks
✅ **Cache aggressively**: Semantic caching saves massive costs
✅ **A/B test everything**: Prompts, models, parameters
✅ **Observability is critical**: LangSmith, W&B, Helicone
✅ **Deploy safely**: Blue-green, canary, feature flags
✅ **Automate alerts**: Budget overruns, quality issues, latency spikes
✅ **Production-first mindset**: Cost, quality, speed trade-offs

---

## Further Reading

### Tools & Platforms
- **LangSmith**: https://smith.langchain.com/
- **Weights & Biases**: https://wandb.ai/
- **Helicone**: https://helicone.ai/
- **PromptLayer**: https://promptlayer.com/
- **LangFuse**: https://langfuse.com/

### Best Practices
- OpenAI: Production Best Practices Guide
- Anthropic: Claude Production Guide
- "LLMOps: Operationalizing Large Language Models" (Paper)
- "Prompt Engineering Guide" (DAIR.AI)

### Monitoring & Observability
- LangChain Callbacks Documentation
- OpenTelemetry for LLMs
- Prometheus + Grafana for LLM Metrics

---

## Next Lesson

In **Module 17: AI Safety, Ethics & Responsible AI**, we'll cover:
- AI bias and fairness in LLMs and generative models
- Safety and alignment techniques
- Privacy, data governance, and compliance
- Building responsible AI systems

See you in the final module! 🛡️

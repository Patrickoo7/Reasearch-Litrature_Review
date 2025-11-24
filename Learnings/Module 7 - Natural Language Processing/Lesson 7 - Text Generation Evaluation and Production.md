# Lesson 7: Text Generation, Evaluation & Production 🚀

**Module 7: Natural Language Processing | Lesson 7 of 7**

Master text generation, evaluation metrics, and production NLP deployment!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Master all decoding strategies (greedy, beam, sampling, top-k, top-p)
2. ✅ Build text summarization systems (extractive & abstractive)
3. ✅ Evaluate NLP models (BLEU, ROUGE, BERTScore, LLM-as-judge)
4. ✅ Implement responsible NLP (bias, toxicity, PII detection)
5. ✅ Deploy production NLP systems (FastAPI, ONNX, quantization)
6. ✅ Monitor models for drift and performance degradation

---

## 1. Decoding Strategies for Text Generation

### Temperature-Controlled Sampling

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The future of artificial intelligence is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

def generate_with_temperature(temperature=1.0):
    """
    Temperature controls randomness:
    - temperature=0.0: Deterministic (greedy)
    - temperature=0.7: Balanced
    - temperature=1.0: Default randomness
    - temperature=2.0: Very creative/random
    """
    output = model.generate(
        input_ids,
        max_length=50,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test different temperatures
for temp in [0.3, 0.7, 1.0, 1.5]:
    print(f"\nTemperature={temp}:")
    print(generate_with_temperature(temp))
```

**Output:**
```
Temperature=0.3:
The future of artificial intelligence is to make the world a better place.

Temperature=0.7:
The future of artificial intelligence is bright, with applications in healthcare...

Temperature=1.5:
The future of artificial intelligence is limitless! From quantum computing to...
```

### Greedy Decoding

```python
def greedy_decode(prompt, max_length=50):
    """
    Greedy: Always pick the highest probability token.
    Fast but can be repetitive.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=False,  # Greedy decoding
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

print(greedy_decode("Machine learning is"))
```

### Beam Search

```python
def beam_search_decode(prompt, num_beams=5, max_length=50):
    """
    Beam search: Keep top-k candidates at each step.
    Better quality than greedy but slower.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=2,  # Prevent 2-gram repetition
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Generate with beam search
result = beam_search_decode("The key to success in AI is", num_beams=5)
print(result)
```

### Top-k Sampling

```python
def topk_sampling(prompt, k=50, max_length=50):
    """
    Top-k: Sample from top k most likely tokens.
    Adds diversity while avoiding very unlikely tokens.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_k=k,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Compare different k values
for k in [10, 50, 100]:
    print(f"\nTop-k={k}:")
    print(topk_sampling("Deep learning enables", k=k))
```

### Top-p (Nucleus) Sampling

```python
def nucleus_sampling(prompt, p=0.9, max_length=50):
    """
    Top-p (Nucleus): Sample from smallest set with cumulative prob >= p.
    Adaptive: Uses more tokens when model is uncertain.

    Best practice: p=0.9-0.95 for most tasks
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=p,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Compare different p values
for p in [0.5, 0.9, 0.95]:
    print(f"\nTop-p={p}:")
    print(nucleus_sampling("Natural language processing", p=p))
```

### Combined Strategy (Best Practice)

```python
def generate_best_practice(prompt, max_length=100):
    """
    Best practice: Combine top-p + temperature for balanced output.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    output = model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,
        top_p=0.92,              # Nucleus sampling
        temperature=0.7,          # Slightly focused
        repetition_penalty=1.2,   # Penalize repetition
        no_repeat_ngram_size=3,   # No 3-gram repeats
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

result = generate_best_practice(
    "The three most important principles of machine learning are"
)
print(result)
```

---

## 2. Text Summarization

### Extractive Summarization (TF-IDF)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extractive_summarize(text, num_sentences=3):
    """
    Extractive: Select most important sentences from original text.
    Uses TF-IDF to score sentences.
    """
    # Split into sentences
    sentences = text.split('. ')

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Score sentences (sum of TF-IDF scores)
    sentence_scores = tfidf_matrix.sum(axis=1).A1

    # Get top sentences
    top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
    top_indices = sorted(top_indices)  # Maintain original order

    summary = '. '.join([sentences[i] for i in top_indices])
    return summary

# Example
article = """
Machine learning is a subset of artificial intelligence. It focuses on
building systems that can learn from data. Deep learning is a type of
machine learning using neural networks. Neural networks are inspired by
the human brain. They can process complex patterns in data. Applications
include image recognition and natural language processing. The field has
grown rapidly in recent years. Many companies now use ML in production.
"""

summary = extractive_summarize(article, num_sentences=3)
print("Extractive Summary:")
print(summary)
```

### Abstractive Summarization with BART

```python
from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model fine-tuned for summarization
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def abstractive_summarize(text, max_length=130, min_length=30):
    """
    Abstractive: Generate new sentences that capture main ideas.
    BART is state-of-the-art for summarization.
    """
    inputs = tokenizer.encode(
        text,
        return_tensors='pt',
        max_length=1024,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=max_length,
        min_length=min_length,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example: Summarize news article
article = """
Artificial intelligence has made remarkable progress in recent years,
particularly in natural language processing. Large language models like
GPT-4 and Claude can now perform tasks that were impossible just a few
years ago. These models are trained on vast amounts of text data and
can generate human-like responses, translate languages, write code, and
answer complex questions. However, they also raise important questions
about bias, safety, and responsible deployment. Researchers are working
on techniques like RLHF and constitutional AI to make these models more
aligned with human values.
"""

summary = abstractive_summarize(article)
print("Abstractive Summary:")
print(summary)
```

### Summarization with T5

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def t5_summarize(text):
    """
    T5: Text-to-text model. Prefix with 'summarize:' for task.
    """
    # T5 requires task prefix
    input_text = "summarize: " + text

    inputs = tokenizer.encode(
        input_text,
        return_tensors='pt',
        max_length=512,
        truncation=True
    )

    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

summary = t5_summarize(article)
print("T5 Summary:")
print(summary)
```

---

## 3. Evaluation Metrics

### BLEU Score (Translation/Generation)

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt')

def calculate_bleu(reference, hypothesis):
    """
    BLEU: Measures n-gram overlap between reference and hypothesis.
    Range: 0-1 (higher is better)

    Use case: Machine translation, text generation
    """
    # Tokenize
    reference_tokens = [reference.lower().split()]
    hypothesis_tokens = hypothesis.lower().split()

    # Calculate BLEU with smoothing
    smoothing = SmoothingFunction().method1

    # BLEU-1, BLEU-2, BLEU-3, BLEU-4
    bleu1 = sentence_bleu(reference_tokens, hypothesis_tokens,
                          weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu2 = sentence_bleu(reference_tokens, hypothesis_tokens,
                          weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu4 = sentence_bleu(reference_tokens, hypothesis_tokens,
                          weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smoothing)

    return {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-4': bleu4
    }

# Example
reference = "The cat sat on the mat"
hypothesis = "A cat is sitting on the mat"

scores = calculate_bleu(reference, hypothesis)
print("BLEU Scores:", scores)
```

### ROUGE Score (Summarization)

```python
from rouge_score import rouge_scorer

def calculate_rouge(reference, hypothesis):
    """
    ROUGE: Recall-oriented metric for summarization.

    - ROUGE-1: Unigram overlap
    - ROUGE-2: Bigram overlap
    - ROUGE-L: Longest common subsequence

    Use case: Summarization, text generation
    """
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'],
        use_stemmer=True
    )

    scores = scorer.score(reference, hypothesis)

    return {
        'ROUGE-1': scores['rouge1'].fmeasure,
        'ROUGE-2': scores['rouge2'].fmeasure,
        'ROUGE-L': scores['rougeL'].fmeasure
    }

# Example
reference = "The quick brown fox jumps over the lazy dog"
hypothesis = "A fast brown fox jumped over a lazy dog"

scores = calculate_rouge(reference, hypothesis)
print("ROUGE Scores:", scores)
```

### BERTScore (Semantic Similarity)

```python
from bert_score import score

def calculate_bertscore(references, hypotheses):
    """
    BERTScore: Measures semantic similarity using BERT embeddings.
    Better than BLEU/ROUGE for capturing meaning.

    Returns: Precision, Recall, F1
    """
    P, R, F1 = score(
        hypotheses,
        references,
        lang='en',
        model_type='bert-base-uncased',
        verbose=False
    )

    return {
        'Precision': P.mean().item(),
        'Recall': R.mean().item(),
        'F1': F1.mean().item()
    }

# Example
references = ["The cat sat on the mat"]
hypotheses = ["A feline was resting on the rug"]

scores = calculate_bertscore(references, hypotheses)
print("BERTScore:", scores)
```

### LLM-as-Judge Evaluation

```python
import openai

def llm_judge_evaluate(text, criteria="quality", model="gpt-4"):
    """
    LLM-as-judge: Use GPT-4 to evaluate text quality.

    Criteria:
    - quality: Overall text quality
    - coherence: Logical flow
    - relevance: Relevance to topic
    - factuality: Factual accuracy
    """
    prompt = f"""
Evaluate the following text on a scale of 1-10 for {criteria}.
Provide a score and brief explanation.

Text: {text}

Evaluation:
Score (1-10):
Explanation:
"""

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

# Example
generated_text = """
Machine learning enables computers to learn from data without being
explicitly programmed. It has applications in healthcare, finance, and more.
"""

evaluation = llm_judge_evaluate(generated_text, criteria="quality")
print(evaluation)
```

### Multi-Metric Evaluation

```python
def comprehensive_evaluation(reference, hypothesis):
    """
    Evaluate using multiple metrics for robust assessment.
    """
    results = {}

    # BLEU
    results['BLEU'] = calculate_bleu(reference, hypothesis)

    # ROUGE
    results['ROUGE'] = calculate_rouge(reference, hypothesis)

    # BERTScore
    results['BERTScore'] = calculate_bertscore([reference], [hypothesis])

    return results

# Example
ref = "Artificial intelligence is transforming healthcare through early disease detection"
hyp = "AI is revolutionizing medicine by detecting diseases earlier"

results = comprehensive_evaluation(ref, hyp)
print("\nComprehensive Evaluation:")
for metric, scores in results.items():
    print(f"\n{metric}:")
    for key, value in scores.items():
        print(f"  {key}: {value:.3f}")
```

---

## 4. Responsible NLP

### Bias Detection in Embeddings

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def detect_gender_bias():
    """
    Detect gender bias by comparing word associations.
    """
    # Professional terms
    careers = ["engineer", "doctor", "nurse", "teacher", "CEO", "secretary"]

    # Gender terms
    male_terms = ["he", "him", "man", "male"]
    female_terms = ["she", "her", "woman", "female"]

    # Get embeddings
    career_embs = model.encode(careers)
    male_embs = model.encode(male_terms)
    female_embs = model.encode(female_terms)

    # Average gender embeddings
    male_avg = np.mean(male_embs, axis=0).reshape(1, -1)
    female_avg = np.mean(female_embs, axis=0).reshape(1, -1)

    # Calculate associations
    results = []
    for career, emb in zip(careers, career_embs):
        emb = emb.reshape(1, -1)
        male_sim = cosine_similarity(emb, male_avg)[0][0]
        female_sim = cosine_similarity(emb, female_avg)[0][0]

        bias = male_sim - female_sim
        results.append({
            'career': career,
            'male_similarity': male_sim,
            'female_similarity': female_sim,
            'bias_score': bias
        })

    return results

bias_results = detect_gender_bias()
print("\nGender Bias Analysis:")
for result in bias_results:
    print(f"{result['career']:12} - Bias: {result['bias_score']:+.3f} "
          f"(Male: {result['male_similarity']:.3f}, "
          f"Female: {result['female_similarity']:.3f})")
```

### Toxicity Detection

```python
from detoxify import Detoxify

# Load toxicity detection model
toxicity_model = Detoxify('original')

def detect_toxicity(text):
    """
    Detect toxic content using Detoxify.

    Categories:
    - toxicity: General toxicity
    - severe_toxicity: Severe toxic content
    - obscene: Obscene language
    - threat: Threats
    - insult: Insults
    - identity_attack: Attacks on identity
    """
    results = toxicity_model.predict(text)
    return results

# Example
texts = [
    "This is a great product!",
    "I hate this stupid thing",
    "You are an idiot"
]

print("\nToxicity Detection:")
for text in texts:
    scores = detect_toxicity(text)
    print(f"\nText: {text}")
    print(f"  Toxicity: {scores['toxicity']:.3f}")
    print(f"  Obscene: {scores['obscene']:.3f}")
    print(f"  Insult: {scores['insult']:.3f}")
```

### PII Detection and Redaction

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# Initialize Presidio
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def detect_and_redact_pii(text):
    """
    Detect and redact Personally Identifiable Information (PII).

    Detects:
    - Names (PERSON)
    - Email addresses (EMAIL_ADDRESS)
    - Phone numbers (PHONE_NUMBER)
    - Credit cards (CREDIT_CARD)
    - SSN (US_SSN)
    - Locations (LOCATION)
    """
    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        language='en',
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
                  "CREDIT_CARD", "LOCATION"]
    )

    # Anonymize PII
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )

    return {
        'original': text,
        'redacted': anonymized.text,
        'entities_found': [(r.entity_type, r.score) for r in results]
    }

# Example
text = "John Doe's email is john.doe@email.com and phone is 555-1234"

result = detect_and_redact_pii(text)
print("\nPII Detection:")
print(f"Original: {result['original']}")
print(f"Redacted: {result['redacted']}")
print(f"Entities: {result['entities_found']}")
```

### Content Moderation Pipeline

```python
def content_moderation_pipeline(text):
    """
    Complete content moderation: toxicity + PII + bias check.
    """
    results = {}

    # 1. Toxicity check
    toxicity = detect_toxicity(text)
    results['toxicity'] = toxicity['toxicity']
    results['is_toxic'] = toxicity['toxicity'] > 0.7

    # 2. PII detection
    pii = detect_and_redact_pii(text)
    results['has_pii'] = len(pii['entities_found']) > 0
    results['redacted_text'] = pii['redacted']

    # 3. Overall safety
    results['is_safe'] = not results['is_toxic'] and not results['has_pii']

    return results

# Example
test_text = "Contact me at john@email.com, you idiot!"

moderation = content_moderation_pipeline(test_text)
print("\nContent Moderation:")
print(f"Is Safe: {moderation['is_safe']}")
print(f"Is Toxic: {moderation['is_toxic']} (score: {moderation['toxicity']:.3f})")
print(f"Has PII: {moderation['has_pii']}")
print(f"Redacted: {moderation['redacted_text']}")
```

---

## 5. Production Deployment

### FastAPI NLP Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import uvicorn

app = FastAPI(title="NLP API")

# Load models at startup
sentiment_classifier = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

class TextRequest(BaseModel):
    text: str
    max_length: int = 130

class SentimentResponse(BaseModel):
    label: str
    score: float

class SummaryResponse(BaseModel):
    summary: str

@app.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: TextRequest):
    """
    Analyze sentiment of text.
    """
    try:
        result = sentiment_classifier(request.text)[0]
        return SentimentResponse(
            label=result['label'],
            score=result['score']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/summarize", response_model=SummaryResponse)
async def summarize_text(request: TextRequest):
    """
    Generate summary of text.
    """
    try:
        summary = summarizer(
            request.text,
            max_length=request.max_length,
            min_length=30,
            do_sample=False
        )[0]['summary_text']

        return SummaryResponse(summary=summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
# API docs at: http://localhost:8000/docs
```

### Batch Processing

```python
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

def batch_process_texts(texts, model, batch_size=32):
    """
    Process large batches of texts efficiently.
    """
    results = []

    # Process in batches
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        batch_results = model(batch)
        results.extend(batch_results)

    return results

# Example: Classify 10,000 texts
texts = ["Sample text " + str(i) for i in range(10000)]

classifier = pipeline("sentiment-analysis", device=0)  # Use GPU

# Batch processing
results = batch_process_texts(texts, classifier, batch_size=64)

# Convert to DataFrame
df = pd.DataFrame(results)
print(df.head())
print(f"\nProcessed {len(results)} texts")
```

### Model Optimization: ONNX Export

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch

# Original PyTorch model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Export to ONNX
from optimum.onnxruntime import ORTModelForSequenceClassification

ort_model = ORTModelForSequenceClassification.from_pretrained(
    model_name,
    from_transformers=True
)

# Save ONNX model
ort_model.save_pretrained("./onnx_model")

# Use ONNX model (faster inference)
def classify_onnx(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = ort_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions

# ONNX is 2-3x faster than PyTorch!
```

### Quantization (Reduce Model Size)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "gpt2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Dynamic quantization (PyTorch)
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize linear layers
    dtype=torch.qint8   # 8-bit integers
)

# Save quantized model
torch.save(quantized_model.state_dict(), "gpt2_quantized.pt")

# Model size comparison
import os

def get_model_size(model_path):
    return os.path.getsize(model_path) / (1024 * 1024)  # MB

print(f"Original size: {get_model_size('pytorch_model.bin'):.2f} MB")
print(f"Quantized size: {get_model_size('gpt2_quantized.pt'):.2f} MB")
# Quantization reduces size by ~4x!
```

---

## 6. Monitoring and Maintenance

### Model Performance Monitoring

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Prometheus metrics
request_count = Counter('nlp_requests_total', 'Total requests')
request_duration = Histogram('nlp_request_duration_seconds', 'Request duration')
model_confidence = Gauge('nlp_model_confidence', 'Average model confidence')

def monitored_predict(text):
    """
    Make prediction with monitoring.
    """
    request_count.inc()

    start_time = time.time()

    # Make prediction
    result = sentiment_classifier(text)[0]

    # Record metrics
    duration = time.time() - start_time
    request_duration.observe(duration)
    model_confidence.set(result['score'])

    return result

# Example
result = monitored_predict("This is great!")
print(result)
```

### Data Drift Detection

```python
from scipy.stats import ks_2samp
import numpy as np

def detect_drift(reference_embeddings, current_embeddings, threshold=0.05):
    """
    Detect distribution drift using Kolmogorov-Smirnov test.

    Returns: True if drift detected
    """
    # Flatten embeddings
    ref_flat = reference_embeddings.flatten()
    cur_flat = current_embeddings.flatten()

    # KS test
    statistic, p_value = ks_2samp(ref_flat, cur_flat)

    drift_detected = p_value < threshold

    return {
        'drift_detected': drift_detected,
        'p_value': p_value,
        'ks_statistic': statistic
    }

# Example: Compare training vs production data
training_embeddings = np.random.randn(1000, 384)
production_embeddings = np.random.randn(1000, 384) + 0.5  # Shifted distribution

drift = detect_drift(training_embeddings, production_embeddings)
print("\nDrift Detection:")
print(f"Drift Detected: {drift['drift_detected']}")
print(f"P-value: {drift['p_value']:.4f}")
```

### Input/Output Distribution Monitoring

```python
from collections import defaultdict
import numpy as np

class NLPMonitor:
    """
    Monitor input/output distributions over time.
    """
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.input_lengths = []
        self.output_confidences = []
        self.predictions = defaultdict(int)

    def log_prediction(self, input_text, prediction, confidence):
        """Log a single prediction."""
        # Track input length
        self.input_lengths.append(len(input_text.split()))
        if len(self.input_lengths) > self.window_size:
            self.input_lengths.pop(0)

        # Track confidence
        self.output_confidences.append(confidence)
        if len(self.output_confidences) > self.window_size:
            self.output_confidences.pop(0)

        # Track prediction distribution
        self.predictions[prediction] += 1

    def get_stats(self):
        """Get monitoring statistics."""
        return {
            'avg_input_length': np.mean(self.input_lengths),
            'avg_confidence': np.mean(self.output_confidences),
            'min_confidence': np.min(self.output_confidences),
            'prediction_distribution': dict(self.predictions),
            'total_predictions': sum(self.predictions.values())
        }

# Example usage
monitor = NLPMonitor()

# Simulate predictions
for i in range(100):
    text = "Sample text " * np.random.randint(5, 20)
    prediction = "POSITIVE" if np.random.rand() > 0.3 else "NEGATIVE"
    confidence = np.random.uniform(0.6, 0.99)

    monitor.log_prediction(text, prediction, confidence)

stats = monitor.get_stats()
print("\nMonitoring Stats:")
for key, value in stats.items():
    print(f"{key}: {value}")
```

### A/B Testing Framework

```python
import random

class ABTest:
    """
    A/B test framework for comparing model versions.
    """
    def __init__(self, model_a, model_b, traffic_split=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.traffic_split = traffic_split

        self.results_a = []
        self.results_b = []

    def predict(self, text):
        """Route to model A or B based on traffic split."""
        use_model_a = random.random() < self.traffic_split

        if use_model_a:
            result = self.model_a(text)[0]
            self.results_a.append(result['score'])
            result['model'] = 'A'
        else:
            result = self.model_b(text)[0]
            self.results_b.append(result['score'])
            result['model'] = 'B'

        return result

    def get_results(self):
        """Compare model performance."""
        return {
            'model_a_avg_confidence': np.mean(self.results_a),
            'model_b_avg_confidence': np.mean(self.results_b),
            'model_a_count': len(self.results_a),
            'model_b_count': len(self.results_b)
        }

# Example
model_a = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
model_b = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

ab_test = ABTest(model_a, model_b, traffic_split=0.5)

# Simulate traffic
test_texts = ["Great product!", "Terrible experience", "It's okay"]
for text in test_texts:
    result = ab_test.predict(text)
    print(f"{text} -> {result['label']} (Model {result['model']})")

print("\nA/B Test Results:")
print(ab_test.get_results())
```

---

## Quick Reference

### Decoding Strategy Selection

| Use Case | Strategy | Parameters |
|----------|----------|------------|
| **Factual/Deterministic** | Greedy or low temp | `temperature=0.1` |
| **Creative writing** | High temperature + top-p | `temp=1.2, top_p=0.95` |
| **Balanced quality** | Beam search | `num_beams=5` |
| **Diverse outputs** | Top-p sampling | `top_p=0.9, temp=0.8` |
| **Production default** | Top-p + low temp | `top_p=0.92, temp=0.7` |

### Evaluation Metric Guide

| Metric | Best For | Range | Strengths | Limitations |
|--------|----------|-------|-----------|-------------|
| **BLEU** | Translation | 0-1 | Fast, established | Ignores semantics |
| **ROUGE** | Summarization | 0-1 | Recall-focused | Lexical matching only |
| **BERTScore** | Any NLP task | 0-1 | Semantic similarity | Slower |
| **LLM-as-judge** | Quality/coherence | Custom | Human-like | Expensive |

### Model Optimization Techniques

| Technique | Size Reduction | Speed Improvement | Quality Impact |
|-----------|----------------|-------------------|----------------|
| **Quantization (INT8)** | 4x | 2-3x | Minimal (<1%) |
| **ONNX Export** | None | 2-3x | None |
| **Pruning** | 2-3x | 1.5x | Small (2-3%) |
| **Distillation** | 2-10x | 2-10x | Moderate (5-10%) |

---

## Practice Exercises

### Exercise 1: Build a Content Moderation API

```python
"""
Create a FastAPI endpoint that:
1. Accepts text input
2. Checks for toxicity
3. Detects and redacts PII
4. Returns safe/unsafe status with explanation

Bonus: Add rate limiting and caching
"""

# Your code here
```

**Solution:**
```python
from fastapi import FastAPI, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
from functools import lru_cache

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)

@lru_cache(maxsize=1000)
def cached_moderation(text: str):
    return content_moderation_pipeline(text)

@app.post("/moderate")
@limiter.limit("10/minute")
async def moderate_content(request: Request, text: str):
    result = cached_moderation(text)
    return result
```

### Exercise 2: Multi-Model Summarization Comparison

```python
"""
Compare 3 summarization approaches:
1. Extractive (TF-IDF)
2. BART
3. T5

Evaluate each with ROUGE and BERTScore.
Determine which is best for short vs long documents.
"""

# Your code here
```

### Exercise 3: Production Monitoring Dashboard

```python
"""
Build a monitoring system that:
1. Tracks latency, throughput, error rate
2. Detects data drift
3. A/B tests two model versions
4. Exports metrics to Prometheus

Display results in a simple dashboard.
"""

# Your code here
```

---

## Key Takeaways

### Essential Concepts

1. **Decoding Strategies** 🎲
   - Temperature controls randomness (0=deterministic, 2=creative)
   - Top-p adapts to model confidence
   - Beam search for quality, sampling for diversity
   - Production: `top_p=0.92, temperature=0.7`

2. **Summarization** 📝
   - Extractive: Select important sentences (fast, preserves text)
   - Abstractive: Generate new summary (better quality, more flexible)
   - BART and T5 are state-of-the-art

3. **Evaluation Metrics** 📊
   - BLEU: Translation and generation (n-gram overlap)
   - ROUGE: Summarization (recall-focused)
   - BERTScore: Semantic similarity (best for quality)
   - LLM-as-judge: Human-like evaluation

4. **Responsible NLP** ⚖️
   - Always check for bias in embeddings
   - Detect toxicity before deployment
   - Redact PII for privacy compliance
   - Use content moderation pipelines

5. **Production Deployment** 🚀
   - FastAPI for REST APIs
   - ONNX export for 2-3x speedup
   - Quantization for 4x size reduction
   - Batch processing for throughput

6. **Monitoring** 📈
   - Track latency, confidence, prediction distribution
   - Detect data drift with statistical tests
   - A/B test model versions
   - Export metrics to Prometheus

### Production Checklist

Before deploying NLP models:
- ✅ Quantize/optimize for inference speed
- ✅ Add toxicity and PII detection
- ✅ Implement rate limiting and caching
- ✅ Set up monitoring and alerting
- ✅ A/B test before full rollout
- ✅ Plan for model updates and retraining

### Best Practices

1. **Model Selection**: Start simple (DistilBERT), scale up if needed
2. **Decoding**: Default to `top_p=0.92, temp=0.7` for balanced output
3. **Evaluation**: Use multiple metrics (BLEU + ROUGE + BERTScore)
4. **Safety**: Always moderate content in production
5. **Optimization**: Quantize models for production deployment
6. **Monitoring**: Track drift, latency, and confidence distributions

---

## 🎓 Congratulations!

You've completed **Module 7: Natural Language Processing**!

### You Now Master:

- ✅ **Lesson 1**: Modern tokenization (BPE, WordPiece, SentencePiece)
- ✅ **Lesson 2**: Embeddings & Transformer architecture
- ✅ **Lesson 3**: Syntax, semantics & linguistic understanding
- ✅ **Lesson 4**: Prompt engineering & few-shot learning
- ✅ **Lesson 5**: Fine-tuning (LoRA, QLoRA, PEFT)
- ✅ **Lesson 6**: RAG & vector search
- ✅ **Lesson 7**: Text generation, evaluation & production

### Skills Acquired:

🔹 Build production NLP systems from scratch
🔹 Fine-tune LLMs efficiently with LoRA/QLoRA
🔹 Implement RAG systems with vector databases
🔹 Deploy models with FastAPI and optimize with ONNX
🔹 Evaluate with BLEU, ROUGE, BERTScore
🔹 Ensure responsible AI with bias/toxicity detection
🔹 Monitor models for drift and performance degradation

### What's Next?

- **Module 8**: Advanced Topics (Multimodal, Agents, Tool Use)
- **Module 9**: Capstone Project (Build end-to-end NLP system)
- **Module 10**: Career Preparation (Portfolio, interviews)

**You're now ready to build production NLP systems!** 🚀

---

## Additional Resources

### Papers
- "Attention is All You Need" (Transformers)
- "BERT: Pre-training of Deep Bidirectional Transformers"
- "LoRA: Low-Rank Adaptation of Large Language Models"
- "RAG: Retrieval-Augmented Generation"

### Libraries
- Hugging Face Transformers
- LangChain
- PEFT (Parameter-Efficient Fine-Tuning)
- Detoxify (Toxicity Detection)
- Presidio (PII Detection)

### Tools
- FastAPI (API deployment)
- Prometheus (Monitoring)
- ONNX Runtime (Optimization)
- Weights & Biases (Experiment tracking)

---

**Next**: Apply everything in Module 8 Capstone Project! 🎯

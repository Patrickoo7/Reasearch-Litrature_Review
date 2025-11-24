# Lesson 1: Vector Embeddings & Representation Learning 🔢

**Module 16: RAG, Vector Databases & AI Agents | Lesson 1 of 8**

Master the foundation of semantic search and modern AI systems - dense vector representations!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand embeddings as dense vector representations of text
2. ✅ Differentiate between word, sentence, and document embeddings
3. ✅ Implement embeddings using OpenAI, Cohere, and open-source models
4. ✅ Apply sentence-transformers and bi-encoders for semantic search
5. ✅ Understand dimensionality, normalization, and pooling strategies
6. ✅ Build practical applications: semantic search, clustering, classification

---

## Prerequisites

- **Module 7 Lesson 2**: Embeddings and Transformer Architecture (cross-reference)
- **Module 7 Lesson 6**: RAG & Vector Search basics
- Python fundamentals and NumPy
- Basic understanding of transformers
- API keys for OpenAI/Cohere (optional for cloud examples)

---

## 1. What Are Embeddings?

### Conceptual Foundation

**Embeddings** convert discrete objects (words, sentences, images) into continuous vector representations in a high-dimensional space where **semantic similarity = geometric proximity**.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Conceptual example: Word embeddings
# In reality, these are 300-1536 dimensions, not 2D!

word_embeddings = {
    'king': np.array([0.5, 0.8]),
    'queen': np.array([0.4, 0.9]),
    'man': np.array([0.6, 0.2]),
    'woman': np.array([0.5, 0.3]),
    'prince': np.array([0.55, 0.75]),
    'princess': np.array([0.45, 0.85])
}

# Visualize
plt.figure(figsize=(10, 8))
for word, vec in word_embeddings.items():
    plt.scatter(vec[0], vec[1], s=100)
    plt.annotate(word, (vec[0], vec[1]), fontsize=12,
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel('Dimension 1 (e.g., "royalty")')
plt.ylabel('Dimension 2 (e.g., "gender")')
plt.title('Word Embeddings in 2D Space\n(Real embeddings are 300-1536 dimensions)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('word_embeddings_2d.png', dpi=150, bbox_inches='tight')
print("✅ Saved word embeddings visualization")
```

**Key Properties:**
1. **Semantic Similarity**: Similar meanings → close vectors
2. **Algebraic Operations**: `king - man + woman ≈ queen`
3. **Fixed Length**: Variable text → fixed-size vector
4. **Learned Representations**: Trained on massive text corpora

---

## 2. Word Embeddings vs Sentence Embeddings vs Document Embeddings

### Word Embeddings (Token-Level)

```python
# Example 1: Word2Vec-style embeddings (conceptual)
import gensim.downloader as api

# Download pre-trained word vectors (100MB download)
# word2vec_model = api.load('word2vec-google-news-300')

# For this example, we'll simulate
class SimpleWordEmbeddings:
    """Simplified word embeddings for demonstration"""

    def __init__(self, dim=300):
        self.dim = dim
        self.embeddings = {}

    def get_embedding(self, word):
        """Get or create embedding for a word"""
        if word not in self.embeddings:
            # In reality, these are learned from data
            # Here we use random vectors for demonstration
            np.random.seed(hash(word) % 2**32)
            self.embeddings[word] = np.random.randn(self.dim)
            # Normalize
            self.embeddings[word] /= np.linalg.norm(self.embeddings[word])
        return self.embeddings[word]

    def similarity(self, word1, word2):
        """Cosine similarity between two words"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        return np.dot(emb1, emb2)

# Usage
word_emb = SimpleWordEmbeddings(dim=300)

# Compute similarities
pairs = [
    ('king', 'queen'),
    ('king', 'man'),
    ('paris', 'france'),
    ('paris', 'berlin'),
    ('python', 'programming'),
    ('python', 'snake')
]

print("Word Similarities:")
for w1, w2 in pairs:
    sim = word_emb.similarity(w1, w2)
    print(f"  {w1} <-> {w2}: {sim:.3f}")
```

### Sentence Embeddings (Sequence-Level)

```python
# Example 2: Sentence embeddings with sentence-transformers
from sentence_transformers import SentenceTransformer
import scipy.spatial.distance as distance

# Load pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions, 80MB

sentences = [
    "The cat sits on the mat",
    "A feline rests on the rug",
    "The dog runs in the park",
    "Python is a programming language",
    "I love coding in Python",
]

# Generate embeddings
embeddings = model.encode(sentences)

print(f"\nSentence Embeddings Shape: {embeddings.shape}")
print(f"Each sentence → {embeddings.shape[1]}-dimensional vector\n")

# Compute pairwise similarities
print("Sentence Similarities (cosine):")
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        sim = 1 - distance.cosine(embeddings[i], embeddings[j])
        print(f"  [{i}] <-> [{j}]: {sim:.3f}")
        print(f"    '{sentences[i]}'")
        print(f"    '{sentences[j]}'")
        print()
```

### Document Embeddings (Long Text)

```python
# Example 3: Document-level embeddings
documents = [
    """
    Machine learning is a subset of artificial intelligence that focuses on
    enabling computers to learn from data. It uses statistical techniques to
    give computer systems the ability to improve performance on tasks through experience.
    """,
    """
    Deep learning is a specialized branch of machine learning that uses neural
    networks with multiple layers. These deep neural networks can automatically
    learn hierarchical representations of data.
    """,
    """
    Natural language processing enables computers to understand, interpret, and
    generate human language. It combines computational linguistics with machine
    learning and deep learning techniques.
    """,
    """
    The weather today is sunny with a chance of rain in the afternoon.
    Temperatures will reach a high of 75°F. Don't forget your umbrella!
    """
]

# Encode long documents
doc_embeddings = model.encode(documents, show_progress_bar=False)

print("Document Embeddings:")
print(f"Shape: {doc_embeddings.shape}")
print(f"\nDocument similarities:")
for i in range(len(documents)):
    for j in range(i + 1, len(documents)):
        sim = 1 - distance.cosine(doc_embeddings[i], doc_embeddings[j])
        print(f"  Doc {i+1} <-> Doc {j+1}: {sim:.3f}")
```

---

## 3. Sentence Transformers & Bi-Encoders

### What Are Bi-Encoders?

**Bi-encoders** encode queries and documents independently, enabling efficient similarity search.

```python
# Example 4: Bi-Encoder architecture explained
"""
Bi-Encoder Architecture:

Query: "What is Python?"           Document: "Python is a language"
    ↓                                  ↓
[BERT Encoder]                    [BERT Encoder]
    ↓                                  ↓
Query Embedding (384-dim)         Doc Embedding (384-dim)
    ↓                                  ↓
    └──────── Cosine Similarity ───────┘
                    ↓
              Similarity Score

Advantages:
✅ Encode documents ONCE, reuse for all queries
✅ Fast inference (just compute query embedding)
✅ Scalable to millions of documents

Disadvantages:
❌ No cross-attention between query and document
❌ Lower accuracy than cross-encoders (covered in Lesson 5)
"""

# Practical bi-encoder usage
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base (encode once)
knowledge_base = [
    "Python is a high-level programming language",
    "Machine learning uses statistical methods",
    "Paris is the capital of France",
    "The Eiffel Tower is in Paris",
    "Deep learning uses neural networks",
]

kb_embeddings = model.encode(knowledge_base, convert_to_tensor=True)

# Query (encode on demand)
query = "What is Python?"
query_embedding = model.encode(query, convert_to_tensor=True)

# Compute similarities
similarities = util.cos_sim(query_embedding, kb_embeddings)[0]

# Get top results
top_k = 3
top_results = np.argsort(-similarities.cpu().numpy())[:top_k]

print(f"Query: '{query}'\n")
print("Top matches:")
for idx in top_results:
    print(f"  [{idx}] Score: {similarities[idx]:.3f}")
    print(f"      {knowledge_base[idx]}")
```

### Pooling Strategies

```python
# Example 5: Different pooling strategies
import torch
from transformers import AutoTokenizer, AutoModel

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentence = "Machine learning is transforming industries"

# Tokenize
inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

# Get model outputs
with torch.no_grad():
    outputs = model(**inputs)

# Last hidden state: [batch_size, seq_len, hidden_dim]
hidden_states = outputs.last_hidden_state
print(f"Hidden states shape: {hidden_states.shape}")

# 1. Mean Pooling (most common for sentence embeddings)
attention_mask = inputs['attention_mask']
mean_pooled = torch.sum(hidden_states * attention_mask.unsqueeze(-1), dim=1)
mean_pooled = mean_pooled / torch.sum(attention_mask, dim=1, keepdim=True)

print(f"\nMean pooling output: {mean_pooled.shape}")

# 2. Max Pooling
max_pooled, _ = torch.max(hidden_states, dim=1)
print(f"Max pooling output: {max_pooled.shape}")

# 3. CLS Token (first token)
cls_pooled = hidden_states[:, 0, :]
print(f"CLS pooling output: {cls_pooled.shape}")

# Compare strategies
print("\nPooling Strategy Comparison:")
print(f"  Mean: First 5 dims = {mean_pooled[0, :5].numpy()}")
print(f"  Max:  First 5 dims = {max_pooled[0, :5].numpy()}")
print(f"  CLS:  First 5 dims = {cls_pooled[0, :5].numpy()}")
```

---

## 4. Creating Embeddings with Different Providers

### OpenAI Embeddings (Cloud API)

```python
# Example 6: OpenAI embeddings (requires API key)
"""
# Uncomment to use with your API key
import openai
import os

openai.api_key = os.getenv('OPENAI_API_KEY')

def get_openai_embedding(text, model="text-embedding-3-small"):
    '''
    OpenAI Models:
    - text-embedding-3-small: 1536 dims, $0.02/1M tokens
    - text-embedding-3-large: 3072 dims, $0.13/1M tokens
    - text-embedding-ada-002: 1536 dims, $0.10/1M tokens (legacy)
    '''
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

# Usage
text = "Machine learning is revolutionizing AI"
embedding = get_openai_embedding(text)

print(f"OpenAI Embedding:")
print(f"  Dimensions: {len(embedding)}")
print(f"  First 10 values: {embedding[:10]}")

# Batch processing
texts = ["Python is great", "I love coding", "AI is the future"]
embeddings = [get_openai_embedding(t) for t in texts]
print(f"\nBatch embeddings: {len(embeddings)} texts")
"""

print("OpenAI Embeddings Example (commented - requires API key)")
print("Features:")
print("  - State-of-the-art quality")
print("  - 1536 or 3072 dimensions")
print("  - Cost: $0.02-$0.13 per 1M tokens")
print("  - Rate limits: 3000 RPM (varies by tier)")
```

### Cohere Embeddings (Cloud API)

```python
# Example 7: Cohere embeddings (requires API key)
"""
# Uncomment to use with your API key
import cohere
import os

co = cohere.Client(os.getenv('COHERE_API_KEY'))

def get_cohere_embedding(texts, model="embed-english-v3.0"):
    '''
    Cohere Models:
    - embed-english-v3.0: 1024 dims, multilingual
    - embed-english-light-v3.0: 384 dims, faster
    - embed-multilingual-v3.0: 1024 dims, 100+ languages
    '''
    response = co.embed(
        texts=texts if isinstance(texts, list) else [texts],
        model=model,
        input_type='search_document'  # or 'search_query', 'classification'
    )
    return response.embeddings

# Usage
texts = [
    "Python is a programming language",
    "Machine learning enables AI systems"
]

embeddings = get_cohere_embedding(texts)
print(f"Cohere Embeddings: {len(embeddings)} texts")
print(f"Dimensions: {len(embeddings[0])}")

# Different input types
query_emb = co.embed(
    texts=["What is Python?"],
    model="embed-english-v3.0",
    input_type='search_query'
).embeddings[0]

doc_emb = co.embed(
    texts=["Python is a programming language"],
    model="embed-english-v3.0",
    input_type='search_document'
).embeddings[0]

print(f"\nQuery embedding dims: {len(query_emb)}")
print(f"Doc embedding dims: {len(doc_emb)}")
"""

print("Cohere Embeddings Example (commented - requires API key)")
print("Features:")
print("  - Optimized for search (separate query/doc embeddings)")
print("  - 384-1024 dimensions")
print("  - Multilingual support (100+ languages)")
print("  - Input type awareness (query vs document)")
```

### Open-Source Models (Free, Run Locally)

```python
# Example 8: Popular open-source embedding models
from sentence_transformers import SentenceTransformer

# Model comparison
models_info = [
    {
        'name': 'all-MiniLM-L6-v2',
        'dims': 384,
        'size': '80MB',
        'speed': 'Very Fast',
        'quality': 'Good',
        'use_case': 'General purpose, fast inference'
    },
    {
        'name': 'all-mpnet-base-v2',
        'dims': 768,
        'size': '420MB',
        'speed': 'Medium',
        'quality': 'Excellent',
        'use_case': 'Best quality for English'
    },
    {
        'name': 'multi-qa-MiniLM-L6-cos-v1',
        'dims': 384,
        'size': '80MB',
        'speed': 'Very Fast',
        'quality': 'Good',
        'use_case': 'Question-answering, semantic search'
    },
    {
        'name': 'paraphrase-multilingual-MiniLM-L12-v2',
        'dims': 384,
        'size': '120MB',
        'speed': 'Fast',
        'quality': 'Good',
        'use_case': '50+ languages'
    },
]

print("Open-Source Embedding Models:\n")
for info in models_info:
    print(f"{info['name']}")
    print(f"  Dimensions: {info['dims']}")
    print(f"  Model Size: {info['size']}")
    print(f"  Speed: {info['speed']}")
    print(f"  Quality: {info['quality']}")
    print(f"  Use Case: {info['use_case']}")
    print()

# Load and test a model
model = SentenceTransformer('all-MiniLM-L6-v2')
test_text = "This is a test sentence"
embedding = model.encode(test_text)

print(f"✅ Loaded model: all-MiniLM-L6-v2")
print(f"   Embedding shape: {embedding.shape}")
print(f"   First 10 values: {embedding[:10]}")
```

### Hugging Face Models (Advanced)

```python
# Example 9: Custom Hugging Face models
from transformers import AutoTokenizer, AutoModel
import torch

def get_hf_embedding(text, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    """Get embeddings from any Hugging Face model"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Mean pooling
    attention_mask = inputs['attention_mask']
    embeddings = outputs.last_hidden_state
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask

    return mean_pooled[0].numpy()

# Test
text = "Hugging Face provides thousands of pre-trained models"
embedding = get_hf_embedding(text)

print(f"Custom HF Embedding:")
print(f"  Shape: {embedding.shape}")
print(f"  Norm: {np.linalg.norm(embedding):.3f}")
```

---

## 5. Dimensionality & Embedding Quality

### Understanding Embedding Dimensions

```python
# Example 10: Dimension analysis
from sentence_transformers import SentenceTransformer
import numpy as np

# Compare different dimension models
models = {
    'MiniLM (384d)': SentenceTransformer('all-MiniLM-L6-v2'),
    'MPNet (768d)': SentenceTransformer('all-mpnet-base-v2'),
}

test_sentences = [
    "Machine learning is amazing",
    "I love deep learning",
    "The weather is nice today"
]

print("Dimensionality Comparison:\n")
for model_name, model in models.items():
    embeddings = model.encode(test_sentences)

    # Compute pairwise similarities
    sim_ml_dl = 1 - distance.cosine(embeddings[0], embeddings[1])
    sim_ml_weather = 1 - distance.cosine(embeddings[0], embeddings[2])

    print(f"{model_name}:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  ML <-> DL similarity: {sim_ml_dl:.3f}")
    print(f"  ML <-> Weather similarity: {sim_ml_weather:.3f}")
    print(f"  Discrimination: {sim_ml_dl - sim_ml_weather:.3f}")
    print()

# Dimension statistics
embedding = models['MiniLM (384d)'].encode("Sample text")
print(f"Embedding Statistics (384d):")
print(f"  Mean: {np.mean(embedding):.4f}")
print(f"  Std: {np.std(embedding):.4f}")
print(f"  Min: {np.min(embedding):.4f}")
print(f"  Max: {np.max(embedding):.4f}")
print(f"  L2 Norm: {np.linalg.norm(embedding):.4f}")
```

### Dimensionality Reduction

```python
# Example 11: Reduce embedding dimensions
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    # Cluster 1: Programming
    "Python is a programming language",
    "Java is used for software development",
    "JavaScript runs in web browsers",
    # Cluster 2: Machine Learning
    "Machine learning uses neural networks",
    "Deep learning is a subset of ML",
    "AI systems can learn from data",
    # Cluster 3: Nature
    "The sun rises in the east",
    "Trees produce oxygen",
    "Rivers flow to the ocean"
]

embeddings = model.encode(sentences)

# PCA: Linear reduction
pca = PCA(n_components=2)
embeddings_2d_pca = pca.fit_transform(embeddings)

# t-SNE: Non-linear reduction
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d_tsne = tsne.fit_transform(embeddings)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# PCA plot
colors = ['red']*3 + ['blue']*3 + ['green']*3
for i, (x, y) in enumerate(embeddings_2d_pca):
    ax1.scatter(x, y, c=colors[i], s=100)
    ax1.annotate(f"{i}", (x, y), fontsize=9)
ax1.set_title(f'PCA Reduction (384d → 2d)\nExplained Variance: {pca.explained_variance_ratio_.sum():.1%}')
ax1.set_xlabel('PC1')
ax1.set_ylabel('PC2')
ax1.grid(True, alpha=0.3)

# t-SNE plot
for i, (x, y) in enumerate(embeddings_2d_tsne):
    ax2.scatter(x, y, c=colors[i], s=100)
    ax2.annotate(f"{i}", (x, y), fontsize=9)
ax2.set_title('t-SNE Reduction (384d → 2d)\nNon-linear manifold')
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dimensionality_reduction.png', dpi=150, bbox_inches='tight')
print("✅ Saved dimensionality reduction visualization")

# Print variance explained
print(f"\nPCA Variance Explained:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.1%}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.1%}")
print(f"  Total (2 components): {pca.explained_variance_ratio_.sum():.1%}")
```

---

## 6. Normalization Strategies

### L2 Normalization

```python
# Example 12: L2 normalization for cosine similarity
import numpy as np

def normalize_l2(embeddings):
    """L2 normalize embeddings to unit length"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
sentences = ["Python programming", "Machine learning", "Data science"]

embeddings = model.encode(sentences)
embeddings_normalized = normalize_l2(embeddings)

print("L2 Normalization:")
print(f"Original norms: {[np.linalg.norm(e) for e in embeddings]}")
print(f"Normalized norms: {[np.linalg.norm(e) for e in embeddings_normalized]}")

# Cosine similarity with normalized vectors = dot product
sim_cosine = 1 - distance.cosine(embeddings[0], embeddings[1])
sim_dot = np.dot(embeddings_normalized[0], embeddings_normalized[1])

print(f"\nCosine similarity: {sim_cosine:.4f}")
print(f"Dot product (normalized): {sim_dot:.4f}")
print(f"Difference: {abs(sim_cosine - sim_dot):.6f}")
```

### Why Normalize?

```python
# Example 13: Benefits of normalization
"""
Benefits of L2 Normalization:

1. ✅ Cosine Similarity = Dot Product
   - Faster computation (no division needed)
   - Better for vector databases (optimized for dot product)

2. ✅ Consistent Scale
   - All vectors have same magnitude
   - Prevents length bias

3. ✅ Better for Distance Metrics
   - Euclidean distance ≈ Angular distance
   - More meaningful for semantic similarity
"""

# Demonstrate the speed benefit
import time

model = SentenceTransformer('all-MiniLM-L6-v2')
docs = ["Document " + str(i) for i in range(1000)]
query = "Search query"

# Encode
doc_embeddings = model.encode(docs)
query_embedding = model.encode(query)

# Normalize
doc_embeddings_norm = normalize_l2(doc_embeddings)
query_embedding_norm = normalize_l2(query_embedding.reshape(1, -1))[0]

# Benchmark: Cosine similarity
start = time.time()
for doc_emb in doc_embeddings:
    sim = 1 - distance.cosine(query_embedding, doc_emb)
cosine_time = time.time() - start

# Benchmark: Dot product (normalized)
start = time.time()
for doc_emb in doc_embeddings_norm:
    sim = np.dot(query_embedding_norm, doc_emb)
dot_time = time.time() - start

print(f"Cosine similarity time: {cosine_time:.4f}s")
print(f"Dot product time: {dot_time:.4f}s")
print(f"Speedup: {cosine_time/dot_time:.2f}x faster")
```

---

## 7. Applications: Semantic Search

### Building a Semantic Search Engine

```python
# Example 14: Production-grade semantic search
from sentence_transformers import SentenceTransformer, util
import numpy as np

class SemanticSearchEngine:
    """
    Simple but production-ready semantic search engine.

    Features:
    - L2 normalized embeddings
    - Efficient similarity search
    - Metadata support
    - Batched encoding
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.documents = []
        self.embeddings = None

    def index_documents(self, documents, batch_size=32):
        """Index documents for search"""
        self.documents = documents

        # Batch encode for efficiency
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True  # L2 normalize
        )

        print(f"✅ Indexed {len(documents)} documents")
        print(f"   Embedding shape: {self.embeddings.shape}")

    def search(self, query, top_k=5):
        """Search for most relevant documents"""
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

        # Compute similarities (dot product because normalized)
        similarities = util.dot_score(query_embedding, self.embeddings)[0]

        # Get top-k
        top_results = torch.topk(similarities, k=min(top_k, len(self.documents)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            results.append({
                'document': self.documents[idx],
                'score': score.item(),
                'index': idx.item()
            })

        return results

# Usage example
search_engine = SemanticSearchEngine()

# Knowledge base
documents = [
    "Python is a high-level programming language known for its simplicity",
    "Machine learning is a subset of AI that learns from data",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing enables computers to understand text",
    "Computer vision allows machines to interpret visual information",
    "Reinforcement learning trains agents through trial and error",
    "Paris is the capital city of France, known for the Eiffel Tower",
    "The Pacific Ocean is the largest ocean on Earth",
    "Photosynthesis is how plants convert sunlight into energy",
    "Quantum computing uses quantum mechanics for computation"
]

search_engine.index_documents(documents)

# Search queries
queries = [
    "What is Python?",
    "Tell me about AI and neural networks",
    "Information about France"
]

for query in queries:
    print(f"\n🔍 Query: '{query}'")
    results = search_engine.search(query, top_k=3)
    for i, result in enumerate(results, 1):
        print(f"  {i}. [Score: {result['score']:.3f}] {result['document']}")
```

---

## 8. Applications: Clustering

### Document Clustering with Embeddings

```python
# Example 15: K-means clustering on embeddings
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Documents from different topics
documents = [
    # Programming (cluster 0)
    "Python is a versatile programming language",
    "JavaScript is essential for web development",
    "Java is widely used in enterprise applications",
    "C++ offers high performance for system programming",
    # Machine Learning (cluster 1)
    "Neural networks are the foundation of deep learning",
    "Machine learning models learn patterns from data",
    "Supervised learning requires labeled training data",
    "Unsupervised learning finds hidden patterns",
    # Space (cluster 2)
    "The Mars rover explores the red planet",
    "Black holes have immense gravitational pull",
    "The International Space Station orbits Earth",
    "Galaxies contain billions of stars",
]

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)

# K-means clustering
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Visualize with PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
colors = ['red', 'blue', 'green']
for i, doc in enumerate(documents):
    plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1],
                c=colors[clusters[i]], s=100, alpha=0.6)
    plt.annotate(f"Doc {i}", (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                 fontsize=9, alpha=0.7)

# Plot cluster centers
centers_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
            c='black', marker='X', s=200, label='Centroids')

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Document Clustering with K-Means on Embeddings')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('document_clustering.png', dpi=150, bbox_inches='tight')

# Print clusters
print("\nDocument Clusters:")
for cluster_id in range(n_clusters):
    print(f"\n📁 Cluster {cluster_id}:")
    cluster_docs = [doc for i, doc in enumerate(documents) if clusters[i] == cluster_id]
    for doc in cluster_docs:
        print(f"  - {doc}")
```

---

## 9. Applications: Classification

### Zero-Shot Classification with Embeddings

```python
# Example 16: Zero-shot text classification
from sentence_transformers import SentenceTransformer
import numpy as np

class ZeroShotClassifier:
    """
    Zero-shot classifier using embedding similarity.
    No training required!
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.categories = None
        self.category_embeddings = None

    def fit(self, categories):
        """Define categories (no training data needed!)"""
        self.categories = categories
        self.category_embeddings = self.model.encode(
            categories,
            normalize_embeddings=True
        )

    def predict(self, texts):
        """Predict category for texts"""
        text_embeddings = self.model.encode(
            texts if isinstance(texts, list) else [texts],
            normalize_embeddings=True
        )

        # Compute similarities to all categories
        similarities = np.dot(text_embeddings, self.category_embeddings.T)

        # Get best category
        predictions = np.argmax(similarities, axis=1)

        return [self.categories[pred] for pred in predictions]

    def predict_proba(self, text):
        """Get probabilities for all categories"""
        text_embedding = self.model.encode(text, normalize_embeddings=True)
        similarities = np.dot(text_embedding, self.category_embeddings.T)

        # Softmax for probabilities
        exp_sim = np.exp(similarities - np.max(similarities))
        probs = exp_sim / exp_sim.sum()

        return {cat: prob for cat, prob in zip(self.categories, probs)}

# Usage
classifier = ZeroShotClassifier()

# Define categories (just descriptions!)
categories = [
    "Technology and programming",
    "Health and medicine",
    "Sports and athletics",
    "Finance and economics"
]

classifier.fit(categories)

# Test documents
test_docs = [
    "Python is great for data science and machine learning",
    "Regular exercise reduces the risk of heart disease",
    "The stock market reached new highs today",
    "LeBron James scored 40 points in the game"
]

print("Zero-Shot Classification Results:\n")
for doc in test_docs:
    prediction = classifier.predict(doc)[0]
    probs = classifier.predict_proba(doc)

    print(f"📄 Document: '{doc}'")
    print(f"   Predicted: {prediction}")
    print(f"   Probabilities:")
    for cat, prob in sorted(probs.items(), key=lambda x: -x[1]):
        print(f"     - {cat}: {prob:.3f}")
    print()
```

---

## 10. Multilingual Embeddings

```python
# Example 17: Multilingual semantic search
from sentence_transformers import SentenceTransformer

# Load multilingual model
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Documents in different languages
documents = {
    'en': "Machine learning is transforming industries",
    'es': "El aprendizaje automático está transformando las industrias",
    'fr': "L'apprentissage automatique transforme les industries",
    'de': "Maschinelles Lernen verändert Branchen",
    'zh': "机器学习正在改变各行各业",
    'ja': "機械学習は産業を変革しています"
}

# Encode all documents
embeddings = {lang: model.encode(text) for lang, text in documents.items()}

# Compute cross-lingual similarities
print("Cross-Lingual Similarities:\n")
for lang1 in documents.keys():
    for lang2 in documents.keys():
        if lang1 < lang2:  # Avoid duplicates
            sim = 1 - distance.cosine(embeddings[lang1], embeddings[lang2])
            print(f"{lang1}-{lang2}: {sim:.3f}")

# Cross-lingual search
query_en = "artificial intelligence"
query_embedding = model.encode(query_en)

print(f"\n🔍 Query (English): '{query_en}'")
print("Matches across languages:")
for lang, doc_embedding in embeddings.items():
    sim = 1 - distance.cosine(query_embedding, doc_embedding)
    print(f"  {lang}: {sim:.3f} - {documents[lang]}")
```

---

## 11. Advanced: Custom Training

```python
# Example 18: Fine-tune embeddings for domain-specific use
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Fine-tuning steps (conceptual example):

# 1. Prepare training data
train_examples = [
    InputExample(texts=['Python programming', 'Coding in Python'], label=0.9),
    InputExample(texts=['Python snake', 'Programming language'], label=0.1),
    # ... more examples
]

# 2. Create DataLoader
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# 3. Define loss function
# - CosineSimilarityLoss: For similarity pairs
# - MultipleNegativesRankingLoss: For asymmetric search
# - TripletLoss: For triplet data
train_loss = losses.CosineSimilarityLoss(model)

# 4. Fine-tune
model = SentenceTransformer('all-MiniLM-L6-v2')
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=100,
    output_path='./fine-tuned-model'
)

# 5. Save and load
model.save('./my-custom-embeddings')
custom_model = SentenceTransformer('./my-custom-embeddings')
"""

print("Fine-tuning Embeddings (Conceptual Example)")
print("\nWhen to fine-tune:")
print("  ✅ Domain-specific vocabulary (medical, legal, etc.)")
print("  ✅ Custom similarity definitions")
print("  ✅ Insufficient performance on your task")
print("\nRequired:")
print("  - Training data (pairs, triplets, or labeled examples)")
print("  - Computational resources (GPU recommended)")
print("  - Time (hours to days depending on dataset size)")
```

---

## 12. Embedding Quality Metrics

```python
# Example 19: Evaluate embedding quality
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def embedding_quality_metrics(embeddings, labels):
    """
    Metrics to evaluate embedding quality:
    1. Intra-class similarity (higher = better)
    2. Inter-class similarity (lower = better)
    3. Silhouette score
    """
    from sklearn.metrics import silhouette_score

    # Compute pairwise similarities
    similarities = cosine_similarity(embeddings)

    # Intra-class: average similarity within same class
    unique_labels = np.unique(labels)
    intra_class_sims = []
    for label in unique_labels:
        mask = labels == label
        class_sims = similarities[mask][:, mask]
        # Exclude diagonal (self-similarity)
        np.fill_diagonal(class_sims, 0)
        intra_class_sims.append(class_sims.mean())

    # Inter-class: average similarity between different classes
    inter_class_sims = []
    for i, label1 in enumerate(unique_labels):
        for label2 in unique_labels[i+1:]:
            mask1 = labels == label1
            mask2 = labels == label2
            inter_sim = similarities[mask1][:, mask2].mean()
            inter_class_sims.append(inter_sim)

    # Silhouette score
    silhouette = silhouette_score(embeddings, labels, metric='cosine')

    return {
        'intra_class_similarity': np.mean(intra_class_sims),
        'inter_class_similarity': np.mean(inter_class_sims),
        'silhouette_score': silhouette
    }

# Example usage
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create test dataset with labels
docs_by_class = {
    0: ["Python programming", "Java development", "C++ coding"],
    1: ["Heart disease treatment", "Cancer research", "Medical diagnosis"],
    2: ["Stock market analysis", "Investment strategies", "Economic trends"]
}

all_docs = []
labels = []
for label, docs in docs_by_class.items():
    all_docs.extend(docs)
    labels.extend([label] * len(docs))

embeddings = model.encode(all_docs)
labels = np.array(labels)

metrics = embedding_quality_metrics(embeddings, labels)

print("Embedding Quality Metrics:")
print(f"  Intra-class similarity: {metrics['intra_class_similarity']:.3f} (higher = better)")
print(f"  Inter-class similarity: {metrics['inter_class_similarity']:.3f} (lower = better)")
print(f"  Silhouette score: {metrics['silhouette_score']:.3f} (range: -1 to 1, higher = better)")
print(f"  Discrimination: {metrics['intra_class_similarity'] - metrics['inter_class_similarity']:.3f}")
```

---

## 13. Production Considerations

### Batching for Efficiency

```python
# Example 20: Efficient batch processing
import time
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate test data
docs = [f"Document number {i} with some content" for i in range(1000)]

# Single-doc encoding (SLOW)
start = time.time()
embeddings_single = [model.encode(doc) for doc in docs[:100]]
time_single = time.time() - start

# Batch encoding (FAST)
start = time.time()
embeddings_batch = model.encode(docs[:100], batch_size=32, show_progress_bar=False)
time_batch = time.time() - start

print(f"Encoding 100 documents:")
print(f"  Single: {time_single:.2f}s")
print(f"  Batch (32): {time_batch:.2f}s")
print(f"  Speedup: {time_single/time_batch:.1f}x")
print(f"\nFor 1000 documents (estimated):")
print(f"  Single: ~{time_single*10:.1f}s")
print(f"  Batch: ~{time_batch*10:.1f}s")
```

### Caching Embeddings

```python
# Example 21: Cache embeddings to disk
import pickle
import hashlib

class EmbeddingCache:
    """Cache embeddings to avoid recomputation"""

    def __init__(self, model, cache_file='embedding_cache.pkl'):
        self.model = model
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return {}

    def _save_cache(self):
        """Save cache to disk"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)

    def _hash_text(self, text):
        """Generate hash for text"""
        return hashlib.md5(text.encode()).hexdigest()

    def encode(self, text):
        """Encode with caching"""
        text_hash = self._hash_text(text)

        if text_hash in self.cache:
            return self.cache[text_hash]

        # Compute embedding
        embedding = self.model.encode(text)

        # Cache it
        self.cache[text_hash] = embedding
        self._save_cache()

        return embedding

    def clear_cache(self):
        """Clear all cached embeddings"""
        self.cache = {}
        self._save_cache()

# Usage
model = SentenceTransformer('all-MiniLM-L6-v2')
cached_encoder = EmbeddingCache(model)

# First call: compute
start = time.time()
emb1 = cached_encoder.encode("This is a test document")
time1 = time.time() - start

# Second call: from cache
start = time.time()
emb2 = cached_encoder.encode("This is a test document")
time2 = time.time() - start

print(f"First encoding: {time1*1000:.2f}ms")
print(f"Cached encoding: {time2*1000:.2f}ms")
print(f"Speedup: {time1/time2:.0f}x faster")
print(f"Cache size: {len(cached_encoder.cache)} entries")
```

---

## 14. Real-World Application: FAQ Matching

```python
# Example 22: FAQ matching system
class FAQMatcher:
    """
    Production FAQ matching system.

    Features:
    - Semantic similarity matching
    - Confidence thresholds
    - Fallback responses
    """

    def __init__(self, model_name='all-MiniLM-L6-v2', threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.faqs = []
        self.faq_embeddings = None

    def add_faqs(self, faqs):
        """
        Add FAQ pairs.

        Args:
            faqs: List of dicts with 'question' and 'answer' keys
        """
        self.faqs = faqs
        questions = [faq['question'] for faq in faqs]
        self.faq_embeddings = self.model.encode(
            questions,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    def match(self, user_question):
        """Find best matching FAQ"""
        # Encode user question
        query_emb = self.model.encode(
            user_question,
            normalize_embeddings=True
        )

        # Compute similarities
        similarities = np.dot(self.faq_embeddings, query_emb)

        # Best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score >= self.threshold:
            return {
                'question': self.faqs[best_idx]['question'],
                'answer': self.faqs[best_idx]['answer'],
                'confidence': float(best_score)
            }
        else:
            return {
                'question': None,
                'answer': "I'm not sure about that. Please contact support.",
                'confidence': float(best_score)
            }

# Setup FAQ system
faq_system = FAQMatcher(threshold=0.6)

faqs = [
    {
        'question': "How do I reset my password?",
        'answer': "Go to Settings > Account > Reset Password and follow the instructions."
    },
    {
        'question': "What are your business hours?",
        'answer': "We're open Monday-Friday, 9AM-5PM EST."
    },
    {
        'question': "How do I cancel my subscription?",
        'answer': "Visit Account Settings > Billing > Cancel Subscription."
    },
    {
        'question': "Do you offer refunds?",
        'answer': "Yes, we offer full refunds within 30 days of purchase."
    }
]

faq_system.add_faqs(faqs)

# Test queries
test_queries = [
    "I forgot my password, how can I reset it?",  # Similar to FAQ 1
    "When are you open?",  # Similar to FAQ 2
    "What's the meaning of life?"  # No match
]

print("FAQ Matching System:\n")
for query in test_queries:
    result = faq_system.match(query)
    print(f"❓ User: '{query}'")
    print(f"   Confidence: {result['confidence']:.2f}")
    if result['question']:
        print(f"   Matched: '{result['question']}'")
    print(f"   Answer: {result['answer']}")
    print()
```

---

## 15. Monitoring Embedding Quality in Production

```python
# Example 23: Monitor embedding drift
class EmbeddingMonitor:
    """Monitor embedding quality over time"""

    def __init__(self, reference_embeddings, reference_labels):
        self.reference_embeddings = reference_embeddings
        self.reference_labels = reference_labels
        self.reference_centroid = np.mean(reference_embeddings, axis=0)

    def check_drift(self, new_embeddings):
        """Detect if new embeddings have drifted"""
        new_centroid = np.mean(new_embeddings, axis=0)

        # Cosine similarity between centroids
        drift_score = 1 - distance.cosine(self.reference_centroid, new_centroid)

        # Average distance from reference
        distances = []
        for new_emb in new_embeddings:
            # Find nearest reference embedding
            dists = [distance.cosine(new_emb, ref_emb)
                     for ref_emb in self.reference_embeddings]
            distances.append(min(dists))

        return {
            'centroid_similarity': drift_score,
            'avg_min_distance': np.mean(distances),
            'max_distance': np.max(distances),
            'drifted': drift_score < 0.9 or np.mean(distances) > 0.3
        }

# Example usage
model = SentenceTransformer('all-MiniLM-L6-v2')

# Reference data (from development)
ref_docs = ["Python programming", "Machine learning", "Data science"]
ref_embeddings = model.encode(ref_docs)
ref_labels = [0, 1, 1]

monitor = EmbeddingMonitor(ref_embeddings, ref_labels)

# New production data
new_docs = ["Python coding", "ML algorithms", "AI systems"]
new_embeddings = model.encode(new_docs)

drift_report = monitor.check_drift(new_embeddings)

print("Embedding Drift Report:")
for metric, value in drift_report.items():
    print(f"  {metric}: {value}")
```

---

## Practice Exercises

### Exercise 1: Build a Duplicate Detector
Create a system that finds duplicate documents using embedding similarity.

### Exercise 2: Multi-Language Search
Build a search engine that works across multiple languages.

### Exercise 3: Custom Pooling
Implement different pooling strategies and compare their performance.

### Exercise 4: Embedding Compression
Use PCA to reduce embedding dimensions while maintaining quality.

### Exercise 5: Production Pipeline
Build a complete embedding pipeline with caching, batching, and monitoring.

---

## Key Takeaways

1. **Embeddings** convert text into dense vectors where semantic similarity = geometric proximity
2. **Sentence-transformers** provide pre-trained bi-encoders optimized for semantic search
3. **Normalization** (L2) makes cosine similarity = dot product (faster)
4. **Dimensionality** affects quality (768d better) vs speed (384d faster)
5. **Pooling** strategies extract sentence embeddings from token embeddings
6. **Production** considerations: batching, caching, monitoring are essential
7. **Open-source models** (sentence-transformers) vs cloud APIs (OpenAI, Cohere)
8. **Applications**: semantic search, clustering, classification, FAQ matching

---

## Further Reading

### Essential Papers
- **Sentence-BERT** (Reimers & Gurevych, 2019): https://arxiv.org/abs/1908.10084
- **SimCSE** (Gao et al., 2021): https://arxiv.org/abs/2104.08821

### Documentation
- **sentence-transformers**: https://www.sbert.net/
- **Hugging Face**: https://huggingface.co/models?library=sentence-transformers

### Cross-References
- **Module 7 Lesson 2**: Embeddings and Transformer Architecture
- **Module 7 Lesson 6**: RAG & Vector Search basics
- **Module 16 Lesson 2**: Vector Database Internals (next lesson)

### Tutorials
- Sentence-Transformers Training: https://www.sbert.net/docs/training/overview.html
- OpenAI Embeddings Guide: https://platform.openai.com/docs/guides/embeddings

---

**Next Lesson**: Vector Database Internals & Architecture - Learn how to store and search billions of embeddings efficiently!

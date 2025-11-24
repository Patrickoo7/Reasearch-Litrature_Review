# Lesson 2: Embeddings & Transformer Architecture 🧠

**Module 7: Natural Language Processing | Lesson 2 of 7**

From static word vectors to transformers - understand the architecture that powers GPT, BERT, and modern LLMs!

---

## Evolution of Word Representations 📈

### The Journey

```
One-Hot (1960s) → Word2Vec (2013) → GloVe (2014) → ELMo (2018) → BERT/GPT (2018+)
  Sparse            Dense              Dense          Contextual      Contextual
  No meaning        Semantic           Global         Bidirectional   Task-specific
```

---

## 1. Static Embeddings: Word2Vec 🎯

### How Word2Vec Works

**Two architectures:**
1. **Skip-gram:** Predict context from center word
2. **CBOW (Continuous Bag of Words):** Predict center word from context

```
Skip-gram Example:
Sentence: "The quick brown fox jumps"
Center: "brown"
Context: ["The", "quick", "fox", "jumps"]

Task: Given "brown" → predict "quick", "fox", etc.
```

### Train Word2Vec from Scratch

```python
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
nltk.download('punkt')

# Sample corpus
corpus_text = """
Natural language processing is a field of artificial intelligence.
It focuses on the interaction between computers and human language.
Machine learning models can understand and generate text.
Deep learning has revolutionized NLP in recent years.
Transformers are the foundation of modern language models.
"""

# Tokenize into sentences
sentences = sent_tokenize(corpus_text.lower())
tokenized_sentences = [word_tokenize(sent) for sent in sentences]

print(f"Training on {len(tokenized_sentences)} sentences")

# Train Word2Vec
model = Word2Vec(
    sentences=tokenized_sentences,
    vector_size=100,      # Embedding dimension
    window=5,             # Context window
    min_count=1,          # Minimum word frequency
    workers=4,            # Parallel threads
    sg=1,                 # 1=skip-gram, 0=CBOW
    epochs=100
)

# Get embedding for a word
vector = model.wv['language']
print(f"\nEmbedding for 'language': {vector[:10]}...")  # First 10 dimensions
print(f"Shape: {vector.shape}")

# Save model
model.save("word2vec.model")
```

### Word Similarity & Analogies

```python
# Most similar words
similar_words = model.wv.most_similar('language', topn=5)
print("\nMost similar to 'language':")
for word, score in similar_words:
    print(f"  {word}: {score:.4f}")

# Word analogies: king - man + woman = ?
# Formula: vec(queen) ≈ vec(king) - vec(man) + vec(woman)
try:
    result = model.wv.most_similar(
        positive=['woman', 'king'],
        negative=['man'],
        topn=1
    )
    print(f"\nking - man + woman = {result[0][0]}")
except:
    print("\nNot enough training data for analogies")

# Cosine similarity
from scipy.spatial.distance import cosine

def similarity(word1, word2):
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    return 1 - cosine(vec1, vec2)

print(f"\nSimilarity('language', 'processing'): {similarity('language', 'processing'):.4f}")
print(f"Similarity('language', 'banana'): {similarity('language', 'banana'):.4f}")
```

### Pre-trained Word2Vec

```python
import gensim.downloader as api

# Download pre-trained Google News vectors (1.6GB!)
# word2vec_model = api.load('word2vec-google-news-300')

# Smaller alternative
glove_model = api.load('glove-wiki-gigaword-50')  # 50-dim GloVe

# Test
similar = glove_model.most_similar('computer', topn=10)
print("\nMost similar to 'computer':")
for word, score in similar:
    print(f"  {word}: {score:.4f}")

# Famous analogy
result = glove_model.most_similar(
    positive=['woman', 'king'],
    negative=['man'],
    topn=1
)
print(f"\nking - man + woman = {result[0][0]} (score: {result[0][1]:.4f})")
```

### Visualize Embeddings with t-SNE

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Select words to visualize
words = ['language', 'processing', 'computer', 'learning', 'model',
         'transformer', 'artificial', 'intelligence', 'neural', 'network']

# Get vectors
word_vectors = np.array([glove_model[word] for word in words])

# Reduce to 2D with t-SNE
tsne = TSNE(n_components=2, random_state=42)
vectors_2d = tsne.fit_transform(word_vectors)

# Plot
plt.figure(figsize=(12, 8))
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], s=100, alpha=0.7)

for i, word in enumerate(words):
    plt.annotate(word, 
                xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=12,
                weight='bold')

plt.title('Word Embeddings Visualization (t-SNE)', fontsize=16)
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('word_embeddings_tsne.png', dpi=300)
plt.show()
```

---

## 2. GloVe: Global Vectors 🌐

### How GloVe Differs from Word2Vec

- **Word2Vec:** Local context window
- **GloVe:** Global co-occurrence statistics

### Using Pre-trained GloVe

```python
# GloVe available in multiple sizes
models_available = {
    'glove-wiki-gigaword-50': '50 dimensions',
    'glove-wiki-gigaword-100': '100 dimensions',
    'glove-wiki-gigaword-200': '200 dimensions',
    'glove-wiki-gigaword-300': '300 dimensions',
}

# Load
glove = api.load('glove-wiki-gigaword-100')

# Test semantic relationships
print("GloVe Semantic Tests:")
print("-" * 50)

# Countries and capitals
tests = [
    (['paris', 'france'], ['berlin'], 'germany'),
    (['tokyo', 'japan'], ['beijing'], 'china'),
    (['man', 'king'], ['woman'], 'queen'),
]

for positive, negative, expected in tests:
    result = glove.most_similar(positive=positive, negative=negative, topn=1)
    print(f"{positive} - {negative} = {result[0][0]} (expected: {expected})")
```

---

## 3. Contextual Embeddings 🔄

### Problem with Static Embeddings

```python
# Word "bank" has multiple meanings
sentences = [
    "I deposited money in the bank",      # Financial institution
    "We sat by the river bank",           # River edge
    "The plane began to bank left",       # Tilt/turn
]

# Word2Vec gives same vector for "bank" in all contexts!
# This is a limitation of static embeddings
```

### Solution: Contextual Embeddings

**ELMo, BERT, GPT** create different embeddings based on context!

---

## 4. Transformer Architecture Deep Dive 🏗️

### Self-Attention Mechanism

**Core Idea:** Each word attends to all other words to understand context

### Implement Self-Attention from Scratch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads=1):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert self.head_dim * heads == embed_size, "Embed size must be divisible by heads"
        
        # Linear transformations for Q, K, V
        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        
        # Output projection
        self.fc_out = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        N, seq_len, embed_size = x.shape
        
        # Linear projections
        Q = self.queries(x)  # (N, seq_len, embed_size)
        K = self.keys(x)
        V = self.values(x)
        
        # Reshape for multi-head attention
        Q = Q.reshape(N, seq_len, self.heads, self.head_dim)
        K = K.reshape(N, seq_len, self.heads, self.head_dim)
        V = V.reshape(N, seq_len, self.heads, self.head_dim)
        
        # Scaled dot-product attention
        # Q @ K^T / sqrt(d_k)
        energy = torch.einsum("nqhd,nkhd->nhqk", Q, K)
        
        # Scale
        energy = energy / (self.head_dim ** 0.5)
        
        # Softmax to get attention weights
        attention = F.softmax(energy, dim=3)  # (N, heads, seq_len, seq_len)
        
        # Apply attention to values
        out = torch.einsum("nhql,nlhd->nqhd", attention, V)
        
        # Concatenate heads
        out = out.reshape(N, seq_len, embed_size)
        
        # Final linear projection
        out = self.fc_out(out)
        
        return out, attention

# Test
embed_size = 256
seq_len = 10
batch_size = 2

# Random input embeddings
x = torch.randn(batch_size, seq_len, embed_size)

# Create attention layer
attention_layer = SelfAttention(embed_size, heads=8)

# Forward pass
output, attention_weights = attention_layer(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {attention_weights.shape}")
```

### Visualize Attention Weights

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Get attention weights for first sample, first head
attn = attention_weights[0, 0].detach().numpy()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attn, cmap='viridis', cbar=True, square=True)
plt.title('Self-Attention Weights (Head 1)', fontsize=14)
plt.xlabel('Key Position', fontsize=12)
plt.ylabel('Query Position', fontsize=12)
plt.tight_layout()
plt.savefig('attention_heatmap.png', dpi=300)
plt.show()
```

### Multi-Head Attention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        # Compute attention
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        
        # Apply mask if provided (for padding)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        attention = torch.softmax(energy / (self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", attention, values).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        return out
```

### Position Encoding

```python
class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEncoding, self).__init__()
        
        # Create position encodings
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * 
                           (-np.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        return x + self.pe[:, :x.size(1)]

# Visualize position encodings
pe = PositionalEncoding(embed_size=128, max_len=50)
positions = pe.pe[0].numpy()

plt.figure(figsize=(12, 6))
plt.imshow(positions.T, cmap='RdBu', aspect='auto')
plt.colorbar()
plt.xlabel('Position')
plt.ylabel('Embedding Dimension')
plt.title('Sinusoidal Positional Encodings')
plt.tight_layout()
plt.savefig('positional_encoding.png', dpi=300)
plt.show()
```

### Complete Transformer Encoder Block

```python
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, value, key, query, mask):
        # Multi-head attention
        attention = self.attention(value, key, query, mask)
        
        # Add & Norm
        x = self.dropout(self.norm1(attention + query))
        
        # Feed forward
        forward = self.feed_forward(x)
        
        # Add & Norm
        out = self.dropout(self.norm2(forward + x))
        
        return out

# Test complete transformer block
block = TransformerBlock(
    embed_size=256,
    heads=8,
    dropout=0.1,
    forward_expansion=4
)

x = torch.randn(2, 10, 256)
output = block(x, x, x, mask=None)
print(f"Transformer block output shape: {output.shape}")
```

---

## 5. BERT Embeddings (Contextual) 🎭

### Extract BERT Embeddings

```python
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example sentences with "bank"
sentences = [
    "I deposited money in the bank",
    "We sat by the river bank"
]

# Encode
for sent in sentences:
    inputs = tokenizer(sent, return_tensors='pt')
    
    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Last hidden state
    last_hidden = outputs.last_hidden_state  # (1, seq_len, 768)
    
    # Find "bank" token
    tokens = tokenizer.tokenize(sent)
    bank_idx = tokens.index('bank') + 1  # +1 for [CLS]
    
    bank_embedding = last_hidden[0, bank_idx, :]
    
    print(f"\nSentence: {sent}")
    print(f"Tokens: {tokens}")
    print(f"'bank' embedding shape: {bank_embedding.shape}")
    print(f"First 5 dimensions: {bank_embedding[:5]}")

# The embeddings are DIFFERENT for each context!
```

### Layer-wise Analysis

```python
# Get all layers
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

text = "Natural language processing is fascinating"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

# All hidden states (13 layers: embedding + 12 transformer layers)
all_hidden_states = outputs.hidden_states

print(f"Number of layers: {len(all_hidden_states)}")
print(f"Each layer shape: {all_hidden_states[0].shape}")

# Analyze how "language" is represented in each layer
tokens = tokenizer.tokenize(text)
lang_idx = tokens.index('language') + 1

layer_embeddings = []
for layer_idx, hidden_state in enumerate(all_hidden_states):
    lang_emb = hidden_state[0, lang_idx, :]
    layer_embeddings.append(lang_emb.numpy())

# Visualize embedding changes across layers
layer_norms = [np.linalg.norm(emb) for emb in layer_embeddings]

plt.figure(figsize=(10, 6))
plt.plot(range(len(layer_norms)), layer_norms, marker='o')
plt.xlabel('Layer')
plt.ylabel('Embedding Norm')
plt.title('How "language" embedding evolves across BERT layers')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('bert_layer_evolution.png', dpi=300)
plt.show()
```

---

## 6. Sentence Embeddings 📝

### Problem with Word Embeddings

```python
# Can't directly compare sentences
# Need: Sentence-level representation
```

### Sentence-BERT (SBERT)

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encode sentences
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "The weather is nice today"
]

embeddings = model.encode(sentences)

print(f"Sentence embeddings shape: {embeddings.shape}")  # (3, 384)

# Compute similarities
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(embeddings)

print("\nCosine Similarities:")
for i, sent1 in enumerate(sentences):
    for j, sent2 in enumerate(sentences):
        if i < j:
            print(f"{sent1[:30]:30} <-> {sent2[:30]:30}: {similarities[i][j]:.4f}")
```

### Semantic Search

```python
# Create a document corpus
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Deep learning uses neural networks with multiple layers",
    "Natural language processing helps computers understand text",
    "Computer vision enables machines to interpret images",
    "Reinforcement learning trains agents through rewards"
]

# Encode all documents
doc_embeddings = model.encode(documents)

# Query
query = "How do computers understand language?"
query_embedding = model.encode([query])

# Find most similar document
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

# Rank by similarity
ranked_indices = np.argsort(similarities)[::-1]

print(f"Query: {query}\n")
print("Most similar documents:")
for idx in ranked_indices[:3]:
    print(f"{similarities[idx]:.4f}: {documents[idx]}")
```

### Clustering with Sentence Embeddings

```python
from sklearn.cluster import KMeans

# More documents
docs = [
    "Machine learning algorithms",
    "Deep neural networks",
    "The weather is sunny",
    "It's raining today",
    "Natural language processing",
    "Computer vision applications"
]

# Encode
embeddings = model.encode(docs)

# Cluster
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# Display clusters
for cluster_id in range(2):
    print(f"\nCluster {cluster_id}:")
    cluster_docs = [doc for doc, c in zip(docs, clusters) if c == cluster_id]
    for doc in cluster_docs:
        print(f"  - {doc}")
```

---

## Quick Reference 📖

### Choosing Embedding Method

```
Use Case                         Recommendation
─────────────────────────────────────────────────────────
Single word semantics            Word2Vec, GloVe
Need context-awareness           BERT embeddings
Sentence similarity              Sentence-BERT
Multilingual                     LaBSE, mBERT
Code embeddings                  CodeBERT
Domain-specific                  Fine-tune BERT
```

### Common Embedding Dimensions

```
Model                   Dimensions
──────────────────────────────────
Word2Vec (Google News)  300
GloVe                   50-300
BERT-base              768
BERT-large             1024
Sentence-BERT          384-1024
GPT-2                  768
GPT-3                  12288
```

---

## Practice Exercises 🏋️

### Exercise 1: Word Analogies
Train Word2Vec on a large corpus and test analogies:
- Geography: paris:france :: tokyo:?
- Grammar: walk:walked :: eat:?
- Gender: king:queen :: man:?

### Exercise 2: Context Matters
Extract BERT embeddings for "apple" in:
- "I ate an apple"
- "Apple released a new iPhone"
Compare the embeddings.

### Exercise 3: Build Semantic Search
Create a search engine for your documents using Sentence-BERT.

---

## Key Takeaways 💡

1. **Static embeddings** (Word2Vec, GloVe) give same vector regardless of context
2. **Contextual embeddings** (BERT, GPT) change based on surrounding words
3. **Self-attention** allows words to attend to all other words
4. **Transformers** = Multi-head Attention + Feed-Forward + Residual + LayerNorm
5. **Sentence-BERT** creates sentence-level embeddings for semantic search
6. **Position encoding** adds word order information to transformers

---

**Next:** [Lesson 3 - Syntax, Semantics & Language Understanding →](Lesson%203%20-%20Syntax%20Semantics%20and%20Language%20Understanding.md)

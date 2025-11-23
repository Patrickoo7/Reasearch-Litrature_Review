# Lesson 5: Query Optimization, Reranking & Hybrid Search 🎯

**Module 16: RAG, Vector Databases & AI Agents | Lesson 5 of 8**

Master advanced retrieval techniques - from hybrid search to cross-encoder reranking used in production systems!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Implement query expansion and reformulation strategies
2. ✅ Understand semantic search vs keyword search trade-offs
3. ✅ Build hybrid search combining BM25 + vector similarity
4. ✅ Use cross-encoders for precision reranking
5. ✅ Implement two-stage retrieval (recall + reranking)
6. ✅ Apply Reciprocal Rank Fusion (RRF) to merge results
7. ✅ Perform query understanding and intent detection
8. ✅ Measure retrieval quality with MRR, NDCG, Precision@K

---

## Prerequisites

- **Module 16 Lesson 1**: Vector Embeddings
- **Module 16 Lesson 4**: Advanced RAG Patterns
- Understanding of ranking algorithms
- Python and scikit-learn

---

## 1. Query Expansion & Reformulation

### Query Expansion with Synonyms

```python
# Example 1: Query expansion using synonyms
import nltk
from nltk.corpus import wordnet

# Download wordnet (first time only)
# nltk.download('wordnet')
# nltk.download('omw-1.4')

class QueryExpander:
    """
    Expand query with synonyms to improve recall.

    Example: "car" → "car automobile vehicle"
    """

    def expand_with_synonyms(self, query, max_synonyms=3):
        """Add synonyms to query terms"""
        expanded_terms = []

        for word in query.split():
            # Add original word
            expanded_terms.append(word)

            # Get synonyms from WordNet
            synsets = wordnet.synsets(word)
            synonyms = set()

            for synset in synsets[:2]:  # Limit synsets
                for lemma in synset.lemmas()[:max_synonyms]:
                    syn = lemma.name().replace('_', ' ')
                    if syn.lower() != word.lower():
                        synonyms.add(syn)

            expanded_terms.extend(list(synonyms)[:max_synonyms])

        return ' '.join(expanded_terms)

# Test
expander = QueryExpander()

queries = [
    "car repair",
    "happy customer",
    "fast computer"
]

print("Query Expansion with Synonyms:\n")
for query in queries:
    expanded = expander.expand_with_synonyms(query, max_synonyms=2)
    print(f"Original:  {query}")
    print(f"Expanded:  {expanded}")
    print()
```

### Query Reformulation with LLM

```python
# Example 2: Query reformulation using LLM
"""
In production: Use LLM to reformulate queries

Prompt template:
'''
Reformulate this user query into a better search query:

User query: {original_query}

Reformulated query (be specific, add context):
'''

Example transformations:
  "it" → "machine learning"
  "how to fix?" → "how to fix Python installation error"
  "that thing" → "neural network architecture"
"""

class LLMQueryReformulator:
    """Reformulate queries using LLM"""

    def __init__(self, llm=None):
        self.llm = llm  # In production: OpenAI, Anthropic, etc.

    def reformulate(self, query, context=None):
        """
        Reformulate query to be more specific.

        In production: Call LLM API
        """
        # Mock reformulations for demo
        reformulations = {
            "it": "machine learning and artificial intelligence",
            "how does it work?": "how does machine learning work in detail",
            "that thing": "neural network architecture and training process",
            "fix bug": "debug and fix software bugs step by step"
        }

        reformulated = reformulations.get(query.lower(), query)

        # In production:
        # reformulated = self.llm.generate(
        #     f"Reformulate for search: {query}\nReformulated:"
        # )

        return reformulated

# Demo
reformulator = LLMQueryReformulator()

vague_queries = [
    "it",
    "how does it work?",
    "that thing",
    "fix bug"
]

print("Query Reformulation:\n")
for query in vague_queries:
    reformed = reformulator.reformulate(query)
    print(f"Vague:       {query}")
    print(f"Reformulated: {reformed}")
    print()
```

---

## 2. Semantic Search vs Keyword Search

### Understanding the Trade-offs

```python
# Example 3: Compare semantic vs keyword search
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class SearchComparison:
    """Compare semantic and keyword search"""

    def __init__(self):
        # Semantic: dense embeddings
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Keyword: TF-IDF (sparse vectors)
        self.tfidf = TfidfVectorizer()

    def index_documents(self, documents):
        """Index with both methods"""
        self.documents = documents

        # Semantic embeddings
        self.semantic_embeddings = self.semantic_model.encode(
            documents,
            normalize_embeddings=True
        )

        # TF-IDF vectors
        self.tfidf_matrix = self.tfidf.fit_transform(documents)

    def search_semantic(self, query, top_k=3):
        """Dense vector semantic search"""
        query_emb = self.semantic_model.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.semantic_embeddings, query_emb)
        top_idx = np.argsort(-similarities)[:top_k]
        return [(idx, similarities[idx]) for idx in top_idx]

    def search_keyword(self, query, top_k=3):
        """Sparse keyword search (TF-IDF)"""
        query_vec = self.tfidf.transform([query])
        similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_idx = np.argsort(-similarities)[:top_k]
        return [(idx, similarities[idx]) for idx in top_idx]

# Test
searcher = SearchComparison()

documents = [
    "Python is a programming language used for data science",
    "Machine learning models learn patterns from data",
    "Neural networks are inspired by the human brain",
    "The weather today is sunny and warm",
    "Deep learning uses multiple layers of neural networks",
    "Data scientists use Python for analysis",
]

searcher.index_documents(documents)

# Test queries
queries = [
    "python data analysis",  # Exact keywords
    "AI learning from examples",  # Semantic, different words
    "coding for statistics",  # Semantic similarity
]

print("Semantic vs Keyword Search:\n")
for query in queries:
    print(f"Query: '{query}'")

    # Semantic
    sem_results = searcher.search_semantic(query, top_k=2)
    print("  Semantic search:")
    for idx, score in sem_results:
        print(f"    [{score:.3f}] {documents[idx]}")

    # Keyword
    kw_results = searcher.search_keyword(query, top_k=2)
    print("  Keyword search:")
    for idx, score in kw_results:
        if score > 0:
            print(f"    [{score:.3f}] {documents[idx]}")
        else:
            print(f"    [no match]")
    print()

print("\nKey Differences:")
print("  Semantic: Understands meaning, works with synonyms")
print("  Keyword:  Fast, works with exact terms, no synonyms")
```

---

## 3. Hybrid Search: BM25 + Vector

### BM25 Algorithm

```python
# Example 4: BM25 implementation
from rank_bm25 import BM25Okapi
import numpy as np

class BM25Search:
    """
    BM25: Best Matching 25 - probabilistic ranking function.

    Improvements over TF-IDF:
    - Saturation: term frequency saturates (diminishing returns)
    - Document length normalization
    - Tunable parameters (k1, b)
    """

    def __init__(self, k1=1.5, b=0.75):
        """
        Args:
            k1: Term frequency saturation parameter (1.2-2.0)
            b: Length normalization (0-1, 0.75 typical)
        """
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.documents = None

    def index(self, documents):
        """Index documents with BM25"""
        self.documents = documents

        # Tokenize (simple whitespace split)
        tokenized_docs = [doc.lower().split() for doc in documents]

        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_docs, k1=self.k1, b=self.b)

    def search(self, query, top_k=5):
        """Search with BM25"""
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Top-k
        top_indices = np.argsort(-scores)[:top_k]

        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

# Test BM25
bm25_search = BM25Search()

docs = [
    "Python is a programming language",
    "Machine learning uses Python for data analysis",
    "Python programming is popular in data science",
    "Java is also a programming language",
    "The weather is nice today"
]

bm25_search.index(docs)

query = "python programming"
results = bm25_search.search(query, top_k=3)

print("BM25 Search Results:\n")
print(f"Query: '{query}'")
for idx, score in results:
    print(f"  Score: {score:.3f} - {docs[idx]}")
```

### Hybrid Search Implementation

```python
# Example 5: Combine BM25 (keyword) + vector (semantic)
class HybridSearch:
    """
    Hybrid search: Combine BM25 and dense vector search.

    Strategy:
    1. BM25 for exact keyword matches
    2. Vector for semantic similarity
    3. Combine scores with weights
    """

    def __init__(self, alpha=0.5):
        """
        Args:
            alpha: Weight for vector search (1-alpha for BM25)
                   0.0 = pure BM25
                   1.0 = pure vector
                   0.5 = equal weight
        """
        self.alpha = alpha
        self.bm25_search = BM25Search()
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.documents = None
        self.embeddings = None

    def index(self, documents):
        """Index with both BM25 and vector"""
        self.documents = documents

        # BM25 index
        self.bm25_search.index(documents)

        # Vector index
        self.embeddings = self.semantic_model.encode(
            documents,
            normalize_embeddings=True
        )

    def search(self, query, top_k=5):
        """Hybrid search: combine BM25 + vector"""
        # 1. BM25 scores
        bm25_results = self.bm25_search.search(query, top_k=len(self.documents))
        bm25_scores = np.zeros(len(self.documents))
        for idx, score in bm25_results:
            bm25_scores[idx] = score

        # Normalize BM25 scores to [0, 1]
        if bm25_scores.max() > 0:
            bm25_scores = bm25_scores / bm25_scores.max()

        # 2. Vector scores
        query_emb = self.semantic_model.encode(query, normalize_embeddings=True)
        vector_scores = np.dot(self.embeddings, query_emb)

        # Vector scores already in [0, 1] due to normalization

        # 3. Combine scores
        hybrid_scores = (self.alpha * vector_scores) + ((1 - self.alpha) * bm25_scores)

        # 4. Top-k
        top_indices = np.argsort(-hybrid_scores)[:top_k]

        return [(idx, hybrid_scores[idx]) for idx in top_indices]

# Test hybrid search
hybrid = HybridSearch(alpha=0.5)  # Equal weight

hybrid.index(docs)

queries = [
    "python programming",  # Has exact keywords
    "coding for AI",  # Semantic match
]

print("\n" + "="*60)
print("Hybrid Search (BM25 + Vector):")
print("="*60)

for query in queries:
    print(f"\nQuery: '{query}'")

    # Compare all three
    results_bm25 = bm25_search.search(query, top_k=3)
    results_hybrid = hybrid.search(query, top_k=3)

    print("  BM25 only:")
    for idx, score in results_bm25:
        print(f"    [{score:.3f}] {docs[idx]}")

    print("  Hybrid (BM25 + Vector):")
    for idx, score in results_hybrid:
        print(f"    [{score:.3f}] {docs[idx]}")

print("\n💡 Hybrid search gets best of both worlds!")
```

---

## 4. Cross-Encoders for Reranking

### Bi-Encoder vs Cross-Encoder

```python
# Example 6: Understanding cross-encoders
"""
Bi-Encoder (fast, for initial retrieval):
  Query → Encoder → Query Embedding ┐
                                     ├→ Cosine Similarity → Score
  Doc → Encoder → Doc Embedding ────┘

  - Encodes query and doc SEPARATELY
  - Fast: pre-compute doc embeddings
  - Good for 1st stage (retrieve top-100)

Cross-Encoder (accurate, for reranking):
  [Query | Doc] → Encoder → Relevance Score

  - Encodes query+doc TOGETHER
  - Full attention between query and doc
  - Slow: must encode each pair
  - Excellent for 2nd stage (rerank top-100 → top-10)

Two-Stage Retrieval:
  1. Bi-encoder: 1M docs → top-100 (fast)
  2. Cross-encoder: top-100 → top-10 (accurate)
"""

print("Bi-Encoder vs Cross-Encoder:")
print("\nBi-Encoder:")
print("  Speed: FAST (pre-computed embeddings)")
print("  Accuracy: Good")
print("  Use: Initial retrieval (1M → 100)")

print("\nCross-Encoder:")
print("  Speed: SLOW (compute each pair)")
print("  Accuracy: EXCELLENT")
print("  Use: Reranking (100 → 10)")
```

### Cross-Encoder Implementation

```python
# Example 7: Cross-encoder reranking
from sentence_transformers import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    """
    Rerank results using cross-encoder.

    Cross-encoder directly scores query-document pairs.
    Much more accurate than bi-encoder!
    """

    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        """
        Load cross-encoder model.

        Popular models:
        - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good)
        - cross-encoder/ms-marco-MiniLM-L-12-v2 (slower, better)
        """
        self.model = CrossEncoder(model_name)

    def rerank(self, query, documents, top_k=5):
        """
        Rerank documents for query.

        Args:
            query: Search query
            documents: List of candidate documents
            top_k: Number of results to return

        Returns:
            List of (doc_idx, score) tuples
        """
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]

        # Score all pairs (this is the slow part!)
        scores = self.model.predict(pairs)

        # Sort by score
        ranked = sorted(
            enumerate(scores),
            key=lambda x: -x[1]
        )[:top_k]

        return ranked

# Test cross-encoder
print("\n" + "="*60)
print("Cross-Encoder Reranking:")
print("="*60)

reranker = CrossEncoderReranker()

query = "how to learn machine learning"

candidates = [
    "Machine learning is a subset of AI that focuses on learning from data",
    "To learn machine learning, start with Python and statistics fundamentals",
    "The weather forecast predicts rain tomorrow afternoon",
    "Step-by-step guide to learning ML: 1) Learn Python, 2) Study math, 3) Practice",
    "Deep learning is a branch of machine learning using neural networks"
]

print(f"\nQuery: '{query}'")
print(f"Candidates: {len(candidates)} documents")

# Rerank
import time
start = time.time()
results = reranker.rerank(query, candidates, top_k=3)
rerank_time = time.time() - start

print(f"\nReranking time: {rerank_time*1000:.2f}ms")
print("\nTop results:")
for rank, (idx, score) in enumerate(results, 1):
    print(f"  {rank}. Score: {score:.3f}")
    print(f"     {candidates[idx]}")
```

---

## 5. Two-Stage Retrieval

### Recall then Rerank

```python
# Example 8: Two-stage retrieval pipeline
class TwoStageRetriever:
    """
    Two-stage retrieval for best speed + accuracy.

    Stage 1 (Recall): Bi-encoder retrieves top-N candidates (fast)
    Stage 2 (Rerank): Cross-encoder reranks to top-K (accurate)
    """

    def __init__(self, recall_top_n=100, final_top_k=10):
        # Stage 1: Bi-encoder (fast recall)
        self.bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Stage 2: Cross-encoder (accurate reranking)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        self.recall_top_n = recall_top_n
        self.final_top_k = final_top_k

        self.documents = None
        self.doc_embeddings = None

    def index(self, documents):
        """Index documents (bi-encoder only)"""
        self.documents = documents

        # Pre-compute embeddings (one-time cost)
        self.doc_embeddings = self.bi_encoder.encode(
            documents,
            normalize_embeddings=True,
            show_progress_bar=True
        )

    def retrieve(self, query):
        """
        Two-stage retrieval:
        1. Bi-encoder: Fast recall (top-N)
        2. Cross-encoder: Precise reranking (top-K)
        """
        import time

        # Stage 1: Fast bi-encoder recall
        print(f"\n🔍 Stage 1: Bi-encoder recall (top-{self.recall_top_n})...")
        start = time.time()

        query_emb = self.bi_encoder.encode(query, normalize_embeddings=True)
        similarities = np.dot(self.doc_embeddings, query_emb)

        # Get top-N candidates
        top_n_indices = np.argsort(-similarities)[:self.recall_top_n]
        candidates = [self.documents[i] for i in top_n_indices]

        stage1_time = time.time() - start
        print(f"   Time: {stage1_time*1000:.2f}ms")
        print(f"   Retrieved: {len(candidates)} candidates")

        # Stage 2: Accurate cross-encoder reranking
        print(f"\n🎯 Stage 2: Cross-encoder reranking (top-{self.final_top_k})...")
        start = time.time()

        # Create pairs
        pairs = [[query, doc] for doc in candidates]
        scores = self.cross_encoder.predict(pairs)

        # Rerank
        reranked = sorted(
            zip(top_n_indices, scores),
            key=lambda x: -x[1]
        )[:self.final_top_k]

        stage2_time = time.time() - start
        print(f"   Time: {stage2_time*1000:.2f}ms")

        total_time = stage1_time + stage2_time
        print(f"\n⏱️  Total time: {total_time*1000:.2f}ms")

        return reranked

# Demo (requires larger dataset to see benefit)
print("\n" + "="*60)
print("Two-Stage Retrieval Pipeline:")
print("="*60)

print("\nBenefits:")
print("  ✅ Fast: Bi-encoder handles millions of docs")
print("  ✅ Accurate: Cross-encoder refines top candidates")
print("  ✅ Scalable: Best of both worlds!")

print("\nTypical setup:")
print("  Dataset: 1M documents")
print("  Stage 1: Bi-encoder retrieves top-100 (~50ms)")
print("  Stage 2: Cross-encoder reranks to top-10 (~100ms)")
print("  Total: ~150ms for excellent results")
```

---

## 6. Reciprocal Rank Fusion (RRF)

### Merging Multiple Rankings

```python
# Example 9: Reciprocal Rank Fusion
def reciprocal_rank_fusion(rankings, k=60):
    """
    Merge multiple rankings using RRF.

    RRF score = Σ 1 / (k + rank_i)

    Args:
        rankings: List of rankings (each ranking is list of doc IDs)
        k: Constant (typically 60)

    Returns:
        Merged ranking
    """
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0.0

            # Add reciprocal rank
            scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by score
    merged = sorted(scores.items(), key=lambda x: -x[1])

    return merged

# Example: Merge BM25 and vector search results
print("\n" + "="*60)
print("Reciprocal Rank Fusion (RRF):")
print("="*60)

# Two different rankings
bm25_ranking = [0, 2, 1, 5, 3]  # Doc IDs in order
vector_ranking = [2, 0, 4, 1, 6]

print("\nBM25 ranking:   ", bm25_ranking)
print("Vector ranking: ", vector_ranking)

# Merge with RRF
merged = reciprocal_rank_fusion([bm25_ranking, vector_ranking], k=60)

print("\nMerged (RRF):")
for doc_id, score in merged:
    print(f"  Doc {doc_id}: score = {score:.4f}")

print("\n💡 RRF combines different retrieval methods without tuning!")
```

### Production RRF Implementation

```python
# Example 10: RRF with multiple retrievers
class MultiRetrieverRRF:
    """
    Combine multiple retrievers with RRF.

    Use case: Merge BM25, vector, and reranked results.
    """

    def __init__(self, retrievers, k=60):
        """
        Args:
            retrievers: List of retriever objects
            k: RRF constant
        """
        self.retrievers = retrievers
        self.k = k

    def retrieve(self, query, top_k=10):
        """
        Retrieve from all retrievers, merge with RRF.
        """
        # Get rankings from each retriever
        all_rankings = []

        for i, retriever in enumerate(self.retrievers):
            results = retriever.search(query, top_k=50)  # Get more candidates

            # Extract doc IDs
            ranking = [doc_id for doc_id, _ in results]
            all_rankings.append(ranking)

            print(f"Retriever {i+1}: {len(ranking)} results")

        # Merge with RRF
        merged = reciprocal_rank_fusion(all_rankings, k=self.k)

        return merged[:top_k]

print("\nMulti-Retriever RRF:")
print("  Combine: BM25 + Vector + Reranker")
print("  Benefit: Robust, no weight tuning needed")
print("  Used by: Elasticsearch, many production systems")
```

---

## 7. Query Understanding & Intent Detection

### Query Intent Classification

```python
# Example 11: Detect query intent
class QueryIntentClassifier:
    """
    Classify query intent to route to appropriate retrieval strategy.

    Intents:
    - Factual: "What is Python?"
    - How-to: "How to install Python?"
    - Comparison: "Python vs Java"
    - Troubleshooting: "Fix Python error"
    """

    def classify_intent(self, query):
        """
        Classify query intent (rule-based).

        In production: Use ML classifier or LLM
        """
        query_lower = query.lower()

        # How-to questions
        if any(word in query_lower for word in ['how to', 'how do', 'how can']):
            return 'how-to'

        # Comparison
        if 'vs' in query_lower or 'versus' in query_lower or 'compare' in query_lower:
            return 'comparison'

        # Troubleshooting
        if any(word in query_lower for word in ['fix', 'error', 'problem', 'issue', 'debug']):
            return 'troubleshooting'

        # Factual
        if any(word in query_lower for word in ['what is', 'what are', 'define']):
            return 'factual'

        # Default
        return 'general'

    def route_query(self, query):
        """Route query based on intent"""
        intent = self.classify_intent(query)

        strategies = {
            'how-to': 'Use tutorial/guide corpus, prefer detailed chunks',
            'comparison': 'Retrieve multiple entities, use structured comparison',
            'troubleshooting': 'Prioritize recent docs, error messages, solutions',
            'factual': 'Use definition corpus, prefer concise chunks',
            'general': 'Standard hybrid search'
        }

        return {
            'intent': intent,
            'strategy': strategies[intent]
        }

# Test
classifier = QueryIntentClassifier()

test_queries = [
    "What is machine learning?",
    "How to install Python on Mac?",
    "Python vs Java for data science",
    "Fix Python import error",
    "machine learning applications"
]

print("\n" + "="*60)
print("Query Intent Detection:")
print("="*60)

for query in test_queries:
    result = classifier.route_query(query)
    print(f"\nQuery: '{query}'")
    print(f"  Intent: {result['intent']}")
    print(f"  Strategy: {result['strategy']}")
```

---

## 8. Retrieval Quality Metrics

### Mean Reciprocal Rank (MRR)

```python
# Example 12: Calculate MRR
def mean_reciprocal_rank(results, relevant_docs):
    """
    MRR: Average of reciprocal ranks of first relevant doc.

    MRR = average(1 / rank_of_first_relevant_doc)

    Example:
      Query 1: First relevant at rank 1 → RR = 1/1 = 1.0
      Query 2: First relevant at rank 3 → RR = 1/3 = 0.333
      Query 3: No relevant found → RR = 0
      MRR = (1.0 + 0.333 + 0) / 3 = 0.444
    """
    reciprocal_ranks = []

    for query_results, query_relevant in zip(results, relevant_docs):
        # Find rank of first relevant doc
        for rank, doc_id in enumerate(query_results, 1):
            if doc_id in query_relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            # No relevant doc found
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)

# Example
results = [
    [1, 5, 3, 7],  # Query 1 results (doc IDs)
    [2, 4, 6, 8],  # Query 2 results
    [9, 10, 11, 12]  # Query 3 results
]

relevant = [
    [1, 3],  # Query 1 relevant docs
    [6],     # Query 2 relevant docs
    [20]     # Query 3 relevant docs (not in results!)
]

mrr = mean_reciprocal_rank(results, relevant)

print("\n" + "="*60)
print("Mean Reciprocal Rank (MRR):")
print("="*60)
print(f"\nMRR: {mrr:.3f}")
print("\nQuery 1: First relevant at rank 1 → RR = 1.000")
print("Query 2: First relevant at rank 3 → RR = 0.333")
print("Query 3: No relevant found → RR = 0.000")
print(f"Average: {mrr:.3f}")
```

### Normalized Discounted Cumulative Gain (NDCG)

```python
# Example 13: Calculate NDCG
import numpy as np

def dcg_at_k(relevances, k):
    """
    DCG@k: Discounted Cumulative Gain

    DCG = Σ (relevance_i / log2(i + 1))

    Relevance scores (graded):
      0 = Not relevant
      1 = Somewhat relevant
      2 = Relevant
      3 = Highly relevant
    """
    relevances = np.array(relevances)[:k]
    if relevances.size == 0:
        return 0.0

    # Discount by log(rank)
    discounts = np.log2(np.arange(2, relevances.size + 2))
    return np.sum(relevances / discounts)

def ndcg_at_k(predicted_relevances, ideal_relevances, k):
    """
    NDCG@k: Normalized DCG

    NDCG = DCG / IDCG (ideal DCG)

    Range: 0 to 1 (1 = perfect ranking)
    """
    dcg = dcg_at_k(predicted_relevances, k)
    idcg = dcg_at_k(sorted(ideal_relevances, reverse=True), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

# Example
# Retrieved docs with relevance scores
retrieved_relevances = [3, 2, 0, 1, 3, 0, 2]  # Actual retrieval order

# Ideal order (sorted by relevance)
ideal_relevances = [3, 3, 2, 2, 1, 0, 0]

k = 5
ndcg = ndcg_at_k(retrieved_relevances, ideal_relevances, k)

print("\n" + "="*60)
print("Normalized Discounted Cumulative Gain (NDCG):")
print("="*60)
print(f"\nRetrieved order: {retrieved_relevances[:k]}")
print(f"Ideal order:     {sorted(ideal_relevances, reverse=True)[:k]}")
print(f"\nNDCG@{k}: {ndcg:.3f}")
print("\n1.0 = perfect ranking")
print("0.5 = moderate ranking")
print("0.0 = worst ranking")
```

### Precision@K and Recall@K

```python
# Example 14: Precision and Recall metrics
def precision_at_k(retrieved, relevant, k):
    """
    Precision@k = (# relevant in top-k) / k

    Measures: How many retrieved docs are relevant?
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    relevant_retrieved = retrieved_k & relevant_set

    return len(relevant_retrieved) / k

def recall_at_k(retrieved, relevant, k):
    """
    Recall@k = (# relevant in top-k) / (# total relevant)

    Measures: How many relevant docs were retrieved?
    """
    retrieved_k = set(retrieved[:k])
    relevant_set = set(relevant)

    relevant_retrieved = retrieved_k & relevant_set

    return len(relevant_retrieved) / len(relevant_set)

# Example
retrieved = [1, 3, 5, 7, 9, 2, 4, 6, 8, 10]
relevant = [1, 2, 3, 4]  # 4 relevant docs total

print("\n" + "="*60)
print("Precision@K and Recall@K:")
print("="*60)

for k in [1, 3, 5, 10]:
    prec = precision_at_k(retrieved, relevant, k)
    rec = recall_at_k(retrieved, relevant, k)

    print(f"\nK={k}:")
    print(f"  Precision@{k}: {prec:.3f} ({int(prec*k)}/{k} relevant)")
    print(f"  Recall@{k}:    {rec:.3f} ({int(rec*len(relevant))}/{len(relevant)} found)")
```

---

## 9. Production Optimization Tips

### Caching Query Embeddings

```python
# Example 15: Cache query embeddings
import hashlib
import pickle

class CachedRetriever:
    """Cache query embeddings for faster repeated queries"""

    def __init__(self, model):
        self.model = model
        self.cache = {}

    def _hash_query(self, query):
        """Hash query for cache key"""
        return hashlib.md5(query.encode()).hexdigest()

    def encode_query(self, query):
        """Encode with caching"""
        cache_key = self._hash_query(query)

        if cache_key in self.cache:
            return self.cache[cache_key]

        # Compute embedding
        embedding = self.model.encode(query)

        # Cache it
        self.cache[cache_key] = embedding

        return embedding

print("\nProduction Optimization: Query Caching")
print("  ✅ Cache query embeddings (queries often repeat)")
print("  ✅ Cache reranker results")
print("  ✅ Use Redis/Memcached for distributed cache")
```

### Batching for Throughput

```python
# Example 16: Batch processing for high throughput
class BatchedRetriever:
    """Process multiple queries in batch for efficiency"""

    def __init__(self, model, batch_size=32):
        self.model = model
        self.batch_size = batch_size

    def encode_queries(self, queries):
        """Batch encode queries"""
        # Process in batches
        embeddings = self.model.encode(
            queries,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

        return embeddings

print("\nBatching for Throughput:")
print("  ✅ Batch query encoding (10x faster)")
print("  ✅ Batch reranking")
print("  ✅ Use async/parallel processing")
```

---

## Practice Exercises

### Exercise 1: Build Hybrid Search
Implement hybrid search with tunable alpha parameter. Find optimal alpha for your dataset.

### Exercise 2: Two-Stage Pipeline
Build complete two-stage retrieval: bi-encoder (top-100) → cross-encoder (top-10).

### Exercise 3: Evaluate Metrics
Compute MRR, NDCG@10, Precision@5 for your retrieval system on labeled data.

### Exercise 4: Query Intent Router
Build intent classifier that routes queries to specialized retrievers.

### Exercise 5: RRF Comparison
Compare RRF vs weighted combination for merging multiple rankings.

---

## Key Takeaways

1. **Query expansion** improves recall with synonyms and reformulation
2. **Semantic search** understands meaning; **keyword search** matches terms
3. **Hybrid search** (BM25 + vector) combines strengths of both approaches
4. **Cross-encoders** provide superior accuracy for reranking
5. **Two-stage retrieval** (bi-encoder → cross-encoder) balances speed and accuracy
6. **RRF** merges rankings without manual weight tuning
7. **Query intent** detection enables specialized retrieval strategies
8. **Metrics** (MRR, NDCG, Precision@K) measure retrieval quality objectively
9. **Caching** and **batching** are essential for production performance
10. **Reranking** typically improves metrics by 10-30%

---

## Further Reading

### Papers
- **BM25**: Robertson & Zaragoza (2009) - The Probabilistic Relevance Framework
- **Cross-Encoders**: Nogueira & Cho (2019) - Passage Re-ranking with BERT
- **RRF**: Cormack et al. (2009) - Reciprocal Rank Fusion

### Documentation
- **rank_bm25**: https://github.com/dorianbrown/rank_bm25
- **sentence-transformers Cross-Encoders**: https://www.sbert.net/examples/applications/cross-encoder/README.html

### Cross-References
- **Module 16 Lesson 4**: Advanced RAG Patterns (previous)
- **Module 16 Lesson 6**: AI Agents Fundamentals (next)

---

**Next Lesson**: AI Agents Fundamentals & ReAct Pattern - Build autonomous AI agents with reasoning and action!

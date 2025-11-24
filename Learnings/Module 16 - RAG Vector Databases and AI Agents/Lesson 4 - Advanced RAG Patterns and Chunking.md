# Lesson 4: Advanced RAG Patterns & Chunking Strategies 📚

**Module 16: RAG, Vector Databases & AI Agents | Lesson 4 of 8**

Master production RAG systems - from naive chunking to advanced retrieval patterns used by top companies!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Review basic RAG pipeline (cross-reference Module 7 Lesson 6)
2. ✅ Master chunking strategies: fixed-size, semantic, recursive
3. ✅ Implement parent-child chunk hierarchies
4. ✅ Use sliding windows vs non-overlapping chunks
5. ✅ Enrich chunks with metadata for better filtering
6. ✅ Apply recursive retrieval patterns
7. ✅ Implement HyDE (Hypothetical Document Embeddings)
8. ✅ Use multi-query and query decomposition techniques

---

## Prerequisites

- **Module 7 Lesson 6**: RAG & Vector Search basics (REQUIRED - review first!)
- **Module 16 Lesson 1**: Vector Embeddings
- **Module 16 Lesson 2**: Vector Databases
- LangChain or LlamaIndex basics

---

## 1. Basic RAG Pipeline Review

### Simple RAG Architecture

```python
# Example 1: Review of basic RAG pipeline (from Module 7 Lesson 6)
"""
Basic RAG Pipeline:

1. Indexing Phase:
   Document → Split into chunks → Embed → Store in vector DB

2. Query Phase:
   Query → Embed → Search vector DB → Retrieve top-k → Send to LLM

Limitations of Naive RAG:
❌ Poor chunking loses context
❌ No metadata filtering
❌ Retrieves irrelevant chunks
❌ Can't handle complex queries
❌ No re-ranking of results

This lesson: Fix all these problems!
"""

from sentence_transformers import SentenceTransformer
import numpy as np

class NaiveRAG:
    """
    Simple RAG implementation (baseline).

    Limitations:
    - Fixed-size chunking (may split sentences)
    - No overlap (loses context at boundaries)
    - No metadata
    - No re-ranking
    """

    def __init__(self, chunk_size=200):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = chunk_size
        self.chunks = []
        self.embeddings = None

    def index_document(self, text):
        """Naive chunking: split every N characters"""
        # Simple chunking (NAIVE!)
        chunks = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)

        self.chunks = chunks

        # Embed
        self.embeddings = self.model.encode(chunks, normalize_embeddings=True)

        print(f"✅ Indexed {len(chunks)} chunks")

    def query(self, question, top_k=3):
        """Basic retrieval"""
        query_emb = self.model.encode(question, normalize_embeddings=True)

        # Cosine similarity (dot product for normalized)
        similarities = np.dot(self.embeddings, query_emb)

        # Top-k
        top_indices = np.argsort(-similarities)[:top_k]

        return [self.chunks[i] for i in top_indices]

# Demo
naive_rag = NaiveRAG(chunk_size=200)

document = """
Machine learning is a subset of artificial intelligence that focuses on
building systems that can learn from data. Deep learning is a specialized
branch of machine learning that uses neural networks with multiple layers.
These deep neural networks can automatically learn hierarchical representations
of data, making them powerful for tasks like image recognition and natural
language processing. Transfer learning allows models pre-trained on large
datasets to be fine-tuned for specific tasks with less data.
"""

naive_rag.index_document(document)
results = naive_rag.query("What is deep learning?")

print("\nNaive RAG Results:")
for i, chunk in enumerate(results, 1):
    print(f"{i}. {chunk[:100]}...")
print("\n⚠️  Problem: Chunks may be split awkwardly, losing context!")
```

---

## 2. Chunking Strategies

### Fixed-Size Chunking with Overlap

```python
# Example 2: Fixed-size chunking with overlap
def chunk_text_fixed_size(text, chunk_size=200, overlap=50):
    """
    Split text into fixed-size chunks with overlap.

    Overlap helps preserve context at chunk boundaries.
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Only add non-empty chunks
        if chunk.strip():
            chunks.append({
                'text': chunk,
                'start_idx': start,
                'end_idx': end
            })

        # Move forward by (chunk_size - overlap)
        start += (chunk_size - overlap)

    return chunks

# Test
text = "Python is a programming language. " * 20  # 700 chars

chunks_no_overlap = chunk_text_fixed_size(text, chunk_size=100, overlap=0)
chunks_with_overlap = chunk_text_fixed_size(text, chunk_size=100, overlap=20)

print("Fixed-Size Chunking:")
print(f"  No overlap: {len(chunks_no_overlap)} chunks")
print(f"  With overlap (20 chars): {len(chunks_with_overlap)} chunks")
print(f"\n  Overlap benefit: Context preserved at boundaries")
print(f"  Overlap cost: {len(chunks_with_overlap)/len(chunks_no_overlap):.1f}x more chunks")
```

### Semantic Chunking (Sentence-Aware)

```python
# Example 3: Semantic chunking using sentence boundaries
import re

def chunk_by_sentences(text, max_chunk_size=500):
    """
    Chunk by sentences, respecting semantic boundaries.

    Better than fixed-size: doesn't split sentences!
    """
    # Split into sentences (simple regex)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence)

        # If adding this sentence exceeds max, start new chunk
        if current_length + sentence_length > max_chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

# Test
document = """
Machine learning is powerful. It can learn from data. Deep learning uses neural networks.
These networks have multiple layers. They learn hierarchical features. This makes them
effective for complex tasks. Computer vision is one application. Natural language
processing is another. Both benefit from deep learning approaches.
"""

chunks_semantic = chunk_by_sentences(document, max_chunk_size=100)

print("\nSemantic Chunking:")
print(f"  Created {len(chunks_semantic)} chunks")
for i, chunk in enumerate(chunks_semantic, 1):
    print(f"\n  Chunk {i}: {chunk}")
```

### Recursive Chunking (LangChain Style)

```python
# Example 4: Recursive chunking with multiple separators
class RecursiveChunker:
    """
    Recursively split text by separators (paragraphs → sentences → words).

    This is the strategy used by LangChain's RecursiveCharacterTextSplitter.
    """

    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Separators in priority order
        self.separators = [
            "\n\n",  # Paragraphs
            "\n",    # Lines
            ". ",    # Sentences
            " ",     # Words
            ""       # Characters (last resort)
        ]

    def split(self, text):
        """Recursively split text"""
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text, separators):
        """Recursive splitting logic"""
        if len(text) <= self.chunk_size:
            return [text]

        # Try separators in order
        separator = separators[0] if separators else ""

        if separator:
            splits = text.split(separator)
        else:
            # Base case: split by character
            splits = list(text)

        # Merge splits into chunks
        chunks = []
        current_chunk = []
        current_length = 0

        for split in splits:
            split_length = len(split) + len(separator)

            if current_length + split_length > self.chunk_size:
                if current_chunk:
                    # Join and add chunk
                    chunk_text = separator.join(current_chunk)
                    chunks.append(chunk_text)

                    # Add overlap
                    if self.chunk_overlap > 0 and len(current_chunk) > 1:
                        overlap_chunk = current_chunk[-1]
                        current_chunk = [overlap_chunk, split]
                        current_length = len(overlap_chunk) + split_length
                    else:
                        current_chunk = [split]
                        current_length = split_length
                else:
                    # Split is too large, try next separator
                    if len(separators) > 1:
                        sub_chunks = self._split_recursive(split, separators[1:])
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(split)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(split)
                current_length += split_length

        # Add remaining
        if current_chunk:
            chunks.append(separator.join(current_chunk))

        return chunks

# Test
chunker = RecursiveChunker(chunk_size=150, chunk_overlap=20)

document = """
# Deep Learning

Deep learning is a subset of machine learning.

It uses neural networks with multiple layers. These layers learn hierarchical representations.

## Applications
Computer vision uses CNNs. Natural language processing uses transformers.
"""

chunks = chunker.split(document)

print("\nRecursive Chunking:")
print(f"  Chunks created: {len(chunks)}")
for i, chunk in enumerate(chunks, 1):
    print(f"\n  Chunk {i}:")
    print(f"    Length: {len(chunk)}")
    print(f"    Preview: {chunk[:80]}...")
```

---

## 3. Parent-Child Chunk Hierarchies

### Concept

```python
# Example 5: Parent-child chunking architecture
"""
Parent-Child Chunking:

Document
  ├─ Section 1 (PARENT)
  │   ├─ Paragraph 1.1 (CHILD - searchable)
  │   ├─ Paragraph 1.2 (CHILD - searchable)
  │   └─ Paragraph 1.3 (CHILD - searchable)
  └─ Section 2 (PARENT)
      ├─ Paragraph 2.1 (CHILD - searchable)
      └─ Paragraph 2.2 (CHILD - searchable)

Strategy:
1. Index CHILD chunks (small, specific)
2. Search finds relevant CHILD
3. Return PARENT context to LLM (more complete)

Benefits:
✅ Precise retrieval (small chunks)
✅ Rich context (large chunks to LLM)
✅ Best of both worlds!
"""

class ParentChildChunker:
    """
    Create parent-child chunk hierarchy.
    """

    def __init__(self, parent_size=1000, child_size=200):
        self.parent_size = parent_size
        self.child_size = child_size

    def chunk_document(self, text):
        """Create parent-child chunks"""
        # Parent chunks (large)
        parents = []
        for i in range(0, len(text), self.parent_size):
            parent_text = text[i:i + self.parent_size]
            parents.append({
                'id': f'parent_{len(parents)}',
                'text': parent_text,
                'start': i,
                'end': i + len(parent_text)
            })

        # Child chunks (small, within parents)
        children = []
        for parent in parents:
            parent_text = parent['text']
            parent_id = parent['id']

            # Split parent into children
            for j in range(0, len(parent_text), self.child_size):
                child_text = parent_text[j:j + self.child_size]
                children.append({
                    'id': f'{parent_id}_child_{len(children)}',
                    'parent_id': parent_id,
                    'text': child_text
                })

        return {
            'parents': parents,
            'children': children
        }

# Test
chunker = ParentChildChunker(parent_size=400, child_size=100)

doc = "Machine learning is amazing. " * 50  # ~1400 chars

result = chunker.chunk_document(doc)

print("Parent-Child Chunking:")
print(f"  Parents: {len(result['parents'])}")
print(f"  Children: {len(result['children'])}")
print(f"  Avg children per parent: {len(result['children'])/len(result['parents']):.1f}")

# Show structure
for i, parent in enumerate(result['parents'][:2]):
    print(f"\n  {parent['id']}: {len(parent['text'])} chars")
    # Find children
    children = [c for c in result['children'] if c['parent_id'] == parent['id']]
    for child in children:
        print(f"    └─ {child['id']}: {len(child['text'])} chars")
```

### Parent-Child RAG Implementation

```python
# Example 6: RAG with parent-child retrieval
class ParentChildRAG:
    """
    RAG system using parent-child chunks.

    Search: Find relevant children (precise)
    Retrieve: Return parents (context)
    """

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.parents = []
        self.children = []
        self.child_embeddings = None

    def index_document(self, text, parent_size=500, child_size=150):
        """Index with parent-child structure"""
        chunker = ParentChildChunker(parent_size, child_size)
        result = chunker.chunk_document(text)

        self.parents = result['parents']
        self.children = result['children']

        # Embed CHILDREN (search these)
        child_texts = [c['text'] for c in self.children]
        self.child_embeddings = self.model.encode(
            child_texts,
            normalize_embeddings=True
        )

        print(f"✅ Indexed {len(self.parents)} parents, {len(self.children)} children")

    def query(self, question, top_k=2):
        """Search children, return parents"""
        # Embed query
        query_emb = self.model.encode(question, normalize_embeddings=True)

        # Search children
        similarities = np.dot(self.child_embeddings, query_emb)
        top_child_indices = np.argsort(-similarities)[:top_k]

        # Get parent IDs
        parent_ids = set()
        for idx in top_child_indices:
            child = self.children[idx]
            parent_ids.add(child['parent_id'])

        # Return parent texts
        parents = [p for p in self.parents if p['id'] in parent_ids]

        return {
            'matched_children': [self.children[i] for i in top_child_indices],
            'parent_contexts': parents
        }

# Test
pc_rag = ParentChildRAG()

document = """
Machine learning is a branch of AI that enables systems to learn from data.
Supervised learning uses labeled data to train models. Classification and
regression are common supervised learning tasks.

Deep learning is a subset of machine learning using neural networks. Convolutional
neural networks (CNNs) are effective for image tasks. Recurrent neural networks
(RNNs) handle sequential data like text and time series.

Transfer learning allows models pre-trained on large datasets to be adapted for
new tasks. This is especially useful when labeled data is scarce. Fine-tuning
adjusts pre-trained weights for specific applications.
"""

pc_rag.index_document(document)

result = pc_rag.query("What is transfer learning?", top_k=2)

print("\nParent-Child RAG Query:")
print(f"\nMatched children:")
for child in result['matched_children']:
    print(f"  - {child['text'][:80]}...")

print(f"\nParent contexts (sent to LLM):")
for parent in result['parent_contexts']:
    print(f"  - {parent['text'][:120]}...")
print("\n✅ LLM gets full parent context, not just small child chunk!")
```

---

## 4. Metadata Enrichment

### Adding Metadata to Chunks

```python
# Example 7: Enrich chunks with metadata
import hashlib
from datetime import datetime

class MetadataEnricher:
    """Add rich metadata to chunks for filtering"""

    def enrich_chunk(self, chunk_text, document_metadata):
        """Add metadata to chunk"""
        return {
            'text': chunk_text,
            'chunk_id': hashlib.md5(chunk_text.encode()).hexdigest()[:12],
            'length': len(chunk_text),
            'word_count': len(chunk_text.split()),

            # Document-level metadata
            'doc_id': document_metadata.get('doc_id'),
            'doc_title': document_metadata.get('title'),
            'doc_author': document_metadata.get('author'),
            'doc_date': document_metadata.get('date'),
            'doc_category': document_metadata.get('category'),
            'doc_tags': document_metadata.get('tags', []),

            # Chunk-level metadata
            'chunk_index': document_metadata.get('chunk_index', 0),
            'section': document_metadata.get('section'),

            # Searchability metadata
            'language': document_metadata.get('language', 'en'),
            'embedding_model': 'all-MiniLM-L6-v2',

            # Timestamps
            'indexed_at': datetime.now().isoformat()
        }

# Example
enricher = MetadataEnricher()

chunk = enricher.enrich_chunk(
    chunk_text="Machine learning enables systems to learn from data.",
    document_metadata={
        'doc_id': 'ml_guide_2024',
        'title': 'Machine Learning Guide',
        'author': 'John Doe',
        'date': '2024-01-15',
        'category': 'technology',
        'tags': ['ml', 'ai', 'tutorial'],
        'section': 'Introduction',
        'chunk_index': 0,
        'language': 'en'
    }
)

print("Enriched Chunk Metadata:")
for key, value in chunk.items():
    if key != 'text':
        print(f"  {key}: {value}")
```

### Metadata Filtering in Retrieval

```python
# Example 8: Filter by metadata during retrieval
class MetadataFilteredRAG:
    """RAG with metadata filtering"""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks = []
        self.embeddings = None

    def index_chunks(self, chunks_with_metadata):
        """Index chunks with metadata"""
        self.chunks = chunks_with_metadata

        # Embed
        texts = [c['text'] for c in chunks_with_metadata]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

        print(f"✅ Indexed {len(chunks_with_metadata)} chunks with metadata")

    def query(self, question, top_k=5, filters=None):
        """
        Query with optional metadata filters.

        filters: Dict of metadata conditions
        Example: {'category': 'technology', 'date': {'$gte': '2024-01-01'}}
        """
        # Embed query
        query_emb = self.model.encode(question, normalize_embeddings=True)

        # Compute similarities
        similarities = np.dot(self.embeddings, query_emb)

        # Apply filters
        if filters:
            valid_indices = []
            for i, chunk in enumerate(self.chunks):
                if self._matches_filters(chunk, filters):
                    valid_indices.append(i)

            # Filter similarities
            filtered_similarities = [(i, similarities[i]) for i in valid_indices]
            filtered_similarities.sort(key=lambda x: -x[1])
            top_indices = [i for i, _ in filtered_similarities[:top_k]]
        else:
            # No filters: just top-k
            top_indices = np.argsort(-similarities)[:top_k]

        return [self.chunks[i] for i in top_indices]

    def _matches_filters(self, chunk, filters):
        """Check if chunk matches filters"""
        for key, condition in filters.items():
            if isinstance(condition, dict):
                # Range query (e.g., date >= X)
                if '$gte' in condition:
                    if chunk.get(key, '') < condition['$gte']:
                        return False
                if '$lte' in condition:
                    if chunk.get(key, '') > condition['$lte']:
                        return False
            else:
                # Exact match
                if chunk.get(key) != condition:
                    return False
        return True

# Test
rag = MetadataFilteredRAG()

# Create chunks with metadata
chunks = [
    {
        'text': 'Python is great for ML',
        'category': 'technology',
        'date': '2024-01-15',
        'tags': ['python', 'ml']
    },
    {
        'text': 'Exercise improves health',
        'category': 'health',
        'date': '2024-02-01',
        'tags': ['fitness', 'wellness']
    },
    {
        'text': 'Deep learning uses neural networks',
        'category': 'technology',
        'date': '2024-01-20',
        'tags': ['deep-learning', 'ai']
    },
]

rag.index_chunks(chunks)

# Query with filter
results = rag.query(
    "Tell me about AI",
    top_k=3,
    filters={'category': 'technology'}
)

print("\nFiltered Query Results:")
for r in results:
    print(f"  - {r['text']} (category: {r['category']})")
```

---

## 5. Recursive Retrieval

### Multi-Hop Retrieval

```python
# Example 9: Recursive retrieval for complex queries
class RecursiveRetriever:
    """
    Retrieve → Extract entities → Retrieve again → Combine

    Use case: Multi-hop reasoning
    Example: "What university did the president of France attend?"
      1. Retrieve: "President of France is Emmanuel Macron"
      2. Retrieve: "Emmanuel Macron attended Sciences Po and ENA"
    """

    def __init__(self, rag_system):
        self.rag = rag_system
        self.max_depth = 3

    def recursive_retrieve(self, query, depth=0):
        """Recursively retrieve and extract sub-queries"""
        if depth >= self.max_depth:
            return []

        # Initial retrieval
        results = self.rag.query(query, top_k=3)

        all_contexts = results.copy()

        # Extract entities/concepts (simplified - in production use NER)
        for result in results:
            entities = self._extract_entities(result['text'])

            # Retrieve for each entity
            for entity in entities[:2]:  # Limit sub-queries
                sub_query = f"Information about {entity}"
                sub_results = self.recursive_retrieve(sub_query, depth + 1)
                all_contexts.extend(sub_results)

        return all_contexts

    def _extract_entities(self, text):
        """
        Extract entities (simplified).

        In production: Use spaCy, BERT-NER, or LLM-based extraction
        """
        # Simple heuristic: capitalized words
        import re
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        return list(set(words))[:3]  # Limit entities

print("Recursive Retrieval Pattern:")
print("\n1. Initial query → Retrieve documents")
print("2. Extract entities/concepts from results")
print("3. Retrieve documents for each entity")
print("4. Combine all contexts")
print("\n✅ Enables multi-hop reasoning!")
```

---

## 6. HyDE: Hypothetical Document Embeddings

### HyDE Concept and Implementation

```python
# Example 10: HyDE - improve retrieval with hypothetical answers
"""
HyDE (Hypothetical Document Embeddings):

Problem: Query and document embeddings may be in different spaces
  Query: "How do I reset my password?"
  Document: "Navigate to Settings, click Account, select Reset Password..."

Solution: Generate hypothetical answer, embed that!
  Query → LLM → Hypothetical Answer → Embed → Search

Benefits:
✅ Better semantic matching
✅ Bridges query-document gap
✅ Especially good for how-to and factual queries
"""

class HyDERetriever:
    """
    HyDE: Generate hypothetical document, use for retrieval.
    """

    def __init__(self, rag_system, llm=None):
        self.rag = rag_system
        self.llm = llm  # In production: OpenAI, Anthropic, etc.

    def retrieve_with_hyde(self, query, top_k=5):
        """
        1. Generate hypothetical answer
        2. Embed hypothetical answer
        3. Search with that embedding
        """
        # Generate hypothetical document (mock LLM call)
        hypothetical_doc = self._generate_hypothetical(query)

        print(f"Query: {query}")
        print(f"Hypothetical: {hypothetical_doc}\n")

        # Use hypothetical for retrieval instead of query!
        results = self.rag.query(hypothetical_doc, top_k=top_k)

        return results

    def _generate_hypothetical(self, query):
        """
        Generate hypothetical answer.

        In production: Call LLM with prompt like:
        "Write a detailed answer to: {query}"
        """
        # Mock responses for demo
        mock_responses = {
            "What is Python?": "Python is a high-level programming language known for its simplicity and readability. It is widely used for web development, data science, machine learning, and automation.",
            "How to reset password?": "To reset your password, navigate to the Settings page, click on Account Settings, then select Reset Password. Enter your current password and new password, then click Save."
        }

        # Return mock or generic
        return mock_responses.get(query, f"A detailed explanation of {query} including key concepts and examples.")

# Demo (conceptual)
print("HyDE Example:")
print("\nTraditional RAG:")
print("  Query embedding: [0.2, 0.3, ...] (short, question-like)")
print("  Document embedding: [0.5, 0.6, ...] (long, answer-like)")
print("  → May not match well!")

print("\nHyDE:")
print("  Generate hypothetical answer (detailed)")
print("  Hypothetical embedding: [0.48, 0.58, ...] (answer-like)")
print("  Document embedding: [0.5, 0.6, ...] (answer-like)")
print("  → Better match! ✅")
```

---

## 7. Multi-Query Retrieval

### Query Expansion

```python
# Example 11: Multi-query retrieval with query expansion
class MultiQueryRetriever:
    """
    Generate multiple query variations, retrieve for each, merge results.

    Benefits:
    - Covers different phrasings
    - More robust retrieval
    - Reduces sensitivity to query wording
    """

    def __init__(self, rag_system):
        self.rag = rag_system

    def generate_query_variations(self, original_query):
        """
        Generate query variations.

        In production: Use LLM to generate variations
        """
        # Mock variations for demo
        variations = [
            original_query,
            f"Explain {original_query}",
            f"What are the details of {original_query}",
            f"Tell me about {original_query}"
        ]

        return variations[:3]  # Limit to 3 variations

    def retrieve_multi_query(self, query, top_k=3):
        """
        Retrieve using multiple query variations, then merge.
        """
        # Generate variations
        variations = self.generate_query_variations(query)

        print(f"Original query: {query}")
        print(f"Variations: {variations}\n")

        # Retrieve for each variation
        all_results = []
        seen_chunks = set()

        for variant in variations:
            results = self.rag.query(variant, top_k=top_k)

            for result in results:
                chunk_text = result.get('text', result)
                if chunk_text not in seen_chunks:
                    all_results.append(result)
                    seen_chunks.add(chunk_text)

        return all_results[:top_k]

print("Multi-Query Retrieval:")
print("\nProcess:")
print("  1. Generate 3-5 query variations")
print("  2. Retrieve top-k for each variation")
print("  3. Merge and deduplicate results")
print("  4. Return top-k from merged results")
print("\n✅ More robust than single query!")
```

---

## 8. Query Decomposition

### Break Complex Queries into Sub-Queries

```python
# Example 12: Query decomposition for complex questions
class QueryDecomposer:
    """
    Decompose complex query into simpler sub-queries.

    Example:
      Complex: "Compare Python and Java for machine learning and web development"
      Sub-queries:
        1. "Python for machine learning"
        2. "Java for machine learning"
        3. "Python for web development"
        4. "Java for web development"
    """

    def decompose_query(self, complex_query):
        """
        Decompose query into sub-queries.

        In production: Use LLM with prompt:
        "Break this complex question into simpler sub-questions: {query}"
        """
        # Simplified decomposition logic
        if "compare" in complex_query.lower():
            # Extract comparison subjects
            # This is a simplified heuristic
            parts = complex_query.lower().replace("compare ", "").split(" and ")
            if len(parts) >= 2:
                return [
                    f"Information about {parts[0].strip()}",
                    f"Information about {parts[1].strip()}"
                ]

        # Default: return original
        return [complex_query]

    def retrieve_decomposed(self, complex_query, rag_system, top_k=3):
        """
        Decompose query, retrieve for each, combine results.
        """
        # Decompose
        sub_queries = self.decompose_query(complex_query)

        print(f"Complex query: {complex_query}")
        print(f"Sub-queries: {sub_queries}\n")

        # Retrieve for each sub-query
        all_results = []
        for sq in sub_queries:
            results = rag_system.query(sq, top_k=top_k)
            all_results.extend(results)

        # Deduplicate
        seen = set()
        unique_results = []
        for r in all_results:
            text = r.get('text', r)
            if text not in seen:
                unique_results.append(r)
                seen.add(text)

        return unique_results[:top_k * 2]  # Return more results

# Demo
decomposer = QueryDecomposer()

query = "Compare Python and Java for machine learning"
sub_queries = decomposer.decompose_query(query)

print(f"Decomposed Query:")
print(f"  Original: {query}")
print(f"  Sub-queries:")
for sq in sub_queries:
    print(f"    - {sq}")
```

---

## 9. Production RAG with LangChain

### LangChain RAG Pipeline

```python
# Example 13: Production RAG with LangChain
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load and split documents
from langchain.document_loaders import TextLoader

loader = TextLoader('document.txt')
documents = loader.load()

# 2. Recursive chunking with overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\\n\\n", "\\n", ". ", " ", ""]
)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks")

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory='./chroma_db'
)

# 4. Create retriever with metadata filtering
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5,
        "filter": {"category": "technology"}  # Metadata filter
    }
)

# 5. Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",  # or "map_reduce", "refine"
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain("What is machine learning?")

print(f"Answer: {result['result']}")
print(f"\\nSources:")
for doc in result['source_documents']:
    print(f"  - {doc.page_content[:100]}...")
"""

print("LangChain RAG Production Features:")
print("\n1. Text Splitters:")
print("   - RecursiveCharacterTextSplitter")
print("   - TokenTextSplitter")
print("   - MarkdownTextSplitter")

print("\n2. Vector Stores:")
print("   - Chroma (local)")
print("   - Pinecone (cloud)")
print("   - Weaviate (self-hosted/cloud)")

print("\n3. Retrievers:")
print("   - Similarity search")
print("   - MMR (Maximum Marginal Relevance)")
print("   - Metadata filtering")

print("\n4. Chain Types:")
print("   - stuff: All docs in one prompt")
print("   - map_reduce: Parallel then combine")
print("   - refine: Iterative refinement")
```

---

## 10. Production RAG with LlamaIndex

### LlamaIndex Advanced Features

```python
# Example 14: LlamaIndex for production RAG
"""
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import SentenceSplitter
from llama_index.embeddings import OpenAIEmbedding

# 1. Load documents
documents = SimpleDirectoryReader('data').load_data()

# 2. Configure node parser (chunking)
node_parser = SimpleNodeParser.from_defaults(
    text_splitter=SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
)

# 3. Configure service context
service_context = ServiceContext.from_defaults(
    llm=OpenAI(model="gpt-4"),
    embed_model=OpenAIEmbedding(),
    node_parser=node_parser
)

# 4. Create index
index = VectorStoreIndex.from_documents(
    documents,
    service_context=service_context
)

# 5. Query with advanced retrieval
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"  # or "compact", "refine"
)

# 6. Query
response = query_engine.query("What is machine learning?")

print(f"Answer: {response}")
print(f"\\nSource nodes:")
for node in response.source_nodes:
    print(f"  - Score: {node.score:.3f}")
    print(f"    Text: {node.text[:100]}...")
"""

print("LlamaIndex Production Features:")
print("\n1. Node Parsers:")
print("   - Sentence-aware splitting")
print("   - Markdown/HTML parsing")
print("   - Custom parsers")

print("\n2. Response Modes:")
print("   - tree_summarize: Hierarchical summarization")
print("   - compact: Minimal context")
print("   - refine: Iterative refinement")

print("\n3. Advanced Retrieval:")
print("   - Auto-merging retrieval")
print("   - Hierarchical retrieval")
print("   - Knowledge graph retrieval")
```

---

## Practice Exercises

### Exercise 1: Compare Chunking Strategies
Implement fixed-size, semantic, and recursive chunking. Measure retrieval quality on your dataset.

### Exercise 2: Parent-Child RAG
Build a complete parent-child RAG system. Compare retrieval quality vs simple chunking.

### Exercise 3: Metadata Filtering
Create a RAG system with rich metadata. Implement complex filters (date ranges, categories, tags).

### Exercise 4: HyDE Implementation
Integrate HyDE with an LLM API. Measure improvement over standard retrieval.

### Exercise 5: Multi-Query System
Build a multi-query retriever that generates variations. Compare recall vs single-query.

---

## Key Takeaways

1. **Naive chunking** (fixed-size, no overlap) loses context and splits semantics
2. **Semantic chunking** respects sentence/paragraph boundaries for better quality
3. **Recursive chunking** (LangChain style) handles hierarchical structure
4. **Parent-child** chunks enable precise search with rich context
5. **Metadata enrichment** enables powerful filtering and routing
6. **Recursive retrieval** enables multi-hop reasoning
7. **HyDE** bridges query-document gap with hypothetical answers
8. **Multi-query** makes retrieval more robust to phrasing
9. **Query decomposition** breaks complex queries into manageable parts
10. **LangChain/LlamaIndex** provide production-ready implementations

---

## Further Reading

### Papers
- **HyDE**: Gao et al. (2022) - https://arxiv.org/abs/2212.10496
- **Lost in the Middle**: Liu et al. (2023) - https://arxiv.org/abs/2307.03172

### Documentation
- **LangChain Text Splitters**: https://python.langchain.com/docs/modules/data_connection/document_transformers/
- **LlamaIndex**: https://docs.llamaindex.ai/

### Cross-References
- **Module 7 Lesson 6**: RAG & Vector Search basics (review!)
- **Module 16 Lesson 5**: Query Optimization & Reranking (next lesson)

---

**Next Lesson**: Query Optimization, Reranking & Hybrid Search - Take retrieval quality to the next level!

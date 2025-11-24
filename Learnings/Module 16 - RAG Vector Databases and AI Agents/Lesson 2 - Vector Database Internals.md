# Lesson 2: Vector Database Internals & Architecture 🗄️

**Module 16: RAG, Vector Databases & AI Agents | Lesson 2 of 8**

Deep dive into vector databases - the backbone of production RAG systems and semantic search at scale!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand why traditional databases fail for vector similarity search
2. ✅ Master vector database architecture: storage, indexing, query layers
3. ✅ Implement similarity search with FAISS (local, fast)
4. ✅ Build production systems with Weaviate (GraphQL, cloud-ready)
5. ✅ Use Pinecone for managed cloud vector search
6. ✅ Explore Milvus and Qdrant as open-source alternatives
7. ✅ Understand sharding and replication for scaling to billions of vectors

---

## Prerequisites

- **Module 16 Lesson 1**: Vector Embeddings (REQUIRED)
- Understanding of databases (SQL/NoSQL basics)
- Python and NumPy
- Docker (for running database instances)

---

## 1. Why Vector Databases?

### Traditional Databases vs Vector Stores

```python
import numpy as np
import time
from scipy.spatial import distance

# Problem: Find similar items in traditional database

class TraditionalDatabase:
    """Simulate traditional DB with no vector optimization"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def insert(self, doc_id, text, embedding):
        """Insert document with embedding"""
        self.documents.append({'id': doc_id, 'text': text})
        self.embeddings.append(embedding)

    def search_similarity(self, query_embedding, top_k=5):
        """Brute-force similarity search - O(n) complexity!"""
        similarities = []

        # Compute similarity to EVERY document
        for i, doc_emb in enumerate(self.embeddings):
            sim = 1 - distance.cosine(query_embedding, doc_emb)
            similarities.append((i, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]

# Demonstrate the problem
print("Traditional Database Performance:")
print("=" * 50)

db = TraditionalDatabase()

# Insert 10,000 documents
n_docs = 10000
embedding_dim = 384

print(f"Inserting {n_docs} documents...")
for i in range(n_docs):
    fake_embedding = np.random.randn(embedding_dim)
    fake_embedding /= np.linalg.norm(fake_embedding)
    db.insert(i, f"Document {i}", fake_embedding)

# Search
query_emb = np.random.randn(embedding_dim)
query_emb /= np.linalg.norm(query_emb)

start = time.time()
results = db.search_similarity(query_emb, top_k=5)
search_time = time.time() - start

print(f"✅ Inserted {len(db.documents)} documents")
print(f"⚠️  Search time: {search_time*1000:.2f}ms")
print(f"❌ Problem: Linear search O(n) - scales poorly!")
print(f"\nFor 1M documents: ~{search_time*100:.1f}s per query")
print(f"For 1B documents: ~{search_time*100000:.1f}s per query")
print("\n💡 Solution: Vector databases with ANN algorithms!")
```

### What Vector Databases Provide

```python
# Example 1: Key features of vector databases
"""
Vector Database Advantages:

1. ✅ Approximate Nearest Neighbor (ANN) Search
   - Sub-linear search time: O(log n) instead of O(n)
   - Trade accuracy for speed (99%+ recall typical)

2. ✅ Optimized Storage
   - Compressed vectors (quantization)
   - Memory-mapped files for large datasets
   - Efficient serialization

3. ✅ Metadata Filtering
   - Combine vector search with filters
   - Example: "Similar to X AND category='tech' AND date > 2023"

4. ✅ CRUD Operations
   - Create, Read, Update, Delete vectors
   - Real-time updates without rebuilding index

5. ✅ Scalability
   - Horizontal scaling (sharding)
   - Replication for high availability
   - Distributed queries

6. ✅ Persistence
   - Disk-backed storage
   - ACID transactions (some databases)
   - Backups and snapshots

7. ✅ Production Features
   - Authentication & authorization
   - Monitoring and observability
   - API & client libraries
"""

comparison = {
    'Feature': ['Search Time (1M docs)', 'Metadata Filters', 'Scalability', 'Real-time Updates', 'Persistence'],
    'Traditional DB': ['~10s', '✅', '✅', '✅', '✅'],
    'Vector DB': ['~10ms', '✅', '✅', '✅', '✅'],
    'Speedup': ['1000x', '-', '-', '-', '-']
}

print("\nTraditional DB vs Vector DB:\n")
for i in range(len(comparison['Feature'])):
    print(f"{comparison['Feature'][i]:20} | Traditional: {comparison['Traditional DB'][i]:10} | Vector: {comparison['Vector DB'][i]:10} | {comparison['Speedup'][i]}")
```

---

## 2. Vector Database Architecture

### Core Components

```python
# Example 2: Vector database architecture
"""
┌─────────────────────────────────────────────────────────────┐
│                    VECTOR DATABASE                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Query API  │      │  Ingest API  │                   │
│  │  (REST/gRPC) │      │   (Batch)    │                   │
│  └──────┬───────┘      └──────┬───────┘                   │
│         │                     │                            │
│         v                     v                            │
│  ┌─────────────────────────────────────┐                  │
│  │        Query Planner                │                  │
│  │  - Parse query                      │                  │
│  │  - Apply filters                    │                  │
│  │  - Optimize execution               │                  │
│  └──────────────┬──────────────────────┘                  │
│                 │                                          │
│                 v                                          │
│  ┌─────────────────────────────────────┐                  │
│  │         Index Layer                 │                  │
│  │  - HNSW / IVF / LSH                 │                  │
│  │  - Approximate search               │                  │
│  │  - Multiple indexes                 │                  │
│  └──────────────┬──────────────────────┘                  │
│                 │                                          │
│                 v                                          │
│  ┌─────────────────────────────────────┐                  │
│  │        Storage Layer                │                  │
│  │  - Vector storage                   │                  │
│  │  - Metadata storage                 │                  │
│  │  - Compression (quantization)       │                  │
│  └──────────────┬──────────────────────┘                  │
│                 │                                          │
│                 v                                          │
│  ┌─────────────────────────────────────┐                  │
│  │      Persistence Layer              │                  │
│  │  - Disk I/O                         │                  │
│  │  - Memory mapping                   │                  │
│  │  - Write-ahead logging              │                  │
│  └─────────────────────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
"""

print("Vector Database Architecture Layers:")
print("\n1. Query API Layer")
print("   - REST/gRPC endpoints")
print("   - Query validation")
print("   - Authentication")

print("\n2. Query Planning Layer")
print("   - Parse and optimize queries")
print("   - Combine vector + metadata filters")
print("   - Distributed query routing")

print("\n3. Index Layer")
print("   - ANN algorithms (HNSW, IVF, etc.)")
print("   - Multiple indexes per collection")
print("   - Index selection based on query")

print("\n4. Storage Layer")
print("   - Vector compression")
print("   - Metadata indexing")
print("   - Memory vs disk tradeoffs")

print("\n5. Persistence Layer")
print("   - Durable storage")
print("   - Crash recovery")
print("   - Replication")
```

---

## 3. FAISS: Facebook AI Similarity Search

### Local, Fast Vector Search

```python
# Example 3: FAISS basics - in-memory vector search
import faiss
import numpy as np

# Create sample data
n_docs = 10000
dim = 384

# Generate random vectors (normalized)
vectors = np.random.randn(n_docs, dim).astype('float32')
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

print("FAISS Tutorial:")
print("=" * 50)
print(f"Dataset: {n_docs} vectors of dimension {dim}")

# 1. Flat Index (Exact Search - Baseline)
index_flat = faiss.IndexFlatL2(dim)  # L2 distance
index_flat.add(vectors)

print(f"\n✅ Built Flat Index")
print(f"   Vectors in index: {index_flat.ntotal}")

# Search
k = 5  # top-5 results
query = np.random.randn(1, dim).astype('float32')
query = query / np.linalg.norm(query)

start = time.time()
distances, indices = index_flat.search(query, k)
flat_time = time.time() - start

print(f"\nFlat Index Search:")
print(f"   Time: {flat_time*1000:.2f}ms")
print(f"   Top-5 indices: {indices[0]}")
print(f"   Distances: {distances[0]}")
```

### FAISS with IVF Index (Faster)

```python
# Example 4: FAISS IVF - faster approximate search
# IVF = Inverted File Index (clustering-based)

n_clusters = 100  # Number of Voronoi cells
n_probe = 10      # Number of cells to search

# Create IVF index
quantizer = faiss.IndexFlatL2(dim)
index_ivf = faiss.IndexIVFFlat(quantizer, dim, n_clusters)

# Train the index (clustering)
print(f"\n📊 Training IVF index with {n_clusters} clusters...")
index_ivf.train(vectors)
index_ivf.add(vectors)

print(f"✅ Built IVF Index")
print(f"   Clusters: {n_clusters}")
print(f"   Vectors in index: {index_ivf.ntotal}")

# Search
index_ivf.nprobe = n_probe  # Search in top-10 clusters

start = time.time()
distances_ivf, indices_ivf = index_ivf.search(query, k)
ivf_time = time.time() - start

print(f"\nIVF Index Search:")
print(f"   Time: {ivf_time*1000:.2f}ms")
print(f"   Speedup: {flat_time/ivf_time:.1f}x faster")
print(f"   Top-5 indices: {indices_ivf[0]}")

# Check recall (how many results match exact search)
recall = len(set(indices[0]) & set(indices_ivf[0])) / k
print(f"   Recall: {recall*100:.0f}% (vs exact search)")
```

### FAISS with HNSW (Best Quality/Speed Trade-off)

```python
# Example 5: FAISS HNSW - hierarchical graph index
# HNSW = Hierarchical Navigable Small World

M = 32  # Number of connections per layer
ef_construction = 200  # Quality of index construction
ef_search = 64  # Quality of search

# Create HNSW index
index_hnsw = faiss.IndexHNSWFlat(dim, M)
index_hnsw.hnsw.efConstruction = ef_construction

print(f"\n🏗️  Building HNSW index...")
start = time.time()
index_hnsw.add(vectors)
build_time = time.time() - start

print(f"✅ Built HNSW Index")
print(f"   M (connections): {M}")
print(f"   Build time: {build_time:.2f}s")
print(f"   Vectors in index: {index_hnsw.ntotal}")

# Search
index_hnsw.hnsw.efSearch = ef_search

start = time.time()
distances_hnsw, indices_hnsw = index_hnsw.search(query, k)
hnsw_time = time.time() - start

print(f"\nHNSW Index Search:")
print(f"   Time: {hnsw_time*1000:.2f}ms")
print(f"   Speedup vs Flat: {flat_time/hnsw_time:.1f}x")
print(f"   Top-5 indices: {indices_hnsw[0]}")

recall_hnsw = len(set(indices[0]) & set(indices_hnsw[0])) / k
print(f"   Recall: {recall_hnsw*100:.0f}%")
```

### FAISS Index Comparison

```python
# Example 6: Compare all FAISS index types
import pandas as pd

results = {
    'Index Type': ['Flat (Exact)', 'IVF', 'HNSW'],
    'Search Time (ms)': [
        f"{flat_time*1000:.2f}",
        f"{ivf_time*1000:.2f}",
        f"{hnsw_time*1000:.2f}"
    ],
    'Speedup': ['1x (baseline)', f"{flat_time/ivf_time:.1f}x", f"{flat_time/hnsw_time:.1f}x"],
    'Recall': ['100%', f"{recall*100:.0f}%", f"{recall_hnsw*100:.0f}%"],
    'Memory': ['High', 'Medium', 'High'],
    'Build Time': ['Instant', 'Medium', 'Slow'],
    'Use Case': ['Small datasets', 'Large datasets', 'Best quality/speed']
}

df = pd.DataFrame(results)
print("\n" + "="*80)
print("FAISS Index Comparison:")
print("="*80)
print(df.to_string(index=False))
```

### FAISS: Save and Load

```python
# Example 7: Persist FAISS index to disk
# Save index
faiss.write_index(index_hnsw, "vectors.index")
print("\n✅ Saved FAISS index to 'vectors.index'")

# Load index
loaded_index = faiss.read_index("vectors.index")
print(f"✅ Loaded FAISS index from disk")
print(f"   Vectors in loaded index: {loaded_index.ntotal}")

# Verify it works
distances_loaded, indices_loaded = loaded_index.search(query, k)
print(f"   Search works: {np.array_equal(indices_hnsw, indices_loaded)}")
```

### FAISS with Metadata Filtering (ID Map)

```python
# Example 8: FAISS with IDs for metadata filtering
# Use IndexIDMap to preserve document IDs

# Create base index
base_index = faiss.IndexFlatL2(dim)

# Wrap with ID map
index_with_ids = faiss.IndexIDMap(base_index)

# Add vectors with custom IDs
doc_ids = np.arange(100, 100 + n_docs, dtype='int64')
index_with_ids.add_with_ids(vectors, doc_ids)

print(f"\n✅ FAISS with ID mapping:")
print(f"   Vectors: {index_with_ids.ntotal}")

# Search returns actual document IDs
distances, returned_ids = index_with_ids.search(query, k)
print(f"   Top-5 document IDs: {returned_ids[0]}")
print(f"   (IDs range from {doc_ids[0]} to {doc_ids[-1]})")
```

---

## 4. Weaviate: Production Vector Database

### Setup Weaviate (Docker)

```python
# Example 9: Setup Weaviate with Docker
"""
# Docker Compose file for Weaviate (save as docker-compose.yml):

version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.23.0
    ports:
      - "8080:8080"
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'none'
      CLUSTER_HOSTNAME: 'node1'
    volumes:
      - weaviate_data:/var/lib/weaviate

volumes:
  weaviate_data:

# Start: docker-compose up -d
# Stop: docker-compose down
"""

print("Weaviate Setup Instructions:")
print("1. Save docker-compose.yml file above")
print("2. Run: docker-compose up -d")
print("3. Weaviate will be available at http://localhost:8080")
print("4. Install client: pip install weaviate-client")
```

### Weaviate Client Setup

```python
# Example 10: Connect to Weaviate
"""
import weaviate

# Connect to Weaviate instance
client = weaviate.Client(
    url="http://localhost:8080"
)

# Check connection
print(f"✅ Connected to Weaviate")
print(f"   Ready: {client.is_ready()}")
"""

print("\nWeaviate Client (requires running instance):")
print("  - REST API")
print("  - GraphQL support")
print("  - Python/JavaScript/Go clients")
```

### Create Schema in Weaviate

```python
# Example 11: Define Weaviate schema
"""
# Schema defines the data structure
schema = {
    "classes": [{
        "class": "Document",
        "description": "A document with vector embedding",
        "vectorizer": "none",  # We provide our own vectors
        "properties": [
            {
                "name": "content",
                "dataType": ["text"],
                "description": "Document content"
            },
            {
                "name": "title",
                "dataType": ["string"],
                "description": "Document title"
            },
            {
                "name": "category",
                "dataType": ["string"],
                "description": "Document category"
            },
            {
                "name": "date",
                "dataType": ["date"],
                "description": "Publication date"
            }
        ]
    }]
}

# Create schema
client.schema.create(schema)
print("✅ Created Weaviate schema: 'Document' class")

# Inspect schema
current_schema = client.schema.get()
print(f"   Classes: {[c['class'] for c in current_schema['classes']]}")
"""

print("Weaviate Schema Example:")
print("  - Define classes (like database tables)")
print("  - Specify properties and data types")
print("  - Configure vectorizer (or provide vectors)")
```

### Insert Data into Weaviate

```python
# Example 12: Batch insert into Weaviate
"""
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample documents
documents = [
    {
        "title": "Python Programming",
        "content": "Python is a versatile programming language",
        "category": "technology",
        "date": "2024-01-15T00:00:00Z"
    },
    {
        "title": "Machine Learning",
        "content": "ML algorithms learn patterns from data",
        "category": "technology",
        "date": "2024-01-16T00:00:00Z"
    },
    {
        "title": "Healthy Eating",
        "content": "Vegetables and fruits are essential for health",
        "category": "health",
        "date": "2024-01-17T00:00:00Z"
    }
]

# Batch insert
with client.batch as batch:
    batch.batch_size = 100

    for doc in documents:
        # Generate embedding
        vector = model.encode(doc['content']).tolist()

        # Add to batch
        batch.add_data_object(
            data_object=doc,
            class_name="Document",
            vector=vector
        )

print(f"✅ Inserted {len(documents)} documents into Weaviate")

# Check count
result = client.query.aggregate("Document").with_meta_count().do()
count = result['data']['Aggregate']['Document'][0]['meta']['count']
print(f"   Total documents: {count}")
"""

print("Weaviate Batch Insert:")
print("  - Efficient batch processing")
print("  - Automatic indexing")
print("  - Immediate availability for search")
```

### Query Weaviate with Vector Search

```python
# Example 13: Vector similarity search in Weaviate
"""
# Query
query_text = "What is Python?"
query_vector = model.encode(query_text).tolist()

# Vector search with nearVector
result = client.query.get(
    "Document",
    ["title", "content", "category", "date"]
).with_near_vector({
    "vector": query_vector
}).with_limit(3).with_additional(["distance", "certainty"]).do()

print(f"🔍 Query: '{query_text}'\\n")
print("Results:")
for i, doc in enumerate(result['data']['Get']['Document'], 1):
    print(f"{i}. {doc['title']}")
    print(f"   Category: {doc['category']}")
    print(f"   Distance: {doc['_additional']['distance']:.3f}")
    print(f"   Certainty: {doc['_additional']['certainty']:.3f}")
    print(f"   Content: {doc['content'][:60]}...")
    print()
"""

print("Weaviate Vector Search:")
print("  - nearVector for similarity search")
print("  - Returns distance and certainty scores")
print("  - Combine with filters")
```

### Weaviate with Filters

```python
# Example 14: Combine vector search with metadata filters
"""
# Search: Similar to "programming" AND category = "technology"
result = client.query.get(
    "Document",
    ["title", "content", "category"]
).with_near_vector({
    "vector": query_vector
}).with_where({
    "path": ["category"],
    "operator": "Equal",
    "valueString": "technology"
}).with_limit(5).do()

print("Filtered Search Results:")
for doc in result['data']['Get']['Document']:
    print(f"- {doc['title']} (category: {doc['category']})")

# Complex filter: category = tech AND date > 2024-01-15
result = client.query.get(
    "Document",
    ["title", "date"]
).with_near_vector({
    "vector": query_vector
}).with_where({
    "operator": "And",
    "operands": [
        {
            "path": ["category"],
            "operator": "Equal",
            "valueString": "technology"
        },
        {
            "path": ["date"],
            "operator": "GreaterThan",
            "valueDate": "2024-01-15T00:00:00Z"
        }
    ]
}).do()
"""

print("Weaviate Filters:")
print("  - Equal, NotEqual, GreaterThan, LessThan")
print("  - And, Or operators for complex queries")
print("  - Combine with vector similarity")
```

### Weaviate GraphQL Queries

```python
# Example 15: GraphQL interface
"""
# Weaviate supports GraphQL for complex queries

# GraphQL query
graphql_query = '''
{
  Get {
    Document(
      limit: 3
      nearVector: {
        vector: [0.1, 0.2, ...]  # Your query vector
      }
    ) {
      title
      content
      category
      _additional {
        distance
        certainty
      }
    }
  }
}
'''

# Execute via HTTP or client
result = client.query.raw(graphql_query)
"""

print("Weaviate GraphQL:")
print("  - Flexible query language")
print("  - Nested queries")
print("  - Aggregations")
```

---

## 5. Pinecone: Managed Cloud Vector Database

### Pinecone Setup

```python
# Example 16: Pinecone cloud setup (requires API key)
"""
# Install: pip install pinecone-client

import pinecone
import os

# Initialize
pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment='us-west1-gcp'  # or your region
)

# Create index
index_name = 'document-embeddings'

if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=384,  # Match your embedding model
        metric='cosine',  # or 'euclidean', 'dotproduct'
        pods=1,
        pod_type='p1.x1'  # Performance tier
    )

print(f"✅ Created Pinecone index: {index_name}")
print(f"   Dimension: 384")
print(f"   Metric: cosine")

# Connect to index
index = pinecone.Index(index_name)

# Check stats
stats = index.describe_index_stats()
print(f"\\n   Vectors: {stats['total_vector_count']}")
print(f"   Dimension: {stats['dimension']}")
"""

print("Pinecone Setup (Cloud - requires API key):")
print("  - Fully managed service")
print("  - Auto-scaling")
print("  - High availability")
print("  - Pay per use")
```

### Insert Vectors into Pinecone

```python
# Example 17: Upsert vectors to Pinecone
"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# Prepare data
documents = [
    {"id": "doc1", "text": "Python programming language"},
    {"id": "doc2", "text": "Machine learning algorithms"},
    {"id": "doc3", "text": "Data science with pandas"}
]

# Generate embeddings and upsert
vectors = []
for doc in documents:
    embedding = model.encode(doc['text']).tolist()
    vectors.append({
        'id': doc['id'],
        'values': embedding,
        'metadata': {'text': doc['text']}
    })

# Batch upsert (up to 100 vectors at once)
index.upsert(vectors=vectors, namespace='default')

print(f"✅ Upserted {len(vectors)} vectors to Pinecone")

# Check stats
stats = index.describe_index_stats()
print(f"   Total vectors: {stats['total_vector_count']}")
"""

print("Pinecone Upsert:")
print("  - Batch upsert (up to 100 vectors)")
print("  - Automatic indexing")
print("  - Metadata support")
```

### Query Pinecone

```python
# Example 18: Query Pinecone index
"""
# Query
query_text = "What is Python?"
query_embedding = model.encode(query_text).tolist()

# Search
results = index.query(
    vector=query_embedding,
    top_k=3,
    include_values=False,
    include_metadata=True,
    namespace='default'
)

print(f"🔍 Query: '{query_text}'\\n")
print("Results:")
for match in results['matches']:
    print(f"  ID: {match['id']}")
    print(f"  Score: {match['score']:.3f}")
    print(f"  Text: {match['metadata']['text']}")
    print()

# Filter by metadata
results = index.query(
    vector=query_embedding,
    top_k=3,
    filter={"category": {"$eq": "technology"}},
    include_metadata=True
)
"""

print("Pinecone Query:")
print("  - Fast similarity search")
print("  - Metadata filtering")
print("  - Namespace isolation")
```

### Pinecone Features

```python
# Example 19: Pinecone production features
"""
Pinecone Key Features:

1. ✅ Managed Service
   - No infrastructure to manage
   - Auto-scaling
   - High availability (99.9% SLA)

2. ✅ Performance
   - Single-digit ms latency
   - Handles billions of vectors
   - Global deployment

3. ✅ Metadata Filtering
   - Rich filter queries
   - $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin

4. ✅ Namespaces
   - Logical partitions in single index
   - Isolate different datasets

5. ✅ Updates & Deletes
   - Real-time updates
   - Partial metadata updates
   - Bulk delete operations

6. ✅ Monitoring
   - Built-in metrics dashboard
   - Query latency tracking
   - Usage analytics

Pricing:
  - Starter: Free (1M vectors, 1 pod)
  - Standard: $0.096/hour per pod
  - Enterprise: Custom pricing

Cost Example (1M vectors, 384 dims):
  - Starter pod (p1.x1): ~$70/month
  - Production pod (p2.x1): ~$140/month
"""

print("Pinecone Production Features:")
print("  ✅ Fully managed (no DevOps)")
print("  ✅ Auto-scaling")
print("  ✅ 99.9% SLA")
print("  ✅ Global deployment")
print("  ⚠️  Cost: ~$70-140/month for 1M vectors")
```

---

## 6. Milvus: Open-Source Alternative

### Milvus Overview

```python
# Example 20: Milvus setup and usage
"""
# Install Milvus (Docker):
docker run -d --name milvus_standalone \\
  -p 19530:19530 -p 9091:9091 \\
  milvusdb/milvus:latest

# Install Python client:
pip install pymilvus

from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType

# Connect
connections.connect("default", host="localhost", port="19530")

# Define schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1000)
]
schema = CollectionSchema(fields, "Document embeddings")

# Create collection
collection = Collection("documents", schema)

# Insert data
data = [
    [[0.1, 0.2, ...]],  # embeddings
    ["Document text"]   # metadata
]
collection.insert(data)

# Create index
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# Search
search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=5
)
"""

print("Milvus (Open-Source):")
print("  ✅ Free and open-source")
print("  ✅ Highly scalable (distributed mode)")
print("  ✅ Rich index types (IVF, HNSW, etc.)")
print("  ✅ GPU support")
print("  ⚠️  Requires infrastructure management")
```

---

## 7. Qdrant: Modern Vector Database

```python
# Example 21: Qdrant overview
"""
# Install: pip install qdrant-client

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize client (local or cloud)
client = QdrantClient(":memory:")  # or "http://localhost:6333"

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

# Insert points
points = [
    PointStruct(
        id=1,
        vector=[0.1, 0.2, ...],
        payload={"text": "Document 1", "category": "tech"}
    ),
    # ... more points
]
client.upsert(collection_name="documents", points=points)

# Search
results = client.search(
    collection_name="documents",
    query_vector=[0.1, 0.2, ...],
    limit=5,
    query_filter={
        "must": [
            {"key": "category", "match": {"value": "tech"}}
        ]
    }
)
"""

print("Qdrant Features:")
print("  ✅ REST and gRPC APIs")
print("  ✅ Rich filtering capabilities")
print("  ✅ Hybrid search (dense + sparse)")
print("  ✅ Cloud and self-hosted options")
print("  ✅ Modern Rust implementation (fast)")
```

---

## 8. Sharding and Replication

### Horizontal Scaling with Sharding

```python
# Example 22: Sharding concepts
"""
Sharding: Distribute data across multiple nodes

┌─────────────────────────────────────────────────────┐
│              Vector Database Cluster                │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │   Shard 0    │  │   Shard 1    │  │ Shard 2  │ │
│  │              │  │              │  │          │ │
│  │ Docs 0-333k  │  │ Docs 333k-   │  │ Docs     │ │
│  │              │  │     667k     │  │ 667k-1M  │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│                                                     │
└─────────────────────────────────────────────────────┘

Benefits:
✅ Scale beyond single machine memory
✅ Parallel query processing
✅ Linear scaling with nodes

Challenges:
⚠️  Distribute data evenly
⚠️  Route queries to correct shards
⚠️  Aggregate results from multiple shards
"""

print("Sharding Strategy:")
print("\n1. Hash-based sharding")
print("   - Shard = hash(doc_id) % num_shards")
print("   - Even distribution")
print("   - May split related documents")

print("\n2. Range-based sharding")
print("   - Shard by metadata (e.g., date ranges)")
print("   - Temporal locality")
print("   - Can have imbalanced shards")

print("\n3. Semantic sharding")
print("   - Cluster similar documents")
print("   - Fewer shards to search")
print("   - Complex to maintain")
```

### Replication for High Availability

```python
# Example 23: Replication concepts
"""
Replication: Copy data for redundancy

┌───────────────────────────────────────┐
│            Load Balancer              │
└────────┬─────────────┬────────────────┘
         │             │
    ┌────▼─────┐  ┌───▼──────┐
    │ Primary  │  │ Replica  │
    │  Node    │  │   Node   │
    │          │  │          │
    │ Read +   │  │ Read     │
    │ Write    │  │ Only     │
    └──────────┘  └──────────┘
         │
         │ Sync
         ▼
    ┌──────────┐
    │ Replica  │
    │   Node   │
    │          │
    │ Read     │
    │ Only     │
    └──────────┘

Benefits:
✅ High availability (failover)
✅ Read scaling (distribute queries)
✅ Disaster recovery

Consistency Models:
1. Synchronous: Wait for all replicas (slow, consistent)
2. Asynchronous: Don't wait (fast, eventual consistency)
3. Quorum: Wait for majority (balanced)
"""

print("Replication Strategies:")
print("\n1. Primary-Replica")
print("   - One primary (writes)")
print("   - Multiple replicas (reads)")
print("   - Simple, eventual consistency")

print("\n2. Multi-Primary")
print("   - All nodes accept writes")
print("   - Conflict resolution needed")
print("   - High availability")

print("\n3. Consensus-Based")
print("   - Raft/Paxos protocols")
print("   - Strong consistency")
print("   - More complex")
```

---

## Practice Exercises

### Exercise 1: FAISS Index Comparison
Benchmark Flat, IVF, and HNSW indexes on a 100k vector dataset. Measure speed, recall, and memory usage.

### Exercise 2: Weaviate RAG System
Build a complete RAG system with Weaviate including document ingestion, chunking, and query answering.

### Exercise 3: Pinecone Migration
Migrate an existing FAISS index to Pinecone. Compare performance and costs.

### Exercise 4: Sharding Simulation
Implement a simple sharding layer on top of multiple FAISS indexes.

### Exercise 5: Hybrid Search
Combine BM25 (keyword) search with vector search in Weaviate or Qdrant.

---

## Key Takeaways

1. **Traditional databases** are O(n) for similarity search - unusable at scale
2. **Vector databases** use ANN algorithms for sub-linear search time
3. **FAISS** is perfect for local development and prototyping
4. **Weaviate** offers production features with GraphQL and filters
5. **Pinecone** is fully managed but costs money (~$70-140/month for 1M vectors)
6. **Milvus** and **Qdrant** are excellent open-source alternatives
7. **Sharding** enables horizontal scaling beyond single-machine limits
8. **Replication** provides high availability and read scaling
9. **Production** choice depends on: scale, budget, DevOps capacity

---

## Further Reading

### Essential Resources
- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **Weaviate Docs**: https://weaviate.io/developers/weaviate
- **Pinecone Docs**: https://docs.pinecone.io/
- **Milvus Docs**: https://milvus.io/docs
- **Qdrant Docs**: https://qdrant.tech/documentation/

### Papers
- **FAISS: A Library for Efficient Similarity Search** (Johnson et al., 2017)
- **HNSW: Efficient and robust approximate nearest neighbor search** (Malkov & Yashunin, 2018)

### Cross-References
- **Module 16 Lesson 1**: Vector Embeddings (previous lesson)
- **Module 16 Lesson 3**: Similarity Search & ANN Algorithms (next lesson)
- **Module 7 Lesson 6**: RAG basics

---

**Next Lesson**: Similarity Search & ANN Algorithms - Deep dive into HNSW, IVF, Product Quantization, and LSH!

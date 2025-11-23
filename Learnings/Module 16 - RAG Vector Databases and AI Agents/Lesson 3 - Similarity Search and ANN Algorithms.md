# Lesson 3: Similarity Search & ANN Algorithms 🎯

**Module 16: RAG, Vector Databases & AI Agents | Lesson 3 of 8**

Master the algorithms that power billion-scale vector search - from exact search to approximate methods!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand exact nearest neighbor search and why it doesn't scale
2. ✅ Master Approximate Nearest Neighbor (ANN) trade-offs
3. ✅ Implement HNSW (Hierarchical Navigable Small World) graphs
4. ✅ Build IVF (Inverted File Index) with clustering
5. ✅ Apply Product Quantization for compression
6. ✅ Use Locality-Sensitive Hashing (LSH) for similarity search
7. ✅ Compare distance metrics: cosine, L2, dot product, Manhattan
8. ✅ Benchmark algorithms on speed, accuracy, and memory

---

## Prerequisites

- **Module 16 Lesson 1**: Vector Embeddings (REQUIRED)
- **Module 16 Lesson 2**: Vector Database Internals
- NumPy and basic linear algebra
- Understanding of data structures (graphs, trees)

---

## 1. Exact Nearest Neighbor Search

### Brute Force Search

```python
import numpy as np
import time
from scipy.spatial import distance

# Example 1: Brute force nearest neighbor search
def brute_force_search(query, database, k=5):
    """
    Exact k-nearest neighbors using brute force.

    Time Complexity: O(n * d)
    - n: number of vectors in database
    - d: dimensionality

    This is EXACT but SLOW for large datasets!
    """
    distances = []

    # Compute distance to every vector
    for i, vec in enumerate(database):
        dist = distance.euclidean(query, vec)
        distances.append((i, dist))

    # Sort by distance
    distances.sort(key=lambda x: x[1])

    # Return top-k
    return distances[:k]

# Create test data
n_vectors = 10000
dim = 128

database = np.random.randn(n_vectors, dim).astype('float32')
database = database / np.linalg.norm(database, axis=1, keepdims=True)

query = np.random.randn(dim).astype('float32')
query = query / np.linalg.norm(query)

# Benchmark
start = time.time()
results = brute_force_search(query, database, k=10)
elapsed = time.time() - start

print("Brute Force Search:")
print(f"  Dataset: {n_vectors} vectors, {dim} dimensions")
print(f"  Time: {elapsed*1000:.2f}ms")
print(f"  Top-3 results: {[idx for idx, _ in results[:3]]}")
print(f"\n❌ Problem: For 1M vectors: ~{elapsed*100:.1f}s per query!")
print(f"❌ For 1B vectors: ~{elapsed*100000:.1f}s per query!")
```

### KD-Trees (Limited to Low Dimensions)

```python
# Example 2: KD-Tree for low-dimensional data
from sklearn.neighbors import KDTree

# KD-Trees work well in low dimensions (< 20)
# But suffer "curse of dimensionality" in high dimensions

# Low-dimensional test
low_dim_data = np.random.randn(10000, 3)  # Only 3 dimensions
tree = KDTree(low_dim_data)

query_low = np.random.randn(1, 3)

start = time.time()
distances_kd, indices_kd = tree.query(query_low, k=10)
kd_time = time.time() - start

print("\nKD-Tree Search (Low Dimensions):")
print(f"  Data: 10000 vectors, 3 dimensions")
print(f"  Time: {kd_time*1000:.2f}ms")

# High-dimensional test (KD-Tree degrades)
high_dim_data = np.random.randn(10000, 128)
tree_high = KDTree(high_dim_data)
query_high = np.random.randn(1, 128)

start = time.time()
distances_high, indices_high = tree_high.query(query_high, k=10)
kd_high_time = time.time() - start

print("\nKD-Tree Search (High Dimensions):")
print(f"  Data: 10000 vectors, 128 dimensions")
print(f"  Time: {kd_high_time*1000:.2f}ms")
print(f"  ⚠️  Performance degrades with dimensionality!")
print(f"  Ratio: {kd_high_time/kd_time:.1f}x slower than 3D")
```

### Why Exact Search Doesn't Scale

```python
# Example 3: Scaling analysis
"""
Exact Search Complexity:

1. Brute Force: O(n * d)
   - Must compare query to ALL n vectors
   - Each comparison is O(d) operations

2. KD-Trees: O(d * log n) in low dims, but O(d * n) in high dims
   - "Curse of dimensionality"
   - In 100+ dimensions, degrades to linear search

Real-World Scale:
- 1M vectors, 384 dims, 100 QPS (queries per second)
- Brute force: ~100ms per query → Can't meet 100 QPS!
- 1B vectors: ~100s per query → Unusable!

Solution: Approximate Nearest Neighbor (ANN)
- Trade perfect accuracy for speed
- 95-99% recall is usually sufficient
- 10-1000x faster than exact search
"""

print("Scaling Challenges:")
print("\nDataset Size → Search Time (brute force):")
print("  1K vectors:     ~0.1ms")
print("  10K vectors:    ~1ms")
print("  100K vectors:   ~10ms")
print("  1M vectors:     ~100ms")
print("  10M vectors:    ~1s")
print("  100M vectors:   ~10s")
print("  1B vectors:     ~100s ❌ UNUSABLE!")
print("\n💡 Need sub-linear search: O(log n) or O(√n)")
```

---

## 2. Approximate Nearest Neighbor (ANN)

### ANN Fundamentals

```python
# Example 4: ANN vs Exact Search Trade-offs
"""
Approximate Nearest Neighbor (ANN):
Find "approximately" the nearest neighbors - good enough!

Key Metrics:
1. Recall: % of true nearest neighbors found
   - Recall@10: Are 10 results in the true top-10?
   - 95-99% recall is typical target

2. Speed: Queries per second (QPS)
   - 100-1000x faster than exact search

3. Memory: Index size overhead
   - 1-2x original data size typical

4. Build Time: Time to construct index
   - One-time cost, amortized over queries

Trade-off Knobs:
- Higher accuracy → Slower queries
- Faster build → Lower accuracy
- Less memory → Lower accuracy
"""

def calculate_recall(true_neighbors, approx_neighbors):
    """Calculate recall: how many true neighbors were found?"""
    true_set = set(true_neighbors)
    approx_set = set(approx_neighbors)
    overlap = len(true_set & approx_set)
    return overlap / len(true_set)

# Example
true_top10 = [5, 12, 7, 89, 34, 56, 23, 91, 45, 67]
approx_top10 = [5, 12, 7, 89, 34, 56, 23, 100, 45, 67]  # 1 wrong

recall = calculate_recall(true_top10, approx_top10)
print(f"Recall@10: {recall*100:.0f}%")
print(f"  True neighbors found: {int(recall*10)}/10")
print(f"  Missed: {10 - int(recall*10)}")
```

---

## 3. HNSW: Hierarchical Navigable Small World

### HNSW Intuition

```python
# Example 5: HNSW concept - hierarchical graph search
"""
HNSW: Multi-layer graph for fast navigation

Layer 2 (Top):     Few nodes, long-range connections
      o--------o-----------o
      │        │           │
Layer 1:       More nodes, medium-range
      o----o---o---o-------o
      │    │   │   │       │
Layer 0:       ALL nodes, short-range connections
  o-o-o-o-o-o-o-o-o-o-o-o-o-o

Search Algorithm:
1. Start at top layer (sparse, fast traversal)
2. Greedily navigate toward query
3. Drop to next layer when stuck (local minimum)
4. Repeat until bottom layer
5. Return k nearest neighbors

Why it's fast:
- Top layers: Jump across dataset quickly
- Bottom layer: Refine to exact neighbors
- Time complexity: O(log n) with high probability!
"""

print("HNSW Algorithm:")
print("\n1. Construction:")
print("   - Insert vectors one by one")
print("   - Assign random layer (exponential distribution)")
print("   - Connect to M nearest neighbors per layer")

print("\n2. Search:")
print("   - Start at entry point (top layer)")
print("   - Greedy search: follow edge closer to query")
print("   - Drop to next layer at local minimum")
print("   - Bottom layer: return k best")

print("\n3. Parameters:")
print("   - M: Max connections per node (16-64 typical)")
print("   - efConstruction: Candidates during build (200 typical)")
print("   - efSearch: Candidates during search (64-200 typical)")
```

### HNSW with FAISS

```python
# Example 6: HNSW implementation with FAISS
import faiss
import numpy as np
import time

# Generate test data
n_vectors = 100000
dim = 128

vectors = np.random.randn(n_vectors, dim).astype('float32')
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# Build HNSW index
M = 32  # Number of connections per layer
ef_construction = 200  # Quality during construction
ef_search = 64  # Quality during search

print("Building HNSW Index...")
start = time.time()

index_hnsw = faiss.IndexHNSWFlat(dim, M)
index_hnsw.hnsw.efConstruction = ef_construction
index_hnsw.add(vectors)

build_time = time.time() - start

print(f"✅ Built HNSW index")
print(f"   Vectors: {index_hnsw.ntotal:,}")
print(f"   M: {M}")
print(f"   Build time: {build_time:.2f}s")
print(f"   Build rate: {n_vectors/build_time:.0f} vectors/sec")

# Search with different ef values
query = np.random.randn(1, dim).astype('float32')
query = query / np.linalg.norm(query)

for ef in [16, 32, 64, 128, 256]:
    index_hnsw.hnsw.efSearch = ef

    start = time.time()
    distances, indices = index_hnsw.search(query, k=10)
    search_time = time.time() - start

    print(f"\nefSearch={ef}:")
    print(f"  Search time: {search_time*1000:.2f}ms")
    print(f"  QPS: {1/search_time:.0f}")
    print(f"  Top-3 indices: {indices[0][:3]}")
```

### HNSW Recall vs Speed Trade-off

```python
# Example 7: Measure HNSW recall at different ef values
# First, get ground truth with exact search
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(vectors)

# Ground truth
true_distances, true_indices = index_flat.search(query, k=10)

print("\n" + "="*60)
print("HNSW Recall vs Speed Trade-off:")
print("="*60)

results = []
for ef in [8, 16, 32, 64, 128, 256]:
    index_hnsw.hnsw.efSearch = ef

    start = time.time()
    approx_distances, approx_indices = index_hnsw.search(query, k=10)
    search_time = time.time() - start

    # Calculate recall
    recall = calculate_recall(true_indices[0], approx_indices[0])

    results.append({
        'ef': ef,
        'time_ms': search_time * 1000,
        'recall': recall * 100,
        'qps': int(1 / search_time)
    })

    print(f"ef={ef:3d} | Time: {search_time*1000:5.2f}ms | "
          f"Recall: {recall*100:5.1f}% | QPS: {1/search_time:6.0f}")

print("\n💡 Sweet spot: ef=64 (100% recall, good speed)")
```

---

## 4. IVF: Inverted File Index

### IVF with Clustering

```python
# Example 8: IVF (Inverted File Index) algorithm
"""
IVF: Cluster-based search

1. Training Phase:
   - Cluster vectors into K groups (k-means)
   - Each cluster = "Voronoi cell"
   - Store centroid for each cluster

2. Indexing Phase:
   - Assign each vector to nearest cluster
   - Store vectors in inverted lists

3. Search Phase:
   - Find nprobe nearest cluster centroids to query
   - Search only vectors in those clusters
   - Much faster: search ~1% of database!

Visualization:
     Cluster 0        Cluster 1        Cluster 2
        ●                ●                ●   ← Centroids
      ┌─┴─┐            ┌─┴─┐            ┌─┴─┐
      v v v            v v v            v v v
    Vectors          Vectors          Vectors

Query: Find nearest centroid(s), search only those clusters!
"""

# Build IVF index
n_clusters = 256  # More clusters = finer partitioning
n_probe = 16      # Search top-16 clusters

print("Building IVF Index...")
start = time.time()

# Quantizer: finds nearest cluster
quantizer = faiss.IndexFlatL2(dim)

# IVF index
index_ivf = faiss.IndexIVFFlat(quantizer, dim, n_clusters)

# Train (k-means clustering)
print(f"  Training k-means with {n_clusters} clusters...")
index_ivf.train(vectors)

# Add vectors to clusters
index_ivf.add(vectors)

build_time = time.time() - start

print(f"✅ Built IVF index")
print(f"   Vectors: {index_ivf.ntotal:,}")
print(f"   Clusters: {n_clusters}")
print(f"   Build time: {build_time:.2f}s")

# Search with different nprobe values
for nprobe in [1, 4, 16, 64, 256]:
    index_ivf.nprobe = nprobe

    start = time.time()
    distances_ivf, indices_ivf = index_ivf.search(query, k=10)
    search_time = time.time() - start

    # Calculate recall vs ground truth
    recall = calculate_recall(true_indices[0], indices_ivf[0])

    print(f"\nnprobe={nprobe:3d}:")
    print(f"  Time: {search_time*1000:5.2f}ms")
    print(f"  Recall: {recall*100:5.1f}%")
    print(f"  Searched: {nprobe/n_clusters*100:.1f}% of database")
```

### IVF Variants

```python
# Example 9: Different IVF variants
"""
IVF Variants in FAISS:

1. IndexIVFFlat
   - Store full vectors (no compression)
   - Highest accuracy
   - Most memory

2. IndexIVFPQ (with Product Quantization)
   - Compress vectors (lossy)
   - 8-32x less memory
   - Slight accuracy loss

3. IndexIVFScalarQuantizer
   - Quantize to 8-bit or 4-bit
   - 4-8x less memory
   - Minimal accuracy loss

4. IndexIVFPQR (PQ + Refinement)
   - Initial search with PQ
   - Re-rank with original vectors
   - Best accuracy/speed trade-off
"""

print("IVF Variants:")
print("\n1. IVFFlat:")
print("   Memory: 1x (baseline)")
print("   Accuracy: Highest")
print("   Speed: Fast")

print("\n2. IVFPQ:")
print("   Memory: 0.03-0.12x (8-32x compression)")
print("   Accuracy: Good (95-98% of IVFFlat)")
print("   Speed: Fastest")

print("\n3. IVFScalarQuantizer:")
print("   Memory: 0.12-0.25x")
print("   Accuracy: Very Good")
print("   Speed: Very Fast")
```

---

## 5. Product Quantization (PQ)

### PQ Concept

```python
# Example 10: Product Quantization explained
"""
Product Quantization: Lossy compression for vectors

Idea: Split vector into subvectors, quantize each separately

Original vector (128-dim):
[0.1, 0.5, ..., 0.3, 0.7, ..., 0.2, 0.9, ..., 0.4]
 └─ Subvec 1 ─┘  └─ Subvec 2 ─┘  └─ Subvec 3 ─┘  └─ Subvec 4 ─┘

For each subvector:
1. Cluster into 256 centroids (k-means)
2. Replace subvector with centroid ID (1 byte!)

Compressed:
[  17,      203,      45,      128]  ← Just 4 bytes!
 ↑ ID      ↑ ID      ↑ ID      ↑ ID

Original: 128 floats × 4 bytes = 512 bytes
Compressed: 4 bytes
Compression: 128x smaller!

Distance Computation:
- Precompute query distances to all centroids
- Lookup and sum: O(m) instead of O(d)
"""

print("Product Quantization:")
print("\nOriginal vector: 128 dims × 4 bytes = 512 bytes")
print("PQ compressed:   4 subvectors × 1 byte = 4 bytes")
print("Compression ratio: 128x!")

print("\nAccuracy:")
print("  - 95-98% recall vs full vectors")
print("  - Lossy: cannot reconstruct exact vectors")

print("\nSpeed:")
print("  - Distance computation: O(m) vs O(d)")
print("  - m = number of subvectors (typically 8-64)")
print("  - d = original dimensionality (128-1536)")
```

### PQ with FAISS

```python
# Example 11: Product Quantization implementation
# PQ parameters
n_subvectors = 8  # Split vector into 8 parts
bits_per_subvector = 8  # 2^8 = 256 centroids per subvector

print(f"\nBuilding PQ Index...")
print(f"  Subvectors: {n_subvectors}")
print(f"  Bits per subvector: {bits_per_subvector}")
print(f"  Centroids per subvector: {2**bits_per_subvector}")

start = time.time()

# Create PQ index
index_pq = faiss.IndexPQ(dim, n_subvectors, bits_per_subvector)

# Train
index_pq.train(vectors)
index_pq.add(vectors)

build_time = time.time() - start

print(f"✅ Built PQ index")
print(f"   Build time: {build_time:.2f}s")

# Memory comparison
original_size = n_vectors * dim * 4  # float32
pq_size = n_vectors * n_subvectors * (bits_per_subvector // 8)

print(f"\nMemory Usage:")
print(f"  Original: {original_size / 1024**2:.1f} MB")
print(f"  PQ: {pq_size / 1024**2:.1f} MB")
print(f"  Compression: {original_size/pq_size:.0f}x")

# Search
start = time.time()
distances_pq, indices_pq = index_pq.search(query, k=10)
search_time = time.time() - start

recall_pq = calculate_recall(true_indices[0], indices_pq[0])

print(f"\nPQ Search:")
print(f"  Time: {search_time*1000:.2f}ms")
print(f"  Recall: {recall_pq*100:.1f}%")
```

### IVF + PQ Combined

```python
# Example 12: Combine IVF clustering with PQ compression
# Best of both worlds: speed + memory efficiency

n_clusters = 256
n_subvectors = 8
n_probe = 16

print("\nBuilding IVF+PQ Index...")

start = time.time()

# IVF with PQ compression
quantizer = faiss.IndexFlatL2(dim)
index_ivfpq = faiss.IndexIVFPQ(quantizer, dim, n_clusters, n_subvectors, 8)

# Train
index_ivfpq.train(vectors)
index_ivfpq.add(vectors)

build_time = time.time() - start

print(f"✅ Built IVF+PQ index")
print(f"   Clusters: {n_clusters}")
print(f"   Subvectors: {n_subvectors}")
print(f"   Build time: {build_time:.2f}s")

# Memory
ivfpq_size = n_vectors * n_subvectors
print(f"\nMemory: {ivfpq_size / 1024**2:.1f} MB")
print(f"Compression vs original: {original_size/ivfpq_size:.0f}x")

# Search
index_ivfpq.nprobe = n_probe

start = time.time()
distances_ivfpq, indices_ivfpq = index_ivfpq.search(query, k=10)
search_time = time.time() - start

recall_ivfpq = calculate_recall(true_indices[0], indices_ivfpq[0])

print(f"\nIVF+PQ Search:")
print(f"  Time: {search_time*1000:.2f}ms")
print(f"  Recall: {recall_ivfpq*100:.1f}%")
print(f"  QPS: {1/search_time:.0f}")
```

---

## 6. Locality-Sensitive Hashing (LSH)

### LSH Concept

```python
# Example 13: Locality-Sensitive Hashing intuition
"""
Locality-Sensitive Hashing (LSH):
Hash similar vectors to same bucket with high probability

Key Idea:
- Normal hash: small change → completely different hash
- LSH: similar inputs → same hash with high probability

Example: Random Projection LSH
1. Generate random hyperplane
2. Hash based on which side of hyperplane
   - Above: hash bit = 1
   - Below: hash bit = 0
3. Multiple hyperplanes → multiple hash bits

Visualization:
        Hyperplane 1
            │
    v1  v2  │  v3  v4
    ●   ●   │  ●   ●
            │
    ────────┼────────
            │

Hash bits:
  v1, v2: 0 (below)  → same bucket!
  v3, v4: 1 (above)  → same bucket!

With k hyperplanes:
- 2^k hash buckets
- Similar vectors → high probability of same bucket
"""

print("LSH Algorithm:")
print("\n1. Preprocessing:")
print("   - Generate k random hyperplanes")
print("   - Hash all vectors")
print("   - Build hash table")

print("\n2. Query:")
print("   - Hash query vector")
print("   - Retrieve vectors from same bucket")
print("   - Optionally check nearby buckets")

print("\n3. Trade-offs:")
print("   - More hyperplanes (k) → fewer collisions, slower")
print("   - Fewer hyperplanes → more collisions, faster but lower recall")
```

### LSH Implementation

```python
# Example 14: Simple LSH from scratch
class LSH:
    """
    Locality-Sensitive Hashing using random projections.
    """

    def __init__(self, dim, n_hyperplanes=10):
        self.dim = dim
        self.n_hyperplanes = n_hyperplanes

        # Random hyperplanes (normal vectors)
        self.hyperplanes = np.random.randn(n_hyperplanes, dim)
        self.hyperplanes = self.hyperplanes / np.linalg.norm(
            self.hyperplanes, axis=1, keepdims=True
        )

        # Hash table: hash -> list of vector indices
        self.hash_table = {}
        self.vectors = []

    def _hash(self, vector):
        """Hash a vector to binary string"""
        # Dot product with hyperplanes
        projections = np.dot(self.hyperplanes, vector)

        # Convert to binary hash
        hash_bits = (projections > 0).astype(int)

        # Convert to string
        return ''.join(map(str, hash_bits))

    def add(self, vectors):
        """Add vectors to index"""
        start_idx = len(self.vectors)

        for i, vec in enumerate(vectors):
            # Store vector
            self.vectors.append(vec)

            # Hash and add to table
            hash_val = self._hash(vec)
            if hash_val not in self.hash_table:
                self.hash_table[hash_val] = []
            self.hash_table[hash_val].append(start_idx + i)

    def query(self, query_vec, k=10):
        """Find approximate nearest neighbors"""
        # Hash query
        query_hash = self._hash(query_vec)

        # Get candidates from same bucket
        candidates = self.hash_table.get(query_hash, [])

        # If not enough, check nearby buckets (Hamming distance 1)
        if len(candidates) < k:
            for hash_val, indices in self.hash_table.items():
                # Hamming distance
                hamming = sum(c1 != c2 for c1, c2 in zip(query_hash, hash_val))
                if hamming == 1:  # One bit different
                    candidates.extend(indices)

        # Compute actual distances to candidates
        distances = []
        for idx in candidates:
            dist = np.linalg.norm(query_vec - self.vectors[idx])
            distances.append((idx, dist))

        # Sort and return top-k
        distances.sort(key=lambda x: x[1])
        return distances[:k]

# Test LSH
print("\nTesting LSH Implementation:")

lsh = LSH(dim=dim, n_hyperplanes=16)

# Index vectors
print(f"Indexing {n_vectors:,} vectors...")
lsh.add(vectors)

print(f"  Hash buckets created: {len(lsh.hash_table)}")
print(f"  Average bucket size: {n_vectors / len(lsh.hash_table):.1f}")

# Query
query_vec = query[0]
start = time.time()
lsh_results = lsh.query(query_vec, k=10)
lsh_time = time.time() - start

lsh_indices = [idx for idx, _ in lsh_results]
recall_lsh = calculate_recall(true_indices[0], lsh_indices)

print(f"\nLSH Search:")
print(f"  Time: {lsh_time*1000:.2f}ms")
print(f"  Recall: {recall_lsh*100:.1f}%")
print(f"  Candidates checked: {len([idx for idx, _ in lsh_results])}")
```

---

## 7. Distance Metrics Comparison

### Common Distance Metrics

```python
# Example 15: Compare distance metrics
from scipy.spatial import distance as sp_distance

# Sample vectors
v1 = np.array([1.0, 2.0, 3.0, 4.0])
v2 = np.array([1.5, 2.5, 3.5, 4.5])
v3 = np.array([-1.0, -2.0, -3.0, -4.0])

# Normalize for cosine
v1_norm = v1 / np.linalg.norm(v1)
v2_norm = v2 / np.linalg.norm(v2)
v3_norm = v3 / np.linalg.norm(v3)

print("Distance Metrics Comparison:")
print("="*60)

# 1. Euclidean (L2)
print("\n1. Euclidean Distance (L2):")
print(f"   v1 <-> v2: {sp_distance.euclidean(v1, v2):.3f}")
print(f"   v1 <-> v3: {sp_distance.euclidean(v1, v3):.3f}")
print("   Use case: General purpose, sensitive to magnitude")

# 2. Cosine Distance
print("\n2. Cosine Distance:")
print(f"   v1 <-> v2: {sp_distance.cosine(v1, v2):.3f}")
print(f"   v1 <-> v3: {sp_distance.cosine(v1, v3):.3f}")
print("   Use case: Text embeddings, ignore magnitude")
print("   Note: v1 and v3 are opposite → max distance (2.0)")

# 3. Dot Product (Normalized)
print("\n3. Dot Product (on normalized vectors):")
dot_12 = np.dot(v1_norm, v2_norm)
dot_13 = np.dot(v1_norm, v3_norm)
print(f"   v1 <-> v2: {dot_12:.3f}")
print(f"   v1 <-> v3: {dot_13:.3f}")
print("   Use case: Fast similarity, equivalent to cosine for normalized")

# 4. Manhattan (L1)
print("\n4. Manhattan Distance (L1):")
print(f"   v1 <-> v2: {sp_distance.cityblock(v1, v2):.3f}")
print(f"   v1 <-> v3: {sp_distance.cityblock(v1, v3):.3f}")
print("   Use case: Sparse vectors, robust to outliers")

# 5. Hamming (for binary)
binary1 = np.array([1, 0, 1, 0, 1, 1, 0, 0])
binary2 = np.array([1, 1, 1, 0, 0, 1, 0, 1])
print("\n5. Hamming Distance (for binary vectors):")
print(f"   Binary vectors: {sp_distance.hamming(binary1, binary2):.3f}")
print("   Use case: LSH, binary embeddings")
```

### When to Use Each Metric

```python
# Example 16: Distance metric selection guide
"""
Distance Metric Selection Guide:

1. ✅ Cosine Distance
   - Text embeddings (word2vec, BERT, etc.)
   - When magnitude doesn't matter
   - Normalized vectors
   - Default for most NLP tasks

2. ✅ Euclidean (L2) Distance
   - Image embeddings
   - When magnitude is meaningful
   - General-purpose
   - Better for clustering

3. ✅ Dot Product
   - Maximum Inner Product Search (MIPS)
   - Recommendation systems
   - Fast when vectors are normalized
   - Equivalent to cosine for unit vectors

4. ✅ Manhattan (L1) Distance
   - Sparse vectors
   - Robust to outliers
   - Faster computation than L2

5. ✅ Hamming Distance
   - Binary embeddings
   - LSH hash codes
   - Extremely fast

Performance:
  Dot Product: Fastest (simple multiply-add)
  Manhattan:   Fast (absolute values)
  Euclidean:   Medium (requires square root)
  Cosine:      Slower (requires normalization)
"""

print("Quick Reference:")
print("\nTask → Recommended Metric:")
print("  Sentence embeddings → Cosine")
print("  Image embeddings → Euclidean")
print("  User preferences → Dot Product")
print("  LSH buckets → Hamming")
print("  Sparse features → Manhattan")
```

---

## 8. Benchmarking ANN Algorithms

### Comprehensive Benchmark

```python
# Example 17: Benchmark all algorithms
import pandas as pd

# Setup
n_test = 100000
dim = 128
k = 10

test_vectors = np.random.randn(n_test, dim).astype('float32')
test_vectors = test_vectors / np.linalg.norm(test_vectors, axis=1, keepdims=True)

test_query = np.random.randn(1, dim).astype('float32')
test_query = test_query / np.linalg.norm(test_query)

print(f"Benchmark Setup:")
print(f"  Dataset: {n_test:,} vectors")
print(f"  Dimensions: {dim}")
print(f"  Query k: {k}")
print("\n" + "="*80)

# Ground truth
index_flat = faiss.IndexFlatL2(dim)
index_flat.add(test_vectors)
true_d, true_i = index_flat.search(test_query, k)

benchmarks = []

# 1. Exact (Flat)
start = time.time()
d_flat, i_flat = index_flat.search(test_query, k)
t_flat = time.time() - start

benchmarks.append({
    'Algorithm': 'Exact (Flat)',
    'Build Time (s)': 0,
    'Search Time (ms)': t_flat * 1000,
    'Recall@10 (%)': 100.0,
    'Memory (MB)': n_test * dim * 4 / 1024**2,
    'QPS': int(1 / t_flat)
})

# 2. HNSW
index_hnsw = faiss.IndexHNSWFlat(dim, 32)
index_hnsw.hnsw.efConstruction = 200

start_build = time.time()
index_hnsw.add(test_vectors)
t_build_hnsw = time.time() - start_build

index_hnsw.hnsw.efSearch = 64
start = time.time()
d_hnsw, i_hnsw = index_hnsw.search(test_query, k)
t_hnsw = time.time() - start

recall_hnsw = calculate_recall(true_i[0], i_hnsw[0])

benchmarks.append({
    'Algorithm': 'HNSW',
    'Build Time (s)': round(t_build_hnsw, 2),
    'Search Time (ms)': round(t_hnsw * 1000, 2),
    'Recall@10 (%)': round(recall_hnsw * 100, 1),
    'Memory (MB)': round(n_test * dim * 4 * 1.5 / 1024**2, 1),  # ~1.5x overhead
    'QPS': int(1 / t_hnsw)
})

# 3. IVF
n_clusters = 256
index_ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(dim), dim, n_clusters)

start_build = time.time()
index_ivf.train(test_vectors)
index_ivf.add(test_vectors)
t_build_ivf = time.time() - start_build

index_ivf.nprobe = 16
start = time.time()
d_ivf, i_ivf = index_ivf.search(test_query, k)
t_ivf = time.time() - start

recall_ivf = calculate_recall(true_i[0], i_ivf[0])

benchmarks.append({
    'Algorithm': 'IVF',
    'Build Time (s)': round(t_build_ivf, 2),
    'Search Time (ms)': round(t_ivf * 1000, 2),
    'Recall@10 (%)': round(recall_ivf * 100, 1),
    'Memory (MB)': round(n_test * dim * 4 / 1024**2, 1),
    'QPS': int(1 / t_ivf)
})

# 4. IVF+PQ
index_ivfpq = faiss.IndexIVFPQ(faiss.IndexFlatL2(dim), dim, 256, 8, 8)

start_build = time.time()
index_ivfpq.train(test_vectors)
index_ivfpq.add(test_vectors)
t_build_ivfpq = time.time() - start_build

index_ivfpq.nprobe = 16
start = time.time()
d_ivfpq, i_ivfpq = index_ivfpq.search(test_query, k)
t_ivfpq = time.time() - start

recall_ivfpq = calculate_recall(true_i[0], i_ivfpq[0])

benchmarks.append({
    'Algorithm': 'IVF+PQ',
    'Build Time (s)': round(t_build_ivfpq, 2),
    'Search Time (ms)': round(t_ivfpq * 1000, 2),
    'Recall@10 (%)': round(recall_ivfpq * 100, 1),
    'Memory (MB)': round(n_test * 8 / 1024**2, 1),
    'QPS': int(1 / t_ivfpq)
})

# Results
df = pd.DataFrame(benchmarks)
print("\nBenchmark Results:")
print("="*80)
print(df.to_string(index=False))

print("\n" + "="*80)
print("Summary:")
print("  Exact: 100% recall but slow")
print("  HNSW: Best accuracy/speed trade-off")
print("  IVF: Good speed with decent recall")
print("  IVF+PQ: Best memory efficiency")
```

### Scaling Projections

```python
# Example 18: Project performance at different scales
"""
Projected Performance (based on benchmarks):

Dataset Size: 1M vectors
├─ Exact:    ~1000ms per query (UNUSABLE at scale)
├─ HNSW:     ~2ms per query, 99% recall ✅
├─ IVF:      ~1ms per query, 95% recall
└─ IVF+PQ:   ~0.8ms per query, 90% recall, 16x less memory

Dataset Size: 10M vectors
├─ Exact:    ~10,000ms per query ❌
├─ HNSW:     ~3ms per query, 99% recall ✅
├─ IVF:      ~1.5ms per query, 93% recall
└─ IVF+PQ:   ~1ms per query, 88% recall

Dataset Size: 100M vectors
├─ Exact:    ~100,000ms ❌ IMPOSSIBLE
├─ HNSW:     ~5ms per query, 98% recall ✅
├─ IVF:      ~2ms per query, 90% recall (with more clusters)
└─ IVF+PQ:   ~1.5ms per query, 85% recall

Recommendation:
- < 100K vectors: Use Exact or HNSW
- 100K - 10M: Use HNSW
- 10M - 100M: Use HNSW or IVF
- 100M+: Use IVF+PQ for memory, HNSW for accuracy
"""

scales = [
    {'size': '100K', 'exact': 100, 'hnsw': 1, 'ivf': 0.5, 'ivfpq': 0.4},
    {'size': '1M', 'exact': 1000, 'hnsw': 2, 'ivf': 1, 'ivfpq': 0.8},
    {'size': '10M', 'exact': 10000, 'hnsw': 3, 'ivf': 1.5, 'ivfpq': 1},
    {'size': '100M', 'exact': 100000, 'hnsw': 5, 'ivf': 2, 'ivfpq': 1.5},
]

print("\nScaling Projections (query latency in ms):")
print("-" * 60)
print(f"{'Size':<10} {'Exact':<12} {'HNSW':<10} {'IVF':<10} {'IVF+PQ':<10}")
print("-" * 60)

for scale in scales:
    print(f"{scale['size']:<10} "
          f"{scale['exact']:<12.1f} "
          f"{scale['hnsw']:<10.1f} "
          f"{scale['ivf']:<10.1f} "
          f"{scale['ivfpq']:<10.1f}")
```

---

## 9. Production Recommendations

```python
# Example 19: Algorithm selection guide
"""
Production Algorithm Selection:

┌─────────────────────────────────────────────────────────────┐
│  Dataset Size  │  Memory OK?  │  Recommended Algorithm     │
├─────────────────────────────────────────────────────────────┤
│  < 100K        │  Yes         │  Exact (Flat) or HNSW      │
│  < 100K        │  No          │  IVF+PQ                    │
│  100K - 1M     │  Yes         │  HNSW (M=32, ef=64)        │
│  100K - 1M     │  No          │  IVF+PQ                    │
│  1M - 10M      │  Yes         │  HNSW (M=48, ef=128)       │
│  1M - 10M      │  No          │  IVF+PQ (256-1024 clusters)│
│  10M - 100M    │  Yes         │  HNSW or IVF               │
│  10M - 100M    │  No          │  IVF+PQ                    │
│  100M+         │  Yes         │  IVF (many clusters)       │
│  100M+         │  No          │  IVF+PQ                    │
└─────────────────────────────────────────────────────────────┘

Recall Requirements:
- 99%+:  HNSW (efSearch=200+) or IVF (high nprobe)
- 95-99%: HNSW (efSearch=64) or IVF (medium nprobe)
- 90-95%: IVF+PQ or lower efSearch/nprobe

Latency Requirements:
- < 1ms:    IVF+PQ
- 1-5ms:    HNSW or IVF
- 5-10ms:   HNSW (high accuracy)
- > 10ms:   Use exact search (small dataset)

Cost Optimization:
- Cloud: Use IVF+PQ to reduce memory → reduce instance size
- On-prem: HNSW for performance, IVF+PQ if memory constrained
"""

print("Quick Decision Tree:")
print("\n1. Need 99%+ recall?")
print("   → Yes: Use HNSW")
print("   → No: Continue...")

print("\n2. Limited memory (< 2x data size)?")
print("   → Yes: Use IVF+PQ")
print("   → No: Continue...")

print("\n3. Dataset > 10M vectors?")
print("   → Yes: Use IVF or IVF+PQ")
print("   → No: Use HNSW")

print("\n4. Need < 1ms latency?")
print("   → Yes: Use IVF+PQ")
print("   → No: Use HNSW or IVF")
```

---

## Practice Exercises

### Exercise 1: HNSW Parameter Tuning
Build an HNSW index and find optimal M and efSearch for 99% recall with minimum latency.

### Exercise 2: IVF Clustering
Experiment with different numbers of clusters (64, 256, 1024) and nprobe values. Plot recall vs latency.

### Exercise 3: PQ Compression
Implement different PQ configurations (4, 8, 16 subvectors) and measure compression vs accuracy trade-offs.

### Exercise 4: LSH Implementation
Enhance the LSH implementation with multiple hash tables for better recall.

### Exercise 5: Algorithm Comparison
Benchmark all algorithms on your own dataset and create a recommendation guide.

---

## Key Takeaways

1. **Exact search** is O(n) - unusable for large datasets
2. **ANN algorithms** trade accuracy for speed (95-99% recall typical)
3. **HNSW** offers best accuracy/speed trade-off for most use cases
4. **IVF** uses clustering to partition search space
5. **Product Quantization** compresses vectors 10-100x with minimal accuracy loss
6. **LSH** is simple but less accurate than graph-based methods
7. **Distance metrics** matter: cosine for text, L2 for images
8. **Benchmarking** is essential - test on your data!
9. **Production** choice depends on: dataset size, memory budget, latency requirements

---

## Further Reading

### Essential Papers
- **HNSW**: Malkov & Yashunin (2018) - https://arxiv.org/abs/1603.09320
- **Product Quantization**: Jegou et al. (2011) - https://hal.inria.fr/inria-00514462/document
- **LSH**: Indyk & Motwani (1998) - Classic paper on LSH

### Documentation
- **FAISS Documentation**: https://github.com/facebookresearch/faiss/wiki
- **FAISS Index Selection**: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

### Cross-References
- **Module 16 Lesson 2**: Vector Database Internals (previous lesson)
- **Module 16 Lesson 4**: Advanced RAG Patterns (next lesson)

### Benchmarks
- **ANN Benchmarks**: http://ann-benchmarks.com/ - Compare algorithms on standard datasets

---

**Next Lesson**: Advanced RAG Patterns & Chunking Strategies - Build production-ready RAG systems!

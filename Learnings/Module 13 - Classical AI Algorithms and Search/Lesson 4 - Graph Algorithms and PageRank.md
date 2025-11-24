# Lesson 4: Graph Algorithms & PageRank 🔍

**Module 13: Classical AI Algorithms and Search | Lesson 4 of 5**

Master the algorithms that power Google Search, social networks, and knowledge graphs!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Implement graph representations and traversal algorithms
2. ✅ Build shortest path algorithms (Dijkstra, Bellman-Ford, Floyd-Warshall)
3. ✅ Understand and implement PageRank algorithm
4. ✅ Apply community detection and graph clustering
5. ✅ Build recommendation systems using graph algorithms
6. ✅ Work with NetworkX for real-world graph problems

---

## 1. Graph Fundamentals

### Graph Representations

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import heapq
from typing import List, Dict, Set, Tuple, Optional

class Graph:
    """
    Graph data structure with multiple representations.

    Supports both directed and undirected graphs.
    """
    def __init__(self, directed=False):
        self.directed = directed
        self.adjacency_list = defaultdict(list)  # {node: [(neighbor, weight), ...]}
        self.nodes = set()
        self.edges = []

    def add_node(self, node):
        """Add a node to the graph."""
        self.nodes.add(node)
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []

    def add_edge(self, from_node, to_node, weight=1):
        """Add an edge to the graph."""
        self.add_node(from_node)
        self.add_node(to_node)

        self.adjacency_list[from_node].append((to_node, weight))
        self.edges.append((from_node, to_node, weight))

        # For undirected graphs, add reverse edge
        if not self.directed:
            self.adjacency_list[to_node].append((from_node, weight))

    def get_neighbors(self, node):
        """Get neighbors of a node."""
        return self.adjacency_list.get(node, [])

    def get_adjacency_matrix(self):
        """Convert to adjacency matrix representation."""
        node_list = sorted(self.nodes)
        n = len(node_list)
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        matrix = np.zeros((n, n))

        for node in node_list:
            for neighbor, weight in self.get_neighbors(node):
                i = node_to_idx[node]
                j = node_to_idx[neighbor]
                matrix[i][j] = weight

        return matrix, node_list

    def __repr__(self):
        return f"Graph(nodes={len(self.nodes)}, edges={len(self.edges)}, directed={self.directed})"


# Create example graph
g = Graph(directed=True)

# Add edges (web-like structure)
edges = [
    ('A', 'B', 1), ('A', 'C', 1),
    ('B', 'C', 1), ('B', 'D', 1),
    ('C', 'D', 1), ('C', 'E', 1),
    ('D', 'E', 1), ('E', 'A', 1)
]

for from_node, to_node, weight in edges:
    g.add_edge(from_node, to_node, weight)

print("Graph:", g)
print("\nAdjacency List:")
for node in sorted(g.nodes):
    neighbors = [n for n, w in g.get_neighbors(node)]
    print(f"  {node}: {neighbors}")

# Get adjacency matrix
adj_matrix, node_list = g.get_adjacency_matrix()
print("\nAdjacency Matrix:")
print("Nodes:", node_list)
print(adj_matrix)
```

### Graph Traversal

```python
def bfs(graph: Graph, start):
    """
    Breadth-First Search traversal.

    Returns:
        visited: List of nodes in BFS order
        distances: Dict {node: distance_from_start}
    """
    visited = []
    distances = {start: 0}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        visited.append(node)

        for neighbor, weight in graph.get_neighbors(node):
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    return visited, distances


def dfs(graph: Graph, start, visited=None):
    """
    Depth-First Search traversal (recursive).

    Returns:
        visited: List of nodes in DFS order
    """
    if visited is None:
        visited = []

    visited.append(start)

    for neighbor, weight in graph.get_neighbors(start):
        if neighbor not in visited:
            dfs(graph, neighbor, visited)

    return visited


# Test traversal
print("\n" + "="*60)
print("Graph Traversal")
print("="*60)

bfs_order, distances = bfs(g, 'A')
print(f"BFS from A: {bfs_order}")
print(f"Distances: {distances}")

dfs_order = dfs(g, 'A')
print(f"\nDFS from A: {dfs_order}")
```

---

## 2. Shortest Path Algorithms

### Dijkstra's Algorithm

Single-source shortest paths with non-negative weights.

```python
def dijkstra(graph: Graph, start):
    """
    Dijkstra's shortest path algorithm.

    Returns:
        distances: Dict {node: shortest_distance_from_start}
        previous: Dict {node: previous_node_in_path}
    """
    # Initialize distances
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0

    # Track previous node in shortest path
    previous = {node: None for node in graph.nodes}

    # Priority queue: (distance, node)
    pq = [(0, start)]
    visited = set()

    while pq:
        current_dist, current_node = heapq.heappop(pq)

        if current_node in visited:
            continue

        visited.add(current_node)

        # Check all neighbors
        for neighbor, weight in graph.get_neighbors(current_node):
            distance = current_dist + weight

            # Found shorter path
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))

    return distances, previous


def reconstruct_path(previous, start, end):
    """Reconstruct shortest path from previous dict."""
    path = []
    current = end

    while current is not None:
        path.append(current)
        current = previous[current]

    path.reverse()

    if path[0] == start:
        return path
    return None  # No path exists


# Test Dijkstra
print("\n" + "="*60)
print("Dijkstra's Algorithm")
print("="*60)

# Create weighted graph
g_weighted = Graph(directed=True)
edges_weighted = [
    ('A', 'B', 4), ('A', 'C', 2),
    ('B', 'C', 1), ('B', 'D', 5),
    ('C', 'D', 8), ('C', 'E', 10),
    ('D', 'E', 2), ('E', 'D', 6)
]

for from_node, to_node, weight in edges_weighted:
    g_weighted.add_edge(from_node, to_node, weight)

distances, previous = dijkstra(g_weighted, 'A')

print("Shortest distances from A:")
for node in sorted(distances.keys()):
    print(f"  {node}: {distances[node]}")

print("\nShortest paths from A:")
for node in sorted(g_weighted.nodes):
    if node != 'A':
        path = reconstruct_path(previous, 'A', node)
        print(f"  A → {node}: {' → '.join(path)} (distance: {distances[node]})")
```

### Bellman-Ford Algorithm

Handles negative weights and detects negative cycles.

```python
def bellman_ford(graph: Graph, start):
    """
    Bellman-Ford algorithm.

    Handles negative weights, detects negative cycles.

    Returns:
        distances, previous, has_negative_cycle
    """
    # Initialize
    distances = {node: float('inf') for node in graph.nodes}
    distances[start] = 0
    previous = {node: None for node in graph.nodes}

    # Relax edges V-1 times
    n = len(graph.nodes)

    for _ in range(n - 1):
        for from_node, to_node, weight in graph.edges:
            if distances[from_node] + weight < distances[to_node]:
                distances[to_node] = distances[from_node] + weight
                previous[to_node] = from_node

    # Check for negative cycles
    has_negative_cycle = False
    for from_node, to_node, weight in graph.edges:
        if distances[from_node] + weight < distances[to_node]:
            has_negative_cycle = True
            break

    return distances, previous, has_negative_cycle


# Test Bellman-Ford with negative weights
print("\n" + "="*60)
print("Bellman-Ford Algorithm")
print("="*60)

g_negative = Graph(directed=True)
edges_negative = [
    ('A', 'B', -1), ('A', 'C', 4),
    ('B', 'C', 3), ('B', 'D', 2),
    ('B', 'E', 2), ('D', 'C', 5),
    ('D', 'B', 1), ('E', 'D', -3)
]

for from_node, to_node, weight in edges_negative:
    g_negative.add_edge(from_node, to_node, weight)

distances, previous, has_neg_cycle = bellman_ford(g_negative, 'A')

print(f"Has negative cycle: {has_neg_cycle}")
print("\nShortest distances from A:")
for node in sorted(distances.keys()):
    print(f"  {node}: {distances[node]}")
```

### Floyd-Warshall Algorithm

All-pairs shortest paths.

```python
def floyd_warshall(graph: Graph):
    """
    Floyd-Warshall algorithm.

    Computes shortest paths between ALL pairs of nodes.

    Returns:
        distance_matrix: 2D array of shortest distances
        next_matrix: For path reconstruction
    """
    nodes = sorted(graph.nodes)
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    # Initialize distance matrix
    dist = np.full((n, n), float('inf'))
    next_node = np.full((n, n), -1, dtype=int)

    # Distance from node to itself is 0
    for i in range(n):
        dist[i][i] = 0

    # Add edges
    for from_node, to_node, weight in graph.edges:
        i = node_to_idx[from_node]
        j = node_to_idx[to_node]
        dist[i][j] = weight
        next_node[i][j] = j

    # Floyd-Warshall: Try all intermediate nodes
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return dist, next_node, nodes


# Test Floyd-Warshall
print("\n" + "="*60)
print("Floyd-Warshall Algorithm")
print("="*60)

dist_matrix, next_matrix, nodes = floyd_warshall(g_weighted)

print("All-pairs shortest distances:")
print("     ", " ".join(f"{n:5s}" for n in nodes))
for i, from_node in enumerate(nodes):
    row = " ".join(f"{dist_matrix[i][j]:5.0f}" for j in range(len(nodes)))
    print(f"{from_node}:   {row}")
```

---

## 3. PageRank Algorithm

### Google's Secret Sauce

PageRank models web surfing as a random walk on the web graph.

**Key Ideas:**
- Page importance = sum of importance of pages linking to it
- Random surfer follows links with probability d, teleports with probability (1-d)
- Iteratively compute until convergence

```python
def pagerank(graph: Graph, damping=0.85, max_iterations=100, tolerance=1e-6):
    """
    PageRank algorithm.

    Args:
        damping: Probability of following a link (vs random jump)
        max_iterations: Maximum iterations
        tolerance: Convergence threshold

    Returns:
        ranks: Dict {node: pagerank_score}
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # Initialize: Equal probability for all pages
    ranks = {node: 1.0 / n for node in nodes}

    # Precompute out-degrees
    out_degree = {}
    for node in nodes:
        neighbors = graph.get_neighbors(node)
        out_degree[node] = len(neighbors) if neighbors else 0

    # Iterate until convergence
    for iteration in range(max_iterations):
        new_ranks = {}

        # Compute new ranks
        for node in nodes:
            rank_sum = 0

            # Sum contributions from incoming links
            for other_node in nodes:
                if other_node == node:
                    continue

                # Check if other_node links to node
                neighbors = [n for n, w in graph.get_neighbors(other_node)]
                if node in neighbors and out_degree[other_node] > 0:
                    rank_sum += ranks[other_node] / out_degree[other_node]

            # PageRank formula
            new_ranks[node] = (1 - damping) / n + damping * rank_sum

        # Check convergence
        diff = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks

        if diff < tolerance:
            print(f"Converged after {iteration + 1} iterations")
            break

    # Normalize to sum to 1
    total = sum(ranks.values())
    ranks = {node: rank / total for node, rank in ranks.items()}

    return ranks


# Test PageRank on web graph
print("\n" + "="*60)
print("PageRank Algorithm")
print("="*60)

# Create web-like graph
web_graph = Graph(directed=True)

# Simulate website links
web_edges = [
    ('Home', 'About', 1),
    ('Home', 'Products', 1),
    ('Home', 'Blog', 1),
    ('About', 'Home', 1),
    ('About', 'Contact', 1),
    ('Products', 'Home', 1),
    ('Products', 'Blog', 1),
    ('Blog', 'Home', 1),
    ('Blog', 'Products', 1),
    ('Contact', 'Home', 1)
]

for from_page, to_page, weight in web_edges:
    web_graph.add_edge(from_page, to_page, weight)

ranks = pagerank(web_graph, damping=0.85)

print("\nPageRank scores:")
sorted_ranks = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
for page, rank in sorted_ranks:
    print(f"  {page:12s}: {rank:.4f}")
```

### PageRank with Matrix Formulation

```python
def pagerank_matrix(graph: Graph, damping=0.85, max_iterations=100):
    """
    PageRank using matrix multiplication.

    More efficient for large graphs.
    """
    # Build adjacency matrix and normalize
    adj_matrix, nodes = graph.get_adjacency_matrix()
    n = len(nodes)

    # Compute stochastic matrix (column-normalized)
    # P[i,j] = probability of going from j to i
    P = np.zeros((n, n))

    for j in range(n):
        out_degree = np.sum(adj_matrix[j, :])
        if out_degree > 0:
            for i in range(n):
                if adj_matrix[j, i] > 0:
                    P[i, j] = 1.0 / out_degree
        else:
            # Dead end: Equal probability to all pages
            P[:, j] = 1.0 / n

    # Build Google matrix: G = d*P + (1-d)*E
    # E = 1/n (teleportation matrix)
    E = np.ones((n, n)) / n
    G = damping * P + (1 - damping) * E

    # Power iteration
    r = np.ones(n) / n  # Initial rank vector

    for _ in range(max_iterations):
        r_new = G @ r

        # Check convergence
        if np.linalg.norm(r_new - r, 1) < 1e-6:
            break

        r = r_new

    # Convert to dictionary
    ranks = {nodes[i]: r[i] for i in range(n)}

    return ranks


# Compare implementations
print("\n" + "="*60)
print("PageRank: Iterative vs Matrix")
print("="*60)

ranks_iterative = pagerank(web_graph, damping=0.85, max_iterations=100)
ranks_matrix = pagerank_matrix(web_graph, damping=0.85, max_iterations=100)

print("\nIterative PageRank:")
for page, rank in sorted(ranks_iterative.items(), key=lambda x: x[1], reverse=True):
    print(f"  {page:12s}: {rank:.4f}")

print("\nMatrix PageRank:")
for page, rank in sorted(ranks_matrix.items(), key=lambda x: x[1], reverse=True):
    print(f"  {page:12s}: {rank:.4f}")
```

### Personalized PageRank

```python
def personalized_pagerank(graph: Graph, personalization: Dict[str, float],
                         damping=0.85, max_iterations=100):
    """
    Personalized PageRank.

    Instead of uniform teleportation, bias towards specific nodes.
    Used in recommendation systems.

    Args:
        personalization: Dict {node: probability} for teleportation
    """
    nodes = list(graph.nodes)
    n = len(nodes)

    # Normalize personalization vector
    total = sum(personalization.values())
    personalization = {k: v/total for k, v in personalization.items()}

    # Initialize ranks
    ranks = {node: personalization.get(node, 0) for node in nodes}

    # Out-degrees
    out_degree = {}
    for node in nodes:
        neighbors = graph.get_neighbors(node)
        out_degree[node] = len(neighbors) if neighbors else 0

    # Iterate
    for iteration in range(max_iterations):
        new_ranks = {}

        for node in nodes:
            rank_sum = 0

            for other_node in nodes:
                if other_node == node:
                    continue

                neighbors = [n for n, w in graph.get_neighbors(other_node)]
                if node in neighbors and out_degree[other_node] > 0:
                    rank_sum += ranks[other_node] / out_degree[other_node]

            # Personalized teleportation
            teleport_prob = personalization.get(node, 0)
            new_ranks[node] = (1 - damping) * teleport_prob + damping * rank_sum

        diff = sum(abs(new_ranks[node] - ranks[node]) for node in nodes)
        ranks = new_ranks

        if diff < 1e-6:
            break

    return ranks


# Personalized PageRank example
print("\n" + "="*60)
print("Personalized PageRank")
print("="*60)

# User interested in "Products"
personalization = {'Products': 1.0}

p_ranks = personalized_pagerank(web_graph, personalization, damping=0.85)

print("Personalized PageRank (biased towards Products):")
for page, rank in sorted(p_ranks.items(), key=lambda x: x[1], reverse=True):
    print(f"  {page:12s}: {rank:.4f}")
```

---

## 4. Community Detection

### Finding Clusters in Networks

```python
def label_propagation(graph: Graph, max_iterations=100):
    """
    Label Propagation Algorithm for community detection.

    Each node adopts the most common label among its neighbors.

    Returns:
        communities: Dict {node: community_id}
    """
    # Initialize: Each node in its own community
    labels = {node: i for i, node in enumerate(graph.nodes)}

    nodes_list = list(graph.nodes)

    for iteration in range(max_iterations):
        changed = False

        # Process nodes in random order
        np.random.shuffle(nodes_list)

        for node in nodes_list:
            # Count neighbor labels
            neighbor_labels = []
            for neighbor, weight in graph.get_neighbors(node):
                neighbor_labels.append(labels[neighbor])

            if not neighbor_labels:
                continue

            # Most common label
            most_common = max(set(neighbor_labels), key=neighbor_labels.count)

            if labels[node] != most_common:
                labels[node] = most_common
                changed = True

        if not changed:
            print(f"Converged after {iteration + 1} iterations")
            break

    return labels


# Test community detection
print("\n" + "="*60)
print("Community Detection")
print("="*60)

# Create graph with clear communities
social_graph = Graph(directed=False)

# Community 1: A, B, C
social_graph.add_edge('A', 'B')
social_graph.add_edge('A', 'C')
social_graph.add_edge('B', 'C')

# Community 2: D, E, F
social_graph.add_edge('D', 'E')
social_graph.add_edge('D', 'F')
social_graph.add_edge('E', 'F')

# Bridge between communities
social_graph.add_edge('C', 'D')

communities = label_propagation(social_graph)

print("\nDetected communities:")
community_groups = defaultdict(list)
for node, community_id in communities.items():
    community_groups[community_id].append(node)

for community_id, members in community_groups.items():
    print(f"  Community {community_id}: {sorted(members)}")
```

### Modularity Calculation

```python
def calculate_modularity(graph: Graph, communities: Dict):
    """
    Calculate modularity of community partition.

    Modularity Q ∈ [-1, 1]
    Q > 0.3 indicates significant community structure.
    """
    # Total number of edges
    m = len(graph.edges)
    if not graph.directed:
        m = m / 2  # Each edge counted twice in undirected

    # Degrees
    degrees = {}
    for node in graph.nodes:
        degrees[node] = len(graph.get_neighbors(node))

    # Calculate modularity
    Q = 0

    for node_i in graph.nodes:
        for node_j in graph.nodes:
            if communities[node_i] == communities[node_j]:
                # Check if edge exists
                neighbors_i = [n for n, w in graph.get_neighbors(node_i)]
                A_ij = 1 if node_j in neighbors_i else 0

                # Expected number of edges
                expected = (degrees[node_i] * degrees[node_j]) / (2 * m)

                Q += (A_ij - expected)

    Q = Q / (2 * m)

    return Q


modularity = calculate_modularity(social_graph, communities)
print(f"\nModularity: {modularity:.4f}")
```

---

## 5. Real-World Applications

### Application 1: Social Network Analysis

```python
class SocialNetwork:
    """
    Social network with friend recommendations.

    Uses graph algorithms for recommendations.
    """
    def __init__(self):
        self.graph = Graph(directed=False)

    def add_friendship(self, user1, user2):
        """Add friendship between users."""
        self.graph.add_edge(user1, user2)

    def recommend_friends(self, user, top_k=5):
        """
        Recommend friends using:
        - Common friends (Jaccard similarity)
        - Personalized PageRank
        """
        # Get user's friends
        user_friends = set(n for n, w in self.graph.get_neighbors(user))

        # Calculate scores for potential friends
        scores = {}

        for node in self.graph.nodes:
            if node == user or node in user_friends:
                continue

            # Common friends
            node_friends = set(n for n, w in self.graph.get_neighbors(node))
            common = len(user_friends & node_friends)

            # Jaccard similarity
            union = len(user_friends | node_friends)
            jaccard = common / union if union > 0 else 0

            scores[node] = jaccard

        # Sort by score
        recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

    def find_influencers(self, top_k=5):
        """Find most influential users using PageRank."""
        ranks = pagerank(self.graph)
        influencers = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
        return influencers[:top_k]


# Build social network
print("\n" + "="*60)
print("Social Network Analysis")
print("="*60)

social_net = SocialNetwork()

# Add friendships
friendships = [
    ('Alice', 'Bob'), ('Alice', 'Charlie'), ('Alice', 'Diana'),
    ('Bob', 'Charlie'), ('Bob', 'Eve'),
    ('Charlie', 'Diana'), ('Charlie', 'Frank'),
    ('Diana', 'Frank'),
    ('Eve', 'Frank'), ('Eve', 'Grace'),
    ('Frank', 'Grace')
]

for user1, user2 in friendships:
    social_net.add_friendship(user1, user2)

# Recommend friends for Alice
recommendations = social_net.recommend_friends('Alice', top_k=3)
print("\nFriend recommendations for Alice:")
for user, score in recommendations:
    print(f"  {user}: {score:.3f}")

# Find influencers
influencers = social_net.find_influencers(top_k=3)
print("\nTop influencers:")
for user, rank in influencers:
    print(f"  {user}: {rank:.4f}")
```

### Application 2: Knowledge Graph Reasoning

```python
class KnowledgeGraph:
    """
    Knowledge graph with semantic reasoning.

    Nodes: Entities (people, places, concepts)
    Edges: Relations (is-a, works-at, located-in)
    """
    def __init__(self):
        self.graph = Graph(directed=True)
        self.relations = {}  # (from, to) -> relation_type

    def add_fact(self, subject, relation, object_):
        """Add a fact to knowledge graph."""
        self.graph.add_edge(subject, object_)
        self.relations[(subject, object_)] = relation

    def find_related_entities(self, entity, max_hops=2):
        """Find entities related to given entity."""
        related = {}

        # BFS with hop limit
        queue = deque([(entity, 0)])
        visited = {entity}

        while queue:
            current, hops = queue.popleft()

            if hops >= max_hops:
                continue

            for neighbor, weight in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    relation = self.relations.get((current, neighbor), 'unknown')
                    related[neighbor] = (relation, hops + 1)
                    queue.append((neighbor, hops + 1))

        return related

    def find_path_explanation(self, start, end):
        """Find and explain path between entities."""
        # BFS to find shortest path
        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            current, path = queue.popleft()

            if current == end:
                # Explain path
                explanation = []
                for i in range(len(path) - 1):
                    relation = self.relations.get((path[i], path[i+1]), 'unknown')
                    explanation.append((path[i], relation, path[i+1]))
                return explanation

            for neighbor, weight in self.graph.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found


# Build knowledge graph
print("\n" + "="*60)
print("Knowledge Graph Reasoning")
print("="*60)

kg = KnowledgeGraph()

# Add facts
facts = [
    ('Alice', 'works-at', 'Google'),
    ('Alice', 'lives-in', 'California'),
    ('Google', 'located-in', 'California'),
    ('Google', 'is-a', 'Tech-Company'),
    ('Bob', 'works-at', 'Google'),
    ('Bob', 'knows', 'Alice'),
    ('California', 'is-in', 'USA')
]

for subject, relation, object_ in facts:
    kg.add_fact(subject, relation, object_)

# Find related entities
related = kg.find_related_entities('Alice', max_hops=2)
print("\nEntities related to Alice:")
for entity, (relation, hops) in sorted(related.items()):
    print(f"  {entity} ({hops} hops)")

# Find connection between entities
path = kg.find_path_explanation('Bob', 'California')
if path:
    print("\nHow Bob connects to California:")
    for subject, relation, object_ in path:
        print(f"  {subject} --[{relation}]--> {object_}")
```

### Application 3: Recommendation System

```python
class GraphBasedRecommender:
    """
    Recommendation system using graph algorithms.

    Bipartite graph: Users <-> Items
    """
    def __init__(self):
        self.graph = Graph(directed=False)
        self.users = set()
        self.items = set()

    def add_interaction(self, user, item, weight=1.0):
        """Record user-item interaction."""
        self.graph.add_edge(user, item, weight)
        self.users.add(user)
        self.items.add(item)

    def recommend_items(self, user, top_k=5):
        """
        Recommend items using personalized PageRank.

        Bias towards user's current items.
        """
        # Get user's items
        user_items = set(n for n, w in self.graph.get_neighbors(user))

        # Personalized PageRank from user
        personalization = {user: 1.0}
        ranks = personalized_pagerank(self.graph, personalization, damping=0.85)

        # Filter to items user hasn't interacted with
        recommendations = []
        for item in self.items:
            if item not in user_items:
                recommendations.append((item, ranks[item]))

        # Sort by rank
        recommendations.sort(key=lambda x: x[1], reverse=True)

        return recommendations[:top_k]

    def find_similar_items(self, item, top_k=5):
        """Find similar items based on co-occurrence."""
        # Find users who liked this item
        item_users = set(n for n, w in self.graph.get_neighbors(item))

        # Calculate Jaccard similarity with other items
        similarities = {}

        for other_item in self.items:
            if other_item == item:
                continue

            other_users = set(n for n, w in self.graph.get_neighbors(other_item))

            # Jaccard similarity
            intersection = len(item_users & other_users)
            union = len(item_users | other_users)

            if union > 0:
                similarities[other_item] = intersection / union

        # Sort by similarity
        similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

        return similar[:top_k]


# Build recommendation system
print("\n" + "="*60)
print("Graph-Based Recommendation System")
print("="*60)

recommender = GraphBasedRecommender()

# User-item interactions
interactions = [
    ('User1', 'Movie_A'), ('User1', 'Movie_B'), ('User1', 'Movie_C'),
    ('User2', 'Movie_A'), ('User2', 'Movie_D'),
    ('User3', 'Movie_B'), ('User3', 'Movie_C'), ('User3', 'Movie_E'),
    ('User4', 'Movie_A'), ('User4', 'Movie_B'), ('User4', 'Movie_E'),
    ('User5', 'Movie_D'), ('User5', 'Movie_F')
]

for user, item in interactions:
    recommender.add_interaction(user, item)

# Recommend for User1
recommendations = recommender.recommend_items('User1', top_k=3)
print("\nRecommendations for User1:")
for item, score in recommendations:
    print(f"  {item}: {score:.4f}")

# Find similar items to Movie_A
similar = recommender.find_similar_items('Movie_A', top_k=3)
print("\nMovies similar to Movie_A:")
for item, similarity in similar:
    print(f"  {item}: {similarity:.3f}")
```

---

## 6. NetworkX Integration

### Using Industrial-Strength Graph Library

```python
import networkx as nx

# Create NetworkX graph
G = nx.DiGraph()

# Add edges from our web graph
for from_node, to_node, weight in web_edges:
    G.add_edge(from_node, to_node)

# PageRank with NetworkX
print("\n" + "="*60)
print("NetworkX PageRank")
print("="*60)

nx_pagerank = nx.pagerank(G, alpha=0.85)

print("PageRank scores (NetworkX):")
for page, rank in sorted(nx_pagerank.items(), key=lambda x: x[1], reverse=True):
    print(f"  {page:12s}: {rank:.4f}")

# Shortest paths
print("\nShortest paths (NetworkX):")
for target in ['Products', 'Contact']:
    try:
        path = nx.shortest_path(G, 'Home', target)
        print(f"  Home → {target}: {' → '.join(path)}")
    except nx.NetworkXNoPath:
        print(f"  Home → {target}: No path")

# Centrality measures
print("\nCentrality measures:")

betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)

print("\nBetweenness Centrality:")
for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {node:12s}: {score:.4f}")

print("\nCloseness Centrality:")
for node, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:3]:
    print(f"  {node:12s}: {score:.4f}")
```

---

## 7. Practice Exercises

### Exercise 1: Implement A* for Graphs

```python
"""
Implement A* search on general graphs.

Given weighted graph and heuristic function h(node, goal):
- Use A* to find shortest path
- Compare with Dijkstra

Test on road networks with geographic coordinates.
"""

# Your implementation here
```

### Exercise 2: Implement Louvain Algorithm

```python
"""
Implement Louvain algorithm for community detection.

Better than label propagation:
- Optimize modularity greedily
- Hierarchical communities

Compare with label propagation on real social network.
"""

# Your implementation here
```

### Exercise 3: Build Citation Network Analyzer

```python
"""
Build academic citation network analyzer.

Features:
- Import citation data (papers citing papers)
- Compute paper importance (PageRank)
- Find research communities (clustering)
- Recommend related papers
- Identify influential papers

Bonus: Visualize with NetworkX and matplotlib.
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Graph Representations** 🕸️
   - Adjacency list: O(V + E) space, fast neighbor lookup
   - Adjacency matrix: O(V²) space, fast edge queries
   - Choose based on graph density

2. **Shortest Path Algorithms** 🛤️
   - **Dijkstra**: O((V + E) log V), non-negative weights
   - **Bellman-Ford**: O(VE), handles negative weights
   - **Floyd-Warshall**: O(V³), all-pairs shortest paths

3. **PageRank** ⭐
   - Random walk interpretation
   - Damping factor balances following links vs teleporting
   - Power iteration for efficient computation
   - Personalized PageRank for recommendations

4. **Community Detection** 👥
   - Label propagation: Simple, fast
   - Modularity: Quality metric for partitions
   - Applications: Social networks, biology, marketing

5. **Graph Applications** 🚀
   - Social networks: Friend recommendations, influencer detection
   - Knowledge graphs: Semantic reasoning, question answering
   - Recommendation systems: Collaborative filtering
   - Web search: PageRank for ranking

### Algorithm Comparison

| Algorithm | Problem | Time Complexity | Use Case |
|-----------|---------|----------------|----------|
| BFS | Unweighted shortest path | O(V + E) | Social distance |
| Dijkstra | Weighted shortest path | O((V+E) log V) | GPS navigation |
| PageRank | Node importance | O(iterations × E) | Web search |
| Label Propagation | Community detection | O(iterations × E) | Clustering |

### Real-World Applications

✅ **Search Engines**: PageRank for ranking web pages
✅ **Social Networks**: Friend recommendations, influencer detection
✅ **Navigation**: GPS shortest path routing
✅ **Recommendation Systems**: Item and user similarity
✅ **Knowledge Graphs**: Question answering, semantic search
✅ **Fraud Detection**: Community detection for suspicious patterns

### What's Next?

In Lesson 5, we'll explore **Planning and Constraint Satisfaction**:
- STRIPS planning for robotics
- Constraint Satisfaction Problems (CSP)
- Backtracking and constraint propagation
- Applications: Scheduling, Sudoku, resource allocation

**Graphs are everywhere - master them to unlock powerful AI!** 🚀

---

## Additional Resources

### Papers
- Page et al. (1999): "The PageRank Citation Ranking: Bringing Order to the Web"
- Blondel et al. (2008): "Fast Unfolding of Communities in Large Networks" (Louvain)
- Raghavan et al. (2007): "Near Linear Time Algorithm to Detect Community Structures"

### Books
- **Networks** (Newman) - Comprehensive network science
- **Graph Algorithms** (Sedgewick & Wayne)
- **Mining of Massive Datasets** (Leskovec, Rajaraman, Ullman) - Chapter 5 on PageRank

### Libraries
- **NetworkX**: Python graph library
- **igraph**: Fast graph library (C/Python)
- **graph-tool**: High-performance graph analysis

### Visualizations
- **Gephi**: Interactive graph visualization
- **Cytoscape**: Biological network visualization
- **D3.js**: Web-based graph visualizations

---

**Next**: [Lesson 5 - Planning and Constraint Satisfaction](Lesson%205%20-%20Planning%20and%20Constraint%20Satisfaction.md)

Master automated planning and constraint solving! 🤖

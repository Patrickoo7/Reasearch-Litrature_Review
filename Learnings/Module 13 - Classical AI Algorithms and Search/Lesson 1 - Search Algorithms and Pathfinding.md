# Lesson 1: Search Algorithms & Pathfinding 🗺️

**Module 13: Classical AI Algorithms and Search | Lesson 1 of 5**

Master the fundamental search algorithms that power GPS navigation, game AI, and LLM text generation!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand uninformed search: BFS, DFS, Uniform Cost Search
2. ✅ Implement informed search: A* algorithm with heuristics
3. ✅ Design admissible and consistent heuristics
4. ✅ Apply IDA* for memory-efficient search
5. ✅ Understand beam search for LLM text generation
6. ✅ Build real-world pathfinding applications from scratch

---

## 1. Introduction to Search Problems

### What is a Search Problem?

A search problem consists of:
- **State space**: All possible configurations
- **Initial state**: Where we start
- **Goal state**: Where we want to be
- **Actions**: Ways to transition between states
- **Path cost**: Cost of taking actions

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict, Optional

class SearchProblem:
    """
    Abstract base class for search problems.

    Defines the interface that all search problems must implement.
    """
    def get_start_state(self):
        """Return the initial state."""
        raise NotImplementedError

    def is_goal_state(self, state):
        """Check if state is a goal state."""
        raise NotImplementedError

    def get_successors(self, state):
        """
        Get successor states.

        Returns: List of (next_state, action, cost) tuples
        """
        raise NotImplementedError


class GridWorld(SearchProblem):
    """
    2D grid navigation problem.

    Find path from start to goal, avoiding obstacles.
    """
    def __init__(self, grid, start, goal):
        """
        Args:
            grid: 2D numpy array (0=free, 1=obstacle)
            start: (row, col) tuple
            goal: (row, col) tuple
        """
        self.grid = np.array(grid)
        self.start = start
        self.goal = goal
        self.rows, self.cols = grid.shape

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state == self.goal

    def get_successors(self, state):
        """Get valid neighboring cells (up, down, left, right)."""
        row, col = state
        successors = []

        # Four directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        action_names = ['up', 'down', 'left', 'right']

        for (dr, dc), action in zip(directions, action_names):
            new_row, new_col = row + dr, col + dc

            # Check if valid position
            if (0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row, new_col] == 0):

                next_state = (new_row, new_col)
                cost = 1  # Uniform cost for now
                successors.append((next_state, action, cost))

        return successors


# Example grid: 0 = free, 1 = obstacle
grid = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0]
])

problem = GridWorld(grid, start=(0, 0), goal=(4, 5))
print("Start:", problem.get_start_state())
print("Goal:", problem.goal)
print("\nSuccessors from start:")
for state, action, cost in problem.get_successors((0, 0)):
    print(f"  {action} -> {state} (cost={cost})")
```

---

## 2. Breadth-First Search (BFS)

### The Algorithm

BFS explores all neighbors at the current depth before moving deeper. Guarantees shortest path for unweighted graphs.

```python
def breadth_first_search(problem: SearchProblem):
    """
    Breadth-First Search implementation.

    Returns:
        path: List of states from start to goal
        stats: Dictionary with search statistics
    """
    start = problem.get_start_state()

    # Check if start is goal
    if problem.is_goal_state(start):
        return [start], {'nodes_expanded': 0, 'nodes_generated': 1}

    # Initialize frontier (FIFO queue) and explored set
    frontier = deque([(start, [])])  # (state, path_to_state)
    explored = {start}

    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        state, path = frontier.popleft()
        nodes_expanded += 1

        # Explore successors
        for next_state, action, cost in problem.get_successors(state):
            if next_state not in explored:
                nodes_generated += 1
                explored.add(next_state)

                new_path = path + [action]

                # Check if goal
                if problem.is_goal_state(next_state):
                    full_path = [start] + [s for s, _, _ in
                                          reconstruct_path(problem, start, new_path)]
                    stats = {
                        'nodes_expanded': nodes_expanded,
                        'nodes_generated': nodes_generated,
                        'path_length': len(full_path)
                    }
                    return full_path, stats

                frontier.append((next_state, new_path))

    return None, {'nodes_expanded': nodes_expanded,
                 'nodes_generated': nodes_generated}


def reconstruct_path(problem, start, actions):
    """Helper to reconstruct path from action sequence."""
    path = []
    state = start

    for action in actions:
        for next_state, next_action, _ in problem.get_successors(state):
            if next_action == action:
                path.append((next_state, next_action, _))
                state = next_state
                break

    return path


# Run BFS
grid = np.array([
    [0, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 1],
    [0, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0],
    [1, 1, 0, 0, 0, 0]
])

problem = GridWorld(grid, start=(0, 0), goal=(4, 5))
path, stats = breadth_first_search(problem)

print("BFS Results:")
print(f"Path found: {path}")
print(f"Path length: {stats['path_length']}")
print(f"Nodes expanded: {stats['nodes_expanded']}")
print(f"Nodes generated: {stats['nodes_generated']}")
```

### Visualizing BFS

```python
def visualize_search(grid, path, title="Search Path"):
    """Visualize grid and path."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create visualization grid
    vis_grid = grid.copy().astype(float)

    # Mark path
    if path:
        for i, (row, col) in enumerate(path):
            vis_grid[row, col] = 0.5  # Path cells

        # Mark start and goal
        vis_grid[path[0]] = 0.3   # Start (green)
        vis_grid[path[-1]] = 0.7  # Goal (red)

    # Display
    ax.imshow(vis_grid, cmap='RdYlGn_r', vmin=0, vmax=1)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Add cell coordinates
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                ax.text(j, i, 'X', ha='center', va='center',
                       fontsize=12, color='white')

    plt.tight_layout()
    plt.show()


visualize_search(grid, path, "BFS Path")
```

**BFS Properties:**
- ✅ Complete: Always finds solution if one exists
- ✅ Optimal: Finds shortest path (unweighted graphs)
- ❌ Time: O(b^d) where b=branching factor, d=depth
- ❌ Space: O(b^d) - stores all nodes at current level

---

## 3. Depth-First Search (DFS)

### The Algorithm

DFS explores as deep as possible before backtracking. Memory efficient but not optimal.

```python
def depth_first_search(problem: SearchProblem, max_depth=100):
    """
    Depth-First Search implementation.

    Args:
        problem: SearchProblem instance
        max_depth: Maximum depth to prevent infinite loops

    Returns:
        path: List of states from start to goal
        stats: Dictionary with search statistics
    """
    start = problem.get_start_state()

    if problem.is_goal_state(start):
        return [start], {'nodes_expanded': 0, 'nodes_generated': 1}

    # Initialize frontier (LIFO stack) and explored set
    frontier = [(start, [], 0)]  # (state, path, depth)
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        state, path, depth = frontier.pop()

        if state in explored or depth > max_depth:
            continue

        explored.add(state)
        nodes_expanded += 1

        # Explore successors
        for next_state, action, cost in problem.get_successors(state):
            if next_state not in explored:
                nodes_generated += 1
                new_path = path + [action]

                # Check if goal
                if problem.is_goal_state(next_state):
                    full_path = [start] + [s for s, _, _ in
                                          reconstruct_path(problem, start, new_path)]
                    stats = {
                        'nodes_expanded': nodes_expanded,
                        'nodes_generated': nodes_generated,
                        'path_length': len(full_path)
                    }
                    return full_path, stats

                frontier.append((next_state, new_path, depth + 1))

    return None, {'nodes_expanded': nodes_expanded,
                 'nodes_generated': nodes_generated}


# Run DFS
path_dfs, stats_dfs = depth_first_search(problem)

print("\nDFS Results:")
print(f"Path length: {stats_dfs['path_length']}")
print(f"Nodes expanded: {stats_dfs['nodes_expanded']}")
print(f"Nodes generated: {stats_dfs['nodes_generated']}")

visualize_search(grid, path_dfs, "DFS Path")
```

**DFS Properties:**
- ✅ Complete: Yes (with depth limit)
- ❌ Optimal: No - may find longer paths
- ❌ Time: O(b^m) where m=maximum depth
- ✅ Space: O(bm) - only stores current path

---

## 4. Uniform Cost Search (UCS)

### Dijkstra's Algorithm for General Graphs

UCS expands nodes in order of path cost. Optimal for weighted graphs.

```python
class WeightedGridWorld(GridWorld):
    """Grid with variable movement costs."""

    def __init__(self, grid, start, goal, cost_grid=None):
        super().__init__(grid, start, goal)

        # Cost to enter each cell (default 1)
        if cost_grid is None:
            self.cost_grid = np.ones_like(grid, dtype=float)
        else:
            self.cost_grid = cost_grid

    def get_successors(self, state):
        """Get successors with variable costs."""
        row, col = state
        successors = []

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        action_names = ['up', 'down', 'left', 'right']

        for (dr, dc), action in zip(directions, action_names):
            new_row, new_col = row + dr, col + dc

            if (0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row, new_col] == 0):

                next_state = (new_row, new_col)
                cost = self.cost_grid[new_row, new_col]
                successors.append((next_state, action, cost))

        return successors


def uniform_cost_search(problem: SearchProblem):
    """
    Uniform Cost Search (Dijkstra's algorithm).

    Expands nodes in order of cumulative path cost.
    """
    start = problem.get_start_state()

    if problem.is_goal_state(start):
        return [start], 0, {'nodes_expanded': 0}

    # Priority queue: (cumulative_cost, state, path)
    frontier = [(0, start, [])]
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        cum_cost, state, path = heapq.heappop(frontier)

        if state in explored:
            continue

        explored.add(state)
        nodes_expanded += 1

        # Check if goal
        if problem.is_goal_state(state):
            full_path = [start] + [s for s, _, _ in
                                  reconstruct_path(problem, start, path)]
            stats = {
                'nodes_expanded': nodes_expanded,
                'nodes_generated': nodes_generated,
                'path_length': len(full_path),
                'path_cost': cum_cost
            }
            return full_path, cum_cost, stats

        # Explore successors
        for next_state, action, cost in problem.get_successors(state):
            if next_state not in explored:
                nodes_generated += 1
                new_path = path + [action]
                new_cost = cum_cost + cost
                heapq.heappush(frontier, (new_cost, next_state, new_path))

    return None, float('inf'), {'nodes_expanded': nodes_expanded}


# Create weighted grid (terrain with different costs)
cost_grid = np.array([
    [1, 1, 1, 0, 3, 3],
    [1, 0, 2, 0, 3, 0],
    [1, 0, 2, 2, 2, 1],
    [1, 1, 1, 0, 0, 1],
    [0, 0, 1, 1, 1, 1]
])

problem_weighted = WeightedGridWorld(grid, (0, 0), (4, 5), cost_grid)
path_ucs, cost_ucs, stats_ucs = uniform_cost_search(problem_weighted)

print("\nUniform Cost Search Results:")
print(f"Path cost: {cost_ucs:.2f}")
print(f"Path length: {stats_ucs['path_length']}")
print(f"Nodes expanded: {stats_ucs['nodes_expanded']}")
```

**UCS Properties:**
- ✅ Complete: Yes (if all costs > 0)
- ✅ Optimal: Yes - finds lowest cost path
- ❌ Time: O(b^(C*/ε)) where C*=optimal cost, ε=min cost
- ❌ Space: O(b^(C*/ε))

---

## 5. A* Search Algorithm

### Informed Search with Heuristics

A* combines path cost (g) with heuristic estimate to goal (h): f(n) = g(n) + h(n)

```python
def manhattan_distance(state1, state2):
    """Manhattan distance heuristic for grid."""
    return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])


def euclidean_distance(state1, state2):
    """Euclidean distance heuristic."""
    return np.sqrt((state1[0] - state2[0])**2 +
                   (state1[1] - state2[1])**2)


def a_star_search(problem: SearchProblem, heuristic_fn):
    """
    A* Search implementation.

    f(n) = g(n) + h(n)
    - g(n): Cost from start to n
    - h(n): Estimated cost from n to goal

    Args:
        problem: SearchProblem instance
        heuristic_fn: Function(state, goal) -> estimated_cost
    """
    start = problem.get_start_state()
    goal = problem.goal

    if problem.is_goal_state(start):
        return [start], 0, {'nodes_expanded': 0}

    # Priority queue: (f_cost, g_cost, state, path)
    h_start = heuristic_fn(start, goal)
    frontier = [(h_start, 0, start, [])]

    # Track best g-cost to each state
    g_costs = {start: 0}
    explored = set()

    nodes_expanded = 0
    nodes_generated = 1

    while frontier:
        f_cost, g_cost, state, path = heapq.heappop(frontier)

        if state in explored:
            continue

        explored.add(state)
        nodes_expanded += 1

        # Check if goal
        if problem.is_goal_state(state):
            full_path = [start] + [s for s, _, _ in
                                  reconstruct_path(problem, start, path)]
            stats = {
                'nodes_expanded': nodes_expanded,
                'nodes_generated': nodes_generated,
                'path_length': len(full_path),
                'path_cost': g_cost
            }
            return full_path, g_cost, stats

        # Explore successors
        for next_state, action, cost in problem.get_successors(state):
            new_g = g_cost + cost

            # Only add if we found a better path
            if next_state not in g_costs or new_g < g_costs[next_state]:
                g_costs[next_state] = new_g
                h_cost = heuristic_fn(next_state, goal)
                f_cost = new_g + h_cost

                new_path = path + [action]
                heapq.heappush(frontier, (f_cost, new_g, next_state, new_path))
                nodes_generated += 1

    return None, float('inf'), {'nodes_expanded': nodes_expanded}


# Compare UCS vs A* with different heuristics
print("\n" + "="*60)
print("COMPARING SEARCH ALGORITHMS")
print("="*60)

# UCS (A* with h=0)
path_ucs, cost_ucs, stats_ucs = a_star_search(
    problem_weighted,
    lambda s, g: 0  # Zero heuristic = UCS
)
print("\nUniform Cost Search (h=0):")
print(f"  Path cost: {cost_ucs:.2f}")
print(f"  Nodes expanded: {stats_ucs['nodes_expanded']}")

# A* with Manhattan distance
path_manhattan, cost_manhattan, stats_manhattan = a_star_search(
    problem_weighted,
    manhattan_distance
)
print("\nA* with Manhattan Distance:")
print(f"  Path cost: {cost_manhattan:.2f}")
print(f"  Nodes expanded: {stats_manhattan['nodes_expanded']}")
print(f"  Speedup: {stats_ucs['nodes_expanded'] / stats_manhattan['nodes_expanded']:.2f}x")

# A* with Euclidean distance
path_euclidean, cost_euclidean, stats_euclidean = a_star_search(
    problem_weighted,
    euclidean_distance
)
print("\nA* with Euclidean Distance:")
print(f"  Path cost: {cost_euclidean:.2f}")
print(f"  Nodes expanded: {stats_euclidean['nodes_expanded']}")
print(f"  Speedup: {stats_ucs['nodes_expanded'] / stats_euclidean['nodes_expanded']:.2f}x")
```

**A* Properties:**
- ✅ Complete: Yes (with admissible heuristic)
- ✅ Optimal: Yes (with admissible heuristic)
- ✅ Efficiency: Expands fewer nodes than UCS
- 📊 Performance depends on heuristic quality

---

## 6. Heuristic Design

### Admissible Heuristics

A heuristic h(n) is **admissible** if it never overestimates the true cost to goal.

```python
def test_heuristic_admissibility(problem, heuristic_fn, num_samples=100):
    """
    Test if heuristic is admissible.

    For random states, compute true cost and heuristic estimate.
    """
    goal = problem.goal
    violations = []

    # Sample random states
    for _ in range(num_samples):
        row = np.random.randint(0, problem.rows)
        col = np.random.randint(0, problem.cols)

        if problem.grid[row, col] == 1:
            continue  # Skip obstacles

        state = (row, col)

        # Compute true cost using UCS
        temp_problem = WeightedGridWorld(
            problem.grid, state, goal, problem.cost_grid
        )
        _, true_cost, _ = uniform_cost_search(temp_problem)

        # Compute heuristic estimate
        h_cost = heuristic_fn(state, goal)

        # Check admissibility: h(n) <= true_cost
        if h_cost > true_cost + 1e-6:  # Small epsilon for float errors
            violations.append({
                'state': state,
                'h_cost': h_cost,
                'true_cost': true_cost,
                'overestimate': h_cost - true_cost
            })

    if violations:
        print(f"⚠️  Heuristic is NOT admissible!")
        print(f"   Found {len(violations)} violations")
        print(f"   Example: h({violations[0]['state']}) = {violations[0]['h_cost']:.2f}")
        print(f"            true_cost = {violations[0]['true_cost']:.2f}")
    else:
        print(f"✅ Heuristic is admissible (tested on {num_samples} samples)")

    return len(violations) == 0


# Test Manhattan distance
print("\nTesting Manhattan Distance:")
is_admissible = test_heuristic_admissibility(
    problem_weighted,
    manhattan_distance,
    num_samples=50
)

# Create non-admissible heuristic for comparison
def bad_heuristic(state, goal):
    """Overestimating heuristic (not admissible)."""
    return 10 * manhattan_distance(state, goal)

print("\nTesting Overestimating Heuristic:")
is_admissible = test_heuristic_admissibility(
    problem_weighted,
    bad_heuristic,
    num_samples=50
)
```

### Consistent (Monotonic) Heuristics

A heuristic is **consistent** if: h(n) ≤ cost(n, n') + h(n') for all successors n'

```python
def test_heuristic_consistency(problem, heuristic_fn, num_samples=100):
    """
    Test if heuristic is consistent.

    For random states and successors, check triangle inequality.
    """
    goal = problem.goal
    violations = []

    for _ in range(num_samples):
        row = np.random.randint(0, problem.rows)
        col = np.random.randint(0, problem.cols)

        if problem.grid[row, col] == 1:
            continue

        state = (row, col)
        h_state = heuristic_fn(state, goal)

        # Check all successors
        for next_state, action, cost in problem.get_successors(state):
            h_next = heuristic_fn(next_state, goal)

            # Consistency: h(n) <= cost(n,n') + h(n')
            if h_state > cost + h_next + 1e-6:
                violations.append({
                    'state': state,
                    'next_state': next_state,
                    'h_state': h_state,
                    'cost': cost,
                    'h_next': h_next,
                    'violation': h_state - (cost + h_next)
                })

    if violations:
        print(f"⚠️  Heuristic is NOT consistent!")
        print(f"   Found {len(violations)} violations")
    else:
        print(f"✅ Heuristic is consistent")

    return len(violations) == 0


print("\nTesting Consistency:")
test_heuristic_consistency(problem_weighted, manhattan_distance, num_samples=50)
```

### Designing Better Heuristics

```python
def relaxed_problem_heuristic(state, goal, grid):
    """
    Heuristic from relaxed problem.

    Relax constraint: Can move through obstacles.
    Optimal solution to relaxed problem = admissible heuristic.
    """
    # Manhattan distance (assuming no obstacles)
    return manhattan_distance(state, goal)


def pattern_database_heuristic(state, goal, precomputed_costs):
    """
    Pattern database: Precompute costs for subproblems.

    This is a simplified example - real PDBs are more complex.
    """
    # Look up precomputed cost
    if state in precomputed_costs:
        return precomputed_costs[state]
    return manhattan_distance(state, goal)


def max_heuristic(state, goal, heuristics):
    """
    Combine multiple admissible heuristics.

    max(h1, h2, ..., hn) is still admissible!
    """
    return max(h(state, goal) for h in heuristics)


# Example: Combine Manhattan and Euclidean
def combined_heuristic(state, goal):
    return max(
        manhattan_distance(state, goal),
        euclidean_distance(state, goal)
    )


path_combined, cost_combined, stats_combined = a_star_search(
    problem_weighted,
    combined_heuristic
)
print("\nA* with Combined Heuristic:")
print(f"  Nodes expanded: {stats_combined['nodes_expanded']}")
```

---

## 7. Iterative Deepening A* (IDA*)

### Memory-Efficient A*

IDA* uses iterative deepening with f-cost cutoff. O(d) space instead of O(b^d)!

```python
def ida_star_search(problem: SearchProblem, heuristic_fn):
    """
    Iterative Deepening A* implementation.

    Memory-efficient: Only O(d) space instead of O(b^d).
    """
    start = problem.get_start_state()
    goal = problem.goal

    if problem.is_goal_state(start):
        return [start], 0, {'iterations': 0, 'nodes_expanded': 0}

    # Initial f-cost threshold
    threshold = heuristic_fn(start, goal)
    path = [start]

    iterations = 0
    total_nodes = 0

    while True:
        iterations += 1
        result, nodes = _ida_star_dfs(
            problem, heuristic_fn, path, 0, threshold
        )
        total_nodes += nodes

        if result == 'FOUND':
            stats = {
                'iterations': iterations,
                'nodes_expanded': total_nodes,
                'path_length': len(path)
            }
            return path, sum(problem.cost_grid[s] for s in path[1:]), stats

        if result == float('inf'):
            return None, float('inf'), {'iterations': iterations}

        # Increase threshold for next iteration
        threshold = result


def _ida_star_dfs(problem, heuristic_fn, path, g_cost, threshold):
    """
    DFS with f-cost cutoff.

    Returns:
        - 'FOUND' if goal found
        - float('inf') if no solution
        - next_threshold if exceeded current threshold
    """
    state = path[-1]
    goal = problem.goal

    f_cost = g_cost + heuristic_fn(state, goal)

    if f_cost > threshold:
        return f_cost, 1  # Exceeded threshold

    if problem.is_goal_state(state):
        return 'FOUND', 1

    min_threshold = float('inf')
    nodes_expanded = 1

    for next_state, action, cost in problem.get_successors(state):
        if next_state not in path:  # Avoid cycles
            path.append(next_state)
            result, nodes = _ida_star_dfs(
                problem, heuristic_fn, path,
                g_cost + cost, threshold
            )
            nodes_expanded += nodes

            if result == 'FOUND':
                return 'FOUND', nodes_expanded

            if isinstance(result, (int, float)):
                min_threshold = min(min_threshold, result)

            path.pop()

    return min_threshold, nodes_expanded


# Compare A* vs IDA*
print("\n" + "="*60)
print("A* vs IDA* Comparison")
print("="*60)

path_astar, cost_astar, stats_astar = a_star_search(
    problem_weighted, manhattan_distance
)
print("\nA* Search:")
print(f"  Path cost: {cost_astar:.2f}")
print(f"  Nodes expanded: {stats_astar['nodes_expanded']}")

path_ida, cost_ida, stats_ida = ida_star_search(
    problem_weighted, manhattan_distance
)
print("\nIDA* Search:")
print(f"  Path cost: {cost_ida:.2f}")
print(f"  Nodes expanded: {stats_ida['nodes_expanded']}")
print(f"  Iterations: {stats_ida['iterations']}")
```

**IDA* Properties:**
- ✅ Optimal: Yes (with admissible heuristic)
- ✅ Space: O(d) - linear in depth!
- ❌ Time: May re-expand nodes multiple times
- 📊 Best when memory is limited

---

## 8. Beam Search

### Approximate Search for Large Spaces

Beam search keeps only top-k candidates at each level. Used in LLM text generation!

```python
def beam_search(problem: SearchProblem, heuristic_fn, beam_width=3):
    """
    Beam Search: Keep only top-k best candidates.

    Not optimal, but efficient for large search spaces.
    Used in language model text generation.

    Args:
        beam_width: Number of candidates to keep at each step
    """
    start = problem.get_start_state()
    goal = problem.goal

    if problem.is_goal_state(start):
        return [start], 0

    # Beam: List of (f_cost, g_cost, state, path)
    beam = [(heuristic_fn(start, goal), 0, start, [])]

    nodes_expanded = 0
    max_iterations = 1000

    for iteration in range(max_iterations):
        candidates = []

        # Expand all states in current beam
        for f_cost, g_cost, state, path in beam:
            nodes_expanded += 1

            # Check if goal
            if problem.is_goal_state(state):
                full_path = [start] + [s for s, _, _ in
                                      reconstruct_path(problem, start, path)]
                stats = {
                    'nodes_expanded': nodes_expanded,
                    'beam_width': beam_width,
                    'path_length': len(full_path)
                }
                return full_path, g_cost, stats

            # Generate successors
            for next_state, action, cost in problem.get_successors(state):
                new_g = g_cost + cost
                new_f = new_g + heuristic_fn(next_state, goal)
                new_path = path + [action]

                candidates.append((new_f, new_g, next_state, new_path))

        if not candidates:
            break

        # Keep only top beam_width candidates
        candidates.sort(key=lambda x: x[0])  # Sort by f-cost
        beam = candidates[:beam_width]

    return None, float('inf'), {'nodes_expanded': nodes_expanded}


# Compare different beam widths
print("\n" + "="*60)
print("Beam Search with Different Widths")
print("="*60)

for beam_width in [1, 3, 5, 10]:
    path_beam, cost_beam, stats_beam = beam_search(
        problem_weighted, manhattan_distance, beam_width
    )
    print(f"\nBeam Width = {beam_width}:")
    if path_beam:
        print(f"  Path cost: {cost_beam:.2f}")
        print(f"  Path length: {stats_beam['path_length']}")
        print(f"  Nodes expanded: {stats_beam['nodes_expanded']}")
    else:
        print(f"  No path found")
```

### Beam Search for Text Generation

```python
class TextGenerationProblem:
    """
    Simplified text generation as search problem.

    State: Current token sequence
    Action: Add next token
    Cost: Negative log probability
    """
    def __init__(self, vocab, max_length=10):
        self.vocab = vocab
        self.max_length = max_length
        self.start_token = '<START>'
        self.end_token = '<END>'

    def get_start_state(self):
        return (self.start_token,)

    def is_goal_state(self, state):
        return state[-1] == self.end_token or len(state) >= self.max_length

    def get_successors(self, state):
        """Generate next possible tokens."""
        if self.is_goal_state(state):
            return []

        successors = []

        # Simplified: Random probabilities for demo
        for token in self.vocab:
            if token != self.start_token:
                # Simulate language model probabilities
                prob = np.random.rand() * 0.5 + 0.1
                cost = -np.log(prob)  # Negative log probability
                next_state = state + (token,)
                successors.append((next_state, token, cost))

        return successors


# Simulate beam search for text generation
vocab = ['the', 'cat', 'sat', 'on', 'mat', '<END>']
text_problem = TextGenerationProblem(vocab, max_length=6)

def text_heuristic(state, goal):
    """Estimate cost to completion."""
    # Simplified: Encourage shorter sequences
    return 0  # No heuristic for now


path_text, cost_text, stats_text = beam_search(
    text_problem, text_heuristic, beam_width=3
)

if path_text:
    print("\nGenerated Text (Beam Search):")
    print("  ", ' '.join(path_text[1:]))  # Skip START token
    print(f"  Cost: {cost_text:.2f}")
```

---

## 9. Real-World Applications

### Application 1: GPS Navigation

```python
class RoadNetwork:
    """
    Road network for GPS navigation.

    Nodes: Intersections/locations
    Edges: Roads with distances
    """
    def __init__(self):
        # Graph: adjacency list {node: [(neighbor, distance), ...]}
        self.graph = {}
        self.locations = {}  # {node: (lat, lon)}

    def add_road(self, from_node, to_node, distance):
        """Add bidirectional road."""
        if from_node not in self.graph:
            self.graph[from_node] = []
        if to_node not in self.graph:
            self.graph[to_node] = []

        self.graph[from_node].append((to_node, distance))
        self.graph[to_node].append((from_node, distance))

    def set_location(self, node, lat, lon):
        """Set GPS coordinates for node."""
        self.locations[node] = (lat, lon)


class NavigationProblem(SearchProblem):
    """GPS navigation problem."""

    def __init__(self, network, start, goal):
        self.network = network
        self.start = start
        self.goal = goal

    def get_start_state(self):
        return self.start

    def is_goal_state(self, state):
        return state == self.goal

    def get_successors(self, state):
        """Get neighboring locations."""
        successors = []
        for neighbor, distance in self.network.graph.get(state, []):
            successors.append((neighbor, f'to_{neighbor}', distance))
        return successors


def haversine_distance(loc1, loc2):
    """
    Great-circle distance between GPS coordinates.

    Admissible heuristic for road navigation.
    """
    lat1, lon1 = loc1
    lat2, lon2 = loc2

    # Simplified Euclidean approximation
    return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # km


# Build road network
network = RoadNetwork()

# Add roads (simplified city)
roads = [
    ('A', 'B', 5), ('A', 'C', 3), ('B', 'D', 4),
    ('C', 'D', 2), ('C', 'E', 6), ('D', 'E', 3),
    ('D', 'F', 7), ('E', 'F', 2)
]

for from_node, to_node, dist in roads:
    network.add_road(from_node, to_node, dist)

# Set GPS coordinates
coords = {
    'A': (0, 0), 'B': (1, 4), 'C': (2, 1),
    'D': (3, 3), 'E': (5, 2), 'F': (6, 5)
}
for node, (lat, lon) in coords.items():
    network.set_location(node, lat, lon)

# Find route
nav_problem = NavigationProblem(network, 'A', 'F')

def gps_heuristic(state, goal):
    loc1 = nav_problem.network.locations[state]
    loc2 = nav_problem.network.locations[goal]
    return haversine_distance(loc1, loc2)

route, distance, stats = a_star_search(nav_problem, gps_heuristic)

print("\n" + "="*60)
print("GPS Navigation: Route from A to F")
print("="*60)
print(f"Route: {' -> '.join(route)}")
print(f"Total distance: {distance:.2f} km")
print(f"Nodes explored: {stats['nodes_expanded']}")
```

### Application 2: 8-Puzzle Solver

```python
class PuzzleState:
    """State for sliding puzzle."""

    def __init__(self, board):
        self.board = tuple(tuple(row) for row in board)
        self.size = len(board)

        # Find blank position
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    self.blank = (i, j)

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(self.board)

    def __repr__(self):
        return '\n'.join(' '.join(str(x) for x in row)
                        for row in self.board)


class PuzzleProblem(SearchProblem):
    """8-puzzle problem."""

    def __init__(self, initial, goal):
        self.initial = PuzzleState(initial)
        self.goal_state = PuzzleState(goal)

    def get_start_state(self):
        return self.initial

    def is_goal_state(self, state):
        return state == self.goal_state

    def get_successors(self, state):
        """Move blank tile in 4 directions."""
        successors = []
        blank_r, blank_c = state.blank

        moves = [
            (-1, 0, 'up'), (1, 0, 'down'),
            (0, -1, 'left'), (0, 1, 'right')
        ]

        for dr, dc, action in moves:
            new_r, new_c = blank_r + dr, blank_c + dc

            if 0 <= new_r < state.size and 0 <= new_c < state.size:
                # Swap blank with adjacent tile
                new_board = [list(row) for row in state.board]
                new_board[blank_r][blank_c] = new_board[new_r][new_c]
                new_board[new_r][new_c] = 0

                next_state = PuzzleState(new_board)
                successors.append((next_state, action, 1))

        return successors


def manhattan_puzzle_heuristic(state, goal):
    """
    Manhattan distance for puzzle tiles.

    Sum of distances each tile is from its goal position.
    """
    distance = 0

    # Build goal position map
    goal_pos = {}
    for i in range(goal.size):
        for j in range(goal.size):
            tile = goal.board[i][j]
            if tile != 0:
                goal_pos[tile] = (i, j)

    # Calculate Manhattan distance for each tile
    for i in range(state.size):
        for j in range(state.size):
            tile = state.board[i][j]
            if tile != 0:
                goal_i, goal_j = goal_pos[tile]
                distance += abs(i - goal_i) + abs(j - goal_j)

    return distance


# Solve 8-puzzle
initial = [
    [1, 2, 3],
    [4, 0, 5],
    [7, 8, 6]
]

goal = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 0]
]

puzzle = PuzzleProblem(initial, goal)

print("\n" + "="*60)
print("8-Puzzle Solver")
print("="*60)
print("\nInitial State:")
print(puzzle.initial)
print("\nGoal State:")
print(puzzle.goal_state)

def puzzle_heuristic_wrapper(state, goal):
    return manhattan_puzzle_heuristic(state, puzzle.goal_state)

solution, cost, stats = a_star_search(puzzle, puzzle_heuristic_wrapper)

print(f"\nSolution found!")
print(f"Moves: {cost}")
print(f"Nodes expanded: {stats['nodes_expanded']}")
```

---

## 10. Practice Exercises

### Exercise 1: Bidirectional Search

```python
"""
Implement bidirectional search.

Search from both start and goal simultaneously.
Stop when frontiers meet.

Advantages:
- Can be much faster: O(b^(d/2) + b^(d/2)) vs O(b^d)
- Useful when goal is known
"""

# Your implementation here
```

### Exercise 2: Jump Point Search

```python
"""
Implement Jump Point Search (JPS) for grid pathfinding.

JPS is an optimization of A* for uniform-cost grids.
It "jumps" over symmetrical paths to reduce nodes explored.

Can be 10x faster than A* on large grids!
"""

# Your implementation here
```

### Exercise 3: Anytime Search

```python
"""
Implement Anytime Repairing A* (ARA*).

ARA* finds suboptimal solution quickly, then improves it over time.

Useful when you need:
- Fast initial solution
- Better solution if time permits
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Search Problem Formulation** 🎯
   - State space, initial state, goal state
   - Actions and transition function
   - Path cost function
   - Foundation for all search algorithms

2. **Uninformed Search** 🔍
   - **BFS**: Optimal for unweighted, O(b^d) space
   - **DFS**: Memory efficient, not optimal
   - **UCS**: Optimal for weighted graphs

3. **Informed Search (A*)** ⭐
   - Uses heuristic to guide search
   - f(n) = g(n) + h(n)
   - Optimal with admissible heuristic
   - Dramatically faster than uninformed search

4. **Heuristic Design** 🎨
   - **Admissible**: Never overestimate (h ≤ h*)
   - **Consistent**: Satisfies triangle inequality
   - Better heuristics = fewer nodes expanded
   - Combine multiple heuristics with max()

5. **Memory-Efficient Search** 💾
   - **IDA***: Iterative deepening with f-cutoff
   - O(d) space instead of O(b^d)
   - Trade time for space

6. **Approximate Search** 📊
   - **Beam Search**: Keep top-k candidates
   - Used in LLM text generation
   - Not optimal, but very efficient

### Real-World Applications

✅ **GPS Navigation**: A* with haversine heuristic
✅ **Game AI**: Pathfinding in video games
✅ **Puzzle Solving**: 8-puzzle, Rubik's cube
✅ **Robot Motion Planning**: Navigate obstacles
✅ **Text Generation**: Beam search for LLMs

### Algorithm Selection Guide

| Scenario | Best Algorithm |
|----------|---------------|
| Unweighted graph | BFS |
| Weighted graph | A* or UCS |
| Known heuristic | A* |
| Limited memory | IDA* or DFS |
| Need fast approximate | Beam Search |
| Large branching factor | IDA* or Beam |

### What's Next?

In Lesson 2, we'll explore **Game Playing AI**:
- Minimax algorithm with alpha-beta pruning
- Monte Carlo Tree Search (MCTS)
- How AlphaGo defeated world champions
- Applications to chess, Go, and more

**Search algorithms are everywhere in AI - master them!** 🚀

---

## Additional Resources

### Papers
- Hart, Nilsson & Raphael (1968): "A Formal Basis for the Heuristic Determination of Minimum Cost Paths"
- Korf (1985): "Depth-First Iterative-Deepening: An Optimal Admissible Tree Search"
- Harabor & Grastien (2011): "Online Graph Pruning for Pathfinding on Grid Maps"

### Books
- **Artificial Intelligence: A Modern Approach** (Russell & Norvig) - Chapters 3-4
- **Heuristic Search** (Edelkamp & Schrödl)

### Libraries
- **NetworkX**: Graph algorithms in Python
- **pathfinding**: A* and variants for grids
- **heapq**: Priority queue for efficient search

### Visualizations
- **PathFinding.js**: Interactive A* visualization
- **Red Blob Games**: Excellent pathfinding tutorials

---

**Next**: [Lesson 2 - Game Playing AI and MCTS](Lesson%202%20-%20Game%20Playing%20AI%20and%20MCTS.md)

Let's build AI that plays games! 🎮

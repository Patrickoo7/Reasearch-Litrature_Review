# Lesson 3: Optimization Algorithms 🧬

**Module 13: Classical AI Algorithms and Search | Lesson 3 of 5**

Master evolutionary and swarm-based optimization algorithms that solve problems nature's way!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Implement genetic algorithms from scratch
2. ✅ Apply simulated annealing to optimization problems
3. ✅ Understand particle swarm optimization (PSO)
4. ✅ Explore ant colony optimization for routing
5. ✅ Use these algorithms for hyperparameter tuning
6. ✅ Solve real-world optimization problems (TSP, scheduling)

---

## 1. Introduction to Optimization

### The Optimization Problem

Find x that minimizes (or maximizes) f(x), subject to constraints.

**Types of Optimization:**
- **Continuous**: x ∈ ℝⁿ (gradient descent, BFGS)
- **Discrete**: x ∈ {0,1}ⁿ (genetic algorithms, simulated annealing)
- **Combinatorial**: Permutations, assignments (TSP, scheduling)
- **Black-box**: f(x) unknown, expensive to evaluate

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
import random
from copy import deepcopy
import math

class OptimizationProblem:
    """
    Abstract base class for optimization problems.

    Defines interface for fitness evaluation and constraints.
    """
    def fitness(self, solution):
        """
        Evaluate fitness of solution.

        Higher = better for maximization
        Lower = better for minimization
        """
        raise NotImplementedError

    def is_valid(self, solution):
        """Check if solution satisfies constraints."""
        return True

    def random_solution(self):
        """Generate random valid solution."""
        raise NotImplementedError


class SphereFunction(OptimizationProblem):
    """
    Sphere function: f(x) = Σ x_i²

    Global minimum: f(0, 0, ..., 0) = 0
    """
    def __init__(self, dimensions=2, bounds=(-5, 5)):
        self.dimensions = dimensions
        self.bounds = bounds

    def fitness(self, solution):
        """Minimize sum of squares."""
        return -np.sum(np.array(solution) ** 2)  # Negative for maximization

    def random_solution(self):
        """Random point in search space."""
        return [random.uniform(*self.bounds) for _ in range(self.dimensions)]


class RastriginFunction(OptimizationProblem):
    """
    Rastrigin function: Highly multimodal (many local optima).

    f(x) = 10n + Σ(x_i² - 10*cos(2π*x_i))

    Global minimum: f(0, 0, ..., 0) = 0
    """
    def __init__(self, dimensions=2, bounds=(-5.12, 5.12)):
        self.dimensions = dimensions
        self.bounds = bounds

    def fitness(self, solution):
        """Minimize Rastrigin function."""
        x = np.array(solution)
        A = 10
        n = len(x)
        result = A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
        return -result  # Negative for maximization

    def random_solution(self):
        return [random.uniform(*self.bounds) for _ in range(self.dimensions)]


# Visualize optimization landscapes
def plot_landscape(func, bounds=(-5, 5), title="Optimization Landscape"):
    """Plot 2D optimization function."""
    x = np.linspace(bounds[0], bounds[1], 100)
    y = np.linspace(bounds[0], bounds[1], 100)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = -func.fitness([X[i, j], Y[i, j]])  # Negate to show minimization

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Function Value')
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# Visualize Sphere and Rastrigin functions
sphere = SphereFunction(dimensions=2)
plot_landscape(sphere, bounds=(-5, 5), title="Sphere Function (Easy)")

rastrigin = RastriginFunction(dimensions=2)
plot_landscape(rastrigin, bounds=(-5, 5), title="Rastrigin Function (Hard - Many Local Optima)")
```

---

## 2. Genetic Algorithms (GA)

### Inspired by Natural Evolution

Genetic algorithms mimic biological evolution:
- **Population**: Set of candidate solutions
- **Selection**: Fitter individuals more likely to reproduce
- **Crossover**: Combine parent genes
- **Mutation**: Random changes for diversity

```python
class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization.

    Evolution process:
    1. Initialize random population
    2. Evaluate fitness
    3. Select parents (fitter = higher probability)
    4. Crossover (breed offspring)
    5. Mutate offspring
    6. Replace population
    7. Repeat until convergence
    """
    def __init__(self,
                 problem: OptimizationProblem,
                 population_size=100,
                 mutation_rate=0.1,
                 crossover_rate=0.8,
                 elitism=0.1):
        """
        Args:
            problem: OptimizationProblem instance
            population_size: Number of individuals
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Fraction of top individuals to keep
        """
        self.problem = problem
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = int(elitism * population_size)

        self.population = []
        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

    def initialize_population(self):
        """Create random initial population."""
        self.population = [
            self.problem.random_solution()
            for _ in range(self.population_size)
        ]

    def evaluate_fitness(self):
        """Compute fitness for all individuals."""
        return [self.problem.fitness(ind) for ind in self.population]

    def select_parents(self, fitness_values):
        """
        Tournament selection.

        Randomly sample k individuals, pick the best.
        """
        tournament_size = 3
        parent1 = self._tournament_select(fitness_values, tournament_size)
        parent2 = self._tournament_select(fitness_values, tournament_size)
        return parent1, parent2

    def _tournament_select(self, fitness_values, k):
        """Select best individual from k random samples."""
        indices = random.sample(range(len(self.population)), k)
        best_idx = max(indices, key=lambda i: fitness_values[i])
        return self.population[best_idx]

    def crossover(self, parent1, parent2):
        """
        Single-point crossover.

        Split parents at random point, combine halves.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def mutate(self, individual):
        """
        Gaussian mutation for continuous variables.

        Add random noise to each gene with probability mutation_rate.
        """
        mutated = individual[:]

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                # Add Gaussian noise
                mutated[i] += random.gauss(0, 0.5)

                # Clip to bounds if available
                if hasattr(self.problem, 'bounds'):
                    mutated[i] = np.clip(mutated[i], *self.problem.bounds)

        return mutated

    def evolve(self, generations=100):
        """
        Run genetic algorithm.

        Args:
            generations: Number of generations to evolve

        Returns:
            best_solution, best_fitness, fitness_history
        """
        self.initialize_population()

        for gen in range(generations):
            # Evaluate fitness
            fitness_values = self.evaluate_fitness()

            # Track best solution
            max_fitness_idx = np.argmax(fitness_values)
            if fitness_values[max_fitness_idx] > self.best_fitness:
                self.best_fitness = fitness_values[max_fitness_idx]
                self.best_solution = self.population[max_fitness_idx][:]

            self.fitness_history.append(self.best_fitness)

            # Create new population
            new_population = []

            # Elitism: Keep top individuals
            if self.elitism > 0:
                elite_indices = np.argsort(fitness_values)[-self.elitism:]
                new_population.extend([self.population[i][:] for i in elite_indices])

            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1, parent2 = self.select_parents(fitness_values)

                # Crossover
                child1, child2 = self.crossover(parent1, parent2)

                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.extend([child1, child2])

            # Replace population
            self.population = new_population[:self.population_size]

            if gen % 20 == 0:
                print(f"Generation {gen}: Best fitness = {self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness, self.fitness_history


# Test on Sphere function
print("="*60)
print("Genetic Algorithm on Sphere Function")
print("="*60)

sphere_problem = SphereFunction(dimensions=5)
ga = GeneticAlgorithm(
    sphere_problem,
    population_size=50,
    mutation_rate=0.2,
    crossover_rate=0.8,
    elitism=0.1
)

best_solution, best_fitness, history = ga.evolve(generations=100)

print(f"\nBest solution found: {best_solution}")
print(f"Best fitness: {best_fitness:.6f}")
print(f"Distance from optimal: {np.sqrt(-best_fitness):.6f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Genetic Algorithm Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

### GA for Binary Problems

```python
class BinaryGeneticAlgorithm(GeneticAlgorithm):
    """
    GA for binary optimization (genes are 0 or 1).

    Example: Feature selection, knapsack problem.
    """
    def __init__(self, problem, gene_length, **kwargs):
        super().__init__(problem, **kwargs)
        self.gene_length = gene_length

    def initialize_population(self):
        """Create random binary individuals."""
        self.population = [
            [random.randint(0, 1) for _ in range(self.gene_length)]
            for _ in range(self.population_size)
        ]

    def mutate(self, individual):
        """Bit-flip mutation for binary genes."""
        mutated = individual[:]

        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit

        return mutated


class KnapsackProblem(OptimizationProblem):
    """
    0/1 Knapsack problem.

    Given items with weights and values, select subset
    that maximizes value without exceeding capacity.
    """
    def __init__(self, weights, values, capacity):
        self.weights = np.array(weights)
        self.values = np.array(values)
        self.capacity = capacity
        self.n_items = len(weights)

    def fitness(self, solution):
        """
        Fitness = total value (if valid), else penalty.

        solution: Binary vector (1 = take item, 0 = leave)
        """
        solution = np.array(solution)
        total_weight = np.sum(solution * self.weights)
        total_value = np.sum(solution * self.values)

        # Penalty for exceeding capacity
        if total_weight > self.capacity:
            return -1000

        return total_value

    def is_valid(self, solution):
        """Check weight constraint."""
        total_weight = np.sum(np.array(solution) * self.weights)
        return total_weight <= self.capacity


# Solve knapsack with GA
weights = [10, 20, 30, 40, 50, 60, 70]
values = [100, 300, 200, 400, 500, 350, 450]
capacity = 150

print("\n" + "="*60)
print("Genetic Algorithm on Knapsack Problem")
print("="*60)
print(f"Items: {len(weights)}")
print(f"Capacity: {capacity}")
print(f"Weights: {weights}")
print(f"Values: {values}")

knapsack = KnapsackProblem(weights, values, capacity)
ga_binary = BinaryGeneticAlgorithm(
    knapsack,
    gene_length=len(weights),
    population_size=50,
    mutation_rate=0.1,
    crossover_rate=0.9
)

best_solution, best_fitness, _ = ga_binary.evolve(generations=100)

print(f"\nBest solution: {best_solution}")
print(f"Items selected: {[i for i, x in enumerate(best_solution) if x == 1]}")
print(f"Total value: {best_fitness}")
print(f"Total weight: {np.sum(np.array(best_solution) * knapsack.weights)}")
```

---

## 3. Simulated Annealing

### Inspired by Metallurgy

Simulated annealing mimics the annealing process in metallurgy:
- Start with high "temperature" (accept bad moves)
- Gradually cool (become more selective)
- Escape local optima early, converge to global optimum

```python
class SimulatedAnnealing:
    """
    Simulated Annealing for optimization.

    Accepts worse solutions with probability exp(-ΔE/T).
    Temperature T decreases over time.
    """
    def __init__(self,
                 problem: OptimizationProblem,
                 initial_temp=100.0,
                 cooling_rate=0.95,
                 min_temp=0.01):
        """
        Args:
            initial_temp: Starting temperature
            cooling_rate: Multiplicative cooling (T *= cooling_rate)
            min_temp: Stop when T < min_temp
        """
        self.problem = problem
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp

        self.best_solution = None
        self.best_fitness = -float('inf')
        self.fitness_history = []

    def get_neighbor(self, solution):
        """
        Generate neighbor solution.

        For continuous: Add small random noise
        For discrete: Random local change
        """
        neighbor = solution[:]

        # Random perturbation
        idx = random.randint(0, len(solution) - 1)
        neighbor[idx] += random.gauss(0, 0.5)

        # Clip to bounds
        if hasattr(self.problem, 'bounds'):
            neighbor[idx] = np.clip(neighbor[idx], *self.problem.bounds)

        return neighbor

    def acceptance_probability(self, current_fitness, new_fitness, temperature):
        """
        Probability of accepting worse solution.

        Always accept if better (new > current).
        Accept if worse with probability exp(-ΔE/T).
        """
        if new_fitness > current_fitness:
            return 1.0

        delta = new_fitness - current_fitness
        return math.exp(delta / temperature)

    def optimize(self, max_iterations=10000):
        """
        Run simulated annealing.

        Returns:
            best_solution, best_fitness, fitness_history
        """
        # Initialize
        current_solution = self.problem.random_solution()
        current_fitness = self.problem.fitness(current_solution)

        self.best_solution = current_solution[:]
        self.best_fitness = current_fitness

        temperature = self.initial_temp
        iteration = 0

        while temperature > self.min_temp and iteration < max_iterations:
            # Generate neighbor
            new_solution = self.get_neighbor(current_solution)
            new_fitness = self.problem.fitness(new_solution)

            # Acceptance criterion
            accept_prob = self.acceptance_probability(
                current_fitness, new_fitness, temperature
            )

            if random.random() < accept_prob:
                current_solution = new_solution
                current_fitness = new_fitness

                # Update best
                if current_fitness > self.best_fitness:
                    self.best_solution = current_solution[:]
                    self.best_fitness = current_fitness

            # Cool down
            temperature *= self.cooling_rate
            iteration += 1

            # Track progress
            self.fitness_history.append(self.best_fitness)

            if iteration % 1000 == 0:
                print(f"Iteration {iteration}: T={temperature:.4f}, "
                      f"Best fitness={self.best_fitness:.4f}")

        return self.best_solution, self.best_fitness, self.fitness_history


# Test on Rastrigin (hard multimodal function)
print("\n" + "="*60)
print("Simulated Annealing on Rastrigin Function")
print("="*60)

rastrigin_problem = RastriginFunction(dimensions=5)
sa = SimulatedAnnealing(
    rastrigin_problem,
    initial_temp=100.0,
    cooling_rate=0.95,
    min_temp=0.01
)

best_solution, best_fitness, history = sa.optimize(max_iterations=10000)

print(f"\nBest solution found: {best_solution}")
print(f"Best fitness: {best_fitness:.6f}")
print(f"Distance from optimal: {np.sqrt(-best_fitness):.6f}")

# Plot convergence
plt.figure(figsize=(10, 6))
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Simulated Annealing Convergence')
plt.grid(True, alpha=0.3)
plt.show()
```

### Cooling Schedules

```python
def linear_cooling(initial_temp, iteration, max_iterations):
    """Linear temperature decrease."""
    return initial_temp * (1 - iteration / max_iterations)


def exponential_cooling(temp, cooling_rate=0.95):
    """Exponential temperature decrease (most common)."""
    return temp * cooling_rate


def logarithmic_cooling(initial_temp, iteration):
    """Logarithmic cooling (slower)."""
    return initial_temp / (1 + math.log(1 + iteration))


# Compare cooling schedules
iterations = 1000
initial_T = 100

schedules = {
    'Linear': [linear_cooling(initial_T, i, iterations) for i in range(iterations)],
    'Exponential': [initial_T * (0.95 ** i) for i in range(iterations)],
    'Logarithmic': [logarithmic_cooling(initial_T, i) for i in range(iterations)]
}

plt.figure(figsize=(10, 6))
for name, temps in schedules.items():
    plt.plot(temps, label=name)

plt.xlabel('Iteration')
plt.ylabel('Temperature')
plt.title('Cooling Schedules Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

---

## 4. Particle Swarm Optimization (PSO)

### Inspired by Bird Flocking

PSO simulates social behavior of bird flocks searching for food:
- **Particles**: Candidate solutions moving through search space
- **Velocity**: Direction and speed of movement
- **Personal best**: Best position each particle has seen
- **Global best**: Best position any particle has found

```python
class ParticleSwarmOptimization:
    """
    Particle Swarm Optimization.

    Each particle:
    - Has position (solution) and velocity
    - Remembers personal best
    - Knows global best
    - Updates velocity based on personal and global best
    """
    def __init__(self,
                 problem: OptimizationProblem,
                 n_particles=30,
                 w=0.7,
                 c1=1.5,
                 c2=1.5):
        """
        Args:
            n_particles: Number of particles in swarm
            w: Inertia weight (momentum)
            c1: Cognitive parameter (personal best influence)
            c2: Social parameter (global best influence)
        """
        self.problem = problem
        self.n_particles = n_particles
        self.w = w  # Inertia
        self.c1 = c1  # Cognitive
        self.c2 = c2  # Social

        self.dimensions = problem.dimensions

        # Swarm state
        self.positions = []
        self.velocities = []
        self.personal_best_positions = []
        self.personal_best_fitness = []

        self.global_best_position = None
        self.global_best_fitness = -float('inf')
        self.fitness_history = []

    def initialize_swarm(self):
        """Create random particles."""
        bounds = self.problem.bounds

        for _ in range(self.n_particles):
            # Random position
            position = [random.uniform(*bounds) for _ in range(self.dimensions)]
            self.positions.append(position)

            # Random velocity
            velocity = [random.uniform(-1, 1) for _ in range(self.dimensions)]
            self.velocities.append(velocity)

            # Initialize personal best
            fitness = self.problem.fitness(position)
            self.personal_best_positions.append(position[:])
            self.personal_best_fitness.append(fitness)

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = position[:]

    def update_particle(self, i):
        """
        Update velocity and position of particle i.

        Velocity update:
        v = w*v + c1*r1*(personal_best - x) + c2*r2*(global_best - x)

        Position update:
        x = x + v
        """
        position = self.positions[i]
        velocity = self.velocities[i]
        personal_best = self.personal_best_positions[i]

        # Update velocity
        new_velocity = []
        for d in range(self.dimensions):
            # Inertia
            inertia = self.w * velocity[d]

            # Cognitive component (personal best)
            r1 = random.random()
            cognitive = self.c1 * r1 * (personal_best[d] - position[d])

            # Social component (global best)
            r2 = random.random()
            social = self.c2 * r2 * (self.global_best_position[d] - position[d])

            new_velocity.append(inertia + cognitive + social)

        # Update position
        new_position = [
            position[d] + new_velocity[d]
            for d in range(self.dimensions)
        ]

        # Clip to bounds
        bounds = self.problem.bounds
        new_position = [np.clip(x, *bounds) for x in new_position]

        self.velocities[i] = new_velocity
        self.positions[i] = new_position

        # Update personal best
        fitness = self.problem.fitness(new_position)
        if fitness > self.personal_best_fitness[i]:
            self.personal_best_fitness[i] = fitness
            self.personal_best_positions[i] = new_position[:]

            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best_fitness = fitness
                self.global_best_position = new_position[:]

    def optimize(self, max_iterations=100):
        """
        Run PSO optimization.

        Returns:
            best_solution, best_fitness, fitness_history
        """
        self.initialize_swarm()

        for iteration in range(max_iterations):
            # Update all particles
            for i in range(self.n_particles):
                self.update_particle(i)

            self.fitness_history.append(self.global_best_fitness)

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.global_best_fitness:.4f}")

        return self.global_best_position, self.global_best_fitness, self.fitness_history


# Test PSO on Rastrigin
print("\n" + "="*60)
print("Particle Swarm Optimization on Rastrigin Function")
print("="*60)

rastrigin_problem = RastriginFunction(dimensions=5)
pso = ParticleSwarmOptimization(
    rastrigin_problem,
    n_particles=30,
    w=0.7,
    c1=1.5,
    c2=1.5
)

best_solution, best_fitness, history = pso.optimize(max_iterations=100)

print(f"\nBest solution found: {best_solution}")
print(f"Best fitness: {best_fitness:.6f}")

# Compare GA vs SA vs PSO
plt.figure(figsize=(12, 6))
plt.plot(ga.fitness_history, label='Genetic Algorithm')
plt.plot(sa.fitness_history, label='Simulated Annealing')
plt.plot(pso.fitness_history, label='Particle Swarm')
plt.xlabel('Iteration')
plt.ylabel('Best Fitness')
plt.title('Algorithm Comparison on Rastrigin Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 5. Ant Colony Optimization (ACO)

### Inspired by Ant Foraging

Ants find shortest paths using pheromone trails:
- Shorter paths accumulate more pheromone
- Ants prefer paths with more pheromone
- Pheromones evaporate over time

**Best for:** Routing, TSP, network optimization

```python
class AntColonyOptimization:
    """
    Ant Colony Optimization for Traveling Salesman Problem.

    Ants build solutions probabilistically based on pheromones.
    Shorter paths deposit more pheromone.
    """
    def __init__(self,
                 distances,
                 n_ants=10,
                 alpha=1.0,
                 beta=2.0,
                 evaporation=0.5,
                 pheromone_constant=100):
        """
        Args:
            distances: Distance matrix (n_cities x n_cities)
            n_ants: Number of ants
            alpha: Pheromone importance
            beta: Distance importance (heuristic)
            evaporation: Pheromone evaporation rate
            pheromone_constant: Pheromone deposit amount
        """
        self.distances = np.array(distances)
        self.n_cities = len(distances)
        self.n_ants = n_ants

        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.Q = pheromone_constant

        # Initialize pheromones
        self.pheromones = np.ones((self.n_cities, self.n_cities))

        self.best_path = None
        self.best_distance = float('inf')
        self.distance_history = []

    def construct_solution(self):
        """
        Build tour for one ant.

        Probabilistically choose next city based on pheromones and distance.
        """
        # Start from random city
        current_city = random.randint(0, self.n_cities - 1)
        path = [current_city]
        unvisited = set(range(self.n_cities)) - {current_city}

        while unvisited:
            # Calculate probabilities for next city
            probabilities = []

            for city in unvisited:
                pheromone = self.pheromones[current_city][city] ** self.alpha
                distance = (1.0 / self.distances[current_city][city]) ** self.beta
                probabilities.append(pheromone * distance)

            # Normalize probabilities
            prob_sum = sum(probabilities)
            probabilities = [p / prob_sum for p in probabilities]

            # Choose next city
            next_city = random.choices(list(unvisited), weights=probabilities)[0]

            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city

        return path

    def calculate_path_distance(self, path):
        """Calculate total distance of path."""
        distance = 0
        for i in range(len(path)):
            distance += self.distances[path[i]][path[(i + 1) % len(path)]]
        return distance

    def update_pheromones(self, all_paths, all_distances):
        """
        Update pheromone trails.

        1. Evaporate existing pheromones
        2. Add new pheromones based on solution quality
        """
        # Evaporation
        self.pheromones *= (1 - self.evaporation)

        # Add new pheromones
        for path, distance in zip(all_paths, all_distances):
            pheromone_deposit = self.Q / distance

            for i in range(len(path)):
                from_city = path[i]
                to_city = path[(i + 1) % len(path)]

                self.pheromones[from_city][to_city] += pheromone_deposit
                self.pheromones[to_city][from_city] += pheromone_deposit

    def optimize(self, max_iterations=100):
        """
        Run ACO optimization.

        Returns:
            best_path, best_distance, distance_history
        """
        for iteration in range(max_iterations):
            # Construct solutions for all ants
            all_paths = []
            all_distances = []

            for _ in range(self.n_ants):
                path = self.construct_solution()
                distance = self.calculate_path_distance(path)

                all_paths.append(path)
                all_distances.append(distance)

                # Update best solution
                if distance < self.best_distance:
                    self.best_distance = distance
                    self.best_path = path[:]

            # Update pheromones
            self.update_pheromones(all_paths, all_distances)

            self.distance_history.append(self.best_distance)

            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best distance = {self.best_distance:.2f}")

        return self.best_path, self.best_distance, self.distance_history


# Test ACO on TSP
print("\n" + "="*60)
print("Ant Colony Optimization on TSP")
print("="*60)

# Create random TSP instance
n_cities = 10
np.random.seed(42)

# Random city positions
city_positions = np.random.rand(n_cities, 2) * 100

# Calculate distance matrix
distances = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        if i != j:
            distances[i][j] = np.linalg.norm(city_positions[i] - city_positions[j])
        else:
            distances[i][j] = float('inf')  # Can't go to same city

aco = AntColonyOptimization(
    distances,
    n_ants=20,
    alpha=1.0,
    beta=2.0,
    evaporation=0.5
)

best_path, best_distance, history = aco.optimize(max_iterations=100)

print(f"\nBest path: {best_path}")
print(f"Best distance: {best_distance:.2f}")

# Visualize TSP solution
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Best Distance')
plt.title('ACO Convergence on TSP')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
# Plot cities
plt.scatter(city_positions[:, 0], city_positions[:, 1], s=100, c='red', zorder=2)

# Plot tour
for i in range(len(best_path)):
    from_city = best_path[i]
    to_city = best_path[(i + 1) % len(best_path)]

    plt.plot([city_positions[from_city, 0], city_positions[to_city, 0]],
            [city_positions[from_city, 1], city_positions[to_city, 1]],
            'b-', alpha=0.6)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Best TSP Tour Found')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. Hyperparameter Tuning

### Real-World Application: Optimize ML Models

```python
class HyperparameterOptimization:
    """
    Use optimization algorithms for hyperparameter tuning.

    Example: Optimize learning rate, batch size, layers for neural network.
    """
    def __init__(self, param_ranges):
        """
        Args:
            param_ranges: Dict {param_name: (min, max)}
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.dimensions = len(param_ranges)

    def decode_solution(self, solution):
        """Convert solution vector to parameter dict."""
        params = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            # Map [0, 1] to [min, max]
            params[name] = solution[i]
        return params

    def objective_function(self, params):
        """
        Evaluate model with given hyperparameters.

        In practice: Train model, return validation accuracy.
        Here: Simplified synthetic function.
        """
        # Simulate training a model
        # Higher learning rate + more layers = better (up to a point)
        lr = params.get('learning_rate', 0.01)
        layers = params.get('num_layers', 2)
        dropout = params.get('dropout', 0.5)

        # Synthetic fitness (replace with real model training)
        score = (
            -((lr - 0.001) ** 2) * 1000 +
            -(layers - 3) ** 2 +
            -((dropout - 0.3) ** 2) * 10
        )

        return score


# Define search space
param_ranges = {
    'learning_rate': (0.0001, 0.1),
    'num_layers': (1, 5),
    'dropout': (0.0, 0.8)
}

hp_opt = HyperparameterOptimization(param_ranges)


class HyperparameterProblem(OptimizationProblem):
    """Optimization problem for hyperparameter tuning."""

    def __init__(self, hp_opt):
        self.hp_opt = hp_opt
        self.dimensions = hp_opt.dimensions
        self.bounds = (0, 1)  # Normalized space

    def fitness(self, solution):
        """Evaluate hyperparameters."""
        # Denormalize to actual parameter ranges
        params = {}
        for i, name in enumerate(self.hp_opt.param_names):
            min_val, max_val = self.hp_opt.param_ranges[name]
            params[name] = solution[i] * (max_val - min_val) + min_val

        return self.hp_opt.objective_function(params)

    def random_solution(self):
        """Random hyperparameters in [0, 1]."""
        return [random.random() for _ in range(self.dimensions)]


# Optimize hyperparameters with PSO
print("\n" + "="*60)
print("Hyperparameter Optimization with PSO")
print("="*60)

hp_problem = HyperparameterProblem(hp_opt)
pso_hp = ParticleSwarmOptimization(
    hp_problem,
    n_particles=20,
    w=0.7,
    c1=1.5,
    c2=1.5
)

best_solution, best_fitness, _ = pso_hp.optimize(max_iterations=50)

# Decode best hyperparameters
best_params = {}
for i, name in enumerate(hp_opt.param_names):
    min_val, max_val = param_ranges[name]
    best_params[name] = best_solution[i] * (max_val - min_val) + min_val

print(f"\nBest hyperparameters found:")
for name, value in best_params.items():
    print(f"  {name}: {value:.6f}")
print(f"Best score: {best_fitness:.4f}")
```

---

## 7. Practice Exercises

### Exercise 1: Differential Evolution

```python
"""
Implement Differential Evolution (DE).

DE is similar to GA but uses vector differences for mutation:
- For each individual x_i:
  1. Select three random individuals a, b, c
  2. Mutant: v = a + F * (b - c)  where F ∈ [0, 2]
  3. Crossover: Mix mutant with x_i
  4. Selection: Keep better of mutant and x_i

Very effective for continuous optimization!
"""

# Your implementation here
```

### Exercise 2: Multi-Objective Optimization

```python
"""
Implement NSGA-II (Non-dominated Sorting GA).

Multi-objective: Optimize multiple conflicting objectives.
Example: Maximize accuracy, minimize model size

Key concepts:
- Pareto dominance: Solution A dominates B if better in all objectives
- Pareto front: Set of non-dominated solutions
- Crowding distance: Maintain diversity

Apply to: Optimize neural network (accuracy vs size vs speed)
"""

# Your implementation here
```

### Exercise 3: Hybrid Algorithm

```python
"""
Create hybrid optimization algorithm.

Combine strengths of multiple algorithms:
- GA for global exploration
- PSO for fast convergence
- SA for escaping local optima

Example: Run GA for 50 generations, then PSO, then SA refinement.

Test on multimodal function (Rastrigin, Ackley).
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Genetic Algorithms** 🧬
   - Population-based evolution
   - Selection, crossover, mutation
   - Works for discrete and continuous problems
   - Good for complex search spaces

2. **Simulated Annealing** 🔥
   - Probabilistic acceptance of worse solutions
   - Temperature controls exploration
   - Escapes local optima
   - Single-solution method (memory efficient)

3. **Particle Swarm Optimization** 🐦
   - Swarm intelligence
   - Particles influenced by personal and global best
   - Fast convergence
   - Few parameters to tune

4. **Ant Colony Optimization** 🐜
   - Pheromone-based pathfinding
   - Best for routing and graph problems
   - Emergent intelligence from simple rules
   - TSP, network optimization

5. **Hyperparameter Tuning** 🎯
   - Practical application of optimization
   - Replace grid search with intelligent search
   - PSO and Bayesian optimization most popular
   - Can improve model performance significantly

### Algorithm Comparison

| Algorithm | Best For | Pros | Cons |
|-----------|----------|------|------|
| GA | Discrete, combinatorial | Flexible, parallel | Slow, many parameters |
| SA | Continuous, few dimensions | Simple, escapes local optima | Single solution, slow |
| PSO | Continuous optimization | Fast, few parameters | Can converge prematurely |
| ACO | Routing, graph problems | Good for TSP, distributed | Problem-specific |

### Real-World Applications

✅ **Neural Network Training**: Hyperparameter optimization
✅ **Feature Selection**: GA for selecting best features
✅ **Scheduling**: GA, ACO for job shop scheduling
✅ **Routing**: ACO for vehicle routing, network design
✅ **Engineering Design**: PSO for antenna design, circuit optimization
✅ **Drug Discovery**: GA for molecular structure optimization

### What's Next?

In Lesson 4, we'll explore **Graph Algorithms**:
- PageRank algorithm (Google's secret sauce)
- Shortest path algorithms (Dijkstra, Bellman-Ford)
- Community detection
- Applications to social networks and knowledge graphs

**Optimization is everywhere - from nature to neural networks!** 🚀

---

## Additional Resources

### Papers
- Holland (1975): "Adaptation in Natural and Artificial Systems" (Genetic Algorithms)
- Kirkpatrick et al. (1983): "Optimization by Simulated Annealing"
- Kennedy & Eberhart (1995): "Particle Swarm Optimization"
- Dorigo (1992): "Optimization, Learning and Natural Algorithms" (Ant Colony)

### Books
- **An Introduction to Genetic Algorithms** (Mitchell)
- **Swarm Intelligence** (Kennedy, Eberhart & Shi)
- **Metaheuristics** (Talbi)

### Libraries
- **DEAP**: Distributed Evolutionary Algorithms in Python
- **PySwarms**: Particle Swarm Optimization
- **scikit-optimize**: Bayesian optimization for hyperparameters
- **Optuna**: Hyperparameter optimization framework

### Visualizations
- **GA Visualizer**: See evolution in action
- **PSO Animation**: Watch particles converge

---

**Next**: [Lesson 4 - Graph Algorithms and PageRank](Lesson%204%20-%20Graph%20Algorithms%20and%20PageRank.md)

Master the algorithms that power Google Search! 🔍

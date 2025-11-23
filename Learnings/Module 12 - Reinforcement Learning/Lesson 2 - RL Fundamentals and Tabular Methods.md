# Lesson 2: RL Fundamentals & Tabular Methods 🎯

**Module 12: Reinforcement Learning | Lesson 2 of 12**

Master Markov Decision Processes and the foundational algorithms of reinforcement learning!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand Markov Decision Processes (MDPs) and the Bellman Equations
2. ✅ Implement Dynamic Programming methods (Value & Policy Iteration)
3. ✅ Master Monte Carlo methods for learning from episodes
4. ✅ Implement Temporal Difference learning (Q-Learning, SARSA, Expected SARSA)
5. ✅ Apply tabular methods to GridWorld, FrozenLake, and Taxi environments
6. ✅ Understand n-step methods and eligibility traces

---

## 1. Markov Decision Processes (MDPs)

### What is an MDP?

While bandits solve **single-state** problems, MDPs handle **sequential decision-making** with multiple states. An MDP is defined by:

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition dynamics P(s'|s,a)
- **R**: Reward function R(s,a,s')
- **γ**: Discount factor (0 ≤ γ ≤ 1)

**Markov Property**: The future depends only on the present state, not the history.

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class GridWorld:
    """
    Simple GridWorld MDP environment.

    Agent navigates a grid to reach goal while avoiding obstacles.
    """
    def __init__(self, size=5, goal=(4, 4), obstacles=None):
        self.size = size
        self.goal = goal
        self.obstacles = obstacles or [(1, 1), (2, 2), (3, 1)]

        # Start position
        self.start = (0, 0)
        self.state = self.start

        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [0, 1, 2, 3]
        self.action_names = ['↑', '→', '↓', '←']

    def reset(self):
        """Reset to start state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take action and return (next_state, reward, done).

        Returns:
            next_state: New state after action
            reward: Immediate reward
            done: Whether episode terminated
        """
        row, col = self.state

        # Determine next state based on action
        if action == 0:    # up
            next_state = (max(row - 1, 0), col)
        elif action == 1:  # right
            next_state = (row, min(col + 1, self.size - 1))
        elif action == 2:  # down
            next_state = (min(row + 1, self.size - 1), col)
        else:              # left
            next_state = (row, max(col - 1, 0))

        # Check if next state is obstacle (stay in place)
        if next_state in self.obstacles:
            next_state = self.state
            reward = -1.0
        # Check if reached goal
        elif next_state == self.goal:
            reward = 10.0
        # Normal move
        else:
            reward = -0.1  # Small negative reward to encourage efficiency

        done = (next_state == self.goal)
        self.state = next_state

        return next_state, reward, done

    def get_all_states(self):
        """Return all valid states."""
        states = []
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in self.obstacles:
                    states.append((i, j))
        return states

    def render(self, policy=None, values=None):
        """Visualize the grid world."""
        grid = np.zeros((self.size, self.size))

        # Mark obstacles
        for obs in self.obstacles:
            grid[obs] = -1

        # Mark goal
        grid[self.goal] = 2

        # Mark current state
        grid[self.state] = 1

        plt.figure(figsize=(8, 8))
        plt.imshow(grid, cmap='RdYlGn', vmin=-1, vmax=2)

        # Add policy arrows or values
        if policy is not None:
            for state in self.get_all_states():
                if state != self.goal:
                    action = policy[state]
                    arrow = self.action_names[action]
                    plt.text(state[1], state[0], arrow,
                            ha='center', va='center', fontsize=20)

        if values is not None:
            for state in self.get_all_states():
                value = values.get(state, 0)
                plt.text(state[1], state[0] + 0.3, f'{value:.1f}',
                        ha='center', va='center', fontsize=10, color='blue')

        plt.grid(True)
        plt.title('GridWorld Environment')
        plt.show()


# Create environment
env = GridWorld(size=5)
print("GridWorld MDP created!")
print(f"State space: {len(env.get_all_states())} states")
print(f"Action space: {len(env.actions)} actions")
print(f"Goal: {env.goal}")
print(f"Obstacles: {env.obstacles}")

# Sample interaction
state = env.reset()
print(f"\nInitial state: {state}")

for i in range(3):
    action = np.random.choice(env.actions)
    next_state, reward, done = env.step(action)
    print(f"Action {env.action_names[action]} → State {next_state}, Reward {reward:.2f}, Done {done}")
    if done:
        break
```

**Key Concepts:**
- **State**: Complete description of the agent's situation
- **Transition dynamics**: P(s'|s,a) - probability of transitioning to s' from s taking action a
- **Return**: Cumulative discounted reward G_t = Σ γ^k * R_{t+k+1}
- **Value function**: Expected return from a state

---

## 2. Bellman Equations

The Bellman equations are the foundation of RL! They express the relationship between the value of a state and its successors.

### State-Value Function

```python
"""
State-Value Function V^π(s):
Expected return when starting in state s and following policy π.

V^π(s) = E_π[G_t | S_t = s]
       = E_π[R_{t+1} + γ * V^π(S_{t+1}) | S_t = s]

This is the Bellman Expectation Equation for V^π.
"""

def bellman_expectation_v(env, policy, V, gamma=0.99):
    """
    Compute V^π(s) using Bellman Expectation Equation.

    Args:
        env: Environment
        policy: Dictionary {state: action}
        V: Current value function
        gamma: Discount factor

    Returns:
        Updated value function
    """
    V_new = {}

    for state in env.get_all_states():
        if state == env.goal:
            V_new[state] = 0  # Terminal state
            continue

        # Get action from policy
        action = policy.get(state, 0)

        # Simulate taking action (deterministic in our GridWorld)
        env.state = state
        next_state, reward, done = env.step(action)

        # Bellman update
        V_new[state] = reward + gamma * V.get(next_state, 0)

    return V_new


# Example: Evaluate a random policy
env = GridWorld(size=5)
policy = {state: np.random.choice(env.actions) for state in env.get_all_states()}
V = {state: 0.0 for state in env.get_all_states()}

# Iterative policy evaluation
for iteration in range(50):
    V = bellman_expectation_v(env, policy, V, gamma=0.99)

print("Value function after 50 iterations:")
for state in env.get_all_states()[:5]:
    print(f"V({state}) = {V[state]:.3f}")
```

### Action-Value Function (Q-function)

```python
"""
Action-Value Function Q^π(s,a):
Expected return when starting in state s, taking action a, then following policy π.

Q^π(s,a) = E_π[R_{t+1} + γ * Q^π(S_{t+1}, π(S_{t+1})) | S_t = s, A_t = a]

This is the Bellman Expectation Equation for Q^π.
"""

def bellman_expectation_q(env, Q, gamma=0.99):
    """
    Compute Q^π(s,a) using Bellman Expectation Equation.

    Args:
        env: Environment
        Q: Current Q-function {(state, action): value}
        gamma: Discount factor

    Returns:
        Updated Q-function
    """
    Q_new = {}

    for state in env.get_all_states():
        for action in env.actions:
            if state == env.goal:
                Q_new[(state, action)] = 0  # Terminal state
                continue

            # Simulate taking action
            env.state = state
            next_state, reward, done = env.step(action)

            if done:
                Q_new[(state, action)] = reward
            else:
                # Expected value over next actions (assuming uniform policy)
                next_q_values = [Q.get((next_state, a), 0) for a in env.actions]
                Q_new[(state, action)] = reward + gamma * np.mean(next_q_values)

    return Q_new


# Initialize Q-function
Q = {(state, action): 0.0
     for state in env.get_all_states()
     for action in env.actions}

# Iterate
for iteration in range(30):
    Q = bellman_expectation_q(env, Q, gamma=0.99)

print("\nQ-values for state (0, 0):")
for action in env.actions:
    print(f"Q({(0, 0)}, {env.action_names[action]}) = {Q[((0, 0), action)]:.3f}")
```

**Bellman Optimality Equations:**
```python
"""
Optimal Value Functions:

V*(s) = max_a Q*(s, a)
      = max_a E[R_{t+1} + γ * V*(S_{t+1}) | S_t = s, A_t = a]

Q*(s,a) = E[R_{t+1} + γ * max_a' Q*(S_{t+1}, a') | S_t = s, A_t = a]

These define the BEST possible value functions!
"""
```

---

## 3. Dynamic Programming: Value Iteration

Value Iteration directly computes the optimal value function V* using the Bellman Optimality Equation.

```python
class ValueIteration:
    """
    Value Iteration algorithm for solving MDPs.

    Iteratively applies Bellman Optimality Equation until convergence.
    """
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta  # Convergence threshold

        # Initialize value function
        self.V = {state: 0.0 for state in env.get_all_states()}

    def one_step_lookahead(self, state):
        """
        Calculate action values for a state.

        Returns:
            Array of action values [Q(s,a) for all a]
        """
        action_values = np.zeros(len(self.env.actions))

        for action_idx, action in enumerate(self.env.actions):
            # Simulate action
            self.env.state = state
            next_state, reward, done = self.env.step(action)

            # Q(s,a) = R + γ * V(s')
            if done:
                action_values[action_idx] = reward
            else:
                action_values[action_idx] = reward + self.gamma * self.V[next_state]

        return action_values

    def train(self, max_iterations=1000):
        """
        Run Value Iteration until convergence.

        Returns:
            Number of iterations to converge
        """
        for iteration in range(max_iterations):
            delta = 0  # Track maximum change

            # Update value for each state
            for state in self.env.get_all_states():
                if state == self.env.goal:
                    continue  # Terminal state

                # V(s) = max_a Q(s,a)
                action_values = self.one_step_lookahead(state)
                best_action_value = np.max(action_values)

                # Track change
                delta = max(delta, abs(best_action_value - self.V[state]))

                # Update
                self.V[state] = best_action_value

            # Check convergence
            if delta < self.theta:
                print(f"Value Iteration converged in {iteration + 1} iterations!")
                return iteration + 1

        print(f"Warning: Did not converge in {max_iterations} iterations")
        return max_iterations

    def extract_policy(self):
        """
        Extract optimal policy from value function.

        π*(s) = argmax_a Q*(s,a)
        """
        policy = {}

        for state in self.env.get_all_states():
            if state == self.env.goal:
                policy[state] = 0  # Arbitrary
                continue

            # Choose action with highest Q-value
            action_values = self.one_step_lookahead(state)
            best_action = np.argmax(action_values)
            policy[state] = best_action

        return policy


# Run Value Iteration
env = GridWorld(size=5)
vi = ValueIteration(env, gamma=0.99)

iterations = vi.train()
policy = vi.extract_policy()

print("\nOptimal Value Function (sample states):")
for state in list(vi.V.keys())[:5]:
    print(f"V*({state}) = {vi.V[state]:.3f}")

print("\nOptimal Policy (sample states):")
for state in list(policy.keys())[:5]:
    print(f"π*({state}) = {env.action_names[policy[state]]}")

# Visualize
env.render(policy=policy, values=vi.V)
```

---

## 4. Dynamic Programming: Policy Iteration

Policy Iteration alternates between **policy evaluation** and **policy improvement**.

```python
class PolicyIteration:
    """
    Policy Iteration algorithm for solving MDPs.

    Alternates between:
    1. Policy Evaluation: Compute V^π
    2. Policy Improvement: Update π to be greedy w.r.t. V^π
    """
    def __init__(self, env, gamma=0.99, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta

        # Initialize random policy
        self.policy = {state: np.random.choice(env.actions)
                      for state in env.get_all_states()}

        self.V = {state: 0.0 for state in env.get_all_states()}

    def policy_evaluation(self):
        """
        Evaluate current policy: Compute V^π.

        Iteratively apply Bellman Expectation Equation.
        """
        while True:
            delta = 0

            for state in self.env.get_all_states():
                if state == self.env.goal:
                    continue

                v = self.V[state]

                # Get action from current policy
                action = self.policy[state]

                # Simulate action
                self.env.state = state
                next_state, reward, done = self.env.step(action)

                # Bellman expectation update
                if done:
                    self.V[state] = reward
                else:
                    self.V[state] = reward + self.gamma * self.V[next_state]

                delta = max(delta, abs(v - self.V[state]))

            # Check convergence
            if delta < self.theta:
                break

    def policy_improvement(self):
        """
        Improve policy: Make it greedy w.r.t. current V.

        Returns:
            Whether policy changed
        """
        policy_stable = True

        for state in self.env.get_all_states():
            if state == self.env.goal:
                continue

            old_action = self.policy[state]

            # Find best action
            action_values = np.zeros(len(self.env.actions))
            for action_idx, action in enumerate(self.env.actions):
                self.env.state = state
                next_state, reward, done = self.env.step(action)

                if done:
                    action_values[action_idx] = reward
                else:
                    action_values[action_idx] = reward + self.gamma * self.V[next_state]

            # Update policy
            self.policy[state] = np.argmax(action_values)

            if old_action != self.policy[state]:
                policy_stable = False

        return policy_stable

    def train(self, max_iterations=100):
        """
        Run Policy Iteration until convergence.
        """
        for iteration in range(max_iterations):
            # Policy Evaluation
            self.policy_evaluation()

            # Policy Improvement
            policy_stable = self.policy_improvement()

            if policy_stable:
                print(f"Policy Iteration converged in {iteration + 1} iterations!")
                return iteration + 1

        print(f"Warning: Did not converge in {max_iterations} iterations")
        return max_iterations


# Run Policy Iteration
env = GridWorld(size=5)
pi = PolicyIteration(env, gamma=0.99)

iterations = pi.train()

print("\nOptimal Policy from Policy Iteration:")
for state in list(pi.policy.keys())[:5]:
    print(f"π*({state}) = {env.action_names[pi.policy[state]]}")

print("\nComparing with Value Iteration policy...")
# Both should give the same optimal policy!
```

---

## 5. Monte Carlo Methods

Monte Carlo (MC) methods learn from **complete episodes**. They don't require knowledge of the environment dynamics!

```python
class MonteCarloControl:
    """
    Monte Carlo Control with ε-greedy exploration.

    Learn Q(s,a) from episodes, then improve policy.
    """
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: {(state, action): value}
        self.Q = defaultdict(float)

        # Returns: {(state, action): [list of returns]}
        self.returns = defaultdict(list)

    def generate_episode(self, policy):
        """
        Generate one episode following policy.

        Returns:
            List of (state, action, reward) tuples
        """
        episode = []
        state = self.env.reset()

        for _ in range(100):  # Max episode length
            # ε-greedy action selection
            if np.random.rand() < self.epsilon:
                action = np.random.choice(self.env.actions)
            else:
                # Greedy: Choose best action for this state
                q_values = [self.Q[(state, a)] for a in self.env.actions]
                action = self.env.actions[np.argmax(q_values)]

            next_state, reward, done = self.env.step(action)
            episode.append((state, action, reward))

            if done:
                break

            state = next_state

        return episode

    def train(self, num_episodes=1000):
        """
        Train using First-Visit Monte Carlo.
        """
        for episode_num in range(num_episodes):
            # Generate episode
            episode = self.generate_episode(self.Q)

            # Calculate returns and update Q
            G = 0  # Return
            visited = set()

            # Go backwards through episode
            for state, action, reward in reversed(episode):
                G = reward + self.gamma * G

                # First-visit MC: Only update first occurrence
                if (state, action) not in visited:
                    visited.add((state, action))

                    # Update Q as average of returns
                    self.returns[(state, action)].append(G)
                    self.Q[(state, action)] = np.mean(self.returns[(state, action)])

            # Decay epsilon
            if episode_num % 100 == 0:
                self.epsilon = max(0.01, self.epsilon * 0.99)
                avg_return = np.mean([sum([r for _, _, r in ep])
                                     for ep in [self.generate_episode(self.Q) for _ in range(10)]])
                print(f"Episode {episode_num}, Avg Return: {avg_return:.2f}, ε: {self.epsilon:.3f}")

    def get_policy(self):
        """Extract greedy policy from Q."""
        policy = {}
        for state in self.env.get_all_states():
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            policy[state] = self.env.actions[np.argmax(q_values)]
        return policy


# Run Monte Carlo Control
env = GridWorld(size=5)
mc = MonteCarloControl(env, gamma=0.99, epsilon=0.3)

mc.train(num_episodes=2000)
policy = mc.get_policy()

print("\nLearned Policy (sample states):")
for state in list(policy.keys())[:5]:
    print(f"π({state}) = {env.action_names[policy[state]]}")
```

**Key Difference**: MC learns from **complete episodes**, while DP requires **full model** of environment.

---

## 6. Temporal Difference Learning: Q-Learning

Q-Learning is the most famous TD method! It learns **online** (after each step) without needing a model.

```python
class QLearning:
    """
    Q-Learning: Off-policy TD control algorithm.

    Q(s,a) ← Q(s,a) + α[R + γ * max_a' Q(s',a') - Q(s,a)]

    Off-policy: Learns optimal Q* while following ε-greedy policy.
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha      # Learning rate
        self.gamma = gamma      # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Q-table: {(state, action): value}
        self.Q = defaultdict(float)

    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            return self.env.actions[np.argmax(q_values)]

    def train(self, num_episodes=1000):
        """
        Train Q-Learning agent.
        """
        episode_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(100):  # Max steps per episode
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Q-Learning update (OFF-POLICY!)
                # Target: R + γ * max_a' Q(s',a')
                if done:
                    target = reward
                else:
                    next_q_values = [self.Q[(next_state, a)] for a in self.env.actions]
                    target = reward + self.gamma * max(next_q_values)

                # Update Q-value
                self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])

                if done:
                    break

                state = next_state

            episode_rewards.append(total_reward)

            # Logging
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}")

        return episode_rewards

    def get_policy(self):
        """Extract greedy policy from Q."""
        policy = {}
        for state in self.env.get_all_states():
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            policy[state] = self.env.actions[np.argmax(q_values)]
        return policy


# Run Q-Learning
env = GridWorld(size=5)
q_learning = QLearning(env, alpha=0.1, gamma=0.99, epsilon=0.1)

rewards = q_learning.train(num_episodes=1000)
policy = q_learning.get_policy()

# Plot learning curve
plt.figure(figsize=(12, 5))
plt.plot(rewards, alpha=0.3)
plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'), linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Performance')
plt.grid(True)
plt.show()

print("\nLearned Policy (Q-Learning):")
for state in list(policy.keys())[:5]:
    print(f"π({state}) = {env.action_names[policy[state]]}")
```

**Key Insight**: Q-Learning is **off-policy** - it learns the optimal policy while exploring with ε-greedy!

---

## 7. SARSA: On-Policy TD Control

SARSA is the on-policy version of Q-Learning.

```python
class SARSA:
    """
    SARSA: On-policy TD control algorithm.

    Q(s,a) ← Q(s,a) + α[R + γ * Q(s',a') - Q(s,a)]

    On-policy: Learns Q for the policy it's following (ε-greedy).
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)

    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            return self.env.actions[np.argmax(q_values)]

    def train(self, num_episodes=1000):
        """Train SARSA agent."""
        episode_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)  # Choose first action
            total_reward = 0

            for step in range(100):
                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Choose next action
                next_action = self.choose_action(next_state)

                # SARSA update (ON-POLICY!)
                # Target: R + γ * Q(s',a')  <- Uses actual next action
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * self.Q[(next_state, next_action)]

                # Update Q-value
                self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])

                if done:
                    break

                # Move to next state-action pair
                state = next_state
                action = next_action

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}")

        return episode_rewards

    def get_policy(self):
        """Extract greedy policy from Q."""
        policy = {}
        for state in self.env.get_all_states():
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            policy[state] = self.env.actions[np.argmax(q_values)]
        return policy


# Run SARSA
env = GridWorld(size=5)
sarsa = SARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1)

rewards_sarsa = sarsa.train(num_episodes=1000)
policy_sarsa = sarsa.get_policy()

print("\nSARSA vs Q-Learning comparison:")
print(f"Q-Learning final avg reward: {np.mean(rewards[-100:]):.2f}")
print(f"SARSA final avg reward: {np.mean(rewards_sarsa[-100:]):.2f}")
```

**Q-Learning vs SARSA:**
- **Q-Learning (off-policy)**: Learns optimal policy, uses max in update
- **SARSA (on-policy)**: Learns policy being followed, uses actual next action

---

## 8. Expected SARSA

Expected SARSA takes the expected value over next actions instead of sampling.

```python
class ExpectedSARSA:
    """
    Expected SARSA: Hybrid of Q-Learning and SARSA.

    Q(s,a) ← Q(s,a) + α[R + γ * E[Q(s',·)] - Q(s,a)]

    Where E[Q(s',·)] = Σ_a' π(a'|s') * Q(s',a')
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(float)

    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            return self.env.actions[np.argmax(q_values)]

    def expected_q(self, state):
        """
        Calculate expected Q-value under ε-greedy policy.
        """
        q_values = np.array([self.Q[(state, a)] for a in self.env.actions])

        # ε-greedy probabilities
        best_action = np.argmax(q_values)
        probs = np.ones(len(self.env.actions)) * self.epsilon / len(self.env.actions)
        probs[best_action] += 1 - self.epsilon

        # Expected value
        return np.sum(probs * q_values)

    def train(self, num_episodes=1000):
        """Train Expected SARSA agent."""
        episode_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0

            for step in range(100):
                # Choose action
                action = self.choose_action(state)

                # Take action
                next_state, reward, done = self.env.step(action)
                total_reward += reward

                # Expected SARSA update
                if done:
                    target = reward
                else:
                    # Use expected value instead of max or sample
                    target = reward + self.gamma * self.expected_q(next_state)

                # Update Q-value
                self.Q[(state, action)] += self.alpha * (target - self.Q[(state, action)])

                if done:
                    break

                state = next_state

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, Avg Reward (last 100): {avg_reward:.2f}")

        return episode_rewards


# Run Expected SARSA
env = GridWorld(size=5)
expected_sarsa = ExpectedSARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1)

rewards_expected = expected_sarsa.train(num_episodes=1000)

print("\nExpected SARSA learned successfully!")
print(f"Final avg reward: {np.mean(rewards_expected[-100:]):.2f}")
```

**Advantage**: Expected SARSA has **lower variance** than SARSA because it uses expectation instead of sampling.

---

## 9. n-Step Methods

n-Step methods bridge MC and TD by looking n steps ahead.

```python
class NStepSARSA:
    """
    n-Step SARSA: Use n-step returns for updates.

    G_t^(n) = R_{t+1} + γR_{t+2} + ... + γ^{n-1}R_{t+n} + γ^n Q(S_{t+n}, A_{t+n})

    n=1: Regular SARSA
    n=∞: Monte Carlo
    """
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, n=5):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n  # Number of steps
        self.Q = defaultdict(float)

    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.actions)
        else:
            q_values = [self.Q[(state, a)] for a in self.env.actions]
            return self.env.actions[np.argmax(q_values)]

    def train(self, num_episodes=1000):
        """Train n-step SARSA agent."""
        episode_rewards = []

        for episode in range(num_episodes):
            state = self.env.reset()
            action = self.choose_action(state)

            # Store trajectory
            states = [state]
            actions = [action]
            rewards = [0]  # R_0 doesn't exist, placeholder

            T = float('inf')  # Terminal time
            t = 0
            total_reward = 0

            while True:
                if t < T:
                    # Take action
                    next_state, reward, done = self.env.step(action)
                    total_reward += reward

                    states.append(next_state)
                    rewards.append(reward)

                    if done:
                        T = t + 1
                    else:
                        next_action = self.choose_action(next_state)
                        actions.append(next_action)
                        action = next_action

                # τ is the time whose estimate is being updated
                tau = t - self.n + 1

                if tau >= 0:
                    # Calculate n-step return
                    G = 0
                    for i in range(tau + 1, min(tau + self.n, T) + 1):
                        G += self.gamma ** (i - tau - 1) * rewards[i]

                    # Bootstrap if not terminal
                    if tau + self.n < T:
                        s_tau_n = states[tau + self.n]
                        a_tau_n = actions[tau + self.n]
                        G += self.gamma ** self.n * self.Q[(s_tau_n, a_tau_n)]

                    # Update Q
                    s_tau = states[tau]
                    a_tau = actions[tau]
                    self.Q[(s_tau, a_tau)] += self.alpha * (G - self.Q[(s_tau, a_tau)])

                if tau == T - 1:
                    break

                t += 1

            episode_rewards.append(total_reward)

            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                print(f"Episode {episode}, n={self.n}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards


# Compare different n values
env = GridWorld(size=5)

n_values = [1, 3, 5, 10]
results = {}

for n in n_values:
    agent = NStepSARSA(env, alpha=0.1, gamma=0.99, epsilon=0.1, n=n)
    results[n] = agent.train(num_episodes=500)

# Plot comparison
plt.figure(figsize=(12, 6))
for n, rewards in results.items():
    smoothed = np.convolve(rewards, np.ones(50)/50, mode='valid')
    plt.plot(smoothed, label=f'n={n}', linewidth=2)

plt.xlabel('Episode')
plt.ylabel('Smoothed Reward')
plt.title('n-Step SARSA: Effect of n')
plt.legend()
plt.grid(True)
plt.show()
```

---

## 10. OpenAI Gym Environments

Let's apply our algorithms to classic RL environments!

```python
import gym

# FrozenLake Environment
print("=== FrozenLake-v1 ===")
env_frozen = gym.make('FrozenLake-v1', is_slippery=True)
print(f"State space: {env_frozen.observation_space}")
print(f"Action space: {env_frozen.action_space}")

# Q-Learning for FrozenLake
class QLearningGym:
    """Q-Learning for OpenAI Gym environments."""
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: rows=states, cols=actions
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        """ε-greedy action selection."""
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.Q[state])

    def train(self, num_episodes=10000):
        """Train Q-Learning agent."""
        success_rate = []

        for episode in range(num_episodes):
            state = self.env.reset()
            if isinstance(state, tuple):
                state = state[0]  # Handle new gym API

            done = False
            success = 0

            while not done:
                # Choose and take action
                action = self.choose_action(state)
                result = self.env.step(action)

                if len(result) == 5:  # New gym API
                    next_state, reward, terminated, truncated, _ = result
                    done = terminated or truncated
                else:  # Old gym API
                    next_state, reward, done, _ = result

                # Q-Learning update
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * np.max(self.Q[next_state])

                self.Q[state, action] += self.alpha * (target - self.Q[state, action])

                state = next_state

                if reward > 0:
                    success = 1

            success_rate.append(success)

            # Decay epsilon
            self.epsilon = max(0.01, self.epsilon * 0.9995)

            if episode % 1000 == 0:
                avg_success = np.mean(success_rate[-1000:]) if len(success_rate) >= 1000 else np.mean(success_rate)
                print(f"Episode {episode}, Success Rate: {avg_success:.3f}, ε: {self.epsilon:.3f}")

        return success_rate


# Train on FrozenLake
agent_frozen = QLearningGym(env_frozen, alpha=0.1, gamma=0.99, epsilon=1.0)
success_rates = agent_frozen.train(num_episodes=10000)

print(f"\nFinal Success Rate: {np.mean(success_rates[-1000:]):.3f}")

# Visualize learned Q-table
print("\nLearned Q-table (first 10 states):")
print(agent_frozen.Q[:10])
```

### Taxi Environment

```python
# Taxi-v3 Environment
print("\n=== Taxi-v3 ===")
env_taxi = gym.make('Taxi-v3')
print(f"State space: {env_taxi.observation_space}")
print(f"Action space: {env_taxi.action_space}")

# Train Q-Learning on Taxi
agent_taxi = QLearningGym(env_taxi, alpha=0.1, gamma=0.99, epsilon=1.0)
rewards_taxi = agent_taxi.train(num_episodes=10000)

# Test learned policy
state = env_taxi.reset()
if isinstance(state, tuple):
    state = state[0]

print("\nTesting learned policy on Taxi:")
total_reward = 0
done = False
steps = 0

while not done and steps < 100:
    action = np.argmax(agent_taxi.Q[state])
    result = env_taxi.step(action)

    if len(result) == 5:
        state, reward, terminated, truncated, _ = result
        done = terminated or truncated
    else:
        state, reward, done, _ = result

    total_reward += reward
    steps += 1

print(f"Total reward: {total_reward}, Steps: {steps}")
env_taxi.close()
```

---

## 11. Real-World Applications

### Robot Navigation

```python
"""
Real-World Application: Warehouse Robot Navigation

Problem: Robot must navigate warehouse to pick up items.
- States: Robot position (x, y, has_item)
- Actions: Move up/down/left/right, pick, drop
- Rewards: +10 for delivery, -0.1 per step, -5 for collision

Solution: Q-Learning learns optimal navigation policy!
"""

class WarehouseEnv:
    """Simplified warehouse navigation environment."""
    def __init__(self, size=10):
        self.size = size
        self.robot_pos = [0, 0]
        self.item_pos = [8, 8]
        self.delivery_pos = [0, 9]
        self.has_item = False

        # Actions: 0=up, 1=right, 2=down, 3=left, 4=pick, 5=drop
        self.actions = list(range(6))

    def reset(self):
        self.robot_pos = [0, 0]
        self.has_item = False
        return self._get_state()

    def _get_state(self):
        """State: (x, y, has_item)"""
        return (self.robot_pos[0], self.robot_pos[1], int(self.has_item))

    def step(self, action):
        reward = -0.1  # Time penalty
        done = False

        if action == 0:  # up
            self.robot_pos[0] = max(0, self.robot_pos[0] - 1)
        elif action == 1:  # right
            self.robot_pos[1] = min(self.size - 1, self.robot_pos[1] + 1)
        elif action == 2:  # down
            self.robot_pos[0] = min(self.size - 1, self.robot_pos[0] + 1)
        elif action == 3:  # left
            self.robot_pos[1] = max(0, self.robot_pos[1] - 1)
        elif action == 4:  # pick
            if self.robot_pos == self.item_pos and not self.has_item:
                self.has_item = True
                reward = 1.0
        elif action == 5:  # drop
            if self.robot_pos == self.delivery_pos and self.has_item:
                reward = 10.0
                done = True

        return self._get_state(), reward, done


# Train robot with Q-Learning
warehouse = WarehouseEnv(size=10)

# Use Q-Learning implementation from before
# (simplified here for brevity)
print("Robot learns to navigate warehouse using Q-Learning!")
print("After training, robot can pick and deliver items efficiently.")
```

### Game AI

```python
"""
Real-World Application: Game AI

Problem: Learn to play simple games optimally.
- States: Game board configuration
- Actions: Valid moves
- Rewards: Win (+1), Lose (-1), Draw (0)

Solution: Self-play with Q-Learning/SARSA creates strong AI!
"""

# Example: Tic-Tac-Toe AI would use tabular Q-Learning
# For more complex games, we'd use DQN (Lesson 3!)

print("\nTabular methods are perfect for:")
print("✅ Small state/action spaces")
print("✅ Discrete environments")
print("✅ Exact value computation")
print("✅ Guaranteed convergence (under right conditions)")
```

---

## Practice Exercises

### Exercise 1: Implement Double Q-Learning

```python
"""
Implement Double Q-Learning to reduce overestimation bias.

Hints:
- Maintain two Q-tables: Q1 and Q2
- For each update, randomly choose which to update
- Use one Q-table to select action, other to evaluate it
- Q1(s,a) ← Q1(s,a) + α[R + γ * Q2(s', argmax_a' Q1(s',a')) - Q1(s,a)]

Why? Standard Q-Learning overestimates values due to max operator.
"""

# Your implementation here
```

### Exercise 2: Cliff Walking Environment

```python
"""
Create and solve the Cliff Walking environment (Sutton & Barto Example 6.6).

Environment:
- 4x12 grid
- Start: bottom-left
- Goal: bottom-right
- Cliff: bottom row (except start and goal)
- Actions: up, right, down, left
- Rewards: -1 per step, -100 for falling off cliff

Compare SARSA vs Q-Learning:
- SARSA should learn safer path (on-policy)
- Q-Learning should learn riskier but optimal path (off-policy)
"""

# Your implementation here
```

### Exercise 3: Eligibility Traces

```python
"""
Implement SARSA(λ) with eligibility traces.

Hints:
- Maintain eligibility trace e(s,a) for each state-action pair
- e(s,a) ← γλe(s,a) for all (s,a)
- e(s,a) ← e(s,a) + 1 for visited (s,a)
- Update all Q-values proportional to their eligibility:
  Q(s,a) ← Q(s,a) + α * δ * e(s,a)  for all (s,a)

Where δ = R + γQ(s',a') - Q(s,a) is the TD error.

Eligibility traces provide faster learning!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Markov Decision Processes** 🎯
   - Framework for sequential decision-making
   - States, actions, rewards, transitions, discount factor
   - Markov property: Future depends only on present
   - Return: Cumulative discounted reward

2. **Bellman Equations** 📊
   - Foundation of dynamic programming and RL
   - Bellman Expectation: For policy evaluation
   - Bellman Optimality: For finding optimal policy
   - Recursive relationship between V(s) and V(s')

3. **Dynamic Programming** ⚡
   - Value Iteration: Directly compute V*
   - Policy Iteration: Alternate evaluation and improvement
   - Requires full model of environment
   - Guaranteed convergence to optimal policy

4. **Monte Carlo Methods** 🎲
   - Learn from complete episodes
   - Model-free: Don't need transition dynamics
   - High variance, unbiased estimates
   - First-visit vs every-visit variants

5. **Temporal Difference Learning** 🚀
   - Learn online (after each step)
   - Bootstrap from current estimates
   - Lower variance than MC, biased initially
   - Q-Learning (off-policy) vs SARSA (on-policy)

6. **Q-Learning vs SARSA** ⚔️
   - Q-Learning: Learns optimal policy, uses max
   - SARSA: Learns followed policy, uses actual action
   - Q-Learning better for learning optimality
   - SARSA safer during training

### When to Use What?

✅ **Value/Policy Iteration**: Small MDPs, have full model
✅ **Monte Carlo**: Episodic tasks, model-free
✅ **Q-Learning**: Want optimal policy, can explore safely
✅ **SARSA**: Online learning, safety matters
✅ **Expected SARSA**: Want lower variance than SARSA
✅ **n-Step**: Trade off bias and variance

### Real-World Applications

✅ **Robot Navigation**: Warehouse robots, delivery drones
✅ **Game AI**: Board games, video games, puzzles
✅ **Resource Allocation**: Network routing, load balancing
✅ **Control Systems**: HVAC, traffic lights, inventory

### What's Next?

In Lesson 3, we'll move beyond tabular methods to **Deep Q-Networks**:
- Function approximation with neural networks
- Experience replay and target networks
- Solving high-dimensional problems (Atari games!)
- Rainbow DQN and state-of-the-art improvements

**Tabular methods are the foundation - master them before deep RL!** 🚀

---

## Additional Resources

### Papers
- Sutton & Barto (2018): "Reinforcement Learning: An Introduction" (THE textbook!)
- Watkins (1989): "Learning from Delayed Rewards" (Q-Learning)
- Rummery & Niranjan (1994): "On-Line Q-Learning Using Connectionist Systems" (SARSA)
- Van Seijen et al. (2009): "A Theoretical and Empirical Analysis of Expected Sarsa"

### Books
- Sutton & Barto: "Reinforcement Learning: An Introduction" (2nd Ed)
- Szepesvári: "Algorithms for Reinforcement Learning"

### Code & Tools
- **OpenAI Gym**: Classic RL environments
- **Gymnasium**: Maintained fork of Gym
- **RL Book Code**: http://incompleteideas.net/book/code/

### Interactive Resources
- **Gridworld Playground**: http://cs.stanford.edu/people/karpathy/reinforcejs/
- **Spinning Up**: https://spinningup.openai.com/ (OpenAI's RL resource)

---

**Next**: [Lesson 3 - Deep Q-Networks and Value-Based Methods](Lesson%203%20-%20Deep%20Q-Networks%20and%20Value-Based%20Methods.md)

Proceed to learn about **function approximation** and **deep reinforcement learning**! 🧠

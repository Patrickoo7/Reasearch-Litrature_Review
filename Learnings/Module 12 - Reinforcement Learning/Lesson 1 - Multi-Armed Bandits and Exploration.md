# Lesson 1: Multi-Armed Bandits & Exploration Fundamentals 🎰

**Module 12: Reinforcement Learning | Lesson 1 of 12**

Master the foundation of exploration vs exploitation - the core dilemma in all of RL!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the exploration vs exploitation tradeoff
2. ✅ Implement ε-greedy, UCB, and Thompson Sampling algorithms
3. ✅ Apply bandits to real-world problems (A/B testing, recommendations)
4. ✅ Understand contextual bandits as a bridge to full RL
5. ✅ Analyze regret bounds and convergence

---

## 1. The Multi-Armed Bandit Problem

### What is a Bandit Problem?

Imagine a casino with multiple slot machines (bandits), each with unknown payout probabilities. You want to **maximize your total winnings** over time. The challenge:
- **Exploration**: Try different machines to find the best one
- **Exploitation**: Play the machine you currently think is best

This is the **exploration-exploitation tradeoff** - fundamental to ALL of reinforcement learning!

### Formal Definition

```python
import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit:
    """
    K-armed bandit problem.

    Each arm has a true mean reward (unknown to agent).
    Agent must balance exploration vs exploitation.
    """
    def __init__(self, k=10, true_means=None):
        self.k = k  # Number of arms

        if true_means is None:
            # Random true means for each arm
            self.true_means = np.random.randn(k)
        else:
            self.true_means = np.array(true_means)

        # Best arm (for evaluation)
        self.optimal_arm = np.argmax(self.true_means)

    def pull(self, arm):
        """
        Pull an arm and get a noisy reward.

        Reward = true_mean + Gaussian noise
        """
        reward = self.true_means[arm] + np.random.randn()
        return reward

    def reset(self):
        """Reset for new experiment."""
        pass


# Example: Create a 10-armed bandit
bandit = MultiArmedBandit(k=10)

print("True means of each arm:")
print(bandit.true_means)
print(f"\nOptimal arm: {bandit.optimal_arm}")
print(f"Optimal mean: {bandit.true_means[bandit.optimal_arm]:.3f}")

# Pull each arm once
print("\nSingle pulls from each arm:")
for arm in range(bandit.k):
    reward = bandit.pull(arm)
    print(f"Arm {arm}: reward = {reward:.3f}")
```

**Key Concepts:**
- **Arms (Actions)**: Different choices available
- **Reward**: Feedback from choosing an arm
- **Stationary**: Reward distributions don't change over time
- **Regret**: Difference between optimal and actual cumulative reward

---

## 2. ε-Greedy Strategy

The simplest exploration strategy: with probability ε, explore randomly; otherwise, exploit the best arm.

### Implementation from Scratch

```python
class EpsilonGreedy:
    """
    ε-greedy algorithm for multi-armed bandits.

    With probability ε: choose random arm (explore)
    With probability 1-ε: choose best arm (exploit)
    """
    def __init__(self, k, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon

        # Estimated value for each arm
        self.Q = np.zeros(k)

        # Number of times each arm was pulled
        self.N = np.zeros(k)

    def select_arm(self):
        """Select an arm using ε-greedy strategy."""
        if np.random.rand() < self.epsilon:
            # Explore: random arm
            return np.random.randint(self.k)
        else:
            # Exploit: best arm so far
            return np.argmax(self.Q)

    def update(self, arm, reward):
        """Update estimates after observing reward."""
        self.N[arm] += 1

        # Incremental mean update
        # Q_new = Q_old + (1/N) * (reward - Q_old)
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


def run_experiment(bandit, agent, num_steps=1000):
    """Run bandit experiment and track performance."""
    rewards = np.zeros(num_steps)
    optimal_actions = np.zeros(num_steps)

    for t in range(num_steps):
        # Select arm
        arm = agent.select_arm()

        # Pull arm and observe reward
        reward = bandit.pull(arm)

        # Update agent
        agent.update(arm, reward)

        # Track performance
        rewards[t] = reward
        optimal_actions[t] = (arm == bandit.optimal_arm)

    return rewards, optimal_actions


# Run experiment
np.random.seed(42)
bandit = MultiArmedBandit(k=10)
agent = EpsilonGreedy(k=10, epsilon=0.1)

rewards, optimal_actions = run_experiment(bandit, agent, num_steps=1000)

print(f"Average reward: {np.mean(rewards):.3f}")
print(f"Optimal action rate: {np.mean(optimal_actions):.3f}")
print(f"\nLearned Q-values:")
print(agent.Q)
print(f"\nTrue means:")
print(bandit.true_means)
```

### Comparing Different ε Values

```python
# Compare ε = 0 (greedy), 0.01, 0.1, 0.3
epsilons = [0, 0.01, 0.1, 0.3]
num_runs = 100
num_steps = 1000

results = {}

for eps in epsilons:
    all_rewards = []

    for run in range(num_runs):
        bandit = MultiArmedBandit(k=10)
        agent = EpsilonGreedy(k=10, epsilon=eps)
        rewards, _ = run_experiment(bandit, agent, num_steps)
        all_rewards.append(rewards)

    # Average across runs
    results[eps] = np.mean(all_rewards, axis=0)

# Plot results
plt.figure(figsize=(10, 6))
for eps in epsilons:
    plt.plot(results[eps], label=f'ε = {eps}')

plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('ε-Greedy: Effect of Exploration Rate')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

**Key Insights:**
- **ε = 0** (pure exploitation): Gets stuck on suboptimal arms
- **ε = 0.1**: Good balance for stationary problems
- **ε = 0.3**: Too much exploration, wastes pulls

---

## 3. Upper Confidence Bound (UCB)

Instead of random exploration, UCB uses **optimism under uncertainty**: prefer arms with high uncertainty!

### The UCB Algorithm

```python
class UCB:
    """
    Upper Confidence Bound (UCB) algorithm.

    Select arm with highest: Q(a) + c * sqrt(ln(t) / N(a))

    - Q(a): Estimated value
    - N(a): Times arm was pulled
    - t: Total time steps
    - c: Exploration constant (typically c=2)
    """
    def __init__(self, k, c=2.0):
        self.k = k
        self.c = c

        self.Q = np.zeros(k)
        self.N = np.zeros(k)
        self.t = 0

    def select_arm(self):
        """Select arm with highest UCB."""
        self.t += 1

        # First, pull each arm once
        if self.t <= self.k:
            return self.t - 1

        # Calculate UCB for each arm
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)

        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """Update estimates."""
        self.N[arm] += 1
        self.Q[arm] += (reward - self.Q[arm]) / self.N[arm]


# Compare ε-greedy vs UCB
np.random.seed(42)
num_runs = 100
num_steps = 1000

eps_greedy_rewards = []
ucb_rewards = []

for run in range(num_runs):
    bandit = MultiArmedBandit(k=10)

    # ε-greedy
    agent1 = EpsilonGreedy(k=10, epsilon=0.1)
    rewards1, _ = run_experiment(bandit, agent1, num_steps)
    eps_greedy_rewards.append(rewards1)

    # UCB
    bandit2 = MultiArmedBandit(k=10, true_means=bandit.true_means)
    agent2 = UCB(k=10, c=2.0)
    rewards2, _ = run_experiment(bandit2, agent2, num_steps)
    ucb_rewards.append(rewards2)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(np.mean(eps_greedy_rewards, axis=0), label='ε-greedy (ε=0.1)')
plt.plot(np.mean(ucb_rewards, axis=0), label='UCB (c=2)')
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('ε-Greedy vs UCB')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cumulative_eps = np.cumsum(np.mean(eps_greedy_rewards, axis=0))
cumulative_ucb = np.cumsum(np.mean(ucb_rewards, axis=0))
plt.plot(cumulative_eps, label='ε-greedy')
plt.plot(cumulative_ucb, label='UCB')
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**UCB Advantages:**
✅ No hyperparameter tuning (c=2 works well)
✅ Theoretically principled (logarithmic regret bound)
✅ Automatically balances exploration/exploitation
✅ Explores less over time (as uncertainty decreases)

---

## 4. Thompson Sampling (Bayesian Approach)

Maintain a probability distribution over each arm's mean reward. Sample from distributions to select arms!

### Implementation with Beta Distribution

```python
class ThompsonSampling:
    """
    Thompson Sampling for Bernoulli bandits.

    Assumes rewards are 0 or 1 (binary).
    Uses Beta distribution as posterior.
    """
    def __init__(self, k):
        self.k = k

        # Beta distribution parameters (α, β)
        # Start with uniform prior: Beta(1, 1)
        self.alpha = np.ones(k)
        self.beta = np.ones(k)

    def select_arm(self):
        """Sample from each arm's posterior and pick best."""
        # Sample from Beta(α, β) for each arm
        samples = np.random.beta(self.alpha, self.beta)

        return np.argmax(samples)

    def update(self, arm, reward):
        """Update Beta distribution based on reward."""
        if reward >= 0.5:  # Success
            self.alpha[arm] += 1
        else:  # Failure
            self.beta[arm] += 1


class BernoulliBandit:
    """Bandit with Bernoulli (binary) rewards."""
    def __init__(self, k, true_probs=None):
        self.k = k

        if true_probs is None:
            self.true_probs = np.random.rand(k)
        else:
            self.true_probs = np.array(true_probs)

        self.optimal_arm = np.argmax(self.true_probs)

    def pull(self, arm):
        """Return 1 with probability p, else 0."""
        return float(np.random.rand() < self.true_probs[arm])


# Run Thompson Sampling
np.random.seed(42)
bandit = BernoulliBandit(k=5, true_probs=[0.3, 0.5, 0.7, 0.4, 0.6])
agent = ThompsonSampling(k=5)

print("True probabilities:", bandit.true_probs)
print("Optimal arm:", bandit.optimal_arm)

# Run experiment
num_steps = 500
rewards = []
arm_counts = np.zeros(5)

for t in range(num_steps):
    arm = agent.select_arm()
    reward = bandit.pull(arm)
    agent.update(arm, reward)

    rewards.append(reward)
    arm_counts[arm] += 1

print(f"\nAverage reward: {np.mean(rewards):.3f}")
print(f"Arm selection counts: {arm_counts}")
print(f"\nLearned means (α/(α+β)):")
print(agent.alpha / (agent.alpha + agent.beta))
```

### Thompson Sampling for Gaussian Rewards

```python
class ThompsonSamplingGaussian:
    """
    Thompson Sampling for Gaussian bandits.

    Assumes rewards ~ N(μ, σ²).
    Uses Gaussian prior with known variance.
    """
    def __init__(self, k, sigma=1.0):
        self.k = k
        self.sigma = sigma

        # Prior: N(0, 1)
        self.mu = np.zeros(k)
        self.precision = np.ones(k)  # 1/variance

    def select_arm(self):
        """Sample from each arm's posterior."""
        samples = np.random.normal(
            self.mu,
            1.0 / np.sqrt(self.precision)
        )
        return np.argmax(samples)

    def update(self, arm, reward):
        """Bayesian update of Gaussian posterior."""
        # Update precision (inverse variance)
        self.precision[arm] += 1.0 / (self.sigma ** 2)

        # Update mean
        self.mu[arm] = (self.mu[arm] * (self.precision[arm] - 1.0 / (self.sigma ** 2)) +
                       reward / (self.sigma ** 2)) / self.precision[arm]


# Compare all three algorithms
np.random.seed(42)
num_runs = 50
num_steps = 1000

algorithms = {
    'ε-greedy': lambda: EpsilonGreedy(k=10, epsilon=0.1),
    'UCB': lambda: UCB(k=10, c=2.0),
    'Thompson': lambda: ThompsonSamplingGaussian(k=10, sigma=1.0)
}

results = {name: [] for name in algorithms}

for run in range(num_runs):
    bandit_base = MultiArmedBandit(k=10)

    for name, agent_fn in algorithms.items():
        bandit = MultiArmedBandit(k=10, true_means=bandit_base.true_means)
        agent = agent_fn()
        rewards, _ = run_experiment(bandit, agent, num_steps)
        results[name].append(rewards)

# Plot comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name in algorithms:
    avg_rewards = np.mean(results[name], axis=0)
    plt.plot(avg_rewards, label=name)
plt.xlabel('Steps')
plt.ylabel('Average Reward')
plt.title('Algorithm Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
for name in algorithms:
    cumulative = np.cumsum(np.mean(results[name], axis=0))
    plt.plot(cumulative, label=name)
plt.xlabel('Steps')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Performance')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Thompson Sampling Advantages:**
✅ Often outperforms ε-greedy and UCB
✅ Naturally balances exploration/exploitation
✅ Bayesian framework (principled uncertainty)
✅ Works well in practice

---

## 5. Contextual Bandits

**Bridge to full RL:** Bandit + state information = Contextual Bandit

### Problem Definition

In contextual bandits:
- **Context (state)**: Additional information available (e.g., user features)
- **Action**: Choose arm based on context
- **Reward**: Depends on context and action

```python
class ContextualBandit:
    """
    Contextual bandit with linear rewards.

    Reward(context, action) = context^T * theta_action + noise
    """
    def __init__(self, num_actions, context_dim):
        self.num_actions = num_actions
        self.context_dim = context_dim

        # True parameters for each action
        self.theta = np.random.randn(num_actions, context_dim)

    def get_context(self):
        """Generate random context."""
        return np.random.randn(self.context_dim)

    def pull(self, context, action):
        """Get reward for (context, action) pair."""
        mean_reward = np.dot(context, self.theta[action])
        reward = mean_reward + 0.1 * np.random.randn()
        return reward

    def optimal_action(self, context):
        """Return best action for given context."""
        expected_rewards = [np.dot(context, self.theta[a])
                          for a in range(self.num_actions)]
        return np.argmax(expected_rewards)


class LinUCB:
    """
    Linear UCB for contextual bandits.

    Maintains linear model for each action.
    Uses confidence bounds for exploration.
    """
    def __init__(self, num_actions, context_dim, alpha=1.0):
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.alpha = alpha

        # For each action: A_a = I, b_a = 0
        self.A = [np.identity(context_dim) for _ in range(num_actions)]
        self.b = [np.zeros(context_dim) for _ in range(num_actions)]

    def select_action(self, context):
        """Select action using UCB."""
        ucb_values = []

        for a in range(self.num_actions):
            # Solve for theta: A * theta = b
            A_inv = np.linalg.inv(self.A[a])
            theta_a = A_inv.dot(self.b[a])

            # UCB = expected reward + confidence bonus
            expected = theta_a.dot(context)
            confidence = self.alpha * np.sqrt(context.dot(A_inv).dot(context))

            ucb_values.append(expected + confidence)

        return np.argmax(ucb_values)

    def update(self, action, context, reward):
        """Update model for selected action."""
        self.A[action] += np.outer(context, context)
        self.b[action] += reward * context


# Run LinUCB experiment
np.random.seed(42)
bandit = ContextualBandit(num_actions=5, context_dim=10)
agent = LinUCB(num_actions=5, context_dim=10, alpha=0.5)

num_steps = 1000
rewards = []
optimal_actions = []

for t in range(num_steps):
    # Get context
    context = bandit.get_context()

    # Select action
    action = agent.select_action(context)

    # Get reward
    reward = bandit.pull(context, action)

    # Update agent
    agent.update(action, context, reward)

    # Track performance
    rewards.append(reward)
    optimal = bandit.optimal_action(context)
    optimal_actions.append(action == optimal)

print(f"Average reward: {np.mean(rewards):.3f}")
print(f"Optimal action rate: {np.mean(optimal_actions):.3f}")

# Plot learning curve
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
window = 50
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
plt.plot(moving_avg)
plt.xlabel('Steps')
plt.ylabel('Average Reward (50-step window)')
plt.title('LinUCB Learning Curve')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
moving_opt = np.convolve(optimal_actions, np.ones(window)/window, mode='valid')
plt.plot(moving_opt)
plt.xlabel('Steps')
plt.ylabel('Optimal Action Rate')
plt.title('Convergence to Optimal Policy')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 6. Real-World Applications

### Application 1: A/B Testing for Web Optimization

```python
class ABTestingBandit:
    """
    A/B testing framework using bandits.

    Goal: Find best website variant while minimizing regret.
    """
    def __init__(self, variants, true_conversion_rates):
        self.variants = variants
        self.true_rates = true_conversion_rates
        self.num_variants = len(variants)

    def show_variant(self, variant_id):
        """Show variant and get conversion (0 or 1)."""
        return float(np.random.rand() < self.true_rates[variant_id])


# Example: Testing 3 website designs
variants = ['Design A', 'Design B', 'Design C']
true_rates = [0.05, 0.08, 0.06]  # True conversion rates

ab_test = ABTestingBandit(variants, true_rates)
agent = ThompsonSampling(k=3)

num_visitors = 5000
conversions_per_variant = np.zeros(3)
visitors_per_variant = np.zeros(3)

for visitor in range(num_visitors):
    # Select variant using Thompson Sampling
    variant = agent.select_arm()

    # Show variant and observe conversion
    converted = ab_test.show_variant(variant)

    # Update agent
    agent.update(variant, converted)

    # Track stats
    visitors_per_variant[variant] += 1
    conversions_per_variant[variant] += converted

# Results
print("A/B Testing Results:")
print("="*50)
for i, var in enumerate(variants):
    observed_rate = conversions_per_variant[i] / visitors_per_variant[i]
    print(f"{var}:")
    print(f"  Visitors: {int(visitors_per_variant[i])}")
    print(f"  Conversions: {int(conversions_per_variant[i])}")
    print(f"  Observed rate: {observed_rate:.3f}")
    print(f"  True rate: {true_rates[i]:.3f}")
    print()

print(f"Best variant: {variants[np.argmax(true_rates)]}")
print(f"Most shown variant: {variants[np.argmax(visitors_per_variant)]}")
```

### Application 2: News Article Recommendation

```python
class NewsRecommendation:
    """
    News article recommendation using contextual bandits.

    Context: User features (age, location, interests)
    Actions: Which article to show
    Reward: Click-through rate
    """
    def __init__(self):
        # Simplified: 5 articles, 3 user features
        self.num_articles = 5
        self.context_dim = 3

        # Each article has affinity with user features
        self.article_weights = np.random.randn(self.num_articles, self.context_dim)

    def get_user_context(self):
        """Generate user features."""
        # [age_group, location_preference, topic_interest]
        return np.random.randn(self.context_dim)

    def show_article(self, user_context, article_id):
        """Show article and observe click (1) or no click (0)."""
        # Click probability based on user-article match
        match_score = np.dot(user_context, self.article_weights[article_id])
        click_prob = 1.0 / (1.0 + np.exp(-match_score))  # Sigmoid

        return float(np.random.rand() < click_prob)


# Run news recommendation
np.random.seed(42)
news_system = NewsRecommendation()
agent = LinUCB(num_articles=5, context_dim=3, alpha=1.0)

num_users = 2000
clicks = []
ctr_over_time = []

for user in range(num_users):
    # Get user context
    context = news_system.get_user_context()

    # Recommend article
    article = agent.select_action(context)

    # Observe click
    clicked = news_system.show_article(context, article)

    # Update model
    agent.update(article, context, clicked)

    # Track performance
    clicks.append(clicked)
    if (user + 1) % 50 == 0:
        recent_ctr = np.mean(clicks[-50:])
        ctr_over_time.append(recent_ctr)

print(f"Final CTR: {np.mean(clicks):.3f}")

plt.figure(figsize=(10, 6))
plt.plot(ctr_over_time)
plt.xlabel('Time (batches of 50 users)')
plt.ylabel('Click-Through Rate')
plt.title('News Recommendation: CTR Over Time')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 7. Regret Analysis

### Understanding Regret

**Regret** = Cumulative reward of optimal strategy - Cumulative reward of our strategy

```python
def calculate_regret(bandit, rewards, optimal_actions):
    """
    Calculate cumulative regret over time.

    Regret at time t = optimal_reward - actual_reward
    """
    optimal_reward = bandit.true_means[bandit.optimal_arm]

    regret = optimal_reward - rewards
    cumulative_regret = np.cumsum(regret)

    return regret, cumulative_regret


# Compare regret for different algorithms
np.random.seed(42)
num_runs = 100
num_steps = 2000

algorithms = {
    'ε-greedy (ε=0.1)': lambda: EpsilonGreedy(k=10, epsilon=0.1),
    'ε-greedy (ε=0.01)': lambda: EpsilonGreedy(k=10, epsilon=0.01),
    'UCB': lambda: UCB(k=10, c=2.0),
    'Thompson Sampling': lambda: ThompsonSamplingGaussian(k=10, sigma=1.0)
}

regret_results = {name: [] for name in algorithms}

for run in range(num_runs):
    bandit_base = MultiArmedBandit(k=10)

    for name, agent_fn in algorithms.items():
        bandit = MultiArmedBandit(k=10, true_means=bandit_base.true_means)
        agent = agent_fn()
        rewards, optimal_actions = run_experiment(bandit, agent, num_steps)
        _, cum_regret = calculate_regret(bandit, rewards, optimal_actions)
        regret_results[name].append(cum_regret)

# Plot cumulative regret
plt.figure(figsize=(12, 6))

for name in algorithms:
    avg_regret = np.mean(regret_results[name], axis=0)
    plt.plot(avg_regret, label=name, linewidth=2)

plt.xlabel('Steps', fontsize=12)
plt.ylabel('Cumulative Regret', fontsize=12)
plt.title('Regret Comparison: Different Bandit Algorithms', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

# Print final regret
print("Final Cumulative Regret (average over 100 runs):")
for name in algorithms:
    final_regret = np.mean([r[-1] for r in regret_results[name]])
    print(f"{name:30s}: {final_regret:.2f}")
```

**Theoretical Regret Bounds:**
- **ε-greedy**: Linear regret O(T)
- **UCB**: Logarithmic regret O(log T)
- **Thompson Sampling**: Logarithmic regret O(log T)

UCB and Thompson Sampling are **asymptotically optimal**!

---

## Practice Exercises

### Exercise 1: Implement Gradient Bandit Algorithm

```python
"""
Implement softmax (Boltzmann) exploration using gradient updates.

Hints:
- Maintain preference H(a) for each arm
- Action probabilities: π(a) = exp(H(a)) / Σ exp(H(a'))
- Update: H(a) ← H(a) + α(R - R_avg)(1 - π(a))  if a was selected
"""

# Your implementation here
```

### Exercise 2: Non-Stationary Bandits

```python
"""
Modify ε-greedy to handle non-stationary bandits (changing means).

Hints:
- Use exponentially weighted average instead of simple average
- Q(a) ← Q(a) + α(R - Q(a))  with constant α (e.g., α=0.1)
- This gives more weight to recent rewards
"""

# Your implementation here
```

### Exercise 3: Clinical Trial Optimization

```python
"""
Design a bandit algorithm for adaptive clinical trials.

Scenario:
- 4 treatment options
- Want to maximize patient outcomes
- Ethical constraint: Minimize patients on bad treatments

Which algorithm would you use and why?
Implement and compare ε-greedy, UCB, and Thompson Sampling.
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Exploration vs Exploitation** 🎯
   - Core tradeoff in ALL of reinforcement learning
   - Pure exploitation: Get stuck on suboptimal actions
   - Pure exploration: Waste time on bad actions
   - Need balance: Explore early, exploit later

2. **ε-Greedy** 🎲
   - Simplest approach: random exploration with probability ε
   - Easy to implement and understand
   - Linear regret (not optimal)
   - Good starting point for any RL problem

3. **UCB (Upper Confidence Bound)** 📊
   - Optimism under uncertainty
   - Automatically reduces exploration over time
   - Logarithmic regret (asymptotically optimal)
   - No hyperparameter tuning needed

4. **Thompson Sampling** 🎰
   - Bayesian approach: Maintain posterior distributions
   - Often best empirical performance
   - Logarithmic regret (asymptotically optimal)
   - Elegant and principled

5. **Contextual Bandits** 🌉
   - Bridge between bandits and full RL
   - Incorporate state/context information
   - LinUCB: Popular algorithm for linear rewards
   - Used in recommendation systems, A/B testing

6. **Regret Bounds** 📈
   - Measure of algorithm efficiency
   - Logarithmic regret = good algorithm
   - Linear regret = suboptimal
   - Theory guides algorithm design

### Real-World Applications

✅ **A/B Testing**: Website optimization, app features
✅ **Recommendation Systems**: News, videos, products
✅ **Online Advertising**: Ad selection, bidding strategies
✅ **Clinical Trials**: Adaptive treatment allocation
✅ **Resource Allocation**: Server load balancing, network routing

### What's Next?

In Lesson 2, we'll extend bandits to **Markov Decision Processes**:
- Sequential decisions (not just one-shot)
- States and state transitions
- Delayed rewards and credit assignment
- Dynamic programming and Q-Learning

**Bandits are the foundation - master them first!** 🚀

---

## Additional Resources

### Papers
- Lai & Robbins (1985): "Asymptotically Efficient Adaptive Allocation Rules"
- Auer et al. (2002): "Finite-time Analysis of the Multiarmed Bandit Problem"
- Thompson (1933): "On the Likelihood that One Unknown Probability Exceeds Another"
- Li et al. (2010): "A Contextual-Bandit Approach to Personalized News Article Recommendation"

### Libraries
- **scikit-learn**: BernoulliTS, ThompsonSampling (basic implementations)
- **vowpal_wabbit**: Industrial-strength contextual bandits
- **mabwiser**: Multi-Armed Bandit library (LinkedIn)

### Interactive Tools
- **OpenAI Gym**: `BanditTenArmedGaussian-v0` environment
- **RL Book Code**: http://incompleteideas.net/book/code/

---

**Next**: [Lesson 2 - RL Fundamentals & Tabular Methods](Lesson%202%20-%20RL%20Fundamentals%20and%20Tabular%20Methods.md)

Proceed to learn about **Markov Decision Processes** and **Q-Learning**! 🎯

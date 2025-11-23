# Lesson 8: Advanced Exploration & Intrinsic Motivation 🔍

**Module 12: Reinforcement Learning | Lesson 8 of 12**

Master curiosity-driven exploration for solving hard exploration problems with sparse rewards!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand limitations of ε-greedy and need for advanced exploration
2. ✅ Implement count-based exploration methods
3. ✅ Master curiosity-driven exploration (ICM, RND, NGU)
4. ✅ Learn prediction-based and information-theoretic exploration
5. ✅ Apply to sparse reward problems (Montezuma's Revenge, robotics)
6. ✅ Understand when exploration is critical

---

## 1. The Hard Exploration Problem

### Why ε-greedy Fails

ε-greedy works when rewards are dense. But many real tasks have **sparse rewards**!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
from collections import deque

"""
Hard Exploration Problems:

1. Sparse Rewards:
   - Reward only at goal (e.g., Montezuma's Revenge)
   - Need to explore extensively
   - Random exploration takes exponential time!

2. Deceptive Rewards:
   - Local optima mislead agent
   - Need sophisticated exploration

3. High-Dimensional State Spaces:
   - Random exploration has no chance
   - Need directed, intelligent exploration

Examples:
- Montezuma's Revenge: First reward after 100+ actions
- Robotics: Complex manipulation tasks
- Scientific discovery: Finding novel solutions
"""

print("=== Hard Exploration Problems ===")
print("\nWhy ε-greedy fails:")
print("❌ Random exploration in sparse reward environments")
print("❌ Exponential time to discover rare rewards")
print("❌ No memory of what's been explored")
print("❌ No notion of novelty or curiosity")

print("\nSolutions:")
print("✅ Count-based exploration")
print("✅ Curiosity-driven exploration")
print("✅ Prediction-based bonuses")
print("✅ Information-theoretic methods")
```

---

## 2. Count-Based Exploration

Bonus for visiting rarely-seen states!

```python
class CountBasedExploration:
    """
    Count-based exploration: Bonus for visiting rare states.

    Exploration bonus:
    r_intrinsic = β / sqrt(N(s))

    Where N(s) = number of visits to state s.

    Encourages visiting novel states!
    """
    def __init__(self, state_discretization=10, beta=0.1):
        self.counts = {}
        self.state_discretization = state_discretization
        self.beta = beta

    def discretize_state(self, state):
        """
        Discretize continuous state for counting.

        Simple approach: Round to nearest grid point.
        """
        discrete_state = tuple(np.round(state * self.state_discretization).astype(int))
        return discrete_state

    def get_bonus(self, state):
        """
        Compute exploration bonus.

        r_bonus = β / sqrt(count + 1)
        """
        discrete_state = self.discretize_state(state)

        count = self.counts.get(discrete_state, 0)
        bonus = self.beta / np.sqrt(count + 1)

        return bonus

    def update_count(self, state):
        """Update visit count."""
        discrete_state = self.discretize_state(state)
        self.counts[discrete_state] = self.counts.get(discrete_state, 0) + 1

    def get_total_reward(self, state, extrinsic_reward):
        """
        Total reward = Extrinsic + Intrinsic.
        """
        intrinsic_reward = self.get_bonus(state)
        total_reward = extrinsic_reward + intrinsic_reward

        # Update count
        self.update_count(state)

        return total_reward, intrinsic_reward


# Example usage
exploration = CountBasedExploration(beta=0.5)

state = np.array([0.1, 0.5, -0.3, 0.8])
extrinsic_reward = 0.0  # Sparse!

for visit in range(5):
    total_reward, intrinsic_reward = exploration.get_total_reward(state, extrinsic_reward)
    print(f"Visit {visit+1}: Total Reward = {total_reward:.3f}, "
          f"Intrinsic = {intrinsic_reward:.3f}")

print("\nCount-based exploration:")
print("✅ Bonus for novel states")
print("✅ Simple and interpretable")
print("✅ Works for tabular/discrete states")
print("❌ Hard to scale to high dimensions")
```

---

## 3. Intrinsic Curiosity Module (ICM)

Reward agent for encountering **surprising** transitions!

```python
class ICM(nn.Module):
    """
    Intrinsic Curiosity Module (ICM).

    Key idea: Reward prediction error of dynamics model.

    Components:
    1. Forward model: Predict s_{t+1} from s_t, a_t
    2. Inverse model: Predict a_t from s_t, s_{t+1}
    3. Feature network: Learn good state representation

    Intrinsic reward: r_i = ||φ(s_{t+1}) - f(φ(s_t), a_t)||²

    Where:
    - φ: Feature network
    - f: Forward model
    """
    def __init__(self, state_dim, action_dim, feature_dim=64):
        super(ICM, self).__init__()

        # Feature network: State → Feature embedding
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Forward model: (feature, action) → next_feature
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Inverse model: (feature, next_feature) → action
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state, action, next_state):
        """
        Compute intrinsic reward and losses.

        Returns:
            intrinsic_reward: Prediction error
            forward_loss: Forward model loss
            inverse_loss: Inverse model loss
        """
        # Encode states
        phi = self.feature_net(state)
        phi_next = self.feature_net(next_state)

        # Forward model prediction
        phi_next_pred = self.forward_model(torch.cat([phi, action], dim=-1))

        # Forward loss (intrinsic reward!)
        forward_loss = F.mse_loss(phi_next_pred, phi_next.detach(), reduction='none').mean(dim=-1)
        intrinsic_reward = forward_loss.detach()

        # Inverse model prediction
        action_pred = self.inverse_model(torch.cat([phi, phi_next], dim=-1))
        inverse_loss = F.mse_loss(action_pred, action)

        return intrinsic_reward, forward_loss.mean(), inverse_loss


class ICMAgent:
    """
    RL agent with ICM for exploration.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.icm = ICM(state_dim, action_dim)
        self.optimizer = optim.Adam(self.icm.parameters(), lr=lr)

        # Policy network (e.g., PPO)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward using ICM."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        with torch.no_grad():
            intrinsic_reward, _, _ = self.icm(state_tensor, action_tensor, next_state_tensor)

        return intrinsic_reward.item()

    def update_icm(self, batch):
        """Update ICM on batch of transitions."""
        states, actions, next_states = batch

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)

        # Compute losses
        intrinsic_rewards, forward_loss, inverse_loss = self.icm(states, actions, next_states)

        # Total ICM loss
        icm_loss = forward_loss + inverse_loss

        # Update
        self.optimizer.zero_grad()
        icm_loss.backward()
        self.optimizer.step()

        return icm_loss.item(), intrinsic_rewards.mean().item()


print("\nICM (Intrinsic Curiosity Module):")
print("✅ Reward prediction error")
print("✅ Learns feature representation")
print("✅ Scales to high dimensions")
print("✅ Used in Montezuma's Revenge")
```

---

## 4. Random Network Distillation (RND)

Reward unpredictability!

```python
class RND(nn.Module):
    """
    Random Network Distillation (RND).

    Key idea: Train predictor to match random target network.
    Prediction error = novelty!

    Components:
    1. Target network (fixed random): s → features
    2. Predictor network (trained): s → features

    Intrinsic reward: r_i = ||target(s) - predictor(s)||²

    Novel states: Hard to predict → High reward!
    """
    def __init__(self, state_dim, feature_dim=128):
        super(RND, self).__init__()

        # Target network (FIXED, random initialization)
        self.target_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

        # Freeze target network
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Predictor network (trained to match target)
        self.predictor_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, state):
        """
        Compute intrinsic reward.

        Returns:
            intrinsic_reward: Prediction error
        """
        # Target features (fixed)
        with torch.no_grad():
            target_features = self.target_net(state)

        # Predicted features
        predicted_features = self.predictor_net(state)

        # Prediction error = intrinsic reward
        intrinsic_reward = F.mse_loss(predicted_features, target_features, reduction='none').mean(dim=-1)

        return intrinsic_reward


class RNDAgent:
    """
    RL agent with RND for exploration.
    """
    def __init__(self, state_dim, lr=1e-3):
        self.rnd = RND(state_dim)
        self.optimizer = optim.Adam(self.rnd.predictor_net.parameters(), lr=lr)

        # Running statistics for reward normalization
        self.reward_mean = 0
        self.reward_std = 1
        self.reward_count = 0

    def compute_intrinsic_reward(self, state):
        """Compute intrinsic reward using RND."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            intrinsic_reward = self.rnd(state_tensor)

        return intrinsic_reward.item()

    def update_rnd(self, states):
        """Update RND predictor network."""
        states = torch.FloatTensor(states)

        # Compute prediction error
        intrinsic_rewards = self.rnd(states)

        # Loss: Mean prediction error
        loss = intrinsic_rewards.mean()

        # Update predictor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def normalize_reward(self, reward):
        """Normalize intrinsic reward for stability."""
        self.reward_count += 1
        old_mean = self.reward_mean

        self.reward_mean = old_mean + (reward - old_mean) / self.reward_count
        self.reward_std = np.sqrt((self.reward_std ** 2 * (self.reward_count - 1) +
                                   (reward - old_mean) * (reward - self.reward_mean)) /
                                  self.reward_count)

        normalized_reward = (reward - self.reward_mean) / (self.reward_std + 1e-8)
        return normalized_reward


print("\nRND (Random Network Distillation):")
print("✅ Simple and effective")
print("✅ No dynamics model needed")
print("✅ Scales well")
print("✅ Used in Atari, hard exploration games")
```

---

## 5. Never Give Up (NGU)

State-of-the-art exploration combining multiple ideas!

```python
"""
Never Give Up (NGU):

Combines:
1. Episodic novelty: Short-term curiosity
2. Lifelong novelty: Long-term curiosity (RND-like)
3. Intrinsic reward combination
4. Universal Value Function Approximators (UVFA)

Exploration bonus:
r_i = r_episodic * min(max(r_lifelong, 1), L)

Where:
- r_episodic: Bonus for novel states in current episode
- r_lifelong: Bonus for novel states across all time
- L: Clipping constant

NGU achieved superhuman performance on Montezuma's Revenge!
"""

class NGU:
    """
    Never Give Up (NGU) - Simplified version.

    Components:
    1. Episodic memory: Store recent states
    2. Episodic novelty: Bonus based on distance to past states
    3. Lifelong novelty: RND-based
    """
    def __init__(self, state_dim):
        # Episodic memory
        self.episodic_memory = deque(maxlen=1000)

        # RND for lifelong novelty
        self.rnd = RND(state_dim)

        # Embedding network
        self.embedding_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def compute_episodic_novelty(self, state):
        """
        Episodic novelty: Compare to recent states.

        r_episodic = 1 / sqrt(count_similar_states + 0.001)
        """
        if len(self.episodic_memory) == 0:
            return 1.0

        # Embed state
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            embedding = self.embedding_net(state_tensor).numpy()[0]

        # Find similar states (k-NN)
        distances = []
        for past_state in self.episodic_memory:
            past_state_tensor = torch.FloatTensor(past_state).unsqueeze(0)
            with torch.no_grad():
                past_embedding = self.embedding_net(past_state_tensor).numpy()[0]

            dist = np.linalg.norm(embedding - past_embedding)
            distances.append(dist)

        # Count similar states (distance < threshold)
        similar_count = sum(1 for d in distances if d < 0.5)

        # Episodic novelty
        episodic_novelty = 1.0 / np.sqrt(similar_count + 0.001)

        # Add to memory
        self.episodic_memory.append(state)

        return episodic_novelty

    def compute_lifelong_novelty(self, state):
        """Lifelong novelty using RND."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            lifelong_novelty = self.rnd(state_tensor).item()

        return lifelong_novelty

    def compute_intrinsic_reward(self, state):
        """
        Combined intrinsic reward.

        r_i = r_episodic * min(max(r_lifelong, 1), L)
        """
        episodic = self.compute_episodic_novelty(state)
        lifelong = self.compute_lifelong_novelty(state)

        # Clip lifelong between 1 and L
        L = 5.0
        lifelong_clipped = min(max(lifelong, 1.0), L)

        intrinsic_reward = episodic * lifelong_clipped

        return intrinsic_reward


print("\nNGU (Never Give Up):")
print("✅ Episodic + lifelong novelty")
print("✅ State-of-the-art exploration")
print("✅ Superhuman on Montezuma's Revenge")
print("✅ Combines multiple exploration ideas")
```

---

## 6. Information Theory-Based Exploration

Use information gain as exploration bonus!

```python
"""
Information-Theoretic Exploration:

1. VIME (Variational Information Maximizing Exploration):
   - Maximize information gain about dynamics
   - I(θ; s', a, s) where θ = dynamics parameters

2. Disagreement-Based:
   - Ensemble of models
   - High disagreement = high uncertainty
   - Explore to reduce disagreement

3. Entropy-Based (SAC):
   - Maximize policy entropy H(π)
   - Encourages stochastic exploration
"""

class DisagreementExploration:
    """
    Disagreement-based exploration with ensemble.

    Train ensemble of dynamics models.
    Intrinsic reward = disagreement among models.
    """
    def __init__(self, state_dim, action_dim, num_models=5):
        # Ensemble of dynamics models
        self.models = [
            nn.Sequential(
                nn.Linear(state_dim + action_dim, 128),
                nn.ReLU(),
                nn.Linear(128, state_dim)
            )
            for _ in range(num_models)
        ]

        self.optimizers = [
            optim.Adam(model.parameters(), lr=1e-3)
            for model in self.models
        ]

    def compute_disagreement(self, state, action):
        """
        Compute disagreement among ensemble predictions.

        High disagreement = high uncertainty = explore!
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        sa = torch.cat([state_tensor, action_tensor], dim=-1)

        # Predictions from all models
        predictions = []
        with torch.no_grad():
            for model in self.models:
                pred = model(sa)
                predictions.append(pred)

        predictions = torch.stack(predictions)  # [num_models, 1, state_dim]

        # Variance across models
        disagreement = predictions.var(dim=0).mean().item()

        return disagreement


print("\nInformation-Theoretic Exploration:")
print("✅ Information gain as objective")
print("✅ Uncertainty-driven")
print("✅ Ensemble disagreement")
print("✅ Principled approach")
```

---

## 7. Practical Implementation: PPO + ICM

Combine PPO with ICM for sparse reward tasks!

```python
class PPO_ICM:
    """
    PPO with Intrinsic Curiosity Module.

    Total reward: r_total = r_extrinsic + β * r_intrinsic
    """
    def __init__(self, state_dim, action_dim, beta=0.01):
        # PPO components (simplified)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.value = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ICM for intrinsic motivation
        self.icm = ICM(state_dim, action_dim)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)
        self.icm_optimizer = optim.Adam(self.icm.parameters(), lr=1e-3)

        self.beta = beta  # Intrinsic reward weight

    def collect_rollout(self, env, num_steps=2048):
        """Collect rollout with intrinsic rewards."""
        states = []
        actions = []
        extrinsic_rewards = []
        intrinsic_rewards = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(num_steps):
            # Select action
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.policy(state_tensor)

            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample().item()

            # Step environment
            result = env.step(action)
            if len(result) == 5:
                next_state, extrinsic_reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, extrinsic_reward, done, _ = result

            # Compute intrinsic reward
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.FloatTensor([action]).unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

            with torch.no_grad():
                intrinsic_reward, _, _ = self.icm(state_tensor, action_tensor, next_state_tensor)
                intrinsic_reward = intrinsic_reward.item()

            # Store
            states.append(state)
            actions.append(action)
            extrinsic_rewards.append(extrinsic_reward)
            intrinsic_rewards.append(intrinsic_reward)

            state = next_state

            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        # Combine rewards
        total_rewards = [e + self.beta * i for e, i in zip(extrinsic_rewards, intrinsic_rewards)]

        return states, actions, total_rewards, extrinsic_rewards, intrinsic_rewards

    def train(self, env, num_iterations=100):
        """Train PPO with ICM."""
        for iteration in range(num_iterations):
            # Collect rollout
            states, actions, total_rewards, ext_rewards, int_rewards = \
                self.collect_rollout(env)

            print(f"Iteration {iteration}, "
                  f"Avg Extrinsic: {np.mean(ext_rewards):.2f}, "
                  f"Avg Intrinsic: {np.mean(int_rewards):.4f}")

            # Update ICM (simplified here)
            # Update PPO policy and value (simplified here)


print("\nPPO + ICM:")
print("✅ Combine policy gradient with curiosity")
print("✅ Works on sparse reward tasks")
print("✅ Used successfully on Montezuma's Revenge")
```

---

## 8. Real-World Applications

```python
"""
Hard Exploration Applications:

1. Montezuma's Revenge:
   - Classic sparse reward Atari game
   - First reward after 100+ actions
   - ICM, RND, NGU achieved superhuman performance

2. Robotics:
   - Manipulation tasks with sparse rewards
   - Object rearrangement, assembly
   - Curiosity-driven exploration helps

3. Scientific Discovery:
   - Drug discovery: Explore molecular space
   - Materials science: Find novel materials
   - Intrinsic motivation drives innovation

4. Game Playing:
   - Explore game mechanics
   - Discover hidden features
   - Meta-learning across games
"""

print("\n=== Applications ===")
print("\n1. Sparse Reward Games:")
print("   - Montezuma's Revenge, Pitfall")
print("   - RND, NGU achieve superhuman")

print("\n2. Robotics:")
print("   - Sparse reward manipulation")
print("   - Exploration for novel solutions")

print("\n3. Scientific Discovery:")
print("   - Drug design, materials")
print("   - Curiosity-driven search")
```

---

## Practice Exercises

### Exercise 1: Implement ICM from Scratch

```python
"""
Implement full ICM and integrate with PPO.

Requirements:
1. Complete ICM implementation (forward + inverse models)
2. Integrate with PPO
3. Train on sparse reward environment (e.g., MountainCar with sparse rewards)
4. Compare: PPO alone vs PPO+ICM

Track:
- Intrinsic rewards over time
- Exploration coverage
- Final performance
"""

# Your implementation here
```

### Exercise 2: RND for Montezuma's Revenge

```python
"""
Implement RND and train on Montezuma's Revenge (or similar).

Requirements:
1. RND with target and predictor networks
2. PPO with intrinsic rewards
3. Reward normalization
4. Train for 100M frames

Compare:
- PPO vs PPO+RND
- Exploration coverage (rooms visited)
- Score progression

Visualize: Intrinsic rewards across environment
"""

# Your implementation here
```

### Exercise 3: Exploration Bonus Comparison

```python
"""
Compare different exploration methods on sparse reward task.

Methods:
1. ε-greedy
2. Count-based
3. ICM
4. RND
5. Ensemble disagreement

Environment: Custom sparse reward gridworld or MountainCar

Compare:
- Sample efficiency
- Final performance
- Exploration coverage

Visualize: Heatmap of state visitation
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Hard Exploration** 🎯
   - Sparse rewards common in real world
   - Random exploration insufficient
   - Need intelligent, directed exploration
   - Critical for real-world applications

2. **Count-Based** 📊
   - Bonus for rare states
   - Simple and interpretable
   - Hard to scale to high dimensions
   - Good baseline

3. **ICM** 🔬
   - Reward prediction error
   - Learns good features
   - Scales to high dimensions
   - First major success on Montezuma's

4. **RND** 🎲
   - Random target network
   - Simple and effective
   - No dynamics model needed
   - State-of-the-art results

5. **NGU** 🏆
   - Episodic + lifelong novelty
   - State-of-the-art exploration
   - Superhuman Montezuma's Revenge
   - Complex but powerful

6. **Information Theory** 📐
   - Uncertainty-driven exploration
   - Ensemble disagreement
   - Principled approach
   - Future of exploration

### When to Use Advanced Exploration?

✅ **Sparse rewards**: Rare, distant rewards
✅ **Large state spaces**: High-dimensional
✅ **Deceptive rewards**: Local optima
✅ **Scientific discovery**: Novel solutions
✅ **Robotics**: Complex manipulation

### Real-World Applications

✅ **Atari Games**: Montezuma's Revenge, Pitfall
✅ **Robotics**: Sparse reward manipulation
✅ **Scientific Discovery**: Drug design, materials
✅ **Game Playing**: Explore mechanics, hidden features

### What's Next?

In Lesson 9, we'll learn **RLHF and Aligning Language Models**:
- Reinforcement Learning from Human Feedback
- Training ChatGPT-style models
- Reward modeling from preferences
- PPO for LLM fine-tuning

**Advanced exploration unlocks hard problems - master it!** 🔍

---

## Additional Resources

### Papers
- Pathak et al. (2017): "Curiosity-driven Exploration by Self-supervised Prediction" (ICM)
- Burda et al. (2019): "Exploration by Random Network Distillation" (RND)
- Badia et al. (2020): "Never Give Up: Learning Directed Exploration Strategies" (NGU)
- Bellemare et al. (2016): "Unifying Count-Based Exploration"
- Houthooft et al. (2016): "VIME: Variational Information Maximizing Exploration"

### Libraries
- **ICM Implementation**: https://github.com/pathak22/noreward-rl
- **RND Implementation**: https://github.com/openai/random-network-distillation
- **Exploration Baselines**: https://github.com/DLR-RM/stable-baselines3-contrib

### Resources
- **OpenAI Blog on RND**: https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/
- **DeepMind NGU**: https://deepmind.com/blog/article/never-give-up
- **Montezuma's Revenge Benchmark**: Papers with Code

---

**Next**: [Lesson 9 - RLHF and Aligning Language Models](Lesson%209%20-%20RLHF%20and%20Aligning%20Language%20Models.md)

Proceed to learn about **RLHF** and how ChatGPT is trained! 🤖

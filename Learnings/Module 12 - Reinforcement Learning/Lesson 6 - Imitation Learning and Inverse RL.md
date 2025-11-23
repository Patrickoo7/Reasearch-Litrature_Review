# Lesson 6: Imitation Learning & Inverse Reinforcement Learning 👨‍🏫

**Module 12: Reinforcement Learning | Lesson 6 of 12**

Master learning from expert demonstrations - the fastest way to bootstrap RL agents!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand when and why to use imitation learning
2. ✅ Implement Behavioral Cloning and address distribution shift
3. ✅ Master DAgger for interactive imitation learning
4. ✅ Learn Inverse Reinforcement Learning (IRL) and MaxEnt IRL
5. ✅ Implement GAIL (Generative Adversarial Imitation Learning)
6. ✅ Apply imitation learning to robotics, autonomous driving, and game AI

---

## 1. Why Imitation Learning?

### The Problem with RL from Scratch

RL requires **exploration** and **reward engineering**:
- ❌ Millions of samples to learn simple tasks
- ❌ Reward function design is hard
- ❌ Exploration in high-dimensional spaces is difficult

**Solution**: Learn from expert demonstrations!

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
Imitation Learning (IL):
Learn policy from expert demonstrations.

When to use IL:
✅ Expert demonstrations available
✅ Reward function hard to specify
✅ Want to bootstrap RL learning
✅ Safety-critical applications

Examples:
- Autonomous driving: Learn from human drivers
- Robotics: Learn from teleoperation
- Game AI: Learn from human/AI experts
- Healthcare: Learn from doctor decisions
"""

print("=== Imitation Learning ===")
print("\nAdvantages:")
print("✅ No reward engineering needed")
print("✅ Faster than RL from scratch")
print("✅ Can bootstrap from expert knowledge")
print("✅ Works in high-dimensional spaces")

print("\nChallenges:")
print("❌ Distribution shift (compounding errors)")
print("❌ Requires expert demonstrations")
print("❌ May not surpass expert performance")
```

---

## 2. Behavioral Cloning (BC)

The simplest imitation learning: supervised learning on expert data!

```python
class BehavioralCloning:
    """
    Behavioral Cloning: Supervised learning on expert demonstrations.

    Treat imitation as supervised learning:
    - Input: States from expert
    - Output: Actions taken by expert
    - Loss: Cross-entropy (discrete) or MSE (continuous)
    """
    def __init__(self, state_dim, action_dim, discrete=True, hidden_dim=128):
        self.discrete = discrete

        if discrete:
            # Discrete actions: Classifier
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # Continuous actions: Regressor
            self.policy = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()  # Assuming actions in [-1, 1]
            )

        self.optimizer = optim.Adam(self.policy.parameters(), lr=1e-3)

    def train(self, expert_states, expert_actions, epochs=100, batch_size=64):
        """
        Train policy on expert demonstrations.

        Args:
            expert_states: Array of expert states [N, state_dim]
            expert_actions: Array of expert actions [N, action_dim or 1]
            epochs: Number of training epochs
            batch_size: Batch size
        """
        expert_states = torch.FloatTensor(expert_states)
        expert_actions = torch.LongTensor(expert_actions) if self.discrete else torch.FloatTensor(expert_actions)

        dataset = torch.utils.data.TensorDataset(expert_states, expert_actions)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        losses = []

        for epoch in range(epochs):
            epoch_loss = 0

            for states, actions in dataloader:
                # Predict actions
                if self.discrete:
                    logits = self.policy(states)
                    loss = F.cross_entropy(logits, actions)
                else:
                    pred_actions = self.policy(states)
                    loss = F.mse_loss(pred_actions, actions)

                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        return losses

    def predict(self, state):
        """Predict action for given state."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if self.discrete:
                logits = self.policy(state_tensor)
                action = logits.argmax(dim=-1).item()
            else:
                action = self.policy(state_tensor).numpy()[0]

        return action


# Collect expert demonstrations
def collect_expert_demos(env, num_episodes=100):
    """
    Collect expert demonstrations using random policy (for demo purposes).
    In practice, use actual expert (human, trained agent, etc.)
    """
    states = []
    actions = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        done = False
        while not done:
            # Expert action (here: random, but use actual expert in practice!)
            action = env.action_space.sample()

            states.append(state)
            actions.append(action)

            result = env.step(action)
            if len(result) == 5:
                next_state, _, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, _, done, _ = result

            state = next_state

    return np.array(states), np.array(actions)


# Train Behavioral Cloning
print("\n=== Behavioral Cloning ===")
env = gym.make('CartPole-v1')

# Collect expert demos
expert_states, expert_actions = collect_expert_demos(env, num_episodes=50)
print(f"Collected {len(expert_states)} expert transitions")

# Train BC
bc_agent = BehavioralCloning(state_dim=4, action_dim=2, discrete=True)
losses = bc_agent.train(expert_states, expert_actions, epochs=100)

# Evaluate
def evaluate_policy(env, agent, num_episodes=10):
    """Evaluate trained policy."""
    total_rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0
        done = False

        while not done:
            action = agent.predict(state)

            result = env.step(action)
            if len(result) == 5:
                state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, reward, done, _ = result

            episode_reward += reward

        total_rewards.append(episode_reward)

    return np.mean(total_rewards), np.std(total_rewards)


mean_reward, std_reward = evaluate_policy(env, bc_agent, num_episodes=20)
print(f"\nBC Performance: {mean_reward:.2f} ± {std_reward:.2f}")

env.close()
```

**BC Problem**: **Distribution Shift**! The agent visits states not seen in expert data, makes mistakes, and compounds errors.

---

## 3. Dataset Aggregation (DAgger)

Fix distribution shift by iteratively collecting data from learned policy!

```python
class DAgger:
    """
    Dataset Aggregation (DAgger): Interactive imitation learning.

    Algorithm:
    1. Train policy on expert data
    2. Execute learned policy, collect states
    3. Ask expert to label these states
    4. Aggregate data, retrain
    5. Repeat

    This fixes distribution shift!
    """
    def __init__(self, state_dim, action_dim, discrete=True):
        self.policy = BehavioralCloning(state_dim, action_dim, discrete)
        self.dataset_states = []
        self.dataset_actions = []

    def initial_training(self, expert_states, expert_actions):
        """Train on initial expert demonstrations."""
        self.dataset_states = list(expert_states)
        self.dataset_actions = list(expert_actions)

        print("Initial training on expert data...")
        self.policy.train(
            np.array(self.dataset_states),
            np.array(self.dataset_actions),
            epochs=50
        )

    def dagger_iteration(self, env, expert_fn, num_episodes=10):
        """
        One DAgger iteration.

        Args:
            env: Environment
            expert_fn: Expert policy function (state → action)
            num_episodes: Number of rollout episodes
        """
        new_states = []
        new_actions = []

        # Rollout learned policy
        for _ in range(num_episodes):
            state = env.reset()
            if isinstance(state, tuple):
                state = state[0]

            done = False
            while not done:
                # Use learned policy to select action
                action = self.policy.predict(state)

                # Get expert label for this state
                expert_action = expert_fn(state)

                new_states.append(state)
                new_actions.append(expert_action)  # Expert label!

                # Execute learned policy action
                result = env.step(action)
                if len(result) == 5:
                    state, _, terminated, truncated, _ = result
                    done = terminated or truncated
                else:
                    state, _, done, _ = result

        # Aggregate data
        self.dataset_states.extend(new_states)
        self.dataset_actions.extend(new_actions)

        # Retrain on aggregated dataset
        print(f"Dataset size: {len(self.dataset_states)}, Retraining...")
        self.policy.train(
            np.array(self.dataset_states),
            np.array(self.dataset_actions),
            epochs=30
        )

    def train(self, env, expert_fn, expert_states, expert_actions, num_iterations=5):
        """
        Full DAgger training.

        Args:
            env: Environment
            expert_fn: Expert policy function
            expert_states: Initial expert states
            expert_actions: Initial expert actions
            num_iterations: Number of DAgger iterations
        """
        # Initial training
        self.initial_training(expert_states, expert_actions)

        # DAgger iterations
        for iteration in range(num_iterations):
            print(f"\n=== DAgger Iteration {iteration + 1} ===")
            self.dagger_iteration(env, expert_fn, num_episodes=10)

            # Evaluate
            mean_reward, _ = evaluate_policy(env, self.policy, num_episodes=10)
            print(f"Performance: {mean_reward:.2f}")


print("\nDAgger:")
print("✅ Fixes distribution shift")
print("✅ Iteratively improves policy")
print("✅ Requires expert to label new states")
print("✅ Better than BC in practice")
```

---

## 4. Inverse Reinforcement Learning (IRL)

Instead of learning policy, learn the **reward function** the expert is optimizing!

```python
"""
Inverse Reinforcement Learning (IRL):

Given: Expert demonstrations
Goal: Infer reward function R(s,a)

Once we have R, we can:
- Use any RL algorithm to learn optimal policy
- Transfer to new environments
- Understand expert's objectives

IRL is harder than imitation, but more powerful!
"""

class MaxEntIRL:
    """
    Maximum Entropy Inverse Reinforcement Learning.

    Idea: Find reward function that makes expert demonstrations have maximum likelihood
    under maximum entropy distribution.

    R(s,a) = θ^T φ(s,a)  (linear in features)
    """
    def __init__(self, state_dim, num_features=16):
        self.num_features = num_features

        # Reward function parameters
        self.theta = np.random.randn(num_features)

        # Feature extractor (simple for demo)
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_features)
        )

    def extract_features(self, state):
        """
        Extract features from state.

        φ(s): State → Feature vector
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_net(state_tensor).numpy()[0]
        return features

    def compute_reward(self, state):
        """
        Compute reward: R(s) = θ^T φ(s)
        """
        features = self.extract_features(state)
        reward = np.dot(self.theta, features)
        return reward

    def compute_expert_feature_expectations(self, expert_trajectories):
        """
        Compute feature expectations from expert demonstrations.

        μ_E = E_{expert}[φ(s)]
        """
        all_features = []

        for traj in expert_trajectories:
            for state in traj:
                features = self.extract_features(state)
                all_features.append(features)

        return np.mean(all_features, axis=0)

    def irl_update(self, expert_feature_exp, learned_feature_exp, lr=0.01):
        """
        Update reward parameters.

        Gradient: μ_E - μ_π  (expert features - learned policy features)
        """
        grad = expert_feature_exp - learned_feature_exp
        self.theta += lr * grad

    def train(self, expert_trajectories, env, num_iterations=50):
        """
        Train IRL.

        Algorithm:
        1. Compute expert feature expectations
        2. Train policy with current reward
        3. Compute policy feature expectations
        4. Update reward parameters
        5. Repeat
        """
        expert_feature_exp = self.compute_expert_feature_expectations(expert_trajectories)

        for iteration in range(num_iterations):
            # TODO: Train policy using current reward function (use any RL algorithm)
            # learned_policy = train_rl(self.compute_reward)

            # Compute learned policy feature expectations
            # learned_feature_exp = ...

            # Update reward
            # self.irl_update(expert_feature_exp, learned_feature_exp)

            if iteration % 10 == 0:
                print(f"IRL Iteration {iteration}")


print("\nInverse Reinforcement Learning:")
print("✅ Learn reward function from demonstrations")
print("✅ More powerful than direct imitation")
print("✅ Can transfer to new tasks")
print("✅ Understand expert's objectives")
```

---

## 5. Generative Adversarial Imitation Learning (GAIL)

Combine imitation learning with GANs!

```python
class GAIL:
    """
    Generative Adversarial Imitation Learning (GAIL).

    Key insight: Match occupancy measure of expert and learned policy.

    Components:
    - Generator: Policy π (like actor in actor-critic)
    - Discriminator: D(s,a) - distinguishes expert from policy

    Loss:
    - Generator: Fool discriminator (max log D(s,a))
    - Discriminator: Classify expert vs policy

    GAIL learns reward function implicitly via discriminator!
    """
    def __init__(self, state_dim, action_dim):
        # Policy (Generator)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        # Discriminator: D(s,a) → [0, 1]
        # 1 = expert, 0 = policy
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=3e-4)

    def update_discriminator(self, expert_batch, policy_batch):
        """
        Update discriminator to distinguish expert from policy.

        Loss: Binary cross-entropy
        """
        expert_states, expert_actions = expert_batch
        policy_states, policy_actions = policy_batch

        expert_states = torch.FloatTensor(expert_states)
        expert_actions = torch.FloatTensor(expert_actions)
        policy_states = torch.FloatTensor(policy_states)
        policy_actions = torch.FloatTensor(policy_actions)

        # Discriminator predictions
        expert_sa = torch.cat([expert_states, expert_actions], dim=-1)
        policy_sa = torch.cat([policy_states, policy_actions], dim=-1)

        expert_preds = self.discriminator(expert_sa)
        policy_preds = self.discriminator(policy_sa)

        # Binary cross-entropy loss
        # Expert should be classified as 1, policy as 0
        expert_loss = F.binary_cross_entropy(expert_preds, torch.ones_like(expert_preds))
        policy_loss = F.binary_cross_entropy(policy_preds, torch.zeros_like(policy_preds))

        discriminator_loss = expert_loss + policy_loss

        # Update
        self.discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        return discriminator_loss.item()

    def compute_reward(self, state, action):
        """
        Compute reward from discriminator.

        r(s,a) = -log(1 - D(s,a))

        This encourages policy to fool discriminator!
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)

        sa = torch.cat([state_tensor, action_tensor], dim=-1)

        with torch.no_grad():
            d_value = self.discriminator(sa)
            reward = -torch.log(1 - d_value + 1e-8).item()

        return reward

    def train_policy(self, env, num_steps=1000):
        """
        Train policy using PPO with discriminator-based rewards.

        This is where GAIL uses RL (typically PPO or TRPO).
        """
        # Collect trajectories
        states = []
        actions = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(num_steps):
            # Sample action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.policy(state_tensor).detach().numpy()[0]

            states.append(state)
            actions.append(action)

            # Take action
            # Use discriminator reward instead of environment reward!
            # reward = self.compute_reward(state, action)

            result = env.step(action.argmax() if len(action) > 1 else action)
            if len(result) == 5:
                state, _, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                state, _, done, _ = result

            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        return states, actions


print("\nGAIL (Generative Adversarial Imitation Learning):")
print("✅ No reward function needed")
print("✅ Matches expert distribution")
print("✅ Combines IL with RL")
print("✅ State-of-the-art imitation learning")
```

---

## 6. Adversarial Inverse Reinforcement Learning (AIRL)

AIRL improves GAIL by learning disentangled reward function!

```python
"""
AIRL (Adversarial Inverse Reinforcement Learning):

Improves GAIL:
1. Disentangles reward from dynamics
2. Learns transferable reward function
3. More robust to environmental changes

Discriminator form:
D(s,a,s') = exp(r(s,a)) / [exp(r(s,a)) + π(a|s)]

This separates reward from policy!
"""

print("\nAIRL (Adversarial IRL):")
print("✅ Learns disentangled reward")
print("✅ Better transferability than GAIL")
print("✅ More interpretable")
print("✅ Used in robotics transfer learning")
```

---

## 7. Real-World Applications

### Autonomous Driving

```python
"""
Autonomous Driving: Learn from Human Drivers

Pipeline:
1. Collect human driving data (states, actions)
2. Train BC/DAgger policy
3. Fine-tune with RL in simulation
4. Sim-to-real transfer

Example: Waymo, Tesla Autopilot use imitation learning
"""

print("\n=== Autonomous Driving ===")
print("✅ Learn from human demonstrations")
print("✅ BC for initial policy")
print("✅ DAgger for online correction")
print("✅ IRL to understand human preferences")
```

### Robotics

```python
"""
Robotics: Learn from Teleoperation

Tasks:
- Grasping: Learn from human demonstrations
- Manipulation: Object rearrangement
- Locomotion: Learn walking from motion capture

Approach:
1. Collect expert demos (teleoperation, kinesthetic teaching)
2. BC or GAIL for initial policy
3. Fine-tune with RL
"""

print("\n=== Robotics ===")
print("✅ Teleoperation demonstrations")
print("✅ Kinesthetic teaching")
print("✅ Motion capture data")
print("✅ Fine-tune with RL")
```

### Game AI

```python
"""
Game AI: Learn from Expert Players

Examples:
- AlphaStar (StarCraft): Initial BC from replays
- OpenAI Five (Dota 2): BC → Self-play
- Gran Turismo: Learn driving from best players

Pipeline:
1. Collect expert replays
2. BC for initial policy
3. Self-play / RL for superhuman performance
"""

print("\n=== Game AI ===")
print("✅ Learn from replay data")
print("✅ Bootstrap with BC")
print("✅ Improve with self-play")
print("✅ Achieve superhuman performance")
```

---

## Practice Exercises

### Exercise 1: BC vs DAgger Comparison

```python
"""
Implement and compare BC and DAgger on a simple task.

Environment: LunarLander-v2
1. Collect expert demonstrations (use trained PPO agent)
2. Train BC policy
3. Train DAgger policy (5 iterations)
4. Compare:
   - Sample efficiency
   - Final performance
   - Distribution shift effects

Visualize: Performance vs number of expert queries
"""

# Your implementation here
```

### Exercise 2: Implement Simple IRL

```python
"""
Implement MaxEnt IRL on GridWorld.

1. Create GridWorld environment
2. Generate expert trajectories (optimal policy)
3. Implement feature extraction
4. IRL algorithm:
   - Compute expert feature expectations
   - Train policy with current reward
   - Update reward parameters
5. Visualize learned reward function

Compare learned rewards with true rewards!
"""

# Your implementation here
```

### Exercise 3: GAIL for Atari

```python
"""
Implement GAIL for an Atari game (e.g., Pong).

Requirements:
- Collect expert demonstrations (DQN agent)
- Implement discriminator network
- Train policy with PPO using discriminator rewards
- Compare with BC baseline

Track:
- Discriminator accuracy
- Policy performance
- Sample efficiency

Bonus: Visualize what discriminator learns to distinguish!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Behavioral Cloning** 📚
   - Simplest imitation learning
   - Supervised learning on expert data
   - Fast to implement
   - Suffers from distribution shift

2. **Distribution Shift** ⚠️
   - Agent visits unseen states
   - Makes mistakes, compounds errors
   - Key challenge in IL
   - DAgger fixes this

3. **DAgger** 🔄
   - Interactive imitation learning
   - Iteratively collect data from learned policy
   - Expert labels new states
   - Fixes distribution shift

4. **Inverse RL** 🔍
   - Learn reward function from demonstrations
   - More powerful than BC
   - Enables transfer learning
   - MaxEnt IRL is popular approach

5. **GAIL** 🎭
   - Combines IL with GANs
   - No explicit reward needed
   - Matches expert distribution
   - State-of-the-art IL

6. **AIRL** 🎯
   - Improves GAIL
   - Disentangles reward from dynamics
   - Better transferability
   - More interpretable

### When to Use What?

✅ **BC**: Simple task, lots of expert data, no distribution shift
✅ **DAgger**: Interactive expert available, want robustness
✅ **IRL**: Need transferable reward function
✅ **GAIL**: No reward function, want state-of-the-art
✅ **AIRL**: Transfer learning, interpretability

### Real-World Applications

✅ **Autonomous Driving**: Learn from human drivers
✅ **Robotics**: Teleoperation, kinesthetic teaching
✅ **Game AI**: Expert replays, superhuman performance
✅ **Healthcare**: Learn from doctor decisions
✅ **Manufacturing**: Learn from skilled workers

### What's Next?

In Lesson 7, we'll learn **Offline Reinforcement Learning**:
- Learn from fixed datasets (no environment interaction!)
- BCQ, CQL, Decision Transformer
- Safety-critical applications
- Healthcare, autonomous vehicles

**Imitation learning is the fastest way to bootstrap - master it!** 👨‍🏫

---

## Additional Resources

### Papers
- Pomerleau (1989): "ALVINN: An Autonomous Land Vehicle in a Neural Network" (BC)
- Ross et al. (2011): "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning" (DAgger)
- Abbeel & Ng (2004): "Apprenticeship Learning via Inverse Reinforcement Learning"
- Ziebart et al. (2008): "Maximum Entropy Inverse Reinforcement Learning" (MaxEnt IRL)
- Ho & Ermon (2016): "Generative Adversarial Imitation Learning" (GAIL)
- Fu et al. (2018): "Learning Robust Rewards with Adversarial Inverse Reinforcement Learning" (AIRL)

### Libraries
- **imitation**: https://github.com/HumanCompatibleAI/imitation (BC, DAgger, GAIL, AIRL)
- **Stable-Baselines3**: Has BC implementation
- **IRL Toolkit**: Various IRL implementations

### Resources
- **Stanford CS336 (Deep Multi-Task and Meta Learning)**: IL lectures
- **Berkeley CS294**: Deep RL course (IL section)
- **Imitation Learning Tutorial**: https://sites.google.com/view/icml2018-imitation-learning/

---

**Next**: [Lesson 7 - Offline Reinforcement Learning](Lesson%207%20-%20Offline%20Reinforcement%20Learning.md)

Proceed to learn about **offline RL** and learning from fixed datasets! 💾

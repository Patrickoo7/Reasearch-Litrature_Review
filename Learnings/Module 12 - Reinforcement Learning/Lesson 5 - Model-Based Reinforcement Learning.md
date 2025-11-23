# Lesson 5: Model-Based Reinforcement Learning 🧭

**Module 12: Reinforcement Learning | Lesson 5 of 12**

Master model-based RL for sample-efficient learning through planning and imagination!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand model-based vs model-free RL trade-offs
2. ✅ Implement dynamics model learning and Model Predictive Control (MPC)
3. ✅ Master Dyna architecture for combining learning and planning
4. ✅ Learn advanced methods (PETS, MBPO, Dreamer, World Models)
5. ✅ Apply model-based RL to robotics and control problems
6. ✅ Understand sample efficiency and uncertainty quantification

---

## 1. Model-Based vs Model-Free RL

### The Key Difference

**Model-Free** (DQN, PPO, SAC):
- Learn policy/value directly from experience
- Don't model environment dynamics
- Sample inefficient but robust

**Model-Based**:
- Learn model of environment: p(s'|s,a)
- Use model for planning
- Sample efficient but model errors compound

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
Model-Free RL:
    Experience → Policy/Value Function

Model-Based RL:
    Experience → Dynamics Model → Planning → Policy

Advantages of Model-Based:
✅ Sample efficiency (learn from fewer real interactions)
✅ Can plan and imagine outcomes
✅ Transfer learning across tasks
✅ Interpretable (can visualize model predictions)

Disadvantages:
❌ Model errors compound over time
❌ Harder to learn accurate models
❌ Computational cost of planning
"""

print("=== Model-Based vs Model-Free ===")
print("\nModel-Free (DQN, PPO, SAC):")
print("- Direct policy/value learning")
print("- 1M+ samples to master Atari")
print("- Robust to model errors (no model!)")

print("\nModel-Based (PETS, Dreamer):")
print("- Learn environment model")
print("- 100K samples to master tasks")
print("- 10x more sample efficient!")
print("- But model errors can hurt performance")
```

---

## 2. Learning Dynamics Models

The core of model-based RL: learn p(s'|s,a) from data!

```python
class DynamicsModel(nn.Module):
    """
    Learned dynamics model: s' = f(s, a)

    Predict next state given current state and action.
    Can be:
    - Deterministic: s' = f(s,a)
    - Probabilistic: s' ~ p(s'|s,a)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()

        # Network predicts state delta: Δs = s' - s
        # This is easier to learn than absolute next state!
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Predict mean and log_std (probabilistic model)
        self.mean = nn.Linear(hidden_dim, state_dim)
        self.log_std = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        """
        Predict next state distribution.

        Returns:
            mean: Predicted state delta mean
            log_std: Predicted state delta log std
        """
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -10, 0.5)  # Stabilize

        return mean, log_std

    def predict(self, state, action):
        """
        Predict next state (sampling from distribution).
        """
        mean, log_std = self.forward(state, action)
        std = log_std.exp()

        # Sample from Gaussian
        noise = torch.randn_like(mean)
        state_delta = mean + std * noise

        # Next state = current state + delta
        next_state = state + state_delta

        return next_state

    def predict_mean(self, state, action):
        """Predict next state (deterministic, using mean)."""
        mean, _ = self.forward(state, action)
        next_state = state + mean
        return next_state


class DynamicsModelTrainer:
    """
    Train dynamics model from experience.
    """
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.model = DynamicsModel(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Replay buffer for model training
        self.buffer = deque(maxlen=100000)

    def collect_data(self, env, num_steps=5000):
        """
        Collect random transitions for model learning.
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(num_steps):
            # Random action
            action = env.action_space.sample()

            # Take step
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Store transition
            self.buffer.append((state, action, next_state, reward, done))

            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]
            else:
                state = next_state

        print(f"Collected {len(self.buffer)} transitions")

    def train(self, num_epochs=100, batch_size=256):
        """
        Train dynamics model on collected data.
        """
        losses = []

        for epoch in range(num_epochs):
            # Sample batch
            batch = [self.buffer[i] for i in np.random.choice(len(self.buffer), batch_size)]
            states, actions, next_states, _, _ = zip(*batch)

            states = torch.FloatTensor(states)
            actions = torch.FloatTensor(actions).unsqueeze(1) if len(actions[0]) == 0 else torch.FloatTensor(actions)
            next_states = torch.FloatTensor(next_states)

            # True state delta
            state_deltas = next_states - states

            # Predict
            mean, log_std = self.model(states, actions)
            std = log_std.exp()

            # Negative log likelihood loss
            loss = 0.5 * ((state_deltas - mean) / std) ** 2 + log_std + 0.5 * np.log(2 * np.pi)
            loss = loss.sum(dim=-1).mean()

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

        return losses


# Example: Learn dynamics model for CartPole
print("\n=== Learning Dynamics Model ===")
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = 1  # Discrete action (encoded as float)

trainer = DynamicsModelTrainer(state_dim, action_dim)

# Collect data
trainer.collect_data(env, num_steps=10000)

# Train model
losses = trainer.train(num_epochs=100, batch_size=256)

# Visualize training
plt.figure(figsize=(10, 4))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Dynamics Model Training')
plt.grid(True)
plt.show()

env.close()
```

---

## 3. Model Predictive Control (MPC)

Use learned model for planning!

```python
class RandomShootingMPC:
    """
    Model Predictive Control with Random Shooting.

    Algorithm:
    1. Sample K random action sequences
    2. Rollout each sequence using learned model
    3. Select sequence with highest predicted reward
    4. Execute first action, replan
    """
    def __init__(self, dynamics_model, horizon=10, num_samples=100):
        self.dynamics_model = dynamics_model
        self.horizon = horizon  # Planning horizon
        self.num_samples = num_samples  # Number of action sequences

    def plan(self, state, action_dim, action_range=(-1, 1)):
        """
        Plan action using random shooting.

        Args:
            state: Current state
            action_dim: Dimension of action space
            action_range: (min, max) action values

        Returns:
            best_action: First action of best sequence
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        best_reward = -float('inf')
        best_action_sequence = None

        # Sample random action sequences
        for _ in range(self.num_samples):
            # Random action sequence
            action_sequence = torch.FloatTensor(
                np.random.uniform(action_range[0], action_range[1],
                                (self.horizon, action_dim))
            )

            # Rollout using learned model
            total_reward = 0
            current_state = state_tensor.clone()

            for t in range(self.horizon):
                action = action_sequence[t:t+1]

                # Predict next state
                next_state = self.dynamics_model.predict_mean(current_state, action)

                # Compute reward (task-specific)
                reward = self.reward_function(next_state)
                total_reward += reward * (0.99 ** t)  # Discounted

                current_state = next_state

            # Track best sequence
            if total_reward > best_reward:
                best_reward = total_reward
                best_action_sequence = action_sequence

        # Return first action of best sequence
        return best_action_sequence[0].numpy()

    def reward_function(self, state):
        """
        Reward function for CartPole.
        Maximize: upright pole, centered cart
        """
        # Extract state components (CartPole specific)
        cart_pos = state[0, 0]
        pole_angle = state[0, 2]

        # Reward: Stay upright and centered
        reward = 1.0 - abs(pole_angle) - 0.1 * abs(cart_pos)

        return reward


print("\nModel Predictive Control (MPC):")
print("✅ Use learned model for planning")
print("✅ Sample action sequences")
print("✅ Execute best action")
print("✅ Replan at each step (closed-loop)")
```

---

## 4. Dyna Architecture

Combine model-free learning with model-based planning!

```python
class DynaQ:
    """
    Dyna-Q: Integrate learning, planning, and acting.

    Algorithm:
    1. Take action, observe transition
    2. Update model (store transition)
    3. Update Q-function (model-free)
    4. Planning: Sample from model, update Q-function

    Best of both worlds!
    """
    def __init__(self, state_dim, action_dim, lr=0.1, gamma=0.99, planning_steps=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        self.gamma = gamma
        self.planning_steps = planning_steps

        # Q-table (for discrete states)
        self.Q = {}

        # Model: Store observed transitions
        self.model = {}  # (s, a) → (r, s')

    def get_q(self, state, action):
        """Get Q-value."""
        key = (state, action)
        return self.Q.get(key, 0.0)

    def update_q(self, state, action, reward, next_state, done):
        """Q-learning update."""
        if done:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in range(self.action_dim)]
            target = reward + self.gamma * max(next_q_values)

        current_q = self.get_q(state, action)
        self.Q[(state, action)] = current_q + self.lr * (target - current_q)

    def update_model(self, state, action, reward, next_state):
        """Update learned model."""
        self.model[(state, action)] = (reward, next_state)

    def planning(self):
        """
        Planning: Sample from model and update Q.

        Perform planning_steps updates using simulated experience.
        """
        if not self.model:
            return

        for _ in range(self.planning_steps):
            # Sample random previously observed (s, a)
            state, action = list(self.model.keys())[np.random.randint(len(self.model))]

            # Get model prediction
            reward, next_state = self.model[(state, action)]

            # Q-learning update (using simulated experience!)
            self.update_q(state, action, reward, next_state, False)

    def train_episode(self, env):
        """Train one episode with Dyna-Q."""
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        # Discretize state for tabular representation
        state = tuple(np.round(state, 1))

        episode_reward = 0
        done = False

        while not done:
            # ε-greedy action selection
            if np.random.rand() < 0.1:
                action = np.random.randint(self.action_dim)
            else:
                q_values = [self.get_q(state, a) for a in range(self.action_dim)]
                action = np.argmax(q_values)

            # Take action
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            next_state = tuple(np.round(next_state, 1))
            episode_reward += reward

            # (a) Update model
            self.update_model(state, action, reward, next_state)

            # (b) Q-learning update (real experience)
            self.update_q(state, action, reward, next_state, done)

            # (c) Planning (simulated experience)
            self.planning()

            state = next_state

        return episode_reward


print("\nDyna-Q Architecture:")
print("✅ Combines model-free (Q-learning) and model-based (planning)")
print("✅ Learn from real AND simulated experience")
print("✅ More sample efficient than pure model-free")
print("✅ More robust than pure model-based")
```

---

## 5. Cross-Entropy Method (CEM) for Planning

Simple but effective planning algorithm!

```python
class CEMPlanner:
    """
    Cross-Entropy Method for trajectory optimization.

    Iteratively refine action sequence distribution:
    1. Sample action sequences from current distribution
    2. Evaluate using learned model
    3. Fit new distribution to best sequences
    4. Repeat
    """
    def __init__(self, dynamics_model, horizon=10, num_samples=100, elite_frac=0.1,
                 num_iterations=5):
        self.dynamics_model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = int(num_samples * elite_frac)
        self.num_iterations = num_iterations

    def plan(self, state, action_dim):
        """
        Plan using CEM.

        Returns:
            best_action: First action of optimized sequence
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Initialize action sequence distribution
        mean = torch.zeros(self.horizon, action_dim)
        std = torch.ones(self.horizon, action_dim)

        for iteration in range(self.num_iterations):
            # Sample action sequences
            action_sequences = mean + std * torch.randn(self.num_samples, self.horizon, action_dim)

            # Clip to valid range
            action_sequences = torch.clamp(action_sequences, -1, 1)

            # Evaluate each sequence
            rewards = []
            for action_seq in action_sequences:
                total_reward = self.evaluate_sequence(state_tensor, action_seq)
                rewards.append(total_reward)

            rewards = torch.FloatTensor(rewards)

            # Select elite sequences
            elite_indices = torch.argsort(rewards, descending=True)[:self.num_elites]
            elite_sequences = action_sequences[elite_indices]

            # Update distribution
            mean = elite_sequences.mean(dim=0)
            std = elite_sequences.std(dim=0) + 1e-6

        # Return first action of mean sequence
        return mean[0].detach().numpy()

    def evaluate_sequence(self, state, action_sequence):
        """Evaluate action sequence using learned model."""
        total_reward = 0
        current_state = state.clone()

        for t in range(self.horizon):
            action = action_sequence[t:t+1]

            # Predict next state
            next_state = self.dynamics_model.predict_mean(current_state, action)

            # Compute reward
            reward = self.reward_function(next_state)
            total_reward += reward * (0.99 ** t)

            current_state = next_state

        return total_reward

    def reward_function(self, state):
        """Task-specific reward function."""
        # Example: Pendulum reward
        cos_theta = state[0, 0]
        sin_theta = state[0, 1]
        theta_dot = state[0, 2]

        reward = cos_theta - 0.1 * theta_dot ** 2
        return reward


print("\nCross-Entropy Method (CEM):")
print("✅ Iterative planning algorithm")
print("✅ Simple and effective")
print("✅ Used in PETS and other MBRL algorithms")
```

---

## 6. Probabilistic Ensembles with Trajectory Sampling (PETS)

State-of-the-art model-based RL for continuous control!

```python
class EnsembleDynamicsModel(nn.Module):
    """
    Ensemble of dynamics models for uncertainty estimation.

    Multiple models → Better uncertainty quantification!
    """
    def __init__(self, state_dim, action_dim, num_models=5, hidden_dim=256):
        super(EnsembleDynamicsModel, self).__init__()
        self.num_models = num_models

        # Create ensemble of models
        self.models = nn.ModuleList([
            DynamicsModel(state_dim, action_dim, hidden_dim)
            for _ in range(num_models)
        ])

    def forward(self, state, action, model_idx=None):
        """
        Forward pass through ensemble.

        Args:
            model_idx: If provided, use specific model. Otherwise, random model.
        """
        if model_idx is None:
            model_idx = np.random.randint(self.num_models)

        return self.models[model_idx](state, action)

    def predict_ensemble(self, state, action):
        """
        Get predictions from all models.

        Returns:
            means: Predictions from each model
            stds: Uncertainties from each model
        """
        means = []
        stds = []

        for model in self.models:
            mean, log_std = model(state, action)
            means.append(mean)
            stds.append(log_std.exp())

        return torch.stack(means), torch.stack(stds)


class PETS:
    """
    Probabilistic Ensembles with Trajectory Sampling.

    Key features:
    1. Ensemble of probabilistic models
    2. CEM for planning
    3. Uncertainty-aware planning
    """
    def __init__(self, state_dim, action_dim, num_models=5):
        self.ensemble = EnsembleDynamicsModel(state_dim, action_dim, num_models)
        self.planner = CEMPlanner(self.ensemble, horizon=15, num_samples=200)

        print("PETS initialized!")
        print(f"✅ {num_models} model ensemble")
        print("✅ CEM planner")
        print("✅ Uncertainty-aware")


print("\nPETS (Probabilistic Ensembles with Trajectory Sampling):")
print("✅ State-of-the-art model-based RL")
print("✅ Sample efficient (100K steps for complex tasks)")
print("✅ Uncertainty quantification with ensembles")
print("✅ Used in robotics")
```

---

## 7. Model-Based Policy Optimization (MBPO)

Combine model-based and model-free for best of both worlds!

```python
"""
MBPO (Model-Based Policy Optimization):

Algorithm:
1. Collect real data, train dynamics model
2. Generate synthetic data using learned model
3. Train model-free policy (SAC) on real + synthetic data
4. Repeat

Advantages:
✅ Sample efficient (model-based)
✅ Robust to model errors (model-free fallback)
✅ State-of-the-art performance
"""

class MBPO:
    """
    Model-Based Policy Optimization.

    Combines:
    - Ensemble dynamics model
    - SAC policy
    - Real + synthetic data
    """
    def __init__(self, state_dim, action_dim):
        # Dynamics model ensemble
        self.dynamics_model = EnsembleDynamicsModel(state_dim, action_dim, num_models=7)

        # Model-free policy (SAC)
        # from stable_baselines3 import SAC
        # self.policy = SAC(...)

        # Replay buffers
        self.real_buffer = deque(maxlen=1000000)
        self.model_buffer = deque(maxlen=1000000)

        print("MBPO initialized!")
        print("✅ 7-model ensemble")
        print("✅ SAC policy")
        print("✅ Real + synthetic data")

    def generate_synthetic_data(self, num_steps=10000):
        """
        Generate synthetic transitions using learned model.

        Start from real states, rollout using model.
        """
        synthetic_data = []

        for _ in range(num_steps):
            # Sample real state
            if not self.real_buffer:
                break

            real_transition = self.real_buffer[np.random.randint(len(self.real_buffer))]
            state = real_transition[0]

            # Short rollout (k-step, k=1 to 5)
            rollout_length = np.random.randint(1, 6)

            for step in range(rollout_length):
                # Get action from policy
                # action = self.policy.predict(state)

                # Predict next state using random model from ensemble
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                # action_tensor = torch.FloatTensor(action).unsqueeze(0)

                # next_state = self.dynamics_model.predict(state_tensor, action_tensor)

                # Compute reward (environment-specific)
                # reward = ...

                # Store synthetic transition
                # synthetic_data.append((state, action, reward, next_state, False))

                # state = next_state

        return synthetic_data


print("\nMBPO:")
print("✅ Combines model-based and model-free")
print("✅ More sample efficient than SAC alone")
print("✅ More robust than pure model-based")
print("✅ State-of-the-art on MuJoCo benchmarks")
```

---

## 8. Dreamer: World Models for Visual Control

Learn latent dynamics model from images!

```python
"""
Dreamer (Dream to Control):

Key ideas:
1. Learn latent world model from pixels
2. Learn behaviors purely in imagination (latent space)
3. Transfer to real environment

Components:
- Encoder: Image → Latent state
- Dynamics: Latent transition model
- Decoder: Latent state → Reconstructed image
- Reward predictor
- Actor-critic in latent space

Dreamer enables learning from pixels with sample efficiency!
"""

class LatentWorldModel(nn.Module):
    """
    Latent world model for Dreamer.

    Learns compact latent representation and dynamics.
    """
    def __init__(self, image_channels=3, latent_dim=64, action_dim=2):
        super(LatentWorldModel, self).__init__()

        # Encoder: Image → Latent
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, latent_dim)
        )

        # Dynamics: (latent, action) → next latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

        # Decoder: Latent → Reconstructed image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, 4, stride=2),
            nn.Sigmoid()
        )

        # Reward predictor
        self.reward_pred = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def encode(self, image):
        """Encode image to latent state."""
        return self.encoder(image)

    def imagine_step(self, latent, action):
        """Predict next latent state."""
        x = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(x)
        return next_latent

    def decode(self, latent):
        """Decode latent to image."""
        return self.decoder(latent)

    def predict_reward(self, latent):
        """Predict reward from latent state."""
        return self.reward_pred(latent)


class Dreamer:
    """
    Dreamer: Learn behaviors in imagination.

    1. Learn world model from experience
    2. Imagine trajectories in latent space
    3. Train actor-critic on imagined trajectories
    4. Act in real environment
    """
    def __init__(self, image_channels=3, latent_dim=64, action_dim=2):
        self.world_model = LatentWorldModel(image_channels, latent_dim, action_dim)

        # Actor-critic in latent space
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        print("Dreamer initialized!")
        print("✅ Latent world model")
        print("✅ Learn from pixels")
        print("✅ Train in imagination")
        print("✅ Sample efficient for visual control")


print("\nDreamer:")
print("✅ Learn from high-dimensional observations (images)")
print("✅ Latent space planning")
print("✅ Sample efficient")
print("✅ State-of-the-art for visual control")
```

---

## 9. Uncertainty Quantification

Critical for safe model-based RL!

```python
def epistemic_uncertainty(ensemble_predictions):
    """
    Epistemic uncertainty: Disagreement among models.

    High disagreement → High uncertainty → Out of distribution!
    """
    # ensemble_predictions: [num_models, batch_size, state_dim]
    mean = ensemble_predictions.mean(dim=0)
    variance = ensemble_predictions.var(dim=0)

    return variance.mean(dim=-1)  # Average over state dimensions


def aleatoric_uncertainty(model_std):
    """
    Aleatoric uncertainty: Model's predicted variance.

    Inherent noise in environment.
    """
    return model_std.mean(dim=-1)


print("\nUncertainty in Model-Based RL:")
print("✅ Epistemic: Model uncertainty (disagreement)")
print("✅ Aleatoric: Environment noise (inherent)")
print("✅ Use ensembles for epistemic uncertainty")
print("✅ Avoid high-uncertainty regions during planning")
```

---

## 10. Sample Efficiency Comparison

```python
"""
Sample Efficiency Comparison (approximate):

Atari (1M frames = 250K steps):
- Model-Free (DQN): 200M frames (50M steps)
- Model-Based (SimPLe): 100K steps (200x improvement!)

MuJoCo Continuous Control:
- Model-Free (SAC): 1M steps
- Model-Based (MBPO): 100K steps (10x improvement)
- Model-Based (PETS): 50K steps (20x improvement!)

Robotics (real robot hours):
- Model-Free: 100+ hours
- Model-Based: 5-10 hours

Model-based RL is MUCH more sample efficient!
"""

print("\nSample Efficiency:")
print("Model-Free (SAC): 1M steps")
print("Model-Based (PETS): 50K steps")
print("Speedup: 20x fewer samples! 🚀")
```

---

## Practice Exercises

### Exercise 1: Implement iLQG

```python
"""
Implement iterative Linear Quadratic Gaussian (iLQG) control.

iLQG is a trajectory optimization method:
1. Linearize dynamics around current trajectory
2. Solve LQR for linearized system
3. Update trajectory
4. Repeat

Used in robotics for model-based control.

Hints:
- Compute Jacobians of dynamics
- Backward pass: Compute gains
- Forward pass: Update trajectory
"""

# Your implementation here
```

### Exercise 2: Train PETS on MuJoCo

```python
"""
Implement and train PETS on a MuJoCo environment (e.g., HalfCheetah).

Requirements:
- Ensemble of 5-7 models
- CEM planner with 200-500 samples
- Compare sample efficiency with SAC
- Track model prediction errors

Bonus: Visualize model predictions vs ground truth!
"""

# Your implementation here
```

### Exercise 3: Model-Based + Model-Free Hybrid

```python
"""
Implement Dyna-style algorithm with neural network model.

Combine:
- Neural network dynamics model
- DQN or PPO policy
- Real + simulated experience

Compare pure model-free vs hybrid on CartPole or LunarLander.

Track:
- Sample efficiency
- Final performance
- Model accuracy
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Model-Based vs Model-Free** ⚖️
   - Model-based: Learn dynamics, plan
   - Model-free: Direct policy/value learning
   - Model-based: 10-100x more sample efficient
   - Model-free: More robust to model errors

2. **Dynamics Models** 🧠
   - Learn p(s'|s,a) from data
   - Probabilistic models capture uncertainty
   - Ensembles for better uncertainty estimation
   - Predict state deltas (easier than absolute states)

3. **Planning Algorithms** 🎯
   - Random Shooting: Simple, effective
   - CEM: Iterative refinement
   - MPC: Replan at each step
   - iLQG: Trajectory optimization

4. **Dyna Architecture** 🔄
   - Integrate learning and planning
   - Real + simulated experience
   - Best of both worlds
   - More robust than pure model-based

5. **PETS** 🚀
   - Ensemble models + CEM planning
   - State-of-the-art sample efficiency
   - 50-100K steps for complex tasks
   - Uncertainty-aware planning

6. **MBPO** 🏆
   - Combine model-based and SAC
   - Real + synthetic data
   - Robust to model errors
   - State-of-the-art performance

7. **Dreamer** 🌠
   - Latent world models
   - Learn from pixels
   - Train in imagination
   - Visual control tasks

### When to Use Model-Based RL?

✅ **Sample efficiency critical**: Robotics, expensive simulations
✅ **Planning useful**: Can imagine outcomes
✅ **Transfer learning**: Multi-task, changing rewards
✅ **Interpretability**: Understand environment dynamics

❌ **Don't use when**:
   - Sample efficiency not critical
   - Environment too complex to model
   - Model errors catastrophic

### Real-World Applications

✅ **Robotics**: Manipulation, locomotion (sample efficient!)
✅ **Industrial Control**: Chemical processes, manufacturing
✅ **Energy Systems**: Building HVAC, power grids
✅ **Autonomous Vehicles**: Trajectory planning
✅ **Healthcare**: Treatment planning

### What's Next?

In Lesson 6, we'll learn **Imitation Learning and Inverse RL**:
- Learn from demonstrations
- Behavioral cloning, DAgger
- Inverse RL and GAIL
- When expert data is available

**Model-based RL is essential for sample efficiency - master it!** 🧭

---

## Additional Resources

### Papers
- Sutton (1990): "Integrated Architectures for Learning, Planning, and Reacting" (Dyna)
- Ha & Schmidhuber (2018): "World Models"
- Chua et al. (2018): "Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models" (PETS)
- Janner et al. (2019): "When to Trust Your Model: Model-Based Policy Optimization" (MBPO)
- Hafner et al. (2020): "Dream to Control: Learning Behaviors by Latent Imagination" (Dreamer)
- Hafner et al. (2021): "Mastering Atari with Discrete World Models" (DreamerV2)

### Libraries
- **mbrl-lib**: https://github.com/facebookresearch/mbrl-lib (Meta's MBRL library)
- **MBPO**: https://github.com/jannerm/mbpo
- **Dreamer**: https://github.com/danijar/dreamer

### Resources
- **Model-Based RL Tutorial**: https://sites.google.com/view/mbrl-tutorial
- **World Models Blog**: https://worldmodels.github.io/
- **BAIR Blog on MBRL**: https://bair.berkeley.edu/blog/2019/12/12/mbpo/

---

**Next**: [Lesson 6 - Imitation Learning and Inverse RL](Lesson%206%20-%20Imitation%20Learning%20and%20Inverse%20RL.md)

Proceed to learn about **learning from demonstrations** and **inverse reinforcement learning**! 👨‍🏫

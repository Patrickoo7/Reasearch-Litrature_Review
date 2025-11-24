# Lesson 3: Deep Q-Networks & Value-Based Deep RL 🧠

**Module 12: Reinforcement Learning | Lesson 3 of 12**

Master deep reinforcement learning with neural network function approximation and solve Atari games!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand function approximation and why we need deep RL
2. ✅ Implement DQN from scratch with Experience Replay and Target Networks
3. ✅ Master advanced DQN variants (Double, Dueling, Prioritized, Noisy, Rainbow)
4. ✅ Train agents on CartPole, Atari games, and continuous control
5. ✅ Use Stable-Baselines3 for production-quality implementations
6. ✅ Understand when to use value-based vs policy-based methods

---

## 1. Why Deep RL? The Curse of Dimensionality

### The Problem with Tabular Methods

Tabular Q-Learning works great for small state spaces (GridWorld, FrozenLake). But what about:
- **Atari games**: 210 × 160 × 3 pixels = 100,800 dimensions!
- **Robotics**: Continuous state spaces (infinite states)
- **Go**: 10^170 possible states

**Solution**: **Function approximation** - use neural networks to approximate Q(s,a)!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
from collections import deque
import random
import matplotlib.pyplot as plt

# Why we need function approximation
print("State space sizes:")
print(f"FrozenLake: {gym.make('FrozenLake-v1').observation_space.n} states")
print(f"CartPole: Continuous (infinite states!)")
print(f"Atari Pong: 210 × 160 × 3 = 100,800-dimensional")
print("\nTabular Q-Learning: Store one value per (state, action)")
print("Deep Q-Network: Use neural network to generalize across states!")
```

### Function Approximation

```python
"""
Tabular Q-Learning:
Q(s,a) stored in table

Deep Q-Network:
Q(s,a) = Q(s,a; θ) where θ are neural network weights

Benefits:
- Handle high-dimensional states
- Generalization across similar states
- Continuous state spaces
"""

class QNetwork(nn.Module):
    """
    Simple Q-Network for function approximation.

    Input: State vector
    Output: Q-values for each action
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass: state → Q-values

        Args:
            state: State tensor [batch_size, state_dim]

        Returns:
            Q-values [batch_size, action_dim]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


# Example usage
state_dim = 4  # CartPole
action_dim = 2
q_net = QNetwork(state_dim, action_dim)

# Random state
state = torch.randn(1, state_dim)
q_values = q_net(state)

print("\nQ-Network Example:")
print(f"Input state: {state.shape}")
print(f"Output Q-values: {q_values}")
print(f"Best action: {torch.argmax(q_values).item()}")
```

---

## 2. Deep Q-Network (DQN) - The Breakthrough

DQN (Mnih et al., 2015) was the first deep RL algorithm to master Atari games! Two key innovations:
1. **Experience Replay**: Break correlations in data
2. **Target Network**: Stabilize training

### DQN Algorithm

```python
class ReplayBuffer:
    """
    Experience Replay Buffer.

    Store transitions and sample random minibatches for training.
    This breaks temporal correlations!
    """
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Add transition to buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample random batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent.

    Key components:
    1. Q-network (policy network)
    2. Target network (for stable targets)
    3. Experience replay buffer
    4. ε-greedy exploration
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Q-network and target network
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=100000)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network.to(self.device)
        self.target_network.to(self.device)

    def select_action(self, state, training=True):
        """
        ε-greedy action selection.

        During training: Explore with probability ε
        During evaluation: Always exploit
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def train_step(self, batch_size=64):
        """
        Perform one training step.

        1. Sample batch from replay buffer
        2. Compute Q-learning targets
        3. Update Q-network
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (using target network!)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from Q-network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# Train DQN on CartPole
def train_dqn(env_name='CartPole-v1', num_episodes=500):
    """Train DQN agent."""
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim)
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0
        done = False

        while not done:
            # Select action
            action = agent.select_action(state)

            # Take action
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            # Store transition
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Train
            loss = agent.train_step(batch_size=64)

            episode_reward += reward
            state = next_state

        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        episode_rewards.append(episode_reward)

        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, ε: {agent.epsilon:.3f}")

    env.close()
    return agent, episode_rewards


# Train DQN
print("\n=== Training DQN on CartPole ===")
agent, rewards = train_dqn(num_episodes=500)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(rewards, alpha=0.3, label='Episode Reward')
plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2, label='Moving Average')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('DQN Training on CartPole')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nFinal 100-episode average: {np.mean(rewards[-100:]):.2f}")
```

**Key DQN Components:**
1. **Neural Network**: Approximate Q(s,a)
2. **Experience Replay**: Store and sample transitions randomly
3. **Target Network**: Separate network for computing targets (updated periodically)
4. **ε-greedy**: Exploration strategy

---

## 3. Double DQN - Reducing Overestimation

Standard DQN overestimates Q-values due to max operator. Double DQN fixes this!

```python
class DoubleDQNAgent(DQNAgent):
    """
    Double DQN: Decouple action selection and evaluation.

    Standard DQN:
        target = r + γ * max_a' Q_target(s', a')

    Double DQN:
        a_max = argmax_a' Q(s', a')       # Select using online network
        target = r + γ * Q_target(s', a_max)  # Evaluate using target network
    """
    def train_step(self, batch_size=64):
        """Training step with Double DQN update."""
        if len(self.replay_buffer) < batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: Select action with online network, evaluate with target network
        with torch.no_grad():
            # Select best action using online network
            next_actions = self.q_network(next_states).argmax(1)

            # Evaluate using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)

            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


print("\n=== Double DQN ===")
print("Reduces overestimation bias by decoupling action selection and evaluation")
print("More stable learning, especially in noisy environments")
```

---

## 4. Dueling DQN - Separate Value and Advantage

Dueling DQN decomposes Q(s,a) into Value and Advantage functions.

```python
class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network Architecture.

    Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Where:
    - V(s): State value (how good is this state?)
    - A(s,a): Advantage (how much better is this action?)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingQNetwork, self).__init__()

        # Shared feature extractor
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        """
        Forward pass with value-advantage decomposition.
        """
        features = self.feature(state)

        # Compute V(s) and A(s,a)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        # Subtracting mean ensures identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


# Example
dueling_net = DuelingQNetwork(state_dim=4, action_dim=2)
state = torch.randn(1, 4)
q_values = dueling_net(state)

print("\nDueling DQN:")
print(f"Q-values: {q_values}")
print("Advantage: Explicitly models which actions are better")
print("Value: State value independent of action choice")
```

---

## 5. Prioritized Experience Replay

Not all experiences are equally important! Prioritize high-error transitions.

```python
class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay.

    Sample transitions proportional to their TD error.
    Transitions with high error are more surprising → sample more often!
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # How much prioritization (0=uniform, 1=full)
        self.beta = beta    # Importance sampling correction
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample batch according to priorities."""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]

        # Compute sampling probabilities
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)

        # Importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize

        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*transitions)

        return (np.array(states),
                np.array(actions),
                np.array(rewards, dtype=np.float32),
                np.array(next_states),
                np.array(dones, dtype=np.float32),
                indices,
                torch.FloatTensor(weights))

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = abs(error) + 1e-6  # Small constant to avoid zero priority

    def __len__(self):
        return len(self.buffer)


print("\nPrioritized Experience Replay:")
print("✅ Sample important transitions more frequently")
print("✅ Faster learning on critical experiences")
print("✅ Use importance sampling to correct bias")
```

---

## 6. Noisy Networks for Exploration

Replace ε-greedy with learned exploration!

```python
class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for exploration.

    Add learnable noise to weights and biases.
    Network learns when/how to explore!
    """
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Register noise buffers
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """Sample new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Factorized Gaussian noise."""
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        """Forward with noisy weights."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """Q-Network with Noisy Layers."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.noisy1 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy2 = NoisyLinear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.noisy1(x))
        q_values = self.noisy2(x)
        return q_values

    def reset_noise(self):
        """Reset noise in all noisy layers."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


print("\nNoisy Networks:")
print("✅ Learned exploration (no ε-greedy needed!)")
print("✅ State-dependent exploration")
print("✅ Better for deep architectures")
```

---

## 7. Distributional RL: C51 and QR-DQN

Instead of learning expected Q-value, learn the **distribution** of returns!

```python
class C51Network(nn.Module):
    """
    C51: Categorical DQN (Bellemare et al., 2017).

    Learn distribution over returns instead of expected value.
    Represents Q-distribution with 51 atoms (support points).
    """
    def __init__(self, state_dim, action_dim, num_atoms=51, v_min=-10, v_max=10):
        super(C51Network, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # Support: discrete values the return can take
        self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Network outputs distribution for each action
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim * num_atoms)

    def forward(self, state):
        """
        Output: Distribution over returns for each action.

        Shape: [batch_size, action_dim, num_atoms]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        # Reshape and apply softmax
        batch_size = state.size(0)
        x = x.view(batch_size, self.action_dim, self.num_atoms)
        distributions = F.softmax(x, dim=2)

        return distributions

    def get_q_values(self, state):
        """Compute Q-values as expected value of distribution."""
        distributions = self.forward(state)
        q_values = (distributions * self.support).sum(dim=2)
        return q_values


# Example
c51_net = C51Network(state_dim=4, action_dim=2)
state = torch.randn(1, 4)
distributions = c51_net(state)
q_values = c51_net.get_q_values(state)

print("\nC51 Distributional RL:")
print(f"Distribution shape: {distributions.shape}")  # [1, 2, 51]
print(f"Q-values: {q_values}")
print("✅ Captures full distribution of returns")
print("✅ Better representation of uncertainty")
print("✅ Improved performance on Atari")
```

---

## 8. Rainbow DQN - Combining Everything!

Rainbow DQN combines 6 DQN extensions for state-of-the-art performance.

```python
"""
Rainbow DQN Components:

1. ✅ DQN: Basic value-based deep RL
2. ✅ Double DQN: Reduce overestimation
3. ✅ Dueling DQN: Value-advantage decomposition
4. ✅ Prioritized Experience Replay: Sample important transitions
5. ✅ Multi-step Learning: n-step returns
6. ✅ Noisy Networks: Learned exploration
7. ✅ Distributional RL (C51): Learn return distribution

Rainbow = ALL OF THE ABOVE!
"""

class RainbowDQN:
    """
    Rainbow DQN: Combines all major DQN improvements.

    This is the state-of-the-art value-based deep RL algorithm!
    """
    def __init__(self, state_dim, action_dim):
        # Dueling + Noisy + Distributional architecture
        self.network = self._build_network(state_dim, action_dim)

        # Prioritized Experience Replay
        self.replay_buffer = PrioritizedReplayBuffer()

        # Multi-step returns
        self.n_step = 3
        self.gamma = 0.99

        print("Rainbow DQN initialized with:")
        print("✅ Double Q-Learning")
        print("✅ Dueling Architecture")
        print("✅ Prioritized Replay")
        print("✅ Multi-step Returns")
        print("✅ Noisy Networks")
        print("✅ Distributional RL (C51)")

    def _build_network(self, state_dim, action_dim):
        """Build Rainbow network architecture."""
        # In practice, combine:
        # - Dueling architecture
        # - Noisy layers
        # - C51 distributional output
        return C51Network(state_dim, action_dim)


print("\n=== Rainbow DQN ===")
print("State-of-the-art value-based deep RL")
print("Human-level performance on Atari games!")
```

---

## 9. Convolutional DQN for Atari

For image-based tasks (Atari), use convolutional networks!

```python
class AtariDQN(nn.Module):
    """
    DQN for Atari games with convolutional layers.

    Input: 84x84x4 stacked grayscale frames
    Output: Q-values for each action
    """
    def __init__(self, num_actions):
        super(AtariDQN, self).__init__()

        # Convolutional layers (from DQN paper)
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        Forward pass for Atari frames.

        Args:
            x: Stacked frames [batch_size, 4, 84, 84]

        Returns:
            Q-values [batch_size, num_actions]
        """
        x = x / 255.0  # Normalize pixel values

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values


# Example (random input)
atari_net = AtariDQN(num_actions=4)  # Pong has 4 actions
frames = torch.randn(1, 4, 84, 84)  # Batch of 1, 4 stacked frames
q_values = atari_net(frames)

print("\nAtari DQN:")
print(f"Input: {frames.shape}")
print(f"Output Q-values: {q_values.shape}")
print("Architecture from 'Playing Atari with Deep Reinforcement Learning' (2013)")
```

---

## 10. Stable-Baselines3: Production-Quality DQN

For real projects, use Stable-Baselines3!

```python
"""
Stable-Baselines3: Production-quality RL implementations.

Installation:
pip install stable-baselines3[extra]
"""

# Example: Train DQN on CartPole with SB3
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Create environment
env = gym.make('CartPole-v1')

# Create DQN agent
model = DQN(
    'MlpPolicy',
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=1000,
    batch_size=64,
    tau=0.005,  # Soft update coefficient
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_initial_eps=1.0,
    exploration_final_eps=0.05,
    verbose=1
)

# Train
print("\n=== Training DQN with Stable-Baselines3 ===")
model.learn(total_timesteps=50000)

# Evaluate
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"\nMean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# Save model
model.save("dqn_cartpole")

# Load model
# model = DQN.load("dqn_cartpole")

env.close()

print("\nStable-Baselines3 provides:")
print("✅ Production-quality implementations")
print("✅ Extensive testing and validation")
print("✅ Easy hyperparameter tuning")
print("✅ Tensorboard logging")
print("✅ Callbacks and custom policies")
```

---

## 11. When to Use Value-Based Methods?

```python
"""
Use DQN and value-based methods when:

✅ Discrete action spaces
   - Atari games, board games, discrete control
   - Need argmax over actions

✅ Off-policy learning desired
   - Learn from replay buffer
   - Sample efficiency important

✅ Q-function more natural than policy
   - Clear action ranking
   - Multi-task learning (different rewards)

❌ DON'T use for:
   - Continuous action spaces (use DDPG, TD3, SAC instead)
   - High-dimensional action spaces (combinatorial explosion)
   - When stochastic policy needed (use policy gradients)

Summary:
- DQN: Discrete actions, off-policy
- Policy Gradients: Continuous actions, on-policy
- Actor-Critic: Best of both worlds (Lesson 4!)
"""
```

---

## Practice Exercises

### Exercise 1: Implement Soft Target Updates

```python
"""
Implement soft target network updates (Polyak averaging).

Instead of hard updates (copying weights):
    θ_target ← θ

Use soft updates:
    θ_target ← τ * θ + (1 - τ) * θ_target

Where τ is small (e.g., 0.005).

Hints:
- More stable than hard updates
- Used in DDPG, TD3, SAC
- Update after every training step

Implement this in DQNAgent class.
"""

# Your implementation here
```

### Exercise 2: Multi-step Returns

```python
"""
Implement n-step DQN.

Instead of 1-step return:
    target = r + γ * max_a' Q(s', a')

Use n-step return:
    target = r_t + γ*r_{t+1} + ... + γ^n * max_a' Q(s_{t+n}, a')

Hints:
- Store n-step transitions in replay buffer
- Reduces bias, increases variance
- n=3 to 5 works well

Modify ReplayBuffer to support n-step returns.
"""

# Your implementation here
```

### Exercise 3: Train DQN on Atari

```python
"""
Train DQN on an Atari game (e.g., Pong, Breakout).

Requirements:
- Use AtariDQN architecture
- Frame preprocessing (grayscale, resize to 84x84)
- Frame stacking (4 frames)
- Reward clipping
- Large replay buffer (1M transitions)

Use gymnasium[atari] or stable-baselines3.

Bonus: Compare DQN vs Double DQN vs Rainbow!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Function Approximation** 🧠
   - Neural networks approximate Q(s,a)
   - Handle high-dimensional/continuous states
   - Generalize across similar states
   - Required for complex tasks (Atari, robotics)

2. **DQN Innovations** 🚀
   - Experience Replay: Break correlations
   - Target Network: Stabilize learning
   - Enables deep RL with stable training
   - First to achieve human-level Atari performance

3. **Double DQN** ⚖️
   - Decouple action selection and evaluation
   - Reduces overestimation bias
   - More stable, better performance
   - Simple modification, big impact

4. **Dueling Architecture** 🏗️
   - Separate V(s) and A(s,a)
   - Better representation of state value
   - Especially useful when actions don't matter much
   - Improves sample efficiency

5. **Prioritized Replay** 📊
   - Sample important transitions more
   - Faster learning on critical experiences
   - Importance sampling for unbiased updates
   - Significant speedup in practice

6. **Noisy Networks** 🎲
   - Learned exploration
   - State-dependent noise
   - No ε-greedy needed
   - Better for complex environments

7. **Distributional RL** 📈
   - Learn full return distribution
   - C51, QR-DQN, IQN
   - Better uncertainty quantification
   - Improved performance and stability

8. **Rainbow DQN** 🌈
   - Combines 6+ extensions
   - State-of-the-art value-based RL
   - Human-level Atari performance
   - Use as baseline for research

### Practical Advice

✅ **Start simple**: Basic DQN first, add extensions gradually
✅ **Hyperparameters matter**: Learning rate, buffer size, update frequency
✅ **Use Stable-Baselines3**: Production-quality, well-tested
✅ **Monitor training**: Loss, Q-values, epsilon, episode rewards
✅ **Preprocess carefully**: Frame stacking, normalization for images

### Real-World Applications

✅ **Game AI**: Atari, board games, video games
✅ **Robotic Control**: Discrete action spaces
✅ **Trading**: Discrete buy/sell/hold decisions
✅ **Recommendation**: Next-item prediction
✅ **Resource Allocation**: Server assignment, routing

### What's Next?

In Lesson 4, we'll learn **Policy Gradient Methods**:
- REINFORCE, Actor-Critic, PPO
- Continuous action spaces
- When to use policy-based vs value-based
- Best of both worlds with actor-critic!

**Value-based methods master discrete control - learn them well!** 🎯

---

## Additional Resources

### Papers
- Mnih et al. (2015): "Human-level control through deep RL" (DQN)
- Van Hasselt et al. (2016): "Deep Reinforcement Learning with Double Q-Learning"
- Wang et al. (2016): "Dueling Network Architectures"
- Schaul et al. (2016): "Prioritized Experience Replay"
- Fortunato et al. (2018): "Noisy Networks for Exploration"
- Bellemare et al. (2017): "A Distributional Perspective on RL" (C51)
- Hessel et al. (2018): "Rainbow: Combining Improvements in Deep RL"

### Libraries
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **Dopamine**: https://github.com/google/dopamine (Rainbow implementation)

### Resources
- **OpenAI Spinning Up**: DQN tutorial
- **Deep RL Course (Hugging Face)**: https://huggingface.co/deep-rl-course
- **Atari Benchmark**: https://paperswithcode.com/sota/atari-games-on-atari-2600

---

**Next**: [Lesson 4 - Policy Gradient Methods and Actor-Critic](Lesson%204%20-%20Policy%20Gradient%20Methods%20and%20Actor-Critic.md)

Proceed to learn about **policy-based methods** for continuous control! 🎮

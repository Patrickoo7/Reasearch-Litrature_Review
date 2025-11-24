# Lesson 4: Policy Gradients & Actor-Critic Methods 🎮

**Module 12: Reinforcement Learning | Lesson 4 of 12**

Master policy-based methods for continuous control and learn the powerful actor-critic framework!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the Policy Gradient Theorem and REINFORCE algorithm
2. ✅ Implement Actor-Critic methods (A2C, A3C) from scratch
3. ✅ Master Proximal Policy Optimization (PPO) - the industry standard
4. ✅ Learn continuous control algorithms (DDPG, TD3, SAC)
5. ✅ Apply to robotics, locomotion, and manipulation tasks
6. ✅ Understand when to use policy-based vs value-based methods

---

## 1. Why Policy Gradient Methods?

### Limitations of Value-Based Methods (DQN)

DQN works great for **discrete** actions, but fails for:
- **Continuous action spaces**: Robot joint angles, throttle/steering
- **High-dimensional actions**: Many degrees of freedom
- **Stochastic policies**: Sometimes randomness is optimal

**Solution**: Directly optimize the policy π(a|s; θ)!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal
import gym
import matplotlib.pyplot as plt

"""
Value-Based (DQN):
- Learn Q(s,a)
- Policy: π(s) = argmax_a Q(s,a)
- Discrete actions only

Policy-Based (Policy Gradient):
- Learn π(a|s; θ) directly
- Policy: Sample from π
- Works for continuous and stochastic policies
"""

print("=== Policy Gradient Methods ===")
print("\nAdvantages:")
print("✅ Continuous action spaces")
print("✅ Stochastic policies")
print("✅ Better convergence properties (sometimes)")
print("✅ Can learn complex behaviors")
print("\nDisadvantages:")
print("❌ High variance")
print("❌ Sample inefficient")
print("❌ Can converge to local optima")
```

---

## 2. Policy Gradient Theorem

The foundation of all policy gradient methods!

```python
"""
Policy Gradient Theorem:

∇_θ J(θ) = E_π [ ∇_θ log π(a|s; θ) * Q^π(s,a) ]

Where:
- J(θ) = Expected return under policy π_θ
- ∇_θ log π(a|s; θ) = Score function (gradient of log probability)
- Q^π(s,a) = Action value (how good is this action?)

Intuition: Increase probability of good actions, decrease probability of bad actions!
"""

def policy_gradient_intuition():
    """
    Visualize policy gradient intuition.
    """
    print("\nPolicy Gradient Intuition:")
    print("1. Sample actions from current policy π(a|s; θ)")
    print("2. Evaluate actions (positive or negative reward)")
    print("3. Increase probability of good actions")
    print("4. Decrease probability of bad actions")
    print("5. Repeat!")
    print("\nMath: ∇_θ J(θ) = E[ ∇_θ log π(a|s) * Q(s,a) ]")

policy_gradient_intuition()
```

---

## 3. REINFORCE Algorithm

REINFORCE is the simplest policy gradient algorithm. Monte Carlo approach!

```python
class PolicyNetwork(nn.Module):
    """
    Policy Network for discrete actions.

    Input: State
    Output: Action probabilities
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass: state → action probabilities
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

    def select_action(self, state):
        """
        Sample action from policy.

        Returns:
            action: Sampled action
            log_prob: Log probability of action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.forward(state_tensor)

        # Sample action
        distribution = Categorical(action_probs)
        action = distribution.sample()

        # Log probability for gradient computation
        log_prob = distribution.log_prob(action)

        return action.item(), log_prob


class REINFORCE:
    """
    REINFORCE algorithm (Monte Carlo Policy Gradient).

    Algorithm:
    1. Generate episode using current policy
    2. Compute returns G_t for each timestep
    3. Update policy: θ ← θ + α * ∇_θ log π(a|s) * G_t
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma

    def compute_returns(self, rewards):
        """
        Compute discounted returns for each timestep.

        G_t = Σ_{k=0}^{∞} γ^k * R_{t+k+1}
        """
        returns = []
        G = 0

        # Compute returns backwards
        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        # Normalize returns (reduces variance)
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def train_episode(self, env):
        """
        Train on one episode.
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        log_probs = []
        rewards = []

        # Generate episode
        done = False
        while not done:
            # Select action
            action, log_prob = self.policy.select_action(state)
            log_probs.append(log_prob)

            # Take action
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            rewards.append(reward)
            state = next_state

        # Compute returns
        returns = self.compute_returns(rewards)

        # Compute policy gradient loss
        policy_loss = []
        for log_prob, G in zip(log_probs, returns):
            # Loss = -log π(a|s) * G  (negative because we maximize)
            policy_loss.append(-log_prob * G)

        policy_loss = torch.stack(policy_loss).sum()

        # Update policy
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return sum(rewards), policy_loss.item()

    def train(self, env, num_episodes=1000):
        """
        Train REINFORCE agent.
        """
        episode_rewards = []

        for episode in range(num_episodes):
            episode_reward, loss = self.train_episode(env)
            episode_rewards.append(episode_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.2f}")

        return episode_rewards


# Train REINFORCE on CartPole
print("\n=== Training REINFORCE on CartPole ===")
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

agent = REINFORCE(state_dim, action_dim, lr=1e-2, gamma=0.99)
rewards = agent.train(env, num_episodes=1000)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(rewards, alpha=0.3)
plt.plot(np.convolve(rewards, np.ones(50)/50, mode='valid'), linewidth=2)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('REINFORCE Training')
plt.grid(True)
plt.show()

env.close()
```

**REINFORCE Characteristics:**
- ✅ Simple and intuitive
- ✅ Unbiased gradient estimates
- ❌ High variance (uses full episode returns)
- ❌ Sample inefficient

---

## 4. REINFORCE with Baseline

Reduce variance by subtracting a baseline (usually state value V(s)).

```python
class REINFORCEWithBaseline:
    """
    REINFORCE with baseline to reduce variance.

    Use value function V(s) as baseline:
    ∇_θ J(θ) = E[ ∇_θ log π(a|s) * (Q(s,a) - V(s)) ]

    The advantage A(s,a) = Q(s,a) - V(s) reduces variance!
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
        self.gamma = gamma

    def compute_advantages(self, rewards, values):
        """
        Compute advantages: A_t = G_t - V(s_t)

        This reduces variance while keeping gradient unbiased!
        """
        returns = []
        G = 0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)
        advantages = returns - values

        return advantages, returns

    def train_episode(self, env):
        """Train on one episode with baseline."""
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        log_probs = []
        values = []
        rewards = []
        states = []

        done = False
        while not done:
            # Select action
            action, log_prob = self.policy.select_action(state)
            log_probs.append(log_prob)

            # Compute value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            value = self.value_net(state_tensor)
            values.append(value)

            states.append(state)

            # Take action
            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            rewards.append(reward)
            state = next_state

        # Compute advantages
        values = torch.cat(values)
        advantages, returns = self.compute_advantages(rewards, values.detach())

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy loss
        policy_loss = []
        for log_prob, advantage in zip(log_probs, advantages):
            policy_loss.append(-log_prob * advantage)
        policy_loss = torch.stack(policy_loss).sum()

        # Value loss
        value_loss = F.mse_loss(values, returns)

        # Update networks
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return sum(rewards)


print("\nREINFORCE with Baseline:")
print("✅ Lower variance than vanilla REINFORCE")
print("✅ Faster learning")
print("✅ Uses advantage A(s,a) = G_t - V(s)")
```

---

## 5. Actor-Critic Methods

Actor-Critic combines policy gradients (actor) with value function learning (critic).

```python
class ActorCritic(nn.Module):
    """
    Actor-Critic Network.

    Actor: Policy π(a|s; θ)
    Critic: Value function V(s; w)
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        Forward pass.

        Returns:
            action_probs: Policy distribution
            value: State value
        """
        features = self.shared(state)
        action_probs = F.softmax(self.actor(features), dim=-1)
        value = self.critic(features)
        return action_probs, value


class A2C:
    """
    Advantage Actor-Critic (A2C).

    Online actor-critic using TD error as advantage estimate:
    A(s,a) ≈ δ = r + γV(s') - V(s)
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.ac_net = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)
        self.gamma = gamma

    def train_step(self, env):
        """
        Train one step (online learning).
        """
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0
        done = False

        while not done:
            # Get action probabilities and value
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.ac_net(state_tensor)

            # Sample action
            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            # Take action
            result = env.step(action.item())
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            episode_reward += reward

            # Compute TD error (advantage)
            if done:
                next_value = 0
            else:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, next_value = self.ac_net(next_state_tensor)
                next_value = next_value.detach()

            # TD error: δ = r + γV(s') - V(s)
            td_error = reward + self.gamma * next_value - value

            # Actor loss: -log π(a|s) * A(s,a)
            actor_loss = -log_prob * td_error.detach()

            # Critic loss: MSE between V(s) and target
            target = reward + self.gamma * next_value
            critic_loss = F.mse_loss(value, target.detach())

            # Combined loss
            loss = actor_loss + critic_loss

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            state = next_state

        return episode_reward

    def train(self, env, num_episodes=1000):
        """Train A2C agent."""
        episode_rewards = []

        for episode in range(num_episodes):
            episode_reward = self.train_step(env)
            episode_rewards.append(episode_reward)

            if episode % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}")

        return episode_rewards


# Train A2C
print("\n=== Training A2C on CartPole ===")
env = gym.make('CartPole-v1')
a2c_agent = A2C(state_dim=4, action_dim=2, lr=1e-3)
a2c_rewards = a2c_agent.train(env, num_episodes=500)
env.close()

print("\nA2C Characteristics:")
print("✅ Lower variance than REINFORCE")
print("✅ Online learning (more sample efficient)")
print("✅ Combines policy and value learning")
```

---

## 6. Generalized Advantage Estimation (GAE)

GAE provides a better advantage estimate by blending n-step returns.

```python
def compute_gae(rewards, values, next_value, gamma=0.99, lambda_=0.95):
    """
    Compute Generalized Advantage Estimation.

    GAE balances bias-variance tradeoff:
    A^GAE_t = Σ_{l=0}^{∞} (γλ)^l * δ_{t+l}

    Where δ_t = r_t + γV(s_{t+1}) - V(s_t) is TD error.

    Args:
        rewards: List of rewards
        values: List of state values V(s_t)
        next_value: V(s_T) for last state
        gamma: Discount factor
        lambda_: GAE parameter (0=high bias/low variance, 1=low bias/high variance)

    Returns:
        advantages: GAE advantages
        returns: TD(λ) returns
    """
    advantages = []
    gae = 0

    # Append next_value for computation
    values = values + [next_value]

    # Compute GAE backwards
    for t in reversed(range(len(rewards))):
        # TD error
        delta = rewards[t] + gamma * values[t + 1] - values[t]

        # GAE recursion
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)

    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(values[:-1])

    return advantages, returns


print("\nGeneralized Advantage Estimation (GAE):")
print("✅ Balances bias-variance tradeoff")
print("✅ λ=0: Use TD error (low variance, high bias)")
print("✅ λ=1: Use Monte Carlo returns (high variance, low bias)")
print("✅ λ=0.95: Sweet spot in practice!")
```

---

## 7. Proximal Policy Optimization (PPO) - Industry Standard!

PPO is the most popular RL algorithm today! Used by OpenAI, DeepMind, and industry.

```python
class PPO:
    """
    Proximal Policy Optimization (PPO).

    Key innovations:
    1. Clipped surrogate objective (prevents large policy updates)
    2. Multiple epochs of minibatch updates
    3. GAE for advantage estimation

    PPO is the industry standard due to:
    - Sample efficiency
    - Stability
    - Easy to tune
    - Great performance
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 lambda_=0.95, epsilon=0.2, epochs=10, batch_size=64):
        self.ac_net = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=lr)

        self.gamma = gamma
        self.lambda_ = lambda_
        self.epsilon = epsilon  # Clipping parameter
        self.epochs = epochs
        self.batch_size = batch_size

    def collect_rollouts(self, env, num_steps=2048):
        """
        Collect rollouts from environment.

        Returns:
            states, actions, log_probs, rewards, dones, values
        """
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        for _ in range(num_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs, value = self.ac_net(state_tensor)

            distribution = Categorical(action_probs)
            action = distribution.sample()
            log_prob = distribution.log_prob(action)

            result = env.step(action.item())
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            dones.append(done)
            values.append(value.item())

            state = next_state
            if done:
                state = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        return states, actions, log_probs, rewards, dones, values

    def compute_advantages_gae(self, rewards, values, dones):
        """Compute advantages using GAE."""
        advantages = []
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            if dones[t]:
                next_value = 0

            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.lambda_ * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages)
        returns = advantages + torch.FloatTensor(values)

        return advantages, returns

    def ppo_update(self, states, actions, old_log_probs, advantages, returns):
        """
        PPO update with clipped objective.

        L^CLIP(θ) = E[ min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t) ]

        Where r_t(θ) = π_θ(a|s) / π_θ_old(a|s) is the probability ratio.
        """
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Multiple epochs of updates
        for _ in range(self.epochs):
            # Get current policy and values
            action_probs, values = self.ac_net(states)
            distribution = Categorical(action_probs)
            log_probs = distribution.log_prob(actions)
            entropy = distribution.entropy().mean()

            # Probability ratio
            ratio = torch.exp(log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            critic_loss = F.mse_loss(values.squeeze(), returns)

            # Total loss (with entropy bonus for exploration)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), 0.5)
            self.optimizer.step()

    def train(self, env, num_iterations=100, steps_per_iteration=2048):
        """Train PPO agent."""
        iteration_rewards = []

        for iteration in range(num_iterations):
            # Collect rollouts
            states, actions, log_probs, rewards, dones, values = \
                self.collect_rollouts(env, steps_per_iteration)

            # Compute advantages
            advantages, returns = self.compute_advantages_gae(rewards, values, dones)

            # PPO update
            self.ppo_update(states, actions, log_probs, advantages, returns)

            # Log results
            avg_reward = np.mean(rewards)
            iteration_rewards.append(avg_reward)

            print(f"Iteration {iteration}, Avg Reward: {avg_reward:.2f}")

        return iteration_rewards


# Train PPO
print("\n=== Training PPO on CartPole ===")
env = gym.make('CartPole-v1')
ppo_agent = PPO(state_dim=4, action_dim=2, lr=3e-4)
ppo_rewards = ppo_agent.train(env, num_iterations=50, steps_per_iteration=2048)
env.close()

print("\nPPO is the INDUSTRY STANDARD:")
print("✅ Used by OpenAI (ChatGPT, Dota 2)")
print("✅ Stable and easy to tune")
print("✅ Great performance across tasks")
print("✅ Sample efficient")
```

---

## 8. Continuous Control: DDPG

For continuous action spaces (robotics!), we need different approaches.

```python
class ContinuousPolicyNetwork(nn.Module):
    """
    Policy network for continuous actions.

    Outputs mean and log_std for Gaussian policy.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ContinuousPolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # Stabilize
        return mean, log_std

    def sample(self, state):
        """Sample action from Gaussian policy."""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action).sum(dim=-1)
        return action, log_prob


class DDPG:
    """
    Deep Deterministic Policy Gradient (DDPG).

    Actor-critic for continuous control:
    - Actor: Deterministic policy μ(s; θ)
    - Critic: Q(s,a; w)

    Like DQN for continuous actions!
    """
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005):
        # Actor network (deterministic policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Bound actions to [-1, 1]
        )

        # Critic network (Q-function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Target networks
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic_target = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Copy weights to target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.gamma = gamma
        self.tau = tau  # Soft update coefficient

    def select_action(self, state, noise_scale=0.1):
        """Select action with exploration noise."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]

        # Add Gaussian noise for exploration
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1, 1)

        return action

    def soft_update(self, target, source):
        """Soft update: θ_target ← τ*θ + (1-τ)*θ_target"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


print("\nDDPG for Continuous Control:")
print("✅ Deterministic policy")
print("✅ Off-policy (uses replay buffer)")
print("✅ Actor-critic architecture")
print("✅ Used for robotics and control")
```

---

## 9. Twin Delayed DDPG (TD3)

TD3 improves DDPG with three key tricks!

```python
"""
TD3 Improvements over DDPG:

1. Twin Q-Networks (Double Q-Learning for continuous actions)
   - Maintain two Q-networks
   - Use minimum for target computation
   - Reduces overestimation

2. Delayed Policy Updates
   - Update actor less frequently than critic
   - More stable learning

3. Target Policy Smoothing
   - Add noise to target actions
   - Smooth out value estimates
   - More robust

TD3 is more stable and performs better than DDPG!
"""

print("\nTD3 (Twin Delayed DDPG):")
print("✅ More stable than DDPG")
print("✅ State-of-the-art continuous control")
print("✅ Three key improvements")
print("✅ Recommended over DDPG for new projects")
```

---

## 10. Soft Actor-Critic (SAC)

SAC is the current state-of-the-art for continuous control!

```python
"""
Soft Actor-Critic (SAC):

Maximum entropy RL:
J(π) = Σ_t E[ r(s_t, a_t) + α * H(π(·|s_t)) ]

Where H(π) is policy entropy (encourages exploration).

Key features:
- Stochastic policy (unlike DDPG/TD3)
- Automatic temperature tuning
- Very sample efficient
- Stable across tasks

SAC is the BEST algorithm for continuous control tasks!
"""

class SAC:
    """
    Soft Actor-Critic (SAC).

    State-of-the-art for continuous control!
    """
    def __init__(self, state_dim, action_dim):
        # Stochastic policy (Gaussian)
        self.actor = ContinuousPolicyNetwork(state_dim, action_dim)

        # Twin Q-networks
        self.critic1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Automatic entropy tuning
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha = self.log_alpha.exp()

        print("SAC initialized!")
        print("✅ Stochastic policy")
        print("✅ Twin Q-networks")
        print("✅ Automatic temperature tuning")
        print("✅ State-of-the-art continuous control")


print("\nSAC is the GOLD STANDARD for continuous control:")
print("✅ Best sample efficiency")
print("✅ Stable across domains")
print("✅ Automatic hyperparameter tuning")
print("✅ Use this for robotics!")
```

---

## 11. Stable-Baselines3: Production PPO and SAC

```python
"""
Using Stable-Baselines3 for production-quality implementations.
"""

from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.evaluation import evaluate_policy

# PPO for discrete/continuous control
print("\n=== Stable-Baselines3 PPO ===")
env = gym.make('CartPole-v1')

model_ppo = PPO(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

model_ppo.learn(total_timesteps=100000)
mean_reward, _ = evaluate_policy(model_ppo, env, n_eval_episodes=10)
print(f"PPO Mean Reward: {mean_reward:.2f}")

env.close()

# SAC for continuous control
print("\n=== Stable-Baselines3 SAC ===")
env = gym.make('Pendulum-v1')

model_sac = SAC(
    'MlpPolicy',
    env,
    learning_rate=3e-4,
    buffer_size=1000000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    verbose=1
)

model_sac.learn(total_timesteps=50000)
mean_reward, _ = evaluate_policy(model_sac, env, n_eval_episodes=10)
print(f"SAC Mean Reward: {mean_reward:.2f}")

env.close()

print("\nStable-Baselines3 Algorithms:")
print("✅ PPO: Discrete/continuous, on-policy")
print("✅ SAC: Continuous, off-policy, SOTA")
print("✅ TD3: Continuous, off-policy")
print("✅ A2C: Discrete/continuous, on-policy")
```

---

## Practice Exercises

### Exercise 1: Implement TRPO

```python
"""
Implement Trust Region Policy Optimization (TRPO).

TRPO is the predecessor to PPO. Instead of clipping, it uses:
- KL divergence constraint between old and new policy
- Conjugate gradient for optimization
- Line search for step size

More principled than PPO, but more complex to implement.

Hints:
- Use KL(π_old || π_new) ≤ δ constraint
- Fisher Information Matrix approximation
- Natural gradient updates
"""

# Your implementation here
```

### Exercise 2: Multi-Environment PPO (A3C-style)

```python
"""
Implement parallel environment training for PPO.

Run multiple environments in parallel to:
- Increase sample diversity
- Speed up training
- Reduce correlation

Hints:
- Use multiprocessing or vectorized environments
- Collect rollouts from all environments
- Combine data for updates

Check SubprocVecEnv from stable-baselines3!
"""

# Your implementation here
```

### Exercise 3: Apply SAC to Robotics

```python
"""
Train SAC on a robotics environment (e.g., Ant, Humanoid, FetchReach).

Requirements:
- Use Gymnasium robotics environments
- Tune hyperparameters for best performance
- Compare with TD3 and PPO
- Visualize learned policy

Bonus: Try domain randomization for sim-to-real transfer!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Policy Gradient Theorem** 📊
   - ∇J(θ) = E[ ∇log π(a|s) * Q(s,a) ]
   - Increase probability of good actions
   - Foundation of all policy methods
   - Works for continuous and stochastic policies

2. **REINFORCE** 🎲
   - Monte Carlo policy gradient
   - Simple but high variance
   - Use with baseline to reduce variance
   - Good starting point for understanding

3. **Actor-Critic** 🎭
   - Actor: Policy network
   - Critic: Value network
   - Lower variance than REINFORCE
   - Combines best of both worlds

4. **PPO** 🏆
   - Industry standard RL algorithm
   - Clipped objective prevents large updates
   - GAE for advantage estimation
   - Stable, sample efficient, easy to tune

5. **Continuous Control** 🤖
   - DDPG: Deterministic policy, off-policy
   - TD3: Improved DDPG with tricks
   - SAC: Stochastic, entropy-regularized, SOTA
   - Essential for robotics

6. **GAE** 📈
   - Balances bias-variance tradeoff
   - λ parameter controls trade-off
   - λ=0.95 works well in practice
   - Used in PPO and modern algorithms

### Algorithm Selection Guide

✅ **Discrete Actions + Sample Efficiency**: DQN (Lesson 3)
✅ **Discrete Actions + Stability**: PPO
✅ **Continuous Actions + Sample Efficiency**: SAC
✅ **Continuous Actions + Deterministic**: TD3
✅ **Simple Baseline**: A2C
✅ **Research/New Ideas**: PPO (most robust)

### Real-World Applications

✅ **Robotics**: Locomotion (walking, running), manipulation (grasping)
✅ **Autonomous Driving**: Steering, throttle control
✅ **Game AI**: Continuous control games, strategy
✅ **Industrial Control**: Manufacturing, HVAC optimization
✅ **Finance**: Portfolio optimization, trading

### What's Next?

In Lesson 5, we'll learn **Model-Based RL**:
- Learn environment dynamics
- Planning with learned models
- Sample efficiency improvements
- World Models, PETS, Dreamer

**Policy gradients power modern RL - master PPO and SAC!** 🚀

---

## Additional Resources

### Papers
- Williams (1992): "Simple Statistical Gradient-Following Algorithms" (REINFORCE)
- Mnih et al. (2016): "Asynchronous Methods for Deep RL" (A3C)
- Schulman et al. (2015): "Trust Region Policy Optimization" (TRPO)
- Schulman et al. (2017): "Proximal Policy Optimization" (PPO) ⭐
- Lillicrap et al. (2016): "Continuous Control with Deep RL" (DDPG)
- Fujimoto et al. (2018): "Addressing Function Approximation Error" (TD3)
- Haarnoja et al. (2018): "Soft Actor-Critic" (SAC) ⭐

### Libraries
- **Stable-Baselines3**: https://stable-baselines3.readthedocs.io/ ⭐
- **CleanRL**: https://github.com/vwxyzjn/cleanrl
- **RLlib (Ray)**: https://docs.ray.io/en/latest/rllib/

### Resources
- **Spinning Up (OpenAI)**: https://spinningup.openai.com/ ⭐
- **PPO Implementation Guide**: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- **RL Algorithms Comparison**: https://paperswithcode.com/task/continuous-control

---

**Next**: [Lesson 5 - Model-Based Reinforcement Learning](Lesson%205%20-%20Model-Based%20Reinforcement%20Learning.md)

Proceed to learn about **learning environment models** and **planning**! 🧭

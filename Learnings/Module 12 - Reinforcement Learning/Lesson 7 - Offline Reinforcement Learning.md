# Lesson 7: Offline Reinforcement Learning 💾

**Module 12: Reinforcement Learning | Lesson 7 of 12**

Master learning from fixed datasets without environment interaction - critical for real-world deployment!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand offline RL and when it's essential
2. ✅ Implement conservative RL algorithms (BCQ, CQL, IQL)
3. ✅ Master Decision Transformer for sequence modeling approach
4. ✅ Learn offline policy evaluation techniques
5. ✅ Apply offline RL to healthcare, autonomous vehicles, and robotics
6. ✅ Understand the D4RL benchmark and evaluation

---

## 1. Why Offline RL?

### The Problem with Online RL

Standard RL requires **environment interaction**:
- ❌ Expensive: Real robots, medical trials, financial trading
- ❌ Dangerous: Autonomous vehicles, healthcare
- ❌ Time-consuming: Millions of interactions
- ❌ Impossible: Historical data (can't replay)

**Solution**: Learn from **fixed datasets** without interaction!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym
import matplotlib.pyplot as plt
from collections import deque
import random

"""
Offline RL (Batch RL):
Learn policy from fixed dataset D = {(s, a, r, s')}

No environment interaction during training!

When to use Offline RL:
✅ Environment interaction expensive/dangerous
✅ Large historical datasets available
✅ Safety-critical applications
✅ Cannot deploy imperfect policies

Examples:
- Healthcare: Learn from patient records
- Autonomous driving: Learn from driving logs
- Robotics: Learn from teleoperation data
- Finance: Learn from historical trades
"""

print("=== Offline Reinforcement Learning ===")
print("\nAdvantages:")
print("✅ No environment interaction needed")
print("✅ Leverages existing data")
print("✅ Safe (no risky exploration)")
print("✅ Scalable (can use massive datasets)")

print("\nChallenges:")
print("❌ Distribution shift (OOD actions)")
print("❌ Overestimation bias")
print("❌ Limited to data quality")
print("❌ Harder than online RL")
```

---

## 2. The Core Challenge: Distribution Shift

Offline RL faces severe **distribution shift**!

```python
"""
Distribution Shift in Offline RL:

Training data: D ~ π_β (behavior policy)
Learned policy: π_θ
Evaluation: Need to evaluate π_θ, but only have data from π_β!

Problem: Extrapolation Error
- π_θ may take actions not in D
- Q-values overestimated for OOD actions
- Policy diverges!

Solution: Conservative estimation
- Penalize Q-values for OOD actions
- Constrain policy to stay close to data
- Uncertainty-aware learning
"""

def visualize_distribution_shift():
    """
    Visualize distribution shift problem.
    """
    print("\nDistribution Shift Problem:")
    print("1. Dataset: Actions from behavior policy π_β")
    print("2. Learned policy π_θ takes different actions")
    print("3. Q-values unknown for new actions")
    print("4. Overestimation → Policy divergence!")

    print("\nSolution Approaches:")
    print("✅ Conservative Q-Learning (CQL)")
    print("✅ Batch-Constrained Q-Learning (BCQ)")
    print("✅ Implicit Q-Learning (IQL)")
    print("✅ Decision Transformer")

visualize_distribution_shift()
```

---

## 3. Batch-Constrained Q-Learning (BCQ)

BCQ constrains policy to actions similar to those in the dataset!

```python
class BCQ:
    """
    Batch-Constrained Deep Q-Learning (BCQ).

    Key idea: Only select actions likely under behavior policy.

    Components:
    1. Generative model: VAE(s) → samples actions from dataset
    2. Q-network: Q(s, a)
    3. Perturbation network: Small action adjustments

    Policy: π(s) = argmax_a Q(s, a)  s.t. a ~ VAE(s)
    """
    def __init__(self, state_dim, action_dim, latent_dim=32):
        # VAE for action generation
        # Encoder: (s, a) → z
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder: (s, z) → a
        self.decoder = nn.Sequential(
            nn.Linear(state_dim + latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Q-network (Twin Q)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Perturbation network: Small action adjustments
        self.perturbation = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.vae_optimizer = optim.Adam(
            list(self.encoder.parameters()) +
            list(self.decoder.parameters()) +
            [self.fc_mu, self.fc_logvar],
            lr=1e-3
        )
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=1e-3
        )
        self.perturbation_optimizer = optim.Adam(
            self.perturbation.parameters(), lr=1e-3
        )

    def encode(self, state, action):
        """Encode (s,a) to latent distribution."""
        sa = torch.cat([state, action], dim=-1)
        h = self.encoder(sa)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, state, z):
        """Decode (s,z) to action."""
        sz = torch.cat([state, z], dim=-1)
        return self.decoder(sz)

    def sample_action(self, state, num_samples=10):
        """
        Sample action constrained to dataset.

        1. Sample z from latent space
        2. Decode to action
        3. Apply perturbation
        4. Select action with highest Q-value
        """
        state = state.repeat(num_samples, 1)

        # Sample latent
        z = torch.randn(num_samples, 32).to(state.device)

        # Decode to actions
        actions = self.decode(state, z)

        # Apply perturbation
        perturbed_actions = actions + 0.05 * self.perturbation(torch.cat([state, actions], dim=-1))
        perturbed_actions = torch.clamp(perturbed_actions, -1, 1)

        # Select action with highest Q
        q_values = self.q1(torch.cat([state, perturbed_actions], dim=-1))
        idx = q_values.argmax(dim=0)

        return perturbed_actions[idx]

    def train_step(self, batch):
        """
        Train BCQ on offline batch.
        """
        states, actions, rewards, next_states, dones = batch

        # Train VAE
        mu, logvar = self.encode(states, actions)
        z = self.reparameterize(mu, logvar)
        reconstructed_actions = self.decode(states, z)

        # VAE loss
        recon_loss = F.mse_loss(reconstructed_actions, actions)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        vae_loss = recon_loss + 0.5 * kl_loss

        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        # Train Q-networks
        with torch.no_grad():
            # Sample actions for next state
            next_actions = self.sample_action(next_states[0:1]).repeat(len(next_states), 1)

            # Target Q-value
            target_q1 = self.q1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.q2(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards + (1 - dones) * 0.99 * torch.min(target_q1, target_q2)

        # Current Q-values
        current_q1 = self.q1(torch.cat([states, actions], dim=-1))
        current_q2 = self.q2(torch.cat([states, actions], dim=-1))

        # Q loss
        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return vae_loss.item(), q_loss.item()


print("\nBCQ (Batch-Constrained Q-Learning):")
print("✅ Constrains policy to dataset actions")
print("✅ VAE for action generation")
print("✅ Prevents OOD extrapolation")
print("✅ Works well in practice")
```

---

## 4. Conservative Q-Learning (CQL)

CQL learns conservative Q-functions that lower-bound true values!

```python
class CQL:
    """
    Conservative Q-Learning (CQL).

    Key idea: Learn Q-function that is lower-bounded for OOD actions.

    Loss: L_CQL = L_TD + α * (E[Q(s,a')] - E[Q(s,a)])

    Where:
    - First term: Maximize Q for random actions
    - Second term: Minimize Q for dataset actions
    - This makes Q conservative!
    """
    def __init__(self, state_dim, action_dim, alpha=1.0):
        # Twin Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=3e-4
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.alpha = alpha  # CQL regularization weight

    def cql_loss(self, states, actions):
        """
        Compute CQL regularization loss.

        Maximize Q for random actions, minimize for dataset actions.
        """
        batch_size = states.shape[0]
        action_dim = actions.shape[1]

        # Sample random actions
        random_actions = torch.FloatTensor(batch_size, 10, action_dim).uniform_(-1, 1)
        random_actions = random_actions.to(states.device)

        # Expand states for random actions
        states_expanded = states.unsqueeze(1).repeat(1, 10, 1)
        states_expanded = states_expanded.view(-1, states.shape[1])
        random_actions_flat = random_actions.view(-1, action_dim)

        # Q-values for random actions
        q1_random = self.q1(torch.cat([states_expanded, random_actions_flat], dim=-1))
        q2_random = self.q2(torch.cat([states_expanded, random_actions_flat], dim=-1))

        q1_random = q1_random.view(batch_size, 10).mean(dim=1, keepdim=True)
        q2_random = q2_random.view(batch_size, 10).mean(dim=1, keepdim=True)

        # Q-values for dataset actions
        q1_dataset = self.q1(torch.cat([states, actions], dim=-1))
        q2_dataset = self.q2(torch.cat([states, actions], dim=-1))

        # CQL loss: Maximize random - Minimize dataset
        cql_loss = (q1_random - q1_dataset).mean() + (q2_random - q2_dataset).mean()

        return cql_loss

    def train_step(self, batch, gamma=0.99):
        """Train CQL on offline batch."""
        states, actions, rewards, next_states, dones = batch

        # Standard TD loss
        with torch.no_grad():
            next_actions = self.policy(next_states)
            target_q1 = self.q1(torch.cat([next_states, next_actions], dim=-1))
            target_q2 = self.q2(torch.cat([next_states, next_actions], dim=-1))
            target_q = rewards + (1 - dones) * gamma * torch.min(target_q1, target_q2)

        current_q1 = self.q1(torch.cat([states, actions], dim=-1))
        current_q2 = self.q2(torch.cat([states, actions], dim=-1))

        td_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # CQL regularization
        cql_reg = self.cql_loss(states, actions)

        # Total Q loss
        q_loss = td_loss + self.alpha * cql_reg

        # Update Q-networks
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update policy (standard actor-critic)
        policy_actions = self.policy(states)
        policy_loss = -self.q1(torch.cat([states, policy_actions], dim=-1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return q_loss.item(), policy_loss.item()


print("\nCQL (Conservative Q-Learning):")
print("✅ Conservative Q-function estimation")
print("✅ Prevents overestimation for OOD actions")
print("✅ Simple and effective")
print("✅ State-of-the-art offline RL")
```

---

## 5. Implicit Q-Learning (IQL)

IQL avoids out-of-distribution actions entirely!

```python
class IQL:
    """
    Implicit Q-Learning (IQL).

    Key innovation: Expectile regression instead of max.

    Standard Q-learning:
        V(s) = max_a Q(s,a)  ← Causes overestimation!

    IQL:
        V(s) = expectile_τ[Q(s,a)]  ← More conservative!

    No explicit policy constraint needed!
    """
    def __init__(self, state_dim, action_dim, tau=0.7):
        # Value network V(s)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Q-networks
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=3e-4)
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=3e-4
        )
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.tau = tau  # Expectile parameter

    def expectile_loss(self, diff, tau):
        """
        Expectile regression loss.

        Asymmetric squared loss:
        L = |τ - 1{diff < 0}| * diff²
        """
        weight = torch.where(diff > 0, tau, 1 - tau)
        return weight * (diff ** 2)

    def train_step(self, batch, gamma=0.99):
        """Train IQL on offline batch."""
        states, actions, rewards, next_states, dones = batch

        # Update Q-functions
        with torch.no_grad():
            next_v = self.value_net(next_states)
            target_q = rewards + (1 - dones) * gamma * next_v

        current_q1 = self.q1(torch.cat([states, actions], dim=-1))
        current_q2 = self.q2(torch.cat([states, actions], dim=-1))

        q_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update value function with expectile regression
        with torch.no_grad():
            q_pred = torch.min(
                self.q1(torch.cat([states, actions], dim=-1)),
                self.q2(torch.cat([states, actions], dim=-1))
            )

        v_pred = self.value_net(states)
        value_loss = self.expectile_loss(q_pred - v_pred, self.tau).mean()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Update policy with advantage-weighted regression
        with torch.no_grad():
            q_sa = torch.min(
                self.q1(torch.cat([states, actions], dim=-1)),
                self.q2(torch.cat([states, actions], dim=-1))
            )
            v_s = self.value_net(states)
            advantage = q_sa - v_s
            weights = torch.exp(advantage / 0.1).clamp(max=100)

        policy_actions = self.policy(states)
        policy_loss = (weights * F.mse_loss(policy_actions, actions, reduction='none').sum(dim=-1)).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        return q_loss.item(), value_loss.item(), policy_loss.item()


print("\nIQL (Implicit Q-Learning):")
print("✅ Expectile regression (not max!)")
print("✅ No explicit policy constraint")
print("✅ Simple and stable")
print("✅ Strong performance")
```

---

## 6. Decision Transformer

Treat RL as sequence modeling!

```python
class DecisionTransformer(nn.Module):
    """
    Decision Transformer: RL as Sequence Modeling.

    Revolutionary idea: Instead of RL, use Transformer to model:
    (R, s, a, R, s, a, ...) sequence

    At test time: Condition on desired return R!

    No Q-learning, no policy gradients - just next-action prediction!
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, n_layers=3, n_heads=4):
        super(DecisionTransformer, self).__init__()

        # Embed returns, states, actions
        self.ret_emb = nn.Linear(1, hidden_dim)
        self.state_emb = nn.Linear(state_dim, hidden_dim)
        self.action_emb = nn.Linear(action_dim, hidden_dim)

        # Positional embedding
        self.pos_emb = nn.Parameter(torch.zeros(1, 1000, hidden_dim))

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=4*hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Predict action
        self.action_pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

    def forward(self, returns, states, actions, timesteps):
        """
        Forward pass.

        Args:
            returns: [batch, seq_len, 1]
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len, action_dim]
            timesteps: [batch, seq_len]
        """
        batch_size, seq_len = states.shape[0], states.shape[1]

        # Embed each modality
        ret_embeddings = self.ret_emb(returns)
        state_embeddings = self.state_emb(states)
        action_embeddings = self.action_emb(actions)

        # Interleave: (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # Stack: [batch, seq_len*3, hidden_dim]
        sequence = torch.stack([ret_embeddings, state_embeddings, action_embeddings], dim=2)
        sequence = sequence.reshape(batch_size, seq_len * 3, -1)

        # Add positional embedding
        sequence = sequence + self.pos_emb[:, :seq_len*3, :]

        # Transformer
        output = self.transformer(sequence)

        # Extract state embeddings (every 3rd token starting from index 1)
        state_outputs = output[:, 1::3, :]  # [batch, seq_len, hidden_dim]

        # Predict actions
        action_preds = self.action_pred(state_outputs)

        return action_preds


class DecisionTransformerAgent:
    """Agent using Decision Transformer."""
    def __init__(self, state_dim, action_dim):
        self.model = DecisionTransformer(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

        print("Decision Transformer initialized!")
        print("✅ Sequence modeling approach")
        print("✅ No Q-learning or policy gradients")
        print("✅ Condition on desired return")
        print("✅ Competitive with CQL/IQL")


print("\nDecision Transformer:")
print("✅ RL as sequence modeling")
print("✅ Uses Transformer architecture")
print("✅ Condition on return-to-go")
print("✅ Simple and effective")
```

---

## 7. D4RL Benchmark

Standard benchmark for offline RL!

```python
"""
D4RL (Datasets for Deep Data-Driven RL):

Benchmark datasets:
- MuJoCo locomotion (walker, hopper, halfcheetah)
- Maze navigation (antmaze)
- Adroit hand manipulation (pen, door, hammer)

Dataset types:
- Random: Random policy
- Medium: Partially trained policy
- Expert: Fully trained policy
- Medium-Replay: Mix of all training data
- Medium-Expert: Mix of medium and expert

Installation:
pip install git+https://github.com/Farama-Foundation/d4rl@master

Usage:
import gym
import d4rl
env = gym.make('halfcheetah-medium-v2')
dataset = env.get_dataset()
"""

print("\n=== D4RL Benchmark ===")
print("Standard evaluation for offline RL")
print("\nDatasets:")
print("✅ MuJoCo: Locomotion tasks")
print("✅ AntMaze: Navigation")
print("✅ Adroit: Dexterous manipulation")
print("\nMetrics:")
print("- Normalized score (0-100)")
print("- 0 = random, 100 = expert")
```

---

## 8. Offline Policy Evaluation (OPE)

How to evaluate policies without deployment?

```python
"""
Offline Policy Evaluation (OPE):

Problem: Evaluate policy π without environment interaction.
Have: Dataset D from behavior policy π_β

Approaches:
1. Importance Sampling (IS)
2. Doubly Robust (DR)
3. Model-Based (FQE, MB-OPE)

Importance Sampling:
V^π = E_D [ Π_t (π(a_t|s_t) / π_β(a_t|s_t)) * R ]

High variance! Use doubly robust for lower variance.
"""

def importance_sampling_ope(trajectories, target_policy, behavior_policy):
    """
    Estimate policy value using importance sampling.

    Args:
        trajectories: List of (states, actions, rewards)
        target_policy: Policy to evaluate
        behavior_policy: Policy that collected data
    """
    values = []

    for states, actions, rewards in trajectories:
        # Compute importance weights
        importance_weights = 1.0

        for s, a in zip(states, actions):
            pi_ratio = target_policy(a, s) / (behavior_policy(a, s) + 1e-8)
            importance_weights *= pi_ratio

        # Weighted return
        weighted_return = importance_weights * sum(rewards)
        values.append(weighted_return)

    return np.mean(values)


print("\nOffline Policy Evaluation:")
print("✅ Evaluate without deployment")
print("✅ Importance sampling")
print("✅ Doubly robust estimators")
print("✅ Critical for real-world deployment")
```

---

## 9. Real-World Applications

```python
"""
Offline RL Applications:

1. Healthcare:
   - Learn treatment policies from patient records
   - Cannot experiment on real patients!
   - Example: Sepsis treatment, diabetes management

2. Autonomous Vehicles:
   - Learn from human driving logs
   - Cannot deploy unsafe policies on real roads
   - Example: Waymo, Tesla logs

3. Robotics:
   - Learn from teleoperation data
   - Real robot time expensive
   - Example: Robotic manipulation, assembly

4. Finance:
   - Learn trading policies from historical data
   - Cannot risk real money during training
   - Example: Algorithmic trading, portfolio optimization

5. Recommender Systems:
   - Learn from user interaction logs
   - Cannot show bad recommendations
   - Example: YouTube, Netflix, TikTok
"""

print("\n=== Real-World Applications ===")
print("\n1. Healthcare:")
print("   - Patient records → Treatment policies")
print("   - Safety critical!")

print("\n2. Autonomous Driving:")
print("   - Driving logs → Driving policies")
print("   - Cannot deploy unsafe policies")

print("\n3. Robotics:")
print("   - Teleoperation → Autonomous policies")
print("   - Real robot time expensive")

print("\n4. Finance:")
print("   - Historical trades → Trading policies")
print("   - Cannot risk capital during training")
```

---

## Practice Exercises

### Exercise 1: Implement CQL on D4RL

```python
"""
Implement CQL and train on D4RL halfcheetah-medium-v2.

Requirements:
1. Load D4RL dataset
2. Implement full CQL algorithm
3. Train for 1M steps
4. Evaluate on environment
5. Compare with behavioral cloning baseline

Track:
- Q-values over training
- CQL penalty magnitude
- Policy performance
"""

# Your implementation here
```

### Exercise 2: Decision Transformer from Scratch

```python
"""
Implement Decision Transformer and train on offline dataset.

Requirements:
1. Implement full Transformer architecture
2. Sequence data loader (returns-to-go, states, actions)
3. Training loop
4. Test-time conditioning on different returns
5. Compare with CQL

Bonus: Visualize attention weights!
"""

# Your implementation here
```

### Exercise 3: Offline Policy Evaluation

```python
"""
Implement and compare OPE methods.

Methods:
1. Importance Sampling
2. Per-Decision IS
3. Doubly Robust
4. FQE (Fitted Q Evaluation)

Dataset: D4RL medium dataset
Policies: Random, medium, expert

Compare:
- Estimate accuracy
- Variance
- Bias
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Offline RL** 💾
   - Learn from fixed datasets
   - No environment interaction
   - Critical for real-world deployment
   - Much harder than online RL

2. **Distribution Shift** ⚠️
   - Core challenge in offline RL
   - OOD actions → Overestimation
   - Need conservative methods
   - Constrain policy or Q-function

3. **BCQ** 🔒
   - Constrain to dataset actions
   - VAE for action generation
   - Prevents OOD extrapolation
   - Effective but complex

4. **CQL** 🛡️
   - Conservative Q-functions
   - Lower-bound for OOD actions
   - Simple and effective
   - State-of-the-art

5. **IQL** 🎯
   - Expectile regression
   - No explicit constraints
   - Simple and stable
   - Competitive with CQL

6. **Decision Transformer** 🤖
   - RL as sequence modeling
   - Transformer architecture
   - No Q-learning needed
   - Promising new direction

7. **D4RL** 📊
   - Standard benchmark
   - Various task types
   - Multiple dataset qualities
   - Normalized evaluation

### Algorithm Selection Guide

✅ **Simple baseline**: Behavioral Cloning
✅ **State-of-the-art**: CQL or IQL
✅ **Constrained policy**: BCQ
✅ **Sequence modeling**: Decision Transformer
✅ **Real-world safety-critical**: CQL with OPE

### Real-World Applications

✅ **Healthcare**: Treatment policies from patient records
✅ **Autonomous Driving**: Policies from driving logs
✅ **Robotics**: Learn from demonstrations
✅ **Finance**: Trading policies from historical data
✅ **Recommender Systems**: User interaction logs

### What's Next?

In Lesson 8, we'll learn **Advanced Exploration**:
- Beyond ε-greedy
- Curiosity-driven exploration
- Intrinsic motivation (ICM, RND, NGU)
- Sparse reward environments

**Offline RL enables safe real-world deployment - essential skill!** 💾

---

## Additional Resources

### Papers
- Fujimoto et al. (2019): "Off-Policy Deep Reinforcement Learning without Exploration" (BCQ)
- Kumar et al. (2020): "Conservative Q-Learning for Offline Reinforcement Learning" (CQL)
- Kostrikov et al. (2021): "Offline Reinforcement Learning with Implicit Q-Learning" (IQL)
- Chen et al. (2021): "Decision Transformer: Reinforcement Learning via Sequence Modeling"
- Fu et al. (2020): "D4RL: Datasets for Deep Data-Driven Reinforcement Learning"

### Libraries
- **d4rl**: https://github.com/Farama-Foundation/D4RL
- **rlkit**: https://github.com/rail-berkeley/rlkit (Offline RL implementations)
- **decision-transformer**: https://github.com/kzl/decision-transformer

### Resources
- **Offline RL Tutorial (NeurIPS)**: https://sites.google.com/view/offlinerltutorial-neurips2020
- **Sergey Levine's Course**: http://rail.eecs.berkeley.edu/deeprlcourse/
- **D4RL Benchmark**: https://sites.google.com/view/d4rl/home

---

**Next**: [Lesson 8 - Advanced Exploration and Intrinsic Motivation](Lesson%208%20-%20Advanced%20Exploration%20and%20Intrinsic%20Motivation.md)

Proceed to learn about **curiosity-driven exploration** and **intrinsic rewards**! 🔍

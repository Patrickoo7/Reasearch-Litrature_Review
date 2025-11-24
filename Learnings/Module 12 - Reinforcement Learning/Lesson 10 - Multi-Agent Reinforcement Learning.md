# Lesson 10: Multi-Agent Reinforcement Learning 🎮

**Module 12: Reinforcement Learning | Lesson 10 of 12**

Master multi-agent systems, game theory, and emergent behavior - from self-play to swarm intelligence!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand multi-agent RL settings and challenges
2. ✅ Implement independent learners and joint action learning
3. ✅ Master centralized training with decentralized execution (CTDE)
4. ✅ Learn MADDPG, QMIX, and communication protocols
5. ✅ Apply self-play for game AI (AlphaGo, Dota 2, StarCraft)
6. ✅ Understand emergent behavior and cooperation

---

## 1. Why Multi-Agent RL?

### Single Agent vs Multi-Agent

Single-agent RL: One agent, static environment
Multi-agent RL (MARL): Multiple agents, dynamic environment!

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque

"""
Multi-Agent RL (MARL):

Multiple agents learning simultaneously:
- Agents observe environment
- Agents take actions
- Environment dynamics depend on ALL agents

Challenges:
❌ Non-stationarity: Other agents are learning too!
❌ Credit assignment: Which agent caused reward?
❌ Scalability: Exponential joint action space
❌ Communication: How should agents coordinate?

Examples:
- Game playing: Chess, Go, Dota 2, StarCraft
- Robotics: Multi-robot coordination, swarms
- Autonomous vehicles: Traffic, intersections
- Economics: Markets, auctions, negotiations
"""

print("=== Multi-Agent RL ===")
print("\nChallenges:")
print("❌ Non-stationarity (moving target)")
print("❌ Credit assignment")
print("❌ Exponential action spaces")
print("❌ Partial observability")

print("\nApplications:")
print("✅ Game AI (AlphaGo, Dota 2, StarCraft)")
print("✅ Multi-robot systems")
print("✅ Autonomous vehicles")
print("✅ Swarm intelligence")
```

---

## 2. Game Theory Basics

MARL is deeply connected to game theory!

```python
"""
Game Theory for MARL:

1. Normal Form Games:
   - Finite players, actions, rewards
   - Simultaneous actions
   - Payoff matrix

2. Nash Equilibrium:
   - No agent can improve by changing strategy alone
   - Not unique, may not exist
   - Central concept in MARL

3. Game Types:
   - Cooperative: Shared reward (team games)
   - Competitive: Zero-sum (chess, poker)
   - Mixed: Both cooperation and competition

4. Stochastic Games (Markov Games):
   - Extension of MDPs to multiple agents
   - State transitions depend on joint actions
"""

class GameTheory:
    """
    Game theory concepts for MARL.
    """
    @staticmethod
    def prisoners_dilemma():
        """
        Classic Prisoner's Dilemma.

        Payoff matrix:
                 Cooperate    Defect
        Cooperate  (-1,-1)    (-3,0)
        Defect     (0,-3)     (-2,-2)

        Nash equilibrium: (Defect, Defect)
        But (Cooperate, Cooperate) is better!
        """
        payoffs = {
            ("C", "C"): (-1, -1),
            ("C", "D"): (-3, 0),
            ("D", "C"): (0, -3),
            ("D", "D"): (-2, -2)
        }

        print("Prisoner's Dilemma:")
        print("Nash Equilibrium: (Defect, Defect)")
        print("Socially optimal: (Cooperate, Cooperate)")
        print("Tension between individual and group interests!")

        return payoffs

    @staticmethod
    def find_nash_equilibrium(payoff_matrix):
        """
        Find Nash equilibrium in normal form game.

        (Simplified for 2x2 games)
        """
        # In practice: Use game theory solvers
        print("\nNash Equilibrium:")
        print("No agent benefits from unilateral deviation")

        return None


# Example
game = GameTheory()
payoffs = game.prisoners_dilemma()

print("\n✅ Game theory provides foundation for MARL")
print("✅ Nash equilibrium is key solution concept")
```

---

## 3. Independent Q-Learning

Simplest MARL: Each agent learns independently!

```python
class IndependentQLearning:
    """
    Independent Q-Learning: Agents learn separately.

    Each agent:
    - Maintains own Q-table
    - Learns with standard Q-learning
    - Treats other agents as part of environment

    Pros:
    ✅ Simple
    ✅ Scalable
    ✅ Decentralized

    Cons:
    ❌ Non-stationarity (other agents learning)
    ❌ No explicit coordination
    ❌ May not converge
    """
    def __init__(self, num_agents, state_dim, action_dim, lr=0.1, gamma=0.99):
        self.num_agents = num_agents

        # Each agent has own Q-table
        self.Q_tables = [
            {}  # State-action dictionary
            for _ in range(num_agents)
        ]

        self.lr = lr
        self.gamma = gamma

    def get_q(self, agent_id, state, action):
        """Get Q-value for agent."""
        key = (state, action)
        return self.Q_tables[agent_id].get(key, 0.0)

    def select_actions(self, state, epsilon=0.1):
        """Each agent selects action independently."""
        actions = []

        for agent_id in range(self.num_agents):
            if np.random.rand() < epsilon:
                action = np.random.randint(2)  # Random action
            else:
                # Greedy
                q_values = [self.get_q(agent_id, state, a) for a in range(2)]
                action = np.argmax(q_values)

            actions.append(action)

        return actions

    def update(self, agent_id, state, action, reward, next_state, done):
        """Update Q-table for one agent (standard Q-learning)."""
        current_q = self.get_q(agent_id, state, action)

        if done:
            target = reward
        else:
            next_q_values = [self.get_q(agent_id, next_state, a) for a in range(2)]
            target = reward + self.gamma * max(next_q_values)

        # Q-learning update
        new_q = current_q + self.lr * (target - current_q)
        self.Q_tables[agent_id][(state, action)] = new_q


print("\nIndependent Q-Learning:")
print("✅ Each agent learns independently")
print("✅ Simple and scalable")
print("❌ Treats other agents as environment")
print("❌ Non-stationary problem")
```

---

## 4. Centralized Training Decentralized Execution (CTDE)

Key paradigm in MARL!

```python
"""
CTDE (Centralized Training, Decentralized Execution):

Training:
- Access to global state and all agent information
- Can use centralized critic
- Learn coordination

Execution:
- Each agent acts based on local observations only
- Decentralized (practical for deployment)

Advantages:
✅ Stable training (centralized)
✅ Scalable execution (decentralized)
✅ Best of both worlds

Algorithms: MADDPG, QMIX, COMA
"""

print("\nCTDE Paradigm:")
print("Training: Centralized (use global info)")
print("Execution: Decentralized (local observations)")
print("✅ Combines stability and scalability")
```

---

## 5. Multi-Agent DDPG (MADDPG)

Actor-critic for multi-agent continuous control!

```python
class MADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient.

    CTDE algorithm:
    - Centralized critic: Q(s, a1, a2, ..., aN)
    - Decentralized actors: πi(ai | oi)

    Each agent:
    - Actor sees only local observations
    - Critic sees global state + all actions

    Training: Centralized (use global info)
    Execution: Decentralized (local policies)
    """
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=128):
        self.num_agents = num_agents

        # Decentralized actors (one per agent)
        self.actors = [
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Tanh()
            )
            for _ in range(num_agents)
        ]

        # Centralized critics (one per agent, sees global info)
        critic_input_dim = num_agents * (obs_dim + action_dim)
        self.critics = [
            nn.Sequential(
                nn.Linear(critic_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            for _ in range(num_agents)
        ]

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=1e-3)
            for actor in self.actors
        ]

        self.critic_optimizers = [
            optim.Adam(critic.parameters(), lr=1e-3)
            for critic in self.critics
        ]

        print(f"MADDPG initialized with {num_agents} agents")

    def select_actions(self, observations, noise=0.1):
        """
        Each agent selects action based on LOCAL observation.

        Args:
            observations: List of observations [agent_1_obs, ..., agent_N_obs]

        Returns:
            actions: List of actions
        """
        actions = []

        for i, obs in enumerate(observations):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            with torch.no_grad():
                action = self.actors[i](obs_tensor).numpy()[0]

            # Add noise for exploration
            action = action + np.random.normal(0, noise, size=action.shape)
            action = np.clip(action, -1, 1)

            actions.append(action)

        return actions

    def update(self, batch):
        """
        Update actors and critics.

        Args:
            batch: (observations, actions, rewards, next_observations, dones)
                   All are lists of length num_agents
        """
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = batch

        # Convert to tensors
        obs = [torch.FloatTensor(o) for o in obs_list]
        actions = [torch.FloatTensor(a) for a in actions_list]
        rewards = [torch.FloatTensor(r) for r in rewards_list]
        next_obs = [torch.FloatTensor(no) for no in next_obs_list]

        # Update each agent
        for agent_id in range(self.num_agents):
            # Critic update
            # Centralized: Use all observations and actions
            global_obs = torch.cat(obs, dim=-1)
            global_actions = torch.cat(actions, dim=-1)
            global_next_obs = torch.cat(next_obs, dim=-1)

            # Current Q-value
            current_q = self.critics[agent_id](
                torch.cat([global_obs, global_actions], dim=-1)
            )

            # Target Q-value
            with torch.no_grad():
                # Next actions from all agents
                next_actions = [
                    self.actors[i](next_obs[i])
                    for i in range(self.num_agents)
                ]
                global_next_actions = torch.cat(next_actions, dim=-1)

                target_q = self.critics[agent_id](
                    torch.cat([global_next_obs, global_next_actions], dim=-1)
                )

                target = rewards[agent_id] + 0.99 * target_q

            # Critic loss
            critic_loss = F.mse_loss(current_q, target)

            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[agent_id].step()

            # Actor update
            # Policy gradient: Maximize Q-value
            policy_actions = actions.copy()
            policy_actions[agent_id] = self.actors[agent_id](obs[agent_id])

            global_policy_actions = torch.cat(policy_actions, dim=-1)

            actor_loss = -self.critics[agent_id](
                torch.cat([global_obs, global_policy_actions], dim=-1)
            ).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()


print("\nMADDPG:")
print("✅ CTDE for continuous control")
print("✅ Centralized critics, decentralized actors")
print("✅ Stable multi-agent learning")
print("✅ Used in robotics, cooperative tasks")
```

---

## 6. QMIX: Value Decomposition

Learn decentralized policies from centralized Q-function!

```python
"""
QMIX: Monotonic Value Function Factorization.

Key idea: Decompose joint Q-function into agent Q-functions.

Q_tot(s, a1, ..., aN) = f(Q1(o1, a1), ..., QN(oN, aN))

Where f is monotonic mixing function (ensures consistency).

Architecture:
1. Agent networks: Qi(oi, ai)
2. Mixing network: Combines agent Q-values
3. Hypernetwork: Generates mixing network weights from state

Advantages:
✅ Decentralized execution
✅ Centralized training
✅ Credit assignment
✅ Scalable

Used in StarCraft II!
"""

class QMIXNetwork(nn.Module):
    """
    QMIX Mixing Network.

    Combines agent Q-values into joint Q-value.
    """
    def __init__(self, num_agents, state_dim, mixing_embed_dim=32):
        super(QMIXNetwork, self).__init__()
        self.num_agents = num_agents

        # Hypernetwork: Generate weights from state
        self.hyper_w1 = nn.Linear(state_dim, num_agents * mixing_embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, mixing_embed_dim)

        # Bias hypernetworks
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed_dim),
            nn.ReLU(),
            nn.Linear(mixing_embed_dim, 1)
        )

    def forward(self, agent_qs, state):
        """
        Mix agent Q-values.

        Args:
            agent_qs: [batch_size, num_agents]
            state: [batch_size, state_dim]

        Returns:
            q_tot: [batch_size, 1]
        """
        batch_size = agent_qs.shape[0]

        # Generate weights from state
        w1 = torch.abs(self.hyper_w1(state))  # Ensure monotonicity
        w1 = w1.view(batch_size, self.num_agents, -1)

        b1 = self.hyper_b1(state)
        b1 = b1.view(batch_size, 1, -1)

        # First layer
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)

        # Second layer
        w2 = torch.abs(self.hyper_w2(state))
        w2 = w2.view(batch_size, -1, 1)

        b2 = self.hyper_b2(state)

        # Output
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.squeeze()


print("\nQMIX:")
print("✅ Value function factorization")
print("✅ Decentralized execution")
print("✅ Credit assignment")
print("✅ Used in StarCraft II")
```

---

## 7. Self-Play

Learn by playing against yourself!

```python
class SelfPlay:
    """
    Self-Play: Agent plays against copies of itself.

    Used by:
    - AlphaGo: Self-play → Superhuman Go
    - AlphaZero: Chess, Shogi, Go
    - OpenAI Five: Dota 2
    - AlphaStar: StarCraft II

    Process:
    1. Initialize policy
    2. Generate games by self-play
    3. Train on self-play games
    4. Update policy
    5. Repeat

    Advantages:
    ✅ No human data needed
    ✅ Unlimited training data
    ✅ Curriculum learning (opponent improves)
    ✅ Can surpass human performance
    """
    def __init__(self, policy, game_env):
        self.policy = policy
        self.game_env = game_env

        # Store past policies for diversity
        self.policy_history = deque(maxlen=10)

        print("Self-Play initialized")

    def play_game(self):
        """
        Play one game: Policy vs itself (or past version).
        """
        state = self.game_env.reset()
        trajectory = []

        done = False
        while not done:
            # Current player selects action
            action = self.policy.select_action(state)

            # Take action
            next_state, reward, done, _ = self.game_env.step(action)

            trajectory.append((state, action, reward))
            state = next_state

        return trajectory

    def train_iteration(self, num_games=100):
        """
        One self-play training iteration.

        1. Generate games
        2. Train policy
        3. Update policy pool
        """
        # Generate self-play games
        trajectories = []
        for _ in range(num_games):
            traj = self.play_game()
            trajectories.append(traj)

        # Train policy on self-play data
        # self.policy.train(trajectories)

        # Add current policy to history
        self.policy_history.append(self.policy.copy())

        print(f"Self-play iteration: {num_games} games")


print("\nSelf-Play:")
print("✅ AlphaGo, AlphaZero, OpenAI Five")
print("✅ Unlimited training data")
print("✅ Curriculum learning")
print("✅ Superhuman performance")
```

---

## 8. Communication and Cooperation

Agents learn to communicate!

```python
"""
Communication in MARL:

1. Explicit Communication:
   - Agents send messages
   - Communication channels
   - Learn what to communicate

2. Emergent Communication:
   - No pre-defined language
   - Agents develop own protocols
   - Emerges from cooperation pressure

3. CommNet, IC3Net, TarMAC:
   - Neural architectures for communication
   - Learn communication and action jointly

Applications:
- Multi-robot coordination
- Autonomous vehicles
- Network routing
"""

class CommNet(nn.Module):
    """
    Communication Network for MARL.

    Agents:
    - Encode observations
    - Share hidden states (communication)
    - Decode to actions
    """
    def __init__(self, num_agents, obs_dim, action_dim, hidden_dim=64):
        super(CommNet, self).__init__()
        self.num_agents = num_agents

        # Encoder: Observation → Hidden
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )

        # Communication: Average hidden states
        # (Built-in as mean operation)

        # Decoder: Hidden → Action
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, observations):
        """
        Forward pass with communication.

        Args:
            observations: [num_agents, obs_dim]

        Returns:
            actions: [num_agents, action_dim]
        """
        # Encode observations
        hidden_states = torch.stack([
            self.encoder(obs) for obs in observations
        ])  # [num_agents, hidden_dim]

        # Communication: Average hidden states
        avg_hidden = hidden_states.mean(dim=0, keepdim=True)

        # Each agent gets averaged communication
        comm_hidden = avg_hidden.repeat(self.num_agents, 1)

        # Combine own hidden with communication
        combined_hidden = hidden_states + comm_hidden

        # Decode to actions
        actions = torch.stack([
            self.decoder(h) for h in combined_hidden
        ])

        return actions


print("\nCommunication in MARL:")
print("✅ Explicit communication channels")
print("✅ Emergent communication protocols")
print("✅ CommNet, IC3Net architectures")
print("✅ Multi-robot coordination")
```

---

## 9. Real-World Applications

```python
"""
Multi-Agent RL Applications:

1. Game AI:
   - AlphaGo: Superhuman Go (self-play)
   - OpenAI Five: Dota 2 (5v5 MOBA)
   - AlphaStar: StarCraft II (QMIX, self-play)
   - Pluribus: Poker (6-player)

2. Robotics:
   - Multi-robot warehouses (Amazon)
   - Drone swarms
   - Autonomous vehicle fleets

3. Traffic Control:
   - Intersection management
   - Adaptive traffic lights
   - Vehicle platooning

4. Economics:
   - Market making
   - Auction design
   - Resource allocation

5. Smart Grids:
   - Distributed energy management
   - Load balancing
   - Microgrid coordination
"""

print("\n=== Applications ===")
print("\n1. Game AI:")
print("   - AlphaGo, OpenAI Five, AlphaStar")
print("   - Superhuman performance")

print("\n2. Robotics:")
print("   - Multi-robot coordination")
print("   - Warehouse automation")
print("   - Drone swarms")

print("\n3. Autonomous Vehicles:")
print("   - Traffic coordination")
print("   - Intersection management")
```

---

## Practice Exercises

### Exercise 1: Implement MADDPG

```python
"""
Implement MADDPG and train on multi-agent particle environment.

Requirements:
1. Full MADDPG implementation
2. Replay buffer for multi-agent
3. Train on cooperative navigation task
4. Compare with independent learners

Environment: PettingZoo MPE environments

Metrics:
- Episode rewards
- Coordination quality
- Sample efficiency
"""

# Your implementation here
```

### Exercise 2: Self-Play for Tic-Tac-Toe

```python
"""
Implement self-play for Tic-Tac-Toe (or Connect Four).

Requirements:
1. Game environment
2. Neural network policy
3. Self-play training loop
4. Policy pool for diversity
5. Evaluate against random/minimax

Track:
- Win rate over training
- Policy evolution
- Strategic diversity
"""

# Your implementation here
```

### Exercise 3: Emergent Communication

```python
"""
Train agents to develop emergent communication.

Task: Cooperative navigation with partial observability
- Agent 1 sees goal location
- Agent 2 can move but can't see goal
- Learn communication to succeed

Requirements:
1. Communication channel (discrete or continuous)
2. Reward for reaching goal
3. Analyze emerged communication protocol

Bonus: Visualize what agents communicate!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Multi-Agent Challenges** 🎯
   - Non-stationarity (moving target)
   - Credit assignment
   - Exponential action spaces
   - Partial observability

2. **Game Theory** 📊
   - Nash equilibrium
   - Cooperative vs competitive
   - Stochastic games
   - Foundation for MARL

3. **CTDE Paradigm** 🔄
   - Centralized training
   - Decentralized execution
   - Best of both worlds
   - Industry standard

4. **MADDPG** 🤖
   - Actor-critic for MARL
   - Continuous control
   - Centralized critics
   - Decentralized actors

5. **QMIX** 🎮
   - Value decomposition
   - Credit assignment
   - StarCraft II success
   - Scalable coordination

6. **Self-Play** 🔁
   - AlphaGo, AlphaZero
   - Unlimited data
   - Superhuman performance
   - Curriculum learning

7. **Communication** 💬
   - Emergent protocols
   - CommNet architecture
   - Multi-robot coordination
   - Learned cooperation

### Real-World Impact

✅ **Game AI**: AlphaGo, Dota 2, StarCraft II (superhuman)
✅ **Robotics**: Warehouse automation, drone swarms
✅ **Autonomous Vehicles**: Traffic management, platooning
✅ **Smart Systems**: Grids, markets, auctions

### What's Next?

In Lesson 11, we'll learn **Hierarchical and Meta-RL**:
- Temporal abstraction with options
- Hierarchical policies
- Meta-learning (MAML, RL²)
- Fast adaptation to new tasks

**Multi-agent RL enables coordination and emergence!** 🎮

---

## Additional Resources

### Papers
- Lowe et al. (2017): "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments" (MADDPG)
- Rashid et al. (2018): "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent RL"
- Silver et al. (2017): "Mastering Chess and Shogi by Self-Play with a General RL Algorithm" (AlphaZero)
- Vinyals et al. (2019): "Grandmaster level in StarCraft II using multi-agent RL" (AlphaStar)

### Libraries
- **PettingZoo**: https://pettingzoo.farama.org/ (Multi-agent environments)
- **SMAC**: https://github.com/oxwhirl/smac (StarCraft Multi-Agent Challenge)
- **PyMARL**: https://github.com/oxwhirl/pymarl (MARL algorithms)

### Resources
- **OpenAI Five**: https://openai.com/blog/openai-five/
- **AlphaStar**: https://deepmind.com/blog/article/alphastar-mastering-real-time-strategy-game-starcraft-ii
- **Multi-Agent Tutorial**: https://sites.google.com/view/aamas2020tutorial

---

**Next**: [Lesson 11 - Hierarchical RL and Meta-RL](Lesson%2011%20-%20Hierarchical%20RL%20and%20Meta-RL.md)

Proceed to learn about **temporal abstraction** and **fast adaptation**! 🏗️

# Lesson 11: Hierarchical RL, Meta-RL & Safe RL 🏗️

**Module 12: Reinforcement Learning | Lesson 11 of 12**

Master temporal abstraction, fast adaptation, and safety - essential for real-world deployment!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand hierarchical RL and the options framework
2. ✅ Implement skill discovery and hierarchical policies
3. ✅ Master meta-RL for fast adaptation (MAML, RL²)
4. ✅ Learn Safe RL with constraints and risk-sensitivity
5. ✅ Apply transfer learning and continual learning in RL
6. ✅ Understand when to use hierarchical and meta approaches

---

## 1. Why Hierarchical RL?

### The Temporal Abstraction Problem

Flat RL: Learn primitive action every timestep
Hierarchical RL: Learn skills, then compose them!

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
Hierarchical RL (HRL):

Problem with flat RL:
❌ Long horizons (millions of timesteps)
❌ Sparse rewards
❌ No reusable skills
❌ Sample inefficient

HRL Solution:
✅ Temporal abstraction (skills/options)
✅ Reusable behaviors
✅ Faster learning
✅ Interpretable policies

Examples:
- Robotics: Pick, place, push (reusable skills)
- Navigation: Go-to-door, open-door, go-through-door
- Games: Attack, defend, scout (strategic options)
"""

print("=== Hierarchical RL ===")
print("\nWhy HRL:")
print("✅ Temporal abstraction")
print("✅ Reusable skills")
print("✅ Faster learning")
print("✅ Transfer across tasks")

print("\nApplications:")
print("✅ Robotics: Complex manipulation")
print("✅ Navigation: Multi-room environments")
print("✅ Games: Strategic decision-making")
```

---

## 2. Options Framework

Foundational framework for temporal abstraction!

```python
class Option:
    """
    Option: Temporally extended action.

    Components:
    - Initiation set I: Where can option start?
    - Policy π: What actions to take?
    - Termination β: When does option end?

    Example: "Go to door"
    - I: Any room
    - π: Navigate toward door
    - β: Terminate when at door
    """
    def __init__(self, name, policy, initiation_fn, termination_fn):
        self.name = name
        self.policy = policy  # Option policy
        self.initiation_fn = initiation_fn  # I(s) → bool
        self.termination_fn = termination_fn  # β(s) → [0,1]

    def can_initiate(self, state):
        """Check if option can be initiated in state."""
        return self.initiation_fn(state)

    def select_action(self, state):
        """Select primitive action using option policy."""
        return self.policy(state)

    def should_terminate(self, state):
        """Check if option should terminate."""
        termination_prob = self.termination_fn(state)
        return np.random.rand() < termination_prob


class HierarchicalAgent:
    """
    Agent using options for hierarchical decision-making.

    Two-level hierarchy:
    - High-level: Select option
    - Low-level: Execute option (select primitive actions)
    """
    def __init__(self, options, meta_policy):
        self.options = options
        self.meta_policy = meta_policy  # Policy over options

        self.current_option = None

    def select_option(self, state):
        """High-level: Select which option to use."""
        available_options = [
            opt for opt in self.options if opt.can_initiate(state)
        ]

        if not available_options:
            return None

        # Meta-policy selects option
        option_idx = self.meta_policy(state, available_options)
        return available_options[option_idx]

    def act(self, state):
        """
        Hierarchical action selection.

        1. If no current option or option terminated, select new option
        2. Execute current option policy
        """
        # Check if need to select new option
        if self.current_option is None or self.current_option.should_terminate(state):
            self.current_option = self.select_option(state)

        if self.current_option is None:
            # Fallback: Random action
            return 0

        # Execute option
        action = self.current_option.select_action(state)
        return action


# Example: Define simple options
def make_navigation_options():
    """Create navigation options for GridWorld."""

    # Option 1: Go North
    north_policy = lambda s: 0  # Action 0 = North

    go_north = Option(
        name="GoNorth",
        policy=north_policy,
        initiation_fn=lambda s: True,  # Can initiate anywhere
        termination_fn=lambda s: 0.1  # 10% termination probability
    )

    # Option 2: Go South
    south_policy = lambda s: 2  # Action 2 = South

    go_south = Option(
        name="GoSouth",
        policy=south_policy,
        initiation_fn=lambda s: True,
        termination_fn=lambda s: 0.1
    )

    return [go_north, go_south]


print("\nOptions Framework:")
print("✅ Temporally extended actions")
print("✅ Initiation, policy, termination")
print("✅ Reusable across tasks")
print("✅ Foundation of HRL")
```

---

## 3. Skill Discovery

Automatically discover useful skills!

```python
"""
Skill Discovery:

Instead of hand-designing options, learn them!

Approaches:
1. Diversity-based: Learn diverse skills (DIAYN)
2. Empowerment: Maximize influence on environment
3. Bottleneck-based: Identify subgoals (betweenness)
4. Successor features: Task-independent skills

DIAYN (Diversity is All You Need):
- Learn skills that visit different states
- Discriminator distinguishes skills
- Encourages behavioral diversity
"""

class DIAYN:
    """
    DIAYN: Diversity is All You Need.

    Learn diverse skills without task rewards!

    Components:
    1. Skill-conditioned policy π(a|s,z)
    2. Discriminator q(z|s) - predict skill from state
    3. Intrinsic reward: log q(z|s) - log p(z)

    Skills learn to visit different states!
    """
    def __init__(self, state_dim, action_dim, num_skills=8):
        self.num_skills = num_skills

        # Skill-conditioned policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim + num_skills, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        # Discriminator: Predict skill from state
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_skills)
        )

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=3e-4
        )

        print(f"DIAYN initialized with {num_skills} skills")

    def sample_skill(self):
        """Sample random skill z."""
        return np.random.randint(self.num_skills)

    def compute_intrinsic_reward(self, state, skill):
        """
        Intrinsic reward for DIAYN.

        r_i = log q(z|s) - log p(z)

        Encourages skill z to visit distinctive states!
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            logits = self.discriminator(state_tensor)
            log_q_z = F.log_softmax(logits, dim=-1)[0, skill]

        log_p_z = -np.log(self.num_skills)  # Uniform prior

        intrinsic_reward = log_q_z.item() - log_p_z

        return intrinsic_reward

    def update_discriminator(self, states, skills):
        """Train discriminator to predict skill from state."""
        states = torch.FloatTensor(states)
        skills = torch.LongTensor(skills)

        # Predict skill
        logits = self.discriminator(states)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, skills)

        # Update
        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        return loss.item()


print("\nSkill Discovery (DIAYN):")
print("✅ Learn diverse skills automatically")
print("✅ No task rewards needed")
print("✅ Discriminator encourages diversity")
print("✅ Transferable to downstream tasks")
```

---

## 4. Meta-Reinforcement Learning

Learn to learn! Fast adaptation to new tasks.

```python
"""
Meta-RL:

Goal: Learn policies that adapt quickly to new tasks.

Problem:
- Standard RL: Millions of samples per task
- Real world: Need fast adaptation

Meta-RL Solution:
- Train on distribution of tasks
- Learn to adapt with few samples
- Transfer meta-knowledge

Key Algorithms:
1. MAML (Model-Agnostic Meta-Learning)
2. RL² (Fast RL via Slow RL)
3. PEARL (Probabilistic Embeddings for Actor-critic)
"""

class MAML_RL:
    """
    Model-Agnostic Meta-Learning for RL.

    Key idea: Learn initialization that adapts quickly.

    Algorithm:
    1. Sample task
    2. Take gradient steps on task (inner loop)
    3. Evaluate adapted policy
    4. Update meta-parameters (outer loop)

    Result: Policy that adapts in few gradient steps!
    """
    def __init__(self, state_dim, action_dim, lr_inner=0.01, lr_outer=0.001):
        # Meta-policy (shared initialization)
        self.meta_policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        self.lr_inner = lr_inner  # Task adaptation learning rate
        self.lr_outer = lr_outer  # Meta-update learning rate

        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=lr_outer)

        print("MAML-RL initialized")

    def adapt(self, task, num_steps=5):
        """
        Adapt meta-policy to specific task (inner loop).

        Args:
            task: Task-specific environment
            num_steps: Number of gradient steps

        Returns:
            adapted_policy: Policy adapted to task
        """
        # Clone meta-policy
        adapted_policy = type(self.meta_policy)(
            self.meta_policy[0].in_features,
            self.meta_policy[-1].out_features
        )
        adapted_policy.load_state_dict(self.meta_policy.state_dict())

        # Inner loop: Adapt to task
        inner_optimizer = optim.SGD(adapted_policy.parameters(), lr=self.lr_inner)

        for _ in range(num_steps):
            # Collect trajectory from task
            # states, actions, rewards = collect_trajectory(task, adapted_policy)

            # Compute policy gradient loss
            # loss = policy_loss(states, actions, rewards)

            # Gradient step
            # inner_optimizer.zero_grad()
            # loss.backward()
            # inner_optimizer.step()
            pass

        return adapted_policy

    def meta_update(self, task_distribution, num_tasks=10):
        """
        Meta-update (outer loop).

        1. Sample tasks
        2. Adapt to each task
        3. Evaluate adapted policies
        4. Update meta-parameters
        """
        meta_loss = 0

        for _ in range(num_tasks):
            # Sample task
            # task = task_distribution.sample()

            # Adapt to task
            # adapted_policy = self.adapt(task)

            # Evaluate adapted policy on task
            # eval_loss = evaluate_policy(adapted_policy, task)

            # meta_loss += eval_loss
            pass

        # Meta-gradient update
        # self.meta_optimizer.zero_grad()
        # meta_loss.backward()
        # self.meta_optimizer.step()

        print("Meta-update complete")


print("\nMAML for RL:")
print("✅ Learn to adapt quickly")
print("✅ Few-shot RL")
print("✅ Meta-learning across tasks")
print("✅ Used in robotics")
```

---

## 5. Safe Reinforcement Learning

Safety-critical applications need constraints!

```python
"""
Safe RL:

Problem: Standard RL may take dangerous actions during learning.

Real-world needs:
- Autonomous vehicles: No crashes
- Robotics: No damage to robot/environment
- Healthcare: Patient safety

Safe RL Approaches:
1. Constrained RL: Optimize reward subject to constraints
2. Risk-sensitive RL: Avoid high-variance trajectories
3. Shielding: Safety layer prevents unsafe actions
4. Imitation: Learn from safe demonstrations
"""

class ConstrainedPPO:
    """
    Constrained PPO: PPO with cost constraints.

    Objective:
    maximize E[reward]
    subject to E[cost] ≤ threshold

    Uses Lagrangian relaxation:
    L = reward - λ * (cost - threshold)

    λ (Lagrange multiplier) automatically adjusts!
    """
    def __init__(self, state_dim, action_dim, cost_limit=25):
        # Standard PPO policy
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        # Cost critic: Predicts expected cost
        self.cost_critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # Lagrange multiplier (learned!)
        self.lambda_param = nn.Parameter(torch.zeros(1))

        self.cost_limit = cost_limit

        print(f"Constrained PPO initialized (cost limit: {cost_limit})")

    def compute_constrained_loss(self, states, actions, rewards, costs):
        """
        Constrained PPO loss.

        L = reward_loss - λ * (cost - cost_limit)
        """
        # Standard PPO loss (on rewards)
        # reward_loss = ppo_loss(states, actions, rewards)

        # Cost constraint violation
        # predicted_cost = self.cost_critic(states).mean()
        # constraint_violation = predicted_cost - self.cost_limit

        # Lagrangian
        # total_loss = reward_loss - self.lambda_param * constraint_violation

        # Update lambda
        # self.lambda_param = max(0, self.lambda_param + constraint_violation)

        pass


class RiskSensitiveRL:
    """
    Risk-Sensitive RL: Optimize risk-adjusted returns.

    Instead of E[return], optimize:
    - CVaR (Conditional Value at Risk): Worst α% returns
    - Mean-Variance: E[R] - β * Var(R)

    Reduces variance, avoids risky policies!
    """
    def __init__(self, state_dim, action_dim, risk_param=0.1):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()
        )

        self.risk_param = risk_param

        print(f"Risk-Sensitive RL (risk aversion: {risk_param})")

    def risk_adjusted_return(self, returns):
        """
        Compute risk-adjusted return.

        CVaR_α(R) = E[R | R ≤ quantile_α(R)]
        """
        # Compute α-quantile
        quantile = np.percentile(returns, self.risk_param * 100)

        # CVaR: Mean of returns below quantile
        cvar = returns[returns <= quantile].mean()

        return cvar


print("\nSafe RL:")
print("✅ Constrained optimization")
print("✅ Risk-sensitive objectives")
print("✅ Safety layers/shielding")
print("✅ Critical for real-world deployment")
```

---

## 6. Transfer Learning and Continual Learning

Leverage past experience!

```python
"""
Transfer Learning in RL:

Types:
1. Task Transfer: Learn on task A, transfer to task B
2. Domain Transfer: Sim-to-real (see Lesson 12)
3. Knowledge Transfer: Reuse skills/representations

Approaches:
- Progressive networks: Freeze old task columns, add new
- Policy distillation: Compress multi-task knowledge
- Universal value functions: Generalize across goals
"""

class ProgressiveNetworks:
    """
    Progressive Networks: Transfer without forgetting.

    Architecture:
    - Column per task
    - Lateral connections from old to new
    - Old columns frozen

    Advantages:
    ✅ No catastrophic forgetting
    ✅ Transfer from all previous tasks
    ✅ Continual learning
    """
    def __init__(self, state_dim, action_dim):
        # Start with first task column
        self.columns = [
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            )
        ]

        print("Progressive Networks initialized")

    def add_task(self, state_dim, action_dim):
        """
        Add new task column with lateral connections.
        """
        # New column
        new_column = nn.Sequential(
            nn.Linear(state_dim + 128 * len(self.columns), 128),  # Includes lateral
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

        # Freeze previous columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False

        self.columns.append(new_column)

        print(f"Added task {len(self.columns)}, previous tasks frozen")

    def forward(self, state, task_id):
        """Forward pass for specific task."""
        if task_id == 0:
            return self.columns[0](state)

        # Concatenate activations from previous columns
        prev_activations = []
        for i in range(task_id):
            with torch.no_grad():
                hidden = self.columns[i][0](state)  # First layer
                prev_activations.append(hidden)

        # Combine with current input
        combined = torch.cat([state] + prev_activations, dim=-1)

        return self.columns[task_id](combined)


print("\nTransfer & Continual Learning:")
print("✅ Leverage past experience")
print("✅ Avoid catastrophic forgetting")
print("✅ Progressive networks")
print("✅ Lifelong learning")
```

---

## 7. Real-World Applications

```python
"""
HRL, Meta-RL, Safe RL Applications:

1. Robotics:
   - HRL: Complex manipulation (pick-and-place sequences)
   - Meta-RL: Fast adaptation to new objects
   - Safe RL: Constraint satisfaction

2. Autonomous Vehicles:
   - HRL: Hierarchical driving (route → lane → control)
   - Safe RL: Safety guarantees

3. Healthcare:
   - Safe RL: Treatment optimization with safety
   - Meta-RL: Personalization per patient

4. Industrial Control:
   - HRL: Multi-stage processes
   - Safe RL: Operating constraints
"""

print("\n=== Applications ===")
print("\n1. Robotics:")
print("   - HRL: Complex manipulation")
print("   - Meta-RL: Fast adaptation")
print("   - Safe RL: Collision avoidance")

print("\n2. Autonomous Vehicles:")
print("   - HRL: Hierarchical planning")
print("   - Safe RL: Safety guarantees")

print("\n3. Healthcare:")
print("   - Safe RL: Patient safety")
print("   - Meta-RL: Personalization")
```

---

## Practice Exercises

### Exercise 1: Implement Options

```python
"""
Implement options framework and train hierarchical agent.

Requirements:
1. Define 3-5 hand-crafted options for a task
2. Implement option-critic for learning options
3. Train on multi-room navigation
4. Compare with flat RL

Track:
- Learning curves
- Option usage frequencies
- Transfer to new layouts
"""

# Your implementation here
```

### Exercise 2: MAML for Few-Shot RL

```python
"""
Implement MAML and test on distribution of tasks.

Tasks: GridWorld with different goal locations

Requirements:
1. Full MAML implementation
2. Inner loop: Task adaptation (5 gradient steps)
3. Outer loop: Meta-update
4. Evaluate: Few-shot adaptation vs training from scratch

Compare:
- Adaptation speed
- Final performance
- Sample efficiency
"""

# Your implementation here
```

### Exercise 3: Safe RL with Constraints

```python
"""
Implement constrained PPO for safe RL.

Environment: SafetyGym or custom with safety constraints

Requirements:
1. Define cost function (e.g., distance to obstacles)
2. Implement Lagrangian PPO
3. Train with cost constraints
4. Compare with unconstrained PPO

Metrics:
- Reward
- Cost violations
- Safety rate
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Hierarchical RL** 🏗️
   - Temporal abstraction
   - Options framework
   - Reusable skills
   - Faster learning

2. **Skill Discovery** 🔍
   - Automatic discovery (DIAYN)
   - Diversity-based learning
   - No task rewards needed
   - Transferable skills

3. **Meta-RL** 🚀
   - Learn to learn
   - Fast adaptation
   - MAML, RL²
   - Few-shot RL

4. **Safe RL** 🛡️
   - Constrained optimization
   - Risk-sensitive policies
   - Safety layers
   - Real-world critical

5. **Transfer Learning** 🔄
   - Leverage past experience
   - Progressive networks
   - Continual learning
   - Avoid forgetting

### When to Use?

✅ **HRL**: Long horizons, complex tasks, reusable skills
✅ **Meta-RL**: Distribution of related tasks, fast adaptation needed
✅ **Safe RL**: Safety-critical applications, hard constraints
✅ **Transfer**: Multiple related tasks, continual learning

### Real-World Impact

✅ **Robotics**: Complex manipulation, fast adaptation, safety
✅ **Autonomous Vehicles**: Hierarchical planning, safety guarantees
✅ **Healthcare**: Safe treatment optimization, personalization

### What's Next?

In Lesson 12, we'll learn **Production RL**:
- Stable-Baselines3 comprehensive guide
- Deployment with FastAPI
- Sim-to-real transfer
- Distributed training and scaling

**HRL, Meta-RL, and Safe RL enable real-world deployment!** 🏗️

---

## Additional Resources

### Papers
- Sutton et al. (1999): "Between MDPs and semi-MDPs: A framework for temporal abstraction using options"
- Eysenbach et al. (2019): "Diversity is All You Need" (DIAYN)
- Finn et al. (2017): "Model-Agnostic Meta-Learning" (MAML)
- Achiam et al. (2017): "Constrained Policy Optimization" (CPO)
- Rusu et al. (2016): "Progressive Neural Networks"

### Libraries
- **garage**: https://github.com/rlworkgroup/garage (Meta-RL)
- **safety-gym**: https://github.com/openai/safety-gym (Safe RL)
- **learn2learn**: https://github.com/learnables/learn2learn (Meta-learning)

### Resources
- **Berkeley CS294**: Advanced Deep RL (HRL, Meta-RL lectures)
- **Safe RL Tutorial**: https://sites.google.com/view/safe-rl-tutorial-neurips2019
- **Meta-RL Tutorial**: https://sites.google.com/view/icml19metalearning

---

**Next**: [Lesson 12 - Production RL and Sim-to-Real](Lesson%2012%20-%20Production%20RL%20and%20Sim-to-Real.md)

Proceed to learn about **deployment**, **scaling**, and **real-world systems**! 🚀

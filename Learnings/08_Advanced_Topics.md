# Advanced Machine Learning Topics 🎓

Cutting-edge concepts that push the boundaries of ML. These topics are active research areas!

## Table of Contents
1. [Generative Adversarial Networks (GANs)](#generative-adversarial-networks-gans)
2. [Reinforcement Learning](#reinforcement-learning)
3. [Meta-Learning](#meta-learning)
4. [Few-Shot Learning](#few-shot-learning)
5. [Self-Supervised Learning](#self-supervised-learning)
6. [Neural Architecture Search](#neural-architecture-search)
7. [Diffusion Models](#diffusion-models)

---

## Generative Adversarial Networks (GANs) 🎨

**Key Paper:** "Generative Adversarial Networks" (Goodfellow et al., 2014)

### Core Concept: Two Networks Battle

**Generator (G):** Creates fake data
**Discriminator (D):** Distinguishes real vs fake

```
Real Images ──┐
              ├──→ [Discriminator] → Real/Fake?
Noise → [Generator] → Fake Images ──┘
```

**Analogy:** Counterfeiter vs Detective
- Generator: Creates fake money
- Discriminator: Detects fakes
- Both improve through competition!

### Mathematics

**Discriminator Goal:** Maximize ability to classify real/fake
```
max_D V(D) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

**Generator Goal:** Fool discriminator
```
min_G V(G) = E_z[log(1 - D(G(z)))]
```

**Combined (Minimax Game):**
```
min_G max_D V(D,G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]
```

### Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, int(np.prod(img_shape))),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), *self.img_shape)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        return self.model(img_flat)

# Training Loop
generator = Generator()
discriminator = Discriminator()

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion = nn.BCELoss()

for epoch in range(epochs):
    for real_imgs in dataloader:
        batch_size = real_imgs.size(0)

        # Labels
        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        d_optimizer.zero_grad()

        # Real images
        real_loss = criterion(discriminator(real_imgs), real_labels)

        # Fake images
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)

        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------
        g_optimizer.zero_grad()

        # Generator tries to fool discriminator
        z = torch.randn(batch_size, latent_dim)
        fake_imgs = generator(z)
        g_loss = criterion(discriminator(fake_imgs), real_labels)  # Want D to output 1

        g_loss.backward()
        g_optimizer.step()
```

### Popular GAN Variants

**1. DCGAN (Deep Convolutional GAN)**
- Use conv layers instead of fully connected
- BatchNorm in generator, discriminator
- ReLU in generator, LeakyReLU in discriminator
- No pooling, use strided convolutions

**2. StyleGAN / StyleGAN2**
- High-quality image generation
- Style-based generator architecture
- Can control image attributes at different scales
- Powers thispersondoesnotexist.com

**3. CycleGAN**
- Image-to-image translation without paired data
- Horse → Zebra, Summer → Winter
- Uses cycle consistency loss

**4. Pix2Pix**
- Paired image-to-image translation
- Sketch → Photo, Day → Night
- Conditional GAN with U-Net generator

**5. Progressive GAN**
- Gradually increase image resolution during training
- Start: 4×4, End: 1024×1024
- More stable training

### Common GAN Challenges

**1. Mode Collapse**
- Generator produces limited variety
- All outputs look similar
**Solution:** Mini-batch discrimination, unrolled GAN

**2. Instability**
- Training doesn't converge
- Oscillating losses
**Solution:** Wasserstein GAN, Spectral Normalization

**3. Evaluation**
- Hard to measure quality
**Metrics:** Inception Score (IS), Fréchet Inception Distance (FID)

---

## Reinforcement Learning (RL) 🎮

**Key Idea:** Learn by trial and error, maximizing cumulative reward.

### Core Concepts

**Components:**
- **Agent:** The learner/decision maker
- **Environment:** The world
- **State (s):** Current situation
- **Action (a):** What agent can do
- **Reward (r):** Feedback signal
- **Policy (π):** Strategy (state → action mapping)

**Objective:** Maximize cumulative reward
```
G_t = r_t + γr_{t+1} + γ²r_{t+2} + ...

where γ = discount factor (0 to 1)
```

### Markov Decision Process (MDP)

Formal framework for RL:
```
States: S = {s₁, s₂, ..., sₙ}
Actions: A = {a₁, a₂, ..., aₘ}
Transition: P(s'|s,a) = probability of s' given s, a
Reward: R(s, a, s')
Policy: π(a|s) = probability of action a in state s
```

### Value Functions

**State Value Function V(s):**
Expected return starting from state s
```
V^π(s) = E_π[G_t | s_t = s]
```

**Action Value Function Q(s,a):**
Expected return after taking action a in state s
```
Q^π(s,a) = E_π[G_t | s_t = s, a_t = a]
```

### Bellman Equation

**Recursive relationship:**
```
V(s) = max_a [R(s,a) + γ Σ P(s'|s,a) V(s')]

Q(s,a) = R(s,a) + γ Σ P(s'|s,a) max_a' Q(s',a')
```

### Q-Learning Algorithm

**Model-free, off-policy RL algorithm**

```python
import numpy as np

class QLearning:
    def __init__(self, n_states, n_actions, learning_rate=0.1, discount=0.99, epsilon=0.1):
        self.Q = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon

    def get_action(self, state):
        # Epsilon-greedy policy
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.Q[state]))  # Explore
        else:
            return np.argmax(self.Q[state])  # Exploit

    def update(self, state, action, reward, next_state):
        # Q-learning update rule
        best_next_action = np.argmax(self.Q[next_state])
        td_target = reward + self.gamma * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.lr * td_error

# Training loop
env = gym.make('FrozenLake-v1')
agent = QLearning(n_states=env.observation_space.n,
                  n_actions=env.action_space.n)

for episode in range(10000):
    state = env.reset()
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        state = next_state
```

### Deep Q-Network (DQN)

**Use neural network to approximate Q-function**

**Key Innovations:**
1. **Experience Replay:** Store transitions, sample randomly
2. **Target Network:** Separate network for stability

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, state):
        return self.network(state)

# Experience Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

# Training
policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters())
replay_buffer = ReplayBuffer()

for episode in range(num_episodes):
    state = env.reset()

    for t in range(max_steps):
        # Select action
        if random.random() < epsilon:
            action = random.randrange(action_dim)
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()

        # Execute action
        next_state, reward, done, _ = env.step(action)

        # Store transition
        replay_buffer.push(state, action, reward, next_state, done)

        # Sample batch and train
        if len(replay_buffer) > batch_size:
            batch = replay_buffer.sample(batch_size)

            # Compute Q-learning loss
            states, actions, rewards, next_states, dones = zip(*batch)

            current_q = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].detach()
            target_q = rewards + gamma * next_q * (1 - dones)

            loss = F.mse_loss(current_q, target_q)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        state = next_state
        if done:
            break

    # Update target network
    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
```

### Policy Gradient Methods

**Directly optimize policy instead of value function**

**REINFORCE Algorithm:**
```python
def reinforce(policy, optimizer, gamma=0.99):
    episode_rewards = []
    episode_log_probs = []

    # Run episode
    state = env.reset()
    done = False

    while not done:
        # Sample action from policy
        action_probs = policy(state)
        action = torch.multinomial(action_probs, 1)
        log_prob = torch.log(action_probs[action])

        # Take action
        next_state, reward, done, _ = env.step(action.item())

        episode_rewards.append(reward)
        episode_log_probs.append(log_prob)

        state = next_state

    # Compute discounted returns
    returns = []
    G = 0
    for r in reversed(episode_rewards):
        G = r + gamma * G
        returns.insert(0, G)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # Compute policy loss
    policy_loss = []
    for log_prob, G in zip(episode_log_probs, returns):
        policy_loss.append(-log_prob * G)

    # Optimize
    optimizer.zero_grad()
    loss = torch.stack(policy_loss).sum()
    loss.backward()
    optimizer.step()
```

### Modern RL Algorithms

**1. Actor-Critic**
- Actor: Policy network
- Critic: Value network
- Reduces variance, stable training

**2. Proximal Policy Optimization (PPO)**
- Most popular modern algorithm
- Stable, sample efficient
- Used in ChatGPT (RLHF)

**3. Soft Actor-Critic (SAC)**
- Off-policy, maximum entropy
- Continuous action spaces
- Robotics applications

**4. AlphaGo / AlphaZero**
- Self-play + Monte Carlo Tree Search
- Superhuman performance in Go, Chess, Shogi

---

## Meta-Learning (Learning to Learn) 🧠

**Goal:** Learn across multiple tasks to quickly adapt to new tasks.

### Key Idea

Instead of learning one task, learn how to learn tasks!

```
Traditional ML:
Data → Model for Task A

Meta-Learning:
Multiple Tasks → Meta-Model → Quickly adapt to new Task Z
```

### Model-Agnostic Meta-Learning (MAML)

**Key Paper:** "Model-Agnostic Meta-Learning for Fast Adaptation" (Finn et al., 2017)

**Algorithm:**
1. Sample batch of tasks
2. For each task:
   - Train on support set (few examples)
   - Evaluate on query set
3. Meta-update: Find initialization that works well across all tasks

```python
def maml_train(meta_model, tasks, inner_lr=0.01, outer_lr=0.001, inner_steps=5):
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=outer_lr)

    for iteration in range(num_iterations):
        meta_loss = 0

        # Sample batch of tasks
        task_batch = sample_tasks(tasks)

        for task in task_batch:
            # Clone model for task-specific adaptation
            task_model = copy.deepcopy(meta_model)
            task_optimizer = optim.SGD(task_model.parameters(), lr=inner_lr)

            # Inner loop: Adapt to task
            support_x, support_y = task.support_set()
            for _ in range(inner_steps):
                loss = criterion(task_model(support_x), support_y)
                task_optimizer.zero_grad()
                loss.backward()
                task_optimizer.step()

            # Evaluate on query set
            query_x, query_y = task.query_set()
            query_loss = criterion(task_model(query_x), query_y)
            meta_loss += query_loss

        # Outer loop: Meta-update
        meta_loss /= len(task_batch)
        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()
```

**Applications:**
- Few-shot image classification
- Robot learning (adapt to new objects quickly)
- Personalized recommendations

---

## Few-Shot Learning 🎯

**Goal:** Learn from very few examples (1-shot, 5-shot, etc.)

### Approaches

**1. Metric Learning**
Learn embedding space where similar examples are close.

**Siamese Networks:**
```python
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 10),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256)
        )

    def forward(self, x1, x2):
        # Encode both images
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)

        # Compute distance
        distance = F.pairwise_distance(h1, h2)
        return distance

# Contrastive loss
def contrastive_loss(distance, label, margin=1.0):
    # label = 1 for same class, 0 for different
    loss = label * distance.pow(2) + \
           (1 - label) * F.relu(margin - distance).pow(2)
    return loss.mean()
```

**2. Prototypical Networks**
Classify based on distance to class prototype (mean embedding).

**3. Matching Networks**
Attention-based classification using support set.

---

## Self-Supervised Learning 🔄

**Key Idea:** Create labels from the data itself, no human annotation!

### Contrastive Learning

**SimCLR (Simple Framework for Contrastive Learning)**

**Process:**
1. Take an image
2. Apply two different augmentations → create positive pair
3. Other images in batch → negative examples
4. Pull positive pairs together, push negatives apart

```python
def simclr_loss(z_i, z_j, temperature=0.5):
    """
    z_i, z_j: embeddings of two augmented views of same image
    """
    batch_size = z_i.shape[0]

    # Normalize embeddings
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    # Concatenate
    z = torch.cat([z_i, z_j], dim=0)  # 2N x D

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # 2N x 2N

    # Positive pairs: (i, N+i) and (N+i, i)
    # Negatives: all others

    # Create labels for cross-entropy
    labels = torch.arange(batch_size)
    labels = torch.cat([labels + batch_size, labels])

    # Mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # Contrastive loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss
```

**Other Methods:**
- **BERT:** Masked language modeling
- **GPT:** Next token prediction
- **MAE (Masked Autoencoders):** Reconstruct masked image patches
- **CLIP:** Align images and text

---

## Diffusion Models 🌊

**Latest breakthrough in generative modeling!** Powers Stable Diffusion, DALL-E 2.

### Core Idea

**Forward Process:** Gradually add noise to data
```
x_0 → x_1 → x_2 → ... → x_T (pure noise)
```

**Reverse Process:** Learn to denoise
```
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (clean image)
```

### Mathematics

**Forward (Diffusion):**
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)

where β_t is noise schedule
```

**Reverse (Denoising):**
Learn p_θ(x_{t-1} | x_t) to reverse the process

**Training:**
Predict the noise that was added!

```python
class DiffusionModel(nn.Module):
    def __init__(self, noise_steps=1000):
        super().__init__()
        self.noise_steps = noise_steps

        # Beta schedule (amount of noise at each step)
        self.beta = torch.linspace(1e-4, 0.02, noise_steps)
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward_diffusion(self, x_0, t):
        """Add noise to x_0 to get x_t"""
        noise = torch.randn_like(x_0)
        alpha_hat_t = self.alpha_hat[t]

        # x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * noise
        x_t = torch.sqrt(alpha_hat_t) * x_0 + torch.sqrt(1 - alpha_hat_t) * noise

        return x_t, noise

    def sample(self, model, n_samples):
        """Generate samples by denoising"""
        # Start from pure noise
        x = torch.randn(n_samples, 3, 64, 64)

        # Iteratively denoise
        for t in reversed(range(self.noise_steps)):
            t_tensor = torch.tensor([t] * n_samples)

            # Predict noise
            predicted_noise = model(x, t_tensor)

            # Remove predicted noise
            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]
            beta_t = self.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (1 / torch.sqrt(alpha_t)) * (
                x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

        return x
```

**Advantages over GANs:**
- More stable training
- Better sample quality
- Easier to train

**Applications:**
- Image generation (Stable Diffusion)
- Inpainting
- Super-resolution
- Text-to-image (DALL-E 2, Midjourney)

---

## Key Takeaways 💡

1. **GANs:** Two networks compete to generate realistic data
2. **RL:** Learn through trial and error with rewards
3. **Meta-Learning:** Learn to learn, fast adaptation
4. **Few-Shot:** Learn from minimal examples
5. **Self-Supervised:** Create your own labels from data
6. **Diffusion Models:** State-of-the-art generation by iterative denoising

## Practical Resources 🛠️

**GANs:**
- PyTorch GAN Zoo
- StyleGAN2-ADA (NVIDIA)

**RL:**
- OpenAI Gym environments
- Stable Baselines3 (implementations)

**Self-Supervised:**
- VISSL (Facebook)
- SimCLR, MoCo implementations

**Diffusion:**
- Stable Diffusion (HuggingFace)
- Denoising Diffusion Probabilistic Models

## Next Steps 📚

- Implement a simple GAN for MNIST
- Try Q-learning on CartPole
- Experiment with SimCLR on CIFAR-10
- Fine-tune Stable Diffusion

---

**These are active research areas!** Read recent papers on ArXiv to stay updated. 🚀

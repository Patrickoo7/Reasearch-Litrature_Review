# Lesson 2: Diffusion Models Fundamentals - The New State-of-the-Art 🌊

**Module 14: Generative Vision Models | Lesson 2 of 6**

Learn the breakthrough architecture behind DALL-E 2, Stable Diffusion, and Midjourney!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand forward and reverse diffusion processes
2. ✅ Implement DDPM (Denoising Diffusion Probabilistic Models)
3. ✅ Master DDIM for faster sampling
4. ✅ Design noise schedules (linear, cosine, improved)
5. ✅ Build U-Net architecture for diffusion
6. ✅ Apply diffusion models to image generation
7. ✅ Understand why diffusion beats GANs

---

## Prerequisites

- **Lesson 1**: GANs and VAEs (for comparison)
- **Module 6**: Computer Vision fundamentals (U-Net, CNNs)
- **Module 3**: Deep Learning (gradient descent, backprop)
- **Python Libraries**: PyTorch, numpy, matplotlib
- **Math**: Gaussian distributions, conditional probability

---

## 1. The Diffusion Process - Intuition

### From Noise to Images (and Back)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
DIFFUSION MODELS: A Two-Stage Process

FORWARD DIFFUSION (Training):
  x_0 → x_1 → x_2 → ... → x_T
  Image gradually becomes pure Gaussian noise

REVERSE DIFFUSION (Sampling):
  x_T → x_{T-1} → ... → x_1 → x_0
  Noise gradually becomes image

Key Insight: If we can reverse the diffusion, we can generate!
"""

# Set random seed
torch.manual_seed(42)
np.random.seed(42)


def visualize_diffusion_concept():
    """
    Visualize the core idea of diffusion models.
    """
    print("DIFFUSION MODELS: The Core Idea\n")

    print("FORWARD PROCESS (Destroy Information):")
    print("  t=0:   📸 Clear image")
    print("  t=250: 🌫️  Some noise")
    print("  t=500: 🌪️  More noise")
    print("  t=750: ❄️  Mostly noise")
    print("  t=1000: 📊 Pure Gaussian noise")
    print()

    print("REVERSE PROCESS (Recover Information):")
    print("  t=1000: 📊 Sample pure noise")
    print("  t=750: ❄️  Denoise slightly → Learn to remove noise")
    print("  t=500: 🌪️  Denoise more")
    print("  t=250: 🌫️  Image emerging")
    print("  t=0:   📸 Final clear image!")
    print()

    print("Why this works:")
    print("✅ Each denoising step is small (easy to learn)")
    print("✅ Training objective is simple (predict noise)")
    print("✅ Sampling is iterative refinement")
    print("✅ More stable than GANs!")


visualize_diffusion_concept()
```

---

## 2. Forward Diffusion Process

### Adding Noise Over Time

```python
"""
Forward Diffusion: q(x_t | x_{t-1})

At each timestep, add Gaussian noise:
  x_t = √(1-β_t) * x_{t-1} + √β_t * ε
  where ε ~ N(0, I)

β_t: variance schedule (controls noise amount)
"""


class ForwardDiffusion:
    """
    Forward diffusion process.

    Gradually adds noise to images.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        """
        Args:
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise variance
            beta_end: Ending noise variance
        """
        self.num_timesteps = num_timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # For q(x_t | x_0) - direct sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """
        Sample from q(x_t | x_0) - forward process.

        Can jump directly to any timestep!

        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        where ᾱ_t = ∏_{i=1}^t α_i
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Extract coefficients for timestep t
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        # Apply noise
        x_t = sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

        return x_t


# Test forward diffusion
forward_diffusion = ForwardDiffusion(num_timesteps=1000)

# Create sample image
x_0 = torch.randn(1, 1, 28, 28)

# Add noise at different timesteps
timesteps = [0, 250, 500, 750, 999]

print("Forward Diffusion Process:")
for t in timesteps:
    t_tensor = torch.tensor([t])
    x_t = forward_diffusion.q_sample(x_0, t_tensor)

    noise_level = forward_diffusion.sqrt_one_minus_alphas_cumprod[t].item()
    signal_level = forward_diffusion.sqrt_alphas_cumprod[t].item()

    print(f"  t={t:4d}: Signal={signal_level:.3f}, Noise={noise_level:.3f}")
```

### Visualizing the Forward Process

```python
def visualize_forward_diffusion(image, forward_diffusion, timesteps=[0, 100, 300, 600, 999]):
    """
    Visualize forward diffusion at different timesteps.

    Args:
        image: Original image [1, C, H, W]
        forward_diffusion: ForwardDiffusion instance
        timesteps: List of timesteps to visualize
    """
    fig, axes = plt.subplots(1, len(timesteps), figsize=(len(timesteps)*3, 3))

    for i, t in enumerate(timesteps):
        # Sample noisy image
        t_tensor = torch.tensor([t])
        x_t = forward_diffusion.q_sample(image, t_tensor)

        # Plot
        axes[i].imshow(x_t.squeeze().cpu(), cmap='gray')
        axes[i].set_title(f't = {t}')
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('forward_diffusion.png', dpi=150, bbox_inches='tight')
    plt.show()


# Example usage
print("\nVisualization function ready!")
print("Usage: visualize_forward_diffusion(image, forward_diffusion)")
```

---

## 3. Reverse Diffusion Process

### Learning to Denoise

```python
"""
Reverse Diffusion: p_θ(x_{t-1} | x_t)

Goal: Learn to reverse the forward process
  Given noisy x_t, predict x_{t-1}

Key Insight: Predict the noise ε that was added!

Training Objective:
  L = E_{x_0, ε, t} [||ε - ε_θ(x_t, t)||²]

Where:
- x_t = noisy image at timestep t
- ε = actual noise added
- ε_θ = predicted noise (our neural network)
"""


def reverse_diffusion_step(model, x_t, t, forward_diffusion):
    """
    Single reverse diffusion step.

    Args:
        model: Noise prediction model ε_θ
        x_t: Noisy image at timestep t
        t: Current timestep
        forward_diffusion: ForwardDiffusion instance

    Returns:
        x_{t-1}: Less noisy image
    """
    # Predict noise
    predicted_noise = model(x_t, t)

    # Extract parameters
    alpha_t = forward_diffusion.alphas[t].view(-1, 1, 1, 1)
    alpha_cumprod_t = forward_diffusion.alphas_cumprod[t].view(-1, 1, 1, 1)
    beta_t = forward_diffusion.betas[t].view(-1, 1, 1, 1)

    # Compute x_{t-1}
    # Formula: x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

    # Mean of p(x_{t-1} | x_t)
    pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / torch.sqrt(alpha_cumprod_t)
    pred_mean = (torch.sqrt(alpha_t) * beta_t / (1.0 - alpha_cumprod_t)) * pred_x0 + \
                ((1.0 - alpha_t) * torch.sqrt(alpha_cumprod_t) / (1.0 - alpha_cumprod_t)) * x_t

    # Add noise (except at t=0)
    if t[0] > 0:
        noise = torch.randn_like(x_t)
        sigma_t = torch.sqrt(beta_t)
        x_t_minus_1 = pred_mean + sigma_t * noise
    else:
        x_t_minus_1 = pred_mean

    return x_t_minus_1


print("Reverse diffusion step function defined!")
```

---

## 4. U-Net Architecture for Diffusion

### Noise Prediction Network

```python
"""
U-Net for Diffusion Models

Architecture:
1. Downsampling path: Extract features at multiple scales
2. Upsampling path: Reconstruct to original resolution
3. Skip connections: Preserve fine details
4. Time embedding: Condition on timestep t

Input: Noisy image x_t + timestep t
Output: Predicted noise ε_θ
"""


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time embeddings.

    Encodes timestep t as a vector.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        Args:
            t: Timesteps [batch_size]

        Returns:
            Time embeddings [batch_size, dim]
        """
        device = t.device
        half_dim = self.dim // 2

        # Sinusoidal embeddings
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]

        # Concat sin and cos
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        return embeddings


class ResidualBlock(nn.Module):
    """
    Residual block with time conditioning.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)

        # Residual connection
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        """
        Args:
            x: Feature maps [batch, in_channels, H, W]
            t_emb: Time embeddings [batch, time_emb_dim]

        Returns:
            Output features [batch, out_channels, H, W]
        """
        residual = self.residual_conv(x)

        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)

        # Add time embedding
        t_emb_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_emb_proj

        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)

        return h + residual


class SimpleUNet(nn.Module):
    """
    Simplified U-Net for diffusion models.

    Predicts noise ε given noisy image x_t and timestep t.
    """
    def __init__(self, img_channels=1, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_emb_dim = time_emb_dim

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Encoder (downsampling)
        self.down1 = ResidualBlock(img_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)

        # Decoder (upsampling)
        self.up3 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Output
        self.out_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t):
        """
        Args:
            x: Noisy images [batch, channels, H, W]
            t: Timesteps [batch]

        Returns:
            Predicted noise [batch, channels, H, W]
        """
        # Time embedding
        t_emb = self.time_embedding(t.float())

        # Encoder
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)

        # Decoder with skip connections
        u3 = self.up3(torch.cat([self.upsample(b), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample(u3), d2], dim=1), t_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), t_emb)

        # Output
        out = self.out_conv(u1)

        return out


# Test U-Net
unet = SimpleUNet(img_channels=1, base_channels=64)

# Test forward pass
x_test = torch.randn(2, 1, 28, 28)
t_test = torch.tensor([100, 200])

noise_pred = unet(x_test, t_test)

print(f"Input shape: {x_test.shape}")
print(f"Timesteps: {t_test}")
print(f"Predicted noise shape: {noise_pred.shape}")
print(f"\nU-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
```

---

## 5. Training DDPM

### Complete Training Loop

```python
class DDPM:
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Complete implementation with training and sampling.
    """
    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        """
        Args:
            model: Noise prediction network (e.g., U-Net)
            num_timesteps: Number of diffusion steps
            beta_start: Starting noise variance
            beta_end: Ending noise variance
            device: 'cuda' or 'cpu'
        """
        self.model = model.to(device)
        self.num_timesteps = num_timesteps
        self.device = device

        # Noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # For sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: Add noise to images."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise

    def train_step(self, x_0):
        """
        Single training step.

        Args:
            x_0: Clean images [batch, channels, H, W]

        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

        # Sample noise
        noise = torch.randn_like(x_0)

        # Add noise (forward diffusion)
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        predicted_noise = self.model(x_t, t)

        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def p_sample(self, x_t, t):
        """
        Single reverse diffusion step.

        Args:
            x_t: Noisy image at timestep t
            t: Current timestep

        Returns:
            x_{t-1}: Denoised image
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Extract parameters
        alpha_t = self.alphas[t][:, None, None, None]
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        beta_t = self.betas[t][:, None, None, None]

        # Compute mean
        sqrt_alpha_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)

        # Predicted x_0
        pred_x0 = (x_t - sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t

        # Mean of p(x_{t-1} | x_t)
        alpha_cumprod_prev = self.alphas_cumprod[t-1][:, None, None, None] if t[0] > 0 else torch.ones_like(alpha_cumprod_t)
        pred_mean = torch.sqrt(alpha_cumprod_prev) * beta_t / (1.0 - alpha_cumprod_t) * pred_x0 + \
                    torch.sqrt(alpha_t) * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t) * x_t

        # Add noise (except at t=0)
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            sigma_t = torch.sqrt(beta_t)
            return pred_mean + sigma_t * noise
        else:
            return pred_mean

    @torch.no_grad()
    def sample(self, num_samples, img_size=(1, 28, 28)):
        """
        Generate samples by reverse diffusion.

        Args:
            num_samples: Number of samples to generate
            img_size: Size of images (channels, H, W)

        Returns:
            Generated images [num_samples, channels, H, W]
        """
        # Start from pure noise
        x_t = torch.randn(num_samples, *img_size, device=self.device)

        # Reverse diffusion
        for t in tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            x_t = self.p_sample(x_t, t_batch)

        return x_t


# Create DDPM
ddpm = DDPM(unet, num_timesteps=1000, device='cpu')

# Test training step
x_batch = torch.randn(4, 1, 28, 28)
loss = ddpm.train_step(x_batch)

print(f"Training loss: {loss.item():.4f}")
```

### Full Training Function

```python
def train_ddpm(ddpm, dataloader, num_epochs=10, lr=1e-4):
    """
    Train DDPM on image dataset.

    Args:
        ddpm: DDPM instance
        dataloader: DataLoader with images
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(ddpm.model.parameters(), lr=lr)

    losses = []

    print("Training DDPM...")

    for epoch in range(num_epochs):
        ddpm.model.train()
        epoch_loss = 0

        for x, _ in tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            x = x.to(ddpm.device)

            # Training step
            loss = ddpm.train_step(x)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

        # Sample some images
        if (epoch + 1) % 5 == 0:
            ddpm.model.eval()
            samples = ddpm.sample(num_samples=4, img_size=(1, 28, 28))
            print(f"  Generated {len(samples)} samples")

    return losses


print("DDPM training function ready!")
print("Usage: train_ddpm(ddpm, dataloader, num_epochs=10)")
```

---

## 6. DDIM - Faster Sampling

### Deterministic Sampling

```python
"""
DDIM (Denoising Diffusion Implicit Models)

Key Innovation: Deterministic sampling process
- DDPM: 1000 steps required
- DDIM: 50-100 steps sufficient!

How:
- Non-Markovian process (skip timesteps)
- Same training as DDPM
- 10-20x faster sampling
"""


class DDIM(DDPM):
    """
    DDIM: Faster sampling variant of DDPM.

    Same training, different sampling!
    """
    def __init__(self, model, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        super().__init__(model, num_timesteps, beta_start, beta_end, device)

    @torch.no_grad()
    def ddim_sample_step(self, x_t, t, t_prev, eta=0.0):
        """
        Single DDIM sampling step.

        Args:
            x_t: Noisy image at timestep t
            t: Current timestep
            t_prev: Previous timestep (can be non-adjacent!)
            eta: Stochasticity parameter (0=deterministic)

        Returns:
            x_{t_prev}: Image at previous timestep
        """
        # Predict noise
        predicted_noise = self.model(x_t, t)

        # Extract parameters
        alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
        alpha_cumprod_prev = self.alphas_cumprod[t_prev][:, None, None, None] if t_prev[0] >= 0 else torch.ones_like(alpha_cumprod_t)

        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1.0 - alpha_cumprod_t) * predicted_noise) / torch.sqrt(alpha_cumprod_t)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_cumprod_prev - eta**2 * (1.0 - alpha_cumprod_t) / (1.0 - alpha_cumprod_prev) * (1.0 - alpha_cumprod_prev / alpha_cumprod_t)) * predicted_noise

        # DDIM sampling
        x_prev = torch.sqrt(alpha_cumprod_prev) * pred_x0 + dir_xt

        # Add stochasticity (if eta > 0)
        if eta > 0:
            sigma_t = eta * torch.sqrt((1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod_t)) * torch.sqrt(1.0 - alpha_cumprod_t / alpha_cumprod_prev)
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise

        return x_prev

    @torch.no_grad()
    def ddim_sample(self, num_samples, img_size=(1, 28, 28), num_steps=50, eta=0.0):
        """
        Fast sampling with DDIM.

        Args:
            num_samples: Number of samples
            img_size: Image dimensions
            num_steps: Number of sampling steps (much less than training steps!)
            eta: Stochasticity (0=deterministic)

        Returns:
            Generated images
        """
        # Select timesteps (uniform spacing)
        step_size = self.num_timesteps // num_steps
        timesteps = list(range(0, self.num_timesteps, step_size))[::-1]

        # Start from noise
        x_t = torch.randn(num_samples, *img_size, device=self.device)

        # Reverse diffusion
        for i, t in enumerate(tqdm(timesteps, desc=f'DDIM Sampling ({num_steps} steps)')):
            t_batch = torch.full((num_samples,), t, device=self.device, dtype=torch.long)

            # Previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = -1

            t_prev_batch = torch.full((num_samples,), t_prev, device=self.device, dtype=torch.long)

            x_t = self.ddim_sample_step(x_t, t_batch, t_prev_batch, eta=eta)

        return x_t


# Create DDIM
ddim = DDIM(unet, num_timesteps=1000, device='cpu')

print("DDIM Sampling:")
print(f"  Training: {ddim.num_timesteps} timesteps")
print(f"  Sampling: 50-100 timesteps (10-20x faster!)")
print()
print("Usage: ddim.ddim_sample(num_samples=4, num_steps=50)")
```

---

## 7. Noise Schedules

### Linear vs Cosine Schedules

```python
"""
Noise Schedules: How fast to add noise

Common schedules:
1. Linear: β_t increases linearly
2. Cosine: Smoother transition (Nichol & Dhariwal, 2021)
3. Improved: Optimized for specific datasets
"""


def linear_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Linear noise schedule (original DDPM)."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def cosine_beta_schedule(num_timesteps, s=0.008):
    """
    Cosine noise schedule (improved).

    Gives more uniform signal-to-noise ratio across timesteps.
    """
    steps = num_timesteps + 1
    t = torch.linspace(0, num_timesteps, steps)

    alphas_cumprod = torch.cos(((t / num_timesteps) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]

    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Quadratic schedule - slower start, faster end."""
    return torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps) ** 2


def sigmoid_beta_schedule(num_timesteps, beta_start=0.0001, beta_end=0.02):
    """Sigmoid schedule - smooth S-curve."""
    betas = torch.linspace(-6, 6, num_timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


# Visualize different schedules
def plot_beta_schedules(num_timesteps=1000):
    """Compare different noise schedules."""
    schedules = {
        'Linear': linear_beta_schedule(num_timesteps),
        'Cosine': cosine_beta_schedule(num_timesteps),
        'Quadratic': quadratic_beta_schedule(num_timesteps),
        'Sigmoid': sigmoid_beta_schedule(num_timesteps),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot betas
    for name, betas in schedules.items():
        axes[0].plot(betas.numpy(), label=name, alpha=0.7)

    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('β_t (noise variance)')
    axes[0].set_title('Noise Schedules: β_t')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot cumulative alphas (signal preservation)
    for name, betas in schedules.items():
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        axes[1].plot(alphas_cumprod.numpy(), label=name, alpha=0.7)

    axes[1].set_xlabel('Timestep')
    axes[1].set_ylabel('ᾱ_t (signal strength)')
    axes[1].set_title('Signal Preservation: ᾱ_t')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('noise_schedules.png', dpi=150, bbox_inches='tight')
    plt.show()


plot_beta_schedules(num_timesteps=1000)

print("\nCosine Schedule Advantages:")
print("✅ More uniform SNR across timesteps")
print("✅ Better sample quality")
print("✅ Slower noise addition at start/end")
print("✅ Used in Stable Diffusion!")
```

---

## 8. Score-Based Models Connection

### Understanding the Score Function

```python
"""
SCORE-BASED MODELS (Song & Ermon, 2019)

Alternative view of diffusion:
- Score function: ∇_x log p(x)
- Points in direction of higher probability
- Diffusion models implicitly learn the score!

Connection:
  ε_θ(x_t, t) ≈ -σ_t * ∇_x log p(x_t)

Same objective, different interpretation!
"""


def score_function_intuition():
    """
    Explain score-based models.
    """
    print("SCORE-BASED MODELS\n")

    print("Score Function: ∇_x log p(x)")
    print("  → Gradient of log probability")
    print("  → Points toward higher density")
    print("  → Tells us how to move to more likely samples")
    print()

    print("Connection to Diffusion:")
    print("  Predicted noise: ε_θ(x_t, t)")
    print("  Score function: ∇_x log p(x_t)")
    print("  Relationship: ε_θ ≈ -σ_t * ∇_x log p(x_t)")
    print()

    print("Why this matters:")
    print("✅ Unifies diffusion and score-based models")
    print("✅ Theoretical foundation for why diffusion works")
    print("✅ Enables hybrid approaches (SDE solvers)")
    print()

    print("Langevin Dynamics:")
    print("  x_{t+1} = x_t + α * ∇_x log p(x_t) + √(2α) * ε")
    print("  → Sampling by following the score!")


score_function_intuition()
```

---

## 9. Conditional Generation

### Class-Conditional Diffusion

```python
"""
Conditional Diffusion: Generate specific classes

Training:
  ε_θ(x_t, t, y)  ← condition on class y

Sampling:
  Given class y, generate images of that class

Key: Condition the noise prediction on additional information!
"""


class ConditionalUNet(nn.Module):
    """
    U-Net with class conditioning.

    Predicts noise conditioned on timestep AND class.
    """
    def __init__(self, img_channels=1, num_classes=10, base_channels=64, time_emb_dim=256):
        super().__init__()

        # Class embedding
        self.class_embedding = nn.Embedding(num_classes, time_emb_dim)

        # Time embedding
        self.time_embedding = TimeEmbedding(time_emb_dim)

        # Combined embedding (time + class)
        self.combined_emb_dim = time_emb_dim * 2

        # U-Net (simplified - would use full architecture)
        self.down1 = ResidualBlock(img_channels, base_channels, self.combined_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, self.combined_emb_dim)

        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, self.combined_emb_dim)

        self.up2 = ResidualBlock(base_channels * 4, base_channels, self.combined_emb_dim)
        self.up1 = ResidualBlock(base_channels * 2, base_channels, self.combined_emb_dim)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.out_conv = nn.Conv2d(base_channels, img_channels, 1)

    def forward(self, x, t, y):
        """
        Args:
            x: Noisy images [batch, channels, H, W]
            t: Timesteps [batch]
            y: Class labels [batch]

        Returns:
            Predicted noise [batch, channels, H, W]
        """
        # Time embedding
        t_emb = self.time_embedding(t.float())

        # Class embedding
        y_emb = self.class_embedding(y)

        # Combine embeddings
        combined_emb = torch.cat([t_emb, y_emb], dim=1)

        # U-Net forward
        d1 = self.down1(x, combined_emb)
        d2 = self.down2(self.pool(d1), combined_emb)

        b = self.bottleneck(self.pool(d2), combined_emb)

        u2 = self.up2(torch.cat([self.upsample(b), d2], dim=1), combined_emb)
        u1 = self.up1(torch.cat([self.upsample(u2), d1], dim=1), combined_emb)

        out = self.out_conv(u1)

        return out


# Test conditional U-Net
conditional_unet = ConditionalUNet(num_classes=10)

x_test = torch.randn(4, 1, 28, 28)
t_test = torch.tensor([100, 200, 300, 400])
y_test = torch.tensor([0, 1, 2, 3])  # Classes

noise_pred = conditional_unet(x_test, t_test, y_test)

print(f"Input: {x_test.shape}")
print(f"Timesteps: {t_test}")
print(f"Classes: {y_test}")
print(f"Output: {noise_pred.shape}")
```

### Classifier-Free Guidance (Preview)

```python
"""
Classifier-Free Guidance

Key technique for high-quality conditional generation!

Training:
  - Randomly drop class label (unconditional ~10% of time)
  - Model learns both conditional and unconditional

Sampling:
  ε_guided = ε_uncond + w * (ε_cond - ε_uncond)

Where w > 1 strengthens conditioning

Result:
✅ Better sample quality
✅ Stronger adherence to condition
✅ Used in Stable Diffusion, DALL-E 2

We'll explore this deeply in Lesson 3!
"""

def classifier_free_guidance_preview():
    print("CLASSIFIER-FREE GUIDANCE\n")
    print("Formula: ε_guided = ε_uncond + w * (ε_cond - ε_uncond)")
    print()
    print("Guidance scale w:")
    print("  w = 1.0  → Standard conditional generation")
    print("  w = 7.5  → Stable Diffusion default (stronger)")
    print("  w = 15.0 → Very strong (may reduce diversity)")
    print()
    print("Coming in Lesson 3: Full implementation!")

classifier_free_guidance_preview()
```

---

## Practice Exercises

### Exercise 1: Implement Cosine Schedule DDPM

```python
"""
Modify DDPM to use cosine beta schedule.

Tasks:
1. Replace linear schedule with cosine
2. Train on MNIST
3. Compare sample quality with linear schedule

Hypothesis: Cosine should give better results!
"""

# Your implementation here
```

### Exercise 2: Accelerated Sampling

```python
"""
Implement and benchmark different sampling speeds.

Compare:
1. DDPM with 1000 steps
2. DDIM with 100 steps
3. DDIM with 50 steps

Measure:
- Sampling time
- Sample quality (visual inspection)
- FID score (if you have reference dataset)
"""

# Your implementation here
```

### Exercise 3: Conditional MNIST Generation

```python
"""
Train conditional diffusion model on MNIST.

Requirements:
1. Condition on digit class (0-9)
2. Generate specific digits on demand
3. Visualize all 10 classes

Bonus: Implement classifier-free guidance!
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Diffusion Process** 🌊
   - Forward: Gradually add Gaussian noise
   - Reverse: Learn to denoise step-by-step
   - Training objective: Predict added noise
   - Simple, stable, effective!

2. **DDPM Architecture** 🏗️
   - U-Net with time conditioning
   - Residual blocks for stability
   - Skip connections preserve details
   - Sinusoidal time embeddings

3. **Why Diffusion Beats GANs** 🏆
   - More stable training (no adversarial dynamics)
   - Better mode coverage (no collapse)
   - Scalable to high resolution
   - Controllable generation
   - State-of-the-art quality!

4. **DDIM: Speed Breakthrough** ⚡
   - Same training as DDPM
   - Deterministic sampling process
   - 10-20x faster generation
   - Skip timesteps without quality loss
   - Enables real-time applications

5. **Noise Schedules Matter** 📈
   - Linear: Original DDPM
   - Cosine: More uniform SNR (better!)
   - Schedule affects quality significantly
   - Cosine used in modern models

6. **Conditional Generation** 🎯
   - Condition on class, text, or images
   - Classifier-free guidance for quality
   - Foundation for Stable Diffusion
   - Enables controllable creativity

### Diffusion vs Other Generative Models

| Aspect | Diffusion | GAN | VAE |
|--------|-----------|-----|-----|
| **Training** | Stable | Unstable | Stable |
| **Quality** | Excellent | Excellent | Good |
| **Diversity** | Excellent | Mode collapse risk | Good |
| **Speed** | Slow (DDIM helps) | Fast | Fast |
| **Control** | Easy | Hard | Medium |

### What's Next?

In Lesson 3, we'll dive into **Stable Diffusion**:
- Latent Diffusion Models (why compress first)
- Text conditioning with CLIP
- Classifier-Free Guidance in depth
- Production-ready pipelines with Hugging Face
- Real image generation!

---

## Additional Resources

### Papers

**Core Papers:**
- Ho et al. (2020): "Denoising Diffusion Probabilistic Models" (DDPM)
- Song et al. (2021): "Denoising Diffusion Implicit Models" (DDIM)
- Nichol & Dhariwal (2021): "Improved Denoising Diffusion Probabilistic Models"
- Song & Ermon (2019): "Generative Modeling by Estimating Gradients" (Score-based)

**Advanced:**
- Dhariwal & Nichol (2021): "Diffusion Models Beat GANs on Image Synthesis"
- Ho & Salimans (2021): "Classifier-Free Diffusion Guidance"

### Libraries

- **PyTorch**: Foundation for implementation
- **Hugging Face Diffusers**: Production diffusion models (Lesson 3!)
- **diffusers**: https://github.com/huggingface/diffusers

### Tutorials

- **Annotated Diffusion**: https://huggingface.co/blog/annotated-diffusion
- **Lilian Weng's Blog**: https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
- **Assembly AI Tutorial**: https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/

---

**Next**: [Lesson 3 - Stable Diffusion Deep Dive](Lesson%203%20-%20Stable%20Diffusion%20Deep%20Dive.md)

Master production-ready text-to-image generation with **Stable Diffusion**! 🎨

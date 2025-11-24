# Lesson 1: GANs and VAEs - Foundations of Image Generation 🎨

**Module 14: Generative Vision Models | Lesson 1 of 6**

Master the foundational architectures that revolutionized image generation!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the adversarial training dynamics of GANs
2. ✅ Implement Generator and Discriminator networks from scratch
3. ✅ Master VAE architecture with reparameterization trick
4. ✅ Apply GANs and VAEs to face generation and style transfer
5. ✅ Understand mode collapse and how to mitigate it
6. ✅ Compare GANs vs VAEs for different use cases

---

## Prerequisites

- **Module 6**: Computer Vision fundamentals (CNNs, image processing)
- **Module 3**: Deep Learning basics (backpropagation, optimizers)
- **Python Libraries**: PyTorch, torchvision, matplotlib, numpy
- **Math**: Basic probability, KL divergence, Jensen-Shannon divergence

---

## 1. Generative Adversarial Networks (GANs) - The Game Theory Approach

### What is a GAN?

GANs (Goodfellow et al., 2014) frame image generation as a **two-player game**:
- **Generator (G)**: Creates fake images to fool the discriminator
- **Discriminator (D)**: Distinguishes real images from fake ones

As they compete, both improve until G generates realistic images!

### The GAN Objective Function

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

"""
GAN Objective (Minimax Game):

min_G max_D V(D, G) = E_x[log D(x)] + E_z[log(1 - D(G(z)))]

Where:
- D(x) = probability that x is real (not generated)
- G(z) = generated image from noise z
- E_x = expectation over real data
- E_z = expectation over noise distribution

Training alternates:
1. Train D: Maximize V (distinguish real from fake)
2. Train G: Minimize V (fool discriminator)
"""

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("GAN Training Process:")
print("1. D tries to maximize: log D(real) + log(1 - D(fake))")
print("2. G tries to minimize: log(1 - D(fake))")
print("3. At equilibrium: D(real) = D(fake) = 0.5")
```

**Key Insight**: GANs don't need to model explicit probability distributions - they learn implicitly through adversarial training!

---

## 2. Building a GAN from Scratch

### Generator Network

```python
class Generator(nn.Module):
    """
    Generator: Maps random noise to images.

    Architecture: FC layers → Reshape → Transposed Convs → Image
    """
    def __init__(self, latent_dim=100, img_channels=1, img_size=28):
        super(Generator, self).__init__()

        self.latent_dim = latent_dim
        self.img_channels = img_channels
        self.img_size = img_size

        # Initial size after first linear layer
        self.init_size = img_size // 4

        # Linear layer: noise → feature maps
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        # Transposed convolutions: upsample to final image
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),

            # 7x7 → 14x14
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # 14x14 → 28x28
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Final conv: 64 channels → img_channels
            nn.Conv2d(64, img_channels, 3, stride=1, padding=1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, z):
        """
        Args:
            z: Random noise [batch_size, latent_dim]

        Returns:
            Generated images [batch_size, channels, H, W]
        """
        # Linear layer
        out = self.fc(z)

        # Reshape to 2D feature maps
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)

        # Transposed convolutions
        img = self.conv_blocks(out)

        return img


# Test Generator
latent_dim = 100
generator = Generator(latent_dim=latent_dim, img_channels=1, img_size=28)

# Generate sample
z = torch.randn(4, latent_dim)
fake_imgs = generator(z)

print(f"Input noise shape: {z.shape}")
print(f"Generated image shape: {fake_imgs.shape}")
print(f"Generated image range: [{fake_imgs.min():.2f}, {fake_imgs.max():.2f}]")
```

### Discriminator Network

```python
class Discriminator(nn.Module):
    """
    Discriminator: Binary classifier (real vs fake).

    Architecture: Image → Convs → FC → Sigmoid
    """
    def __init__(self, img_channels=1, img_size=28):
        super(Discriminator, self).__init__()

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(img_channels, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 14x14 → 7x7
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 7x7 → 3x3
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            # 3x3 → 1x1
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        # Classifier head
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(
            nn.Linear(128 * ds_size ** 2, 1),
            nn.Sigmoid()  # Probability of being real
        )

    def forward(self, img):
        """
        Args:
            img: Images [batch_size, channels, H, W]

        Returns:
            Probability of being real [batch_size, 1]
        """
        # Extract features
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)

        # Classify
        validity = self.adv_layer(out)

        return validity


# Test Discriminator
discriminator = Discriminator(img_channels=1, img_size=28)

# Classify samples
real_prob = discriminator(fake_imgs)

print(f"Discriminator output shape: {real_prob.shape}")
print(f"Probabilities (should be ~0.5 initially): {real_prob.squeeze()}")
```

---

## 3. Training a GAN

### Complete Training Loop

```python
def train_gan(generator, discriminator, dataloader, num_epochs=50,
              latent_dim=100, device='cpu'):
    """
    Train GAN with alternating updates.

    Args:
        generator: Generator network
        discriminator: Discriminator network
        dataloader: DataLoader with real images
        num_epochs: Number of training epochs
        latent_dim: Dimension of noise vector
        device: 'cuda' or 'cpu'
    """
    # Loss function
    adversarial_loss = nn.BCELoss()

    # Optimizers
    optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Move to device
    generator.to(device)
    discriminator.to(device)
    adversarial_loss.to(device)

    # Training history
    g_losses = []
    d_losses = []

    print("Starting GAN training...")

    for epoch in range(num_epochs):
        epoch_g_loss = 0
        epoch_d_loss = 0

        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)

            # Move to device
            real_imgs = real_imgs.to(device)

            # Labels
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Loss on real images
            real_pred = discriminator(real_imgs)
            d_loss_real = adversarial_loss(real_pred, real_labels)

            # Generate fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_imgs = generator(z)

            # Loss on fake images
            fake_pred = discriminator(fake_imgs.detach())
            d_loss_fake = adversarial_loss(fake_pred, fake_labels)

            # Total discriminator loss
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake images
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)

            # Generator wants discriminator to think fakes are real
            gen_pred = discriminator(gen_imgs)
            g_loss = adversarial_loss(gen_pred, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # Track losses
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()

        # Average losses
        avg_g_loss = epoch_g_loss / len(dataloader)
        avg_d_loss = epoch_d_loss / len(dataloader)

        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)

        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | "
                  f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

    return g_losses, d_losses


# Example: Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Note: Set download=True first time
mnist_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transform, download=False
)

dataloader = torch.utils.data.DataLoader(
    mnist_dataset, batch_size=64, shuffle=True
)

print(f"Dataset size: {len(mnist_dataset)}")
print(f"Number of batches: {len(dataloader)}")
```

### Visualizing Training Progress

```python
def visualize_gan_training(generator, latent_dim=100, device='cpu', n_samples=16):
    """Visualize generated samples during training."""
    generator.eval()

    with torch.no_grad():
        # Generate samples
        z = torch.randn(n_samples, latent_dim, device=device)
        gen_imgs = generator(z).cpu()

        # Denormalize from [-1, 1] to [0, 1]
        gen_imgs = (gen_imgs + 1) / 2

        # Plot
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))

        for i, ax in enumerate(axes.flat):
            ax.imshow(gen_imgs[i].squeeze(), cmap='gray')
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('gan_samples.png', dpi=150, bbox_inches='tight')
        plt.show()

    generator.train()


def plot_gan_losses(g_losses, d_losses):
    """Plot GAN training losses."""
    plt.figure(figsize=(10, 5))

    plt.plot(g_losses, label='Generator Loss', alpha=0.7)
    plt.plot(d_losses, label='Discriminator Loss', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('GAN Training Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig('gan_losses.png', dpi=150, bbox_inches='tight')
    plt.show()


# Demonstrate visualization (without full training)
print("Visualization functions ready!")
print("After training, call:")
print("  visualize_gan_training(generator)")
print("  plot_gan_losses(g_losses, d_losses)")
```

---

## 4. Mode Collapse and Solutions

### Understanding Mode Collapse

```python
"""
MODE COLLAPSE: The major failure mode of GANs

Problem: Generator produces limited variety of outputs
- Collapses to a few "safe" samples that fool discriminator
- Loses diversity of training data

Example:
- Training on digits 0-9
- Generator only produces 1s and 7s (ignores other digits)

Why it happens:
- Generator finds local optimum that fools current discriminator
- Discriminator adapts, but generator gets stuck
- Oscillation instead of convergence
"""

import torch.nn.functional as F


def check_mode_collapse(generator, latent_dim=100, n_samples=1000):
    """
    Simple heuristic to detect mode collapse.

    Generates many samples and checks diversity.
    """
    generator.eval()

    with torch.no_grad():
        # Generate samples
        z = torch.randn(n_samples, latent_dim)
        samples = generator(z)

        # Compute pairwise distances
        samples_flat = samples.view(n_samples, -1)

        # Random subset for efficiency
        subset_size = min(100, n_samples)
        indices = torch.randperm(n_samples)[:subset_size]
        subset = samples_flat[indices]

        # L2 distances
        distances = torch.cdist(subset, subset)

        # Average distance (higher = more diverse)
        avg_distance = distances.mean().item()

        # Std of distances (lower = potential collapse)
        std_distance = distances.std().item()

        print(f"Average pairwise distance: {avg_distance:.4f}")
        print(f"Std of distances: {std_distance:.4f}")

        if avg_distance < 0.1:
            print("⚠️  WARNING: Potential mode collapse detected!")
        else:
            print("✅ Diversity looks good")

    generator.train()


# Test on random generator
check_mode_collapse(generator, latent_dim=100, n_samples=100)
```

### Solutions to Mode Collapse

```python
"""
Techniques to prevent mode collapse:

1. Feature Matching (Salimans et al., 2016)
   - G matches statistics of intermediate D features
   - Instead of fooling D, match feature distributions

2. Minibatch Discrimination
   - D looks at batch statistics, not just individual samples
   - Encourages diversity within batches

3. Unrolled GAN
   - G optimizes against future D (not current)
   - Look ahead in discriminator updates

4. Wasserstein GAN (WGAN)
   - Better loss function (Earth Mover's Distance)
   - More stable training
"""


class WassersteinLoss(nn.Module):
    """
    Wasserstein Loss for WGAN.

    Key idea: Remove sigmoid from D, use linear output.
    D tries to maximize: E[D(real)] - E[D(fake)]
    G tries to minimize: -E[D(fake)]
    """
    def __init__(self):
        super().__init__()

    def forward(self, d_real, d_fake, is_discriminator=True):
        """
        Args:
            d_real: D output on real images
            d_fake: D output on fake images
            is_discriminator: True for D training, False for G
        """
        if is_discriminator:
            # Discriminator: maximize D(real) - D(fake)
            return -(d_real.mean() - d_fake.mean())
        else:
            # Generator: maximize D(fake)
            return -d_fake.mean()


# Example usage
wgan_loss = WassersteinLoss()

# Dummy predictions
d_real_pred = torch.randn(32, 1)
d_fake_pred = torch.randn(32, 1)

d_loss = wgan_loss(d_real_pred, d_fake_pred, is_discriminator=True)
g_loss = wgan_loss(d_real_pred, d_fake_pred, is_discriminator=False)

print(f"WGAN Discriminator Loss: {d_loss.item():.4f}")
print(f"WGAN Generator Loss: {g_loss.item():.4f}")
```

---

## 5. StyleGAN Architecture

### Progressive Growing and Style-Based Generation

```python
"""
StyleGAN (Karras et al., 2019) - State-of-the-art GAN architecture

Key innovations:
1. Progressive Growing: Start at low resolution, gradually increase
2. Style-Based Generator: Inject style at each resolution
3. Adaptive Instance Normalization (AdaIN)
4. Mixing Regularization: Use different styles at different layers

Result: Photorealistic faces at high resolution!
"""


class AdaptiveInstanceNorm(nn.Module):
    """
    AdaIN: Transfer style statistics to content.

    AdaIN(content, style) = σ(style) * normalize(content) + μ(style)
    """
    def __init__(self, num_features, style_dim):
        super().__init__()

        # Learn affine parameters from style
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_bias = nn.Linear(style_dim, num_features)

    def forward(self, content, style):
        """
        Args:
            content: [batch, channels, H, W]
            style: [batch, style_dim]
        """
        # Normalize content
        normalized = F.instance_norm(content)

        # Get style statistics
        scale = self.style_scale(style).unsqueeze(2).unsqueeze(3)
        bias = self.style_bias(style).unsqueeze(2).unsqueeze(3)

        # Apply style
        return scale * normalized + bias


class StyleBlock(nn.Module):
    """
    StyleGAN synthesis block.

    Flow: Upsample → Conv → AdaIN → Noise → Activation
    """
    def __init__(self, in_channels, out_channels, style_dim):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.adain = AdaptiveInstanceNorm(out_channels, style_dim)

        # Learnable noise injection
        self.noise_weight = nn.Parameter(torch.zeros(1))

        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x, style):
        """
        Args:
            x: Feature maps [batch, in_channels, H, W]
            style: Style vector [batch, style_dim]
        """
        # Upsample and convolve
        out = self.upsample(x)
        out = self.conv(out)

        # Apply style
        out = self.adain(out, style)

        # Add noise
        noise = torch.randn(out.size(0), 1, out.size(2), out.size(3),
                           device=out.device)
        out = out + self.noise_weight * noise

        # Activate
        out = self.activation(out)

        return out


# Test StyleBlock
style_dim = 512
style_block = StyleBlock(in_channels=128, out_channels=64, style_dim=style_dim)

# Input
x = torch.randn(2, 128, 8, 8)
style = torch.randn(2, style_dim)

# Forward
out = style_block(x, style)

print(f"Input shape: {x.shape}")
print(f"Style shape: {style.shape}")
print(f"Output shape: {out.shape}")
```

### Style Mixing

```python
def style_mixing_demo(generator, latent_dim=512, mixing_layer=4):
    """
    Demonstrate style mixing in StyleGAN.

    Use style A for coarse features (early layers)
    Use style B for fine features (late layers)
    """
    generator.eval()

    with torch.no_grad():
        # Two random styles
        style_A = torch.randn(1, latent_dim)
        style_B = torch.randn(1, latent_dim)

        print("Style Mixing:")
        print(f"  Layers 0-{mixing_layer}: Style A (coarse features)")
        print(f"  Layers {mixing_layer+1}+: Style B (fine details)")
        print()
        print("This allows:")
        print("  - Face shape from Style A")
        print("  - Hair color/texture from Style B")
        print("  - Disentangled control!")

    generator.train()


style_mixing_demo(generator, latent_dim=100, mixing_layer=3)
```

---

## 6. Variational Autoencoders (VAEs)

### The Probabilistic Perspective

```python
"""
VAE (Kingma & Welling, 2014) - Probabilistic generative model

Key idea: Learn latent distribution p(z|x)
- Encoder: q(z|x) ≈ p(z|x) (approximate posterior)
- Decoder: p(x|z) (likelihood)

Training objective (ELBO):
  L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
       ↑                  ↑
  Reconstruction      Regularization

Advantages over GAN:
✅ Stable training (no adversarial dynamics)
✅ Principled probabilistic framework
✅ Can encode AND generate

Disadvantages:
❌ Blurrier outputs than GANs
❌ Harder to scale to high resolution
"""


class VAEEncoder(nn.Module):
    """
    VAE Encoder: Maps images to latent distribution.

    Outputs: mean and log-variance of q(z|x)
    """
    def __init__(self, img_channels=1, img_size=28, latent_dim=20):
        super().__init__()

        self.latent_dim = latent_dim

        # Convolutional feature extractor
        self.conv_layers = nn.Sequential(
            # 28x28 → 14x14
            nn.Conv2d(img_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),

            # 14x14 → 7x7
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            # 7x7 → 4x4
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Compute flattened size
        self.flatten_size = 128 * 4 * 4

        # Latent distribution parameters
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

    def forward(self, x):
        """
        Args:
            x: Images [batch, channels, H, W]

        Returns:
            mu: Mean of q(z|x) [batch, latent_dim]
            logvar: Log-variance of q(z|x) [batch, latent_dim]
        """
        # Extract features
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)

        # Latent parameters
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


class VAEDecoder(nn.Module):
    """
    VAE Decoder: Maps latent codes to images.

    Samples: p(x|z)
    """
    def __init__(self, latent_dim=20, img_channels=1, img_size=28):
        super().__init__()

        self.latent_dim = latent_dim
        self.img_size = img_size

        # Initial size
        self.init_size = img_size // 8

        # FC layer
        self.fc = nn.Linear(latent_dim, 128 * self.init_size ** 2)

        # Transposed convolutions
        self.conv_layers = nn.Sequential(
            # 4x4 → 7x7
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            # 7x7 → 14x14
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),

            # 14x14 → 28x28
            nn.ConvTranspose2d(32, img_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, z):
        """
        Args:
            z: Latent codes [batch, latent_dim]

        Returns:
            Reconstructed images [batch, channels, H, W]
        """
        # FC layer
        h = self.fc(z)
        h = h.view(h.size(0), 128, self.init_size, self.init_size)

        # Decode to image
        x_recon = self.conv_layers(h)

        return x_recon


# Test VAE components
vae_encoder = VAEEncoder(latent_dim=20)
vae_decoder = VAEDecoder(latent_dim=20)

# Encode
x = torch.randn(4, 1, 28, 28)
mu, logvar = vae_encoder(x)

print(f"Input shape: {x.shape}")
print(f"Latent mean shape: {mu.shape}")
print(f"Latent logvar shape: {logvar.shape}")

# Decode
z = torch.randn(4, 20)
x_recon = vae_decoder(z)

print(f"Latent code shape: {z.shape}")
print(f"Reconstructed shape: {x_recon.shape}")
```

---

## 7. Reparameterization Trick

### Making VAEs Differentiable

```python
"""
THE REPARAMETERIZATION TRICK

Problem: Sampling is not differentiable!
  z ~ N(μ, σ²) ← Can't backpropagate through random sampling

Solution: Reparameterize with deterministic function
  z = μ + σ ⊙ ε, where ε ~ N(0, I)

Now gradients flow through μ and σ!
"""


def reparameterize(mu, logvar):
    """
    Reparameterization trick for VAE.

    Args:
        mu: Mean of q(z|x) [batch, latent_dim]
        logvar: Log-variance of q(z|x) [batch, latent_dim]

    Returns:
        z: Sampled latent code [batch, latent_dim]
    """
    # Standard deviation
    std = torch.exp(0.5 * logvar)

    # Sample epsilon from standard normal
    eps = torch.randn_like(std)

    # Reparameterize
    z = mu + eps * std

    return z


# Example
mu = torch.tensor([[0.0, 1.0, -0.5]])
logvar = torch.tensor([[-1.0, 0.0, 0.5]])

z = reparameterize(mu, logvar)

print("Reparameterization Trick:")
print(f"  Mean: {mu}")
print(f"  Log-variance: {logvar}")
print(f"  Sampled z: {z}")
print()
print("This allows backpropagation through the sampling process!")
```

### Complete VAE Implementation

```python
class VAE(nn.Module):
    """Complete VAE with encoder and decoder."""

    def __init__(self, img_channels=1, img_size=28, latent_dim=20):
        super().__init__()

        self.encoder = VAEEncoder(img_channels, img_size, latent_dim)
        self.decoder = VAEDecoder(latent_dim, img_channels, img_size)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Forward pass: encode, sample, decode.

        Returns:
            x_recon: Reconstructed image
            mu: Latent mean
            logvar: Latent log-variance
        """
        # Encode
        mu, logvar = self.encoder(x)

        # Sample
        z = self.reparameterize(mu, logvar)

        # Decode
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

    def sample(self, num_samples, device='cpu'):
        """Generate new samples from prior."""
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.decoder.latent_dim, device=device)

            # Decode
            samples = self.decoder(z)

            return samples


# Test VAE
vae = VAE(latent_dim=20)

x = torch.randn(4, 1, 28, 28)
x_recon, mu, logvar = vae(x)

print(f"Original: {x.shape}")
print(f"Reconstructed: {x_recon.shape}")
print(f"Latent mean: {mu.shape}")
print(f"Latent logvar: {logvar.shape}")

# Generate samples
samples = vae.sample(num_samples=8)
print(f"Generated samples: {samples.shape}")
```

---

## 8. VAE Loss Function

### ELBO and KL Divergence

```python
def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction + KL divergence.

    Args:
        x_recon: Reconstructed images
        x: Original images
        mu: Latent mean
        logvar: Latent log-variance
        beta: Weight for KL term (β-VAE)

    Returns:
        total_loss: Combined loss
        recon_loss: Reconstruction loss
        kl_loss: KL divergence
    """
    # Reconstruction loss (binary cross-entropy)
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')

    # KL divergence: KL(q(z|x) || p(z))
    # For N(μ, σ²) vs N(0, I):
    # KL = 0.5 * Σ(1 + log(σ²) - μ² - σ²)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # Total loss (ELBO)
    total_loss = recon_loss + beta * kl_loss

    return total_loss, recon_loss, kl_loss


# Example
x = torch.rand(4, 1, 28, 28)
x_recon, mu, logvar = vae(x)

total_loss, recon_loss, kl_loss = vae_loss(x_recon, x, mu, logvar)

print(f"Total Loss: {total_loss.item():.2f}")
print(f"Reconstruction Loss: {recon_loss.item():.2f}")
print(f"KL Divergence: {kl_loss.item():.2f}")
```

### Training VAE

```python
def train_vae(vae, dataloader, num_epochs=10, beta=1.0, device='cpu'):
    """
    Train VAE on image dataset.

    Args:
        vae: VAE model
        dataloader: DataLoader with images
        num_epochs: Number of training epochs
        beta: Weight for KL term
        device: 'cuda' or 'cpu'
    """
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    train_losses = []

    print("Starting VAE training...")

    for epoch in range(num_epochs):
        vae.train()
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Forward
            x_recon, mu, logvar = vae(x)

            # Loss
            loss, recon, kl = vae_loss(x_recon, x, mu, logvar, beta=beta)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track
            epoch_loss += loss.item()
            epoch_recon += recon.item()
            epoch_kl += kl.item()

        # Average losses
        avg_loss = epoch_loss / len(dataloader.dataset)
        avg_recon = epoch_recon / len(dataloader.dataset)
        avg_kl = epoch_kl / len(dataloader.dataset)

        train_losses.append(avg_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    return train_losses


# Demo (would run with actual dataloader)
print("VAE training function ready!")
print("Usage: train_vae(vae, dataloader, num_epochs=10, beta=1.0)")
```

---

## 9. β-VAE for Disentanglement

### Learning Disentangled Representations

```python
"""
β-VAE (Higgins et al., 2017)

Modify VAE loss:
  L = Reconstruction + β * KL

Where β > 1 (typically 2-10)

Effect:
- Encourages latent dimensions to be independent
- Each z_i controls a different factor of variation
- Example: z_1 = pose, z_2 = lighting, z_3 = identity

Trade-off:
✅ More disentangled (better for editing)
❌ Worse reconstruction (higher β = blurrier)
"""


def compare_beta_values():
    """Compare different β values for VAE."""

    betas = [0.5, 1.0, 4.0, 10.0]

    print("β-VAE: Effect of different β values\n")

    for beta in betas:
        print(f"β = {beta}")

        if beta < 1.0:
            print("  → Focus on reconstruction")
            print("  → Sharp images, entangled latents")
        elif beta == 1.0:
            print("  → Standard VAE")
            print("  → Balanced reconstruction and regularization")
        else:
            print("  → Focus on disentanglement")
            print("  → Blurrier images, independent latents")

        print()


compare_beta_values()
```

### Visualizing Latent Traversals

```python
def latent_traversal(vae, img, latent_dim_idx=0, num_steps=10,
                     range_val=3.0, device='cpu'):
    """
    Traverse a single latent dimension.

    Shows what each latent dimension controls.
    """
    vae.eval()
    vae.to(device)
    img = img.to(device)

    with torch.no_grad():
        # Encode image
        mu, logvar = vae.encoder(img)

        # Create traversal values
        traversal_values = torch.linspace(-range_val, range_val, num_steps)

        # Generate images
        traversal_imgs = []

        for val in traversal_values:
            # Modify specific latent dimension
            z = mu.clone()
            z[:, latent_dim_idx] = val

            # Decode
            img_recon = vae.decoder(z)
            traversal_imgs.append(img_recon)

        # Stack
        traversal_imgs = torch.cat(traversal_imgs, dim=0)

    # Plot
    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps*2, 2))

    for i, ax in enumerate(axes):
        ax.imshow(traversal_imgs[i].cpu().squeeze(), cmap='gray')
        ax.set_title(f'z[{latent_dim_idx}]={traversal_values[i]:.1f}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'latent_traversal_dim{latent_dim_idx}.png', dpi=150, bbox_inches='tight')
    plt.show()

    vae.train()


print("Latent traversal function ready!")
print("Usage: latent_traversal(vae, img, latent_dim_idx=0)")
```

---

## 10. Applications: Face Generation

### Generating Faces with GANs

```python
"""
Face Generation Pipeline:

1. Dataset: CelebA, FFHQ (high-quality faces)
2. Preprocessing: Align faces, crop, resize
3. Architecture: DCGAN, StyleGAN, or StyleGAN2
4. Training: 100k-1M iterations
5. Sampling: Generate new faces from noise

Results:
- Realistic faces that don't exist
- Control attributes (age, gender, expression)
- Style mixing (combine features)
"""


def generate_faces_demo(generator, num_faces=16, latent_dim=512):
    """
    Generate diverse faces.

    In practice, would use pre-trained StyleGAN.
    """
    generator.eval()

    with torch.no_grad():
        # Sample random latents
        z = torch.randn(num_faces, latent_dim)

        # Generate faces
        faces = generator(z)

        # Denormalize
        faces = (faces + 1) / 2

        # Plot
        fig, axes = plt.subplots(4, 4, figsize=(12, 12))

        for i, ax in enumerate(axes.flat):
            if i < num_faces:
                # Assuming RGB images
                img = faces[i].permute(1, 2, 0).cpu().numpy()
                ax.imshow(img)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('generated_faces.png', dpi=150, bbox_inches='tight')
        plt.show()

    generator.train()


print("Face generation demo ready!")
print("In practice, use pre-trained StyleGAN2 from:")
print("  https://github.com/NVlabs/stylegan2-ada-pytorch")
```

---

## 11. Applications: Style Transfer

### Neural Style Transfer with VAE

```python
"""
Style Transfer with Generative Models:

1. Traditional NST (Gatys et al.): Optimization-based
2. VAE-based: Learn style and content separately
3. GAN-based: CycleGAN, StyleGAN

Approach: Encode→ Manipulate latent → Decode
"""


def vae_style_transfer(vae, content_img, style_latent, alpha=0.5):
    """
    Blend content and style using VAE latent space.

    Args:
        vae: Trained VAE
        content_img: Image to preserve content
        style_latent: Latent vector with target style
        alpha: Blending factor (0=content, 1=style)
    """
    vae.eval()

    with torch.no_grad():
        # Encode content
        content_mu, _ = vae.encoder(content_img)

        # Interpolate in latent space
        mixed_latent = (1 - alpha) * content_mu + alpha * style_latent

        # Decode
        stylized_img = vae.decoder(mixed_latent)

    vae.train()

    return stylized_img


# Example
content = torch.rand(1, 1, 28, 28)
style_z = torch.randn(1, 20)

stylized = vae_style_transfer(vae, content, style_z, alpha=0.7)

print(f"Content image: {content.shape}")
print(f"Style latent: {style_z.shape}")
print(f"Stylized result: {stylized.shape}")
```

---

## 12. Data Augmentation with Generative Models

### Synthetic Data Generation

```python
"""
Using generative models for data augmentation:

Benefits:
✅ Generate unlimited training data
✅ Balance imbalanced datasets
✅ Privacy-preserving (generate synthetic patients)
✅ Rare event simulation

Challenges:
❌ Distribution mismatch
❌ Model bias amplification
❌ Quality control
"""


def augment_dataset_with_gan(generator, original_dataset,
                              num_synthetic=1000, latent_dim=100):
    """
    Augment dataset with GAN-generated samples.

    Args:
        generator: Trained GAN generator
        original_dataset: Original training data
        num_synthetic: Number of synthetic samples to generate
        latent_dim: Latent dimension

    Returns:
        augmented_dataset: Combined original + synthetic
    """
    generator.eval()

    synthetic_images = []

    with torch.no_grad():
        # Generate in batches
        batch_size = 100
        num_batches = num_synthetic // batch_size

        for _ in range(num_batches):
            z = torch.randn(batch_size, latent_dim)
            fake_imgs = generator(z)
            synthetic_images.append(fake_imgs)

    # Concatenate
    synthetic_images = torch.cat(synthetic_images, dim=0)

    print(f"Generated {len(synthetic_images)} synthetic samples")
    print(f"Original dataset size: {len(original_dataset)}")
    print(f"Augmented dataset size: {len(original_dataset) + len(synthetic_images)}")

    generator.train()

    # In practice, would combine into new dataset
    return synthetic_images


print("Data augmentation function ready!")
```

---

## Practice Exercises

### Exercise 1: Implement Conditional GAN

```python
"""
Extend basic GAN to conditional GAN (cGAN).

Task:
- Add class labels to both G and D
- Generator: G(z, y) where y is class label
- Discriminator: D(x, y)
- Train on MNIST with digit labels

Goal: Generate specific digits on demand!
"""

# Your implementation here
```

### Exercise 2: VAE Interpolation

```python
"""
Implement latent space interpolation.

Task:
1. Encode two images to get z1 and z2
2. Interpolate: z_t = (1-t)*z1 + t*z2, t ∈ [0, 1]
3. Decode interpolated latents
4. Visualize smooth transition

Bonus: Try spherical interpolation (SLERP)
"""

# Your implementation here
```

### Exercise 3: Mode Collapse Detection

```python
"""
Create a robust mode collapse detector.

Implement:
1. Inception Score (IS) - measures diversity
2. Track number of unique clusters in generated samples
3. Alert when diversity drops below threshold

Test on GAN with limited training to induce collapse.
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **GANs: Adversarial Training** ⚔️
   - Two networks compete: Generator vs Discriminator
   - Implicit density modeling (no explicit likelihood)
   - Can generate sharp, realistic images
   - Training unstable, prone to mode collapse
   - StyleGAN achieves photorealistic quality

2. **VAEs: Probabilistic Framework** 📊
   - Learn latent distribution explicitly
   - Reparameterization trick enables backprop
   - Stable training, principled approach
   - Outputs blurrier than GANs
   - Better for representation learning

3. **Mode Collapse** 🚨
   - Generator produces limited variety
   - Major failure mode of GANs
   - Solutions: WGAN, feature matching, minibatch discrimination
   - Monitor diversity during training

4. **β-VAE** 🎯
   - Disentangle latent factors
   - β > 1 encourages independence
   - Trade-off: disentanglement vs reconstruction
   - Useful for interpretable editing

5. **Architecture Innovations** 🏗️
   - DCGAN: Convolutional architecture for stability
   - StyleGAN: Progressive growing + adaptive normalization
   - Latent space structure matters for generation quality

6. **Applications** 🎨
   - Face generation: PhotoStyle, realistic avatars
   - Data augmentation: Synthetic training samples
   - Style transfer: Artistic image manipulation
   - Anomaly detection: VAE reconstruction error

### GAN vs VAE Comparison

| Aspect | GAN | VAE |
|--------|-----|-----|
| **Training** | Adversarial (unstable) | Direct optimization (stable) |
| **Output Quality** | Sharp, realistic | Blurry |
| **Density Modeling** | Implicit | Explicit |
| **Encoding** | No (generation only) | Yes (bidirectional) |
| **Best For** | High-quality synthesis | Representation learning |

### What's Next?

In Lesson 2, we'll explore **Diffusion Models**:
- The new state-of-the-art for image generation
- Forward and reverse diffusion processes
- DDPM and DDIM architectures
- Why diffusion beats GANs for quality and stability

---

## Additional Resources

### Papers

**GANs:**
- Goodfellow et al. (2014): "Generative Adversarial Networks" (original paper)
- Radford et al. (2016): "Unsupervised Representation Learning with DCGANs"
- Karras et al. (2019): "A Style-Based Generator Architecture for GANs" (StyleGAN)
- Arjovsky et al. (2017): "Wasserstein GAN"

**VAEs:**
- Kingma & Welling (2014): "Auto-Encoding Variational Bayes" (original paper)
- Higgins et al. (2017): "β-VAE: Learning Basic Visual Concepts"
- Rezende et al. (2014): "Stochastic Backpropagation"

### Libraries

- **PyTorch**: `torch.nn`, `torchvision` for implementation
- **TensorFlow/Keras**: Alternative GAN/VAE implementations
- **NVIDIA StyleGAN**: https://github.com/NVlabs/stylegan2-ada-pytorch
- **Hugging Face Diffusers**: Will use in Lesson 2-3

### Interactive Resources

- **GAN Lab**: https://poloclub.github.io/ganlab/ (visualize GAN training)
- **VAE Explainer**: https://adamcobb.github.io/journal/papers.html
- **StyleGAN Demo**: https://www.thispers<ondoesnotexist.com/

---

**Next**: [Lesson 2 - Diffusion Models Fundamentals](Lesson%202%20-%20Diffusion%20Models%20Fundamentals.md)

Master the new state-of-the-art: **Denoising Diffusion Models**! 🌟

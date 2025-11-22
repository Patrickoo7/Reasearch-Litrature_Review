# Modern Neural Network Architectures 🚀

State-of-the-art architectures that power current AI systems. These are what you see in production and research today!

## Table of Contents
1. [Transformers](#transformers) - The revolution
2. [Vision Transformers](#vision-transformers) - Transformers for images
3. [Large Language Models](#large-language-models) - GPT, BERT, T5
4. [ResNet & Skip Connections](#resnet--skip-connections) - Deep CNNs
5. [U-Net](#u-net) - Image segmentation
6. [Autoencoders & VAEs](#autoencoders--vaes) - Unsupervised learning

---

## Transformers 🤖

**The most important architecture since 2017!** Powers GPT, BERT, ChatGPT, and more.

### Key Paper
"Attention Is All You Need" (Vaswani et al., 2017)

### Core Innovation: Attention Mechanism

**Problem with RNNs:**
- Sequential processing (slow)
- Long-term dependencies fade
- Can't parallelize

**Transformer Solution:**
- Parallel processing
- Direct connections between all positions
- Attention mechanism to focus on relevant parts

### Self-Attention Mechanism 🎯

**Intuition:** When processing a word, look at all other words to understand context.

**Example:**
```
Sentence: "The animal didn't cross the street because it was too tired"

When processing "it":
- Attention to "animal": HIGH (it = animal)
- Attention to "street": LOW
- Attention to "tired": MEDIUM (explains why)
```

### Mathematics of Attention

**Three components:**
1. **Query (Q):** What I'm looking for
2. **Key (K):** What I have to offer
3. **Value (V):** The actual information

**Computation:**
```python
# For each position in sequence
Q = X @ W_Q  # Query: what to look for
K = X @ W_K  # Key: what I have
V = X @ W_V  # Value: the content

# Attention scores
scores = Q @ K.T / √d_k  # Scaled dot product
attention_weights = softmax(scores)

# Weighted sum of values
output = attention_weights @ V
```

**Visual Example:**
```
Input: "I love machine learning"

Step 1: Convert to Q, K, V vectors
Word    | Query | Key | Value
--------|-------|-----|------
I       | q1    | k1  | v1
love    | q2    | k2  | v2
machine | q3    | k3  | v3
learning| q4    | k4  | v4

Step 2: Compute attention for "machine" (q3)
Score with "I":        q3·k1 = 0.1
Score with "love":     q3·k2 = 0.3
Score with "machine":  q3·k3 = 0.8
Score with "learning": q3·k4 = 0.9

Step 3: Softmax → weights [0.05, 0.15, 0.35, 0.45]

Step 4: Weighted sum
output3 = 0.05×v1 + 0.15×v2 + 0.35×v3 + 0.45×v4
```

### Multi-Head Attention

**Why?** Different heads learn different relationships.

```python
# Instead of 1 attention, use 8 heads
heads = []
for i in range(8):
    Q_i = X @ W_Q_i
    K_i = X @ W_K_i
    V_i = X @ W_V_i

    attention_i = softmax(Q_i @ K_i.T / √d_k) @ V_i
    heads.append(attention_i)

# Concatenate and project
output = concat(heads) @ W_O
```

**Heads learn different patterns:**
- Head 1: Syntactic relationships (subject-verb)
- Head 2: Long-range dependencies
- Head 3: Local context
- ...

### Transformer Block

```
Input
  ↓
Multi-Head Attention
  ↓
Add & Normalize (Residual connection)
  ↓
Feed-Forward Network
  ↓
Add & Normalize
  ↓
Output
```

**Implementation:**
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head attention
        self.attention = nn.MultiheadAttention(d_model, num_heads)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head attention with residual
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

### Positional Encoding

**Problem:** Transformers don't know word order!

**Solution:** Add position information to embeddings.

```python
def positional_encoding(seq_len, d_model):
    pos = torch.arange(seq_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)  # Even indices
    pe[:, 1::2] = torch.cos(pos * div)  # Odd indices

    return pe
```

### Complete Transformer Architecture

```
Encoder Stack (e.g., 6 layers)
Input → Embedding + Positional Encoding
     → [Transformer Block] × 6
     → Encoder Output

Decoder Stack (e.g., 6 layers)
Target → Embedding + Positional Encoding
      → [Transformer Block with Masked Attention] × 6
      → Uses Encoder Output (cross-attention)
      → Final Prediction
```

**Use Cases:**
- Machine translation (original use)
- Text generation
- Question answering
- Summarization

---

## Large Language Models (LLMs) 📚

### BERT (Bidirectional Encoder Representations from Transformers)

**Key Idea:** Encoder-only architecture, bidirectional context.

**Architecture:**
```
[CLS] The cat sat on the mat [SEP]
  ↓     ↓   ↓   ↓  ↓   ↓
Transformer Encoder × 12 (BERT-base) or 24 (BERT-large)
  ↓     ↓   ↓   ↓  ↓   ↓
Contextualized embeddings for each token
```

**Pre-training Tasks:**
1. **Masked Language Modeling (MLM):**
   ```
   Input:  "The cat [MASK] on the mat"
   Predict: "sat"
   ```

2. **Next Sentence Prediction (NSP):**
   ```
   Input:  "The cat sat on the mat [SEP] It was sleeping"
   Predict: True (sentences are related)
   ```

**Fine-tuning for tasks:**
- Classification: Use [CLS] token embedding
- NER: Use each token embedding
- QA: Predict answer span

**Variants:**
- **RoBERTa:** BERT without NSP, better training
- **ALBERT:** Parameter sharing, smaller model
- **DistilBERT:** Smaller, faster, 97% of BERT performance

### GPT (Generative Pre-trained Transformer)

**Key Idea:** Decoder-only architecture, autoregressive generation.

**Architecture:**
```
Decoder-only Transformer Stack
GPT-1: 12 layers, 117M parameters
GPT-2: 48 layers, 1.5B parameters
GPT-3: 96 layers, 175B parameters
GPT-4: Unknown (estimated 1.7T parameters)
```

**Pre-training:** Next token prediction
```
Input:  "The cat sat on"
Predict: "the"

Input:  "The cat sat on the"
Predict: "mat"
```

**Key Innovation (GPT-3):** Few-shot learning
```
Prompt:
"Translate English to French:
 Hello → Bonjour
 Cat → Chat
 House →"

Model: "Maison"
```

**Evolution:**
- **GPT-1** (2018): 117M params, proves pre-training works
- **GPT-2** (2019): 1.5B params, coherent text generation
- **GPT-3** (2020): 175B params, few-shot learning
- **ChatGPT** (2022): GPT-3.5 + RLHF (Reinforcement Learning from Human Feedback)
- **GPT-4** (2023): Multimodal, improved reasoning

### T5 (Text-to-Text Transfer Transformer)

**Key Idea:** Everything is text-to-text!

```
Task: Translation
Input:  "translate English to French: Hello"
Output: "Bonjour"

Task: Classification
Input:  "sentiment: This movie is great!"
Output: "positive"

Task: QA
Input:  "question: What is ML? context: Machine learning..."
Output: "Machine learning is..."
```

**Architecture:** Encoder-decoder (full Transformer)

**Advantages:**
- Unified framework
- Easy to add new tasks
- Transfer learning across tasks

---

## Vision Transformers (ViT) 👁️

**Key Paper:** "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)

### Core Idea

Apply Transformers to images by treating them as sequences of patches.

**Process:**
```
1. Split image into patches (e.g., 16×16 pixels)
   224×224 image → 14×14 = 196 patches

2. Flatten each patch to vector
   16×16×3 → 768-dim vector

3. Add position embeddings
   Patch embeddings + positional encoding

4. Process with Transformer encoder
   Standard Transformer layers

5. Use [CLS] token for classification
```

**Implementation:**
```python
import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 d_model=768, num_heads=12, num_layers=12):
        super().__init__()

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model, num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Classification head
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification from CLS token
        return self.head(x[:, 0])
```

**Advantages:**
- Scales better than CNNs with more data
- Global receptive field from layer 1
- Transfer learning across domains

**Disadvantages:**
- Needs large datasets (ImageNet-21k)
- More compute than CNNs
- Less inductive bias (must learn spatial relationships)

**Variants:**
- **DeiT:** Data-efficient ViT (works with ImageNet-1k)
- **Swin Transformer:** Hierarchical with shifted windows
- **BEiT:** BERT-style pre-training for vision

---

## ResNet & Skip Connections 🏗️

**Key Paper:** "Deep Residual Learning for Image Recognition" (He et al., 2015)

### Problem: Degradation

Deeper networks should be better, but:
```
56-layer network: 74% accuracy
20-layer network: 79% accuracy

Deeper network performs WORSE! (Not just overfitting)
```

### Solution: Residual Connections

**Skip connections** let gradients flow directly.

```
Input x
  ↓
  ├────────────┐ (Skip connection)
  ↓            │
Conv Layer    │
  ↓            │
ReLU          │
  ↓            │
Conv Layer    │
  ↓            │
  └─────(+)────┘ (Add input)
      ↓
    ReLU
      ↓
    Output
```

**Mathematical Form:**
```
Output = F(x) + x

where F(x) = stack of layers
```

**Why it works:**
- Easy to learn identity function (F(x) = 0)
- Gradients flow directly through skip connections
- Layers learn "residuals" (refinements) instead of full mapping

**Implementation:**
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection adjustment if dimensions change
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = torch.relu(out)

        return out
```

**ResNet Variants:**
- **ResNet-18, 34:** Basic blocks
- **ResNet-50, 101, 152:** Bottleneck blocks (1×1 conv for efficiency)
- **ResNeXt:** Multiple parallel paths
- **DenseNet:** Concatenate instead of add

**Impact:** Enabled training of 100+ layer networks!

---

## U-Net 🎯

**Key Paper:** "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)

### Architecture: Encoder-Decoder with Skip Connections

```
Input Image
    ↓
[Encoder: Downsampling]
    ↓        ↘
    ↓          (Skip connection)
    ↓            ↘
    ↓              ↘
Bottleneck         ↓
    ↓            ↗
    ↓          (Skip connection)
    ↓        ↗
[Decoder: Upsampling]
    ↓
Segmentation Map
```

**Visual:**
```
128×128 → 64×64 → 32×32 → 16×16 (Bottleneck)
  ↓         ↓       ↓        ↓
  └─────────┴───────┴────────┘
            ↓
  ┌─────────┬───────┬────────┐
  ↓         ↓       ↓        ↓
16×16 → 32×32 → 64×64 → 128×128
```

**Skip Connections:** Copy features from encoder to decoder
- Preserve spatial information lost during downsampling
- Help decoder recover fine details

**Implementation:**
```python
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder (downsampling)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder (upsampling)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.conv_block(1024, 512)  # 1024 = 512 + 512 (skip)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Output
        self.out = nn.Conv2d(64, out_channels, 1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)  # Skip connection
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)

        return self.out(dec1)
```

**Applications:**
- Medical image segmentation
- Satellite image analysis
- Object segmentation
- Image inpainting

---

## Autoencoders & VAEs 🔄

### Autoencoder

**Goal:** Learn compressed representation (encoding) of data.

**Architecture:**
```
Input → Encoder → Latent Space (Bottleneck) → Decoder → Reconstruction
```

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)  # Latent representation
        x_reconstructed = self.decoder(z)
        return x_reconstructed
```

**Loss:** Reconstruction error (MSE, BCE)
```python
loss = F.mse_loss(reconstructed, original)
```

**Uses:**
- Dimensionality reduction
- Denoising
- Anomaly detection
- Feature learning

### Variational Autoencoder (VAE)

**Key Difference:** Learns probability distribution in latent space.

**Architecture:**
```
Input → Encoder → [μ, σ] → Sample z ~ N(μ, σ²) → Decoder → Reconstruction
```

**Encoder outputs:**
- μ (mean)
- σ (standard deviation)

**Sampling:**
```
z = μ + σ × ε,  where ε ~ N(0,1)
```

**Loss Function:**
```
Loss = Reconstruction Loss + KL Divergence

Reconstruction: How well it reconstructs
KL Divergence: How close latent distribution is to N(0,1)
```

**Implementation:**
```python
class VAE(nn.Module):
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(400, latent_dim)
        self.fc_var = nn.Linear(400, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 400),
            nn.ReLU(),
            nn.Linear(400, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_var(h)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    # Reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return BCE + KLD
```

**Uses:**
- Generate new samples
- Image generation
- Anomaly detection
- Data augmentation

---

## Architecture Selection Guide 🗺️

| Task | Architecture | Why |
|------|-------------|-----|
| Image Classification | ResNet, ViT | Proven, scalable |
| Object Detection | Faster R-CNN, YOLO | Real-time performance |
| Semantic Segmentation | U-Net, DeepLab | Pixel-level accuracy |
| Text Classification | BERT | Bidirectional context |
| Text Generation | GPT | Autoregressive generation |
| Translation | T5, mBART | Encoder-decoder |
| Question Answering | BERT, RoBERTa | Understanding context |
| Image Generation | VAE, GAN, Diffusion | Creative applications |
| Speech Recognition | Wav2Vec 2.0, Whisper | Audio understanding |

---

## Key Takeaways 💡

1. **Transformers** revolutionized NLP and now vision
2. **Attention** lets models focus on relevant information
3. **Skip connections** (ResNet) enable very deep networks
4. **Pre-training + fine-tuning** is the modern paradigm
5. **Scaling** (more data, parameters) often improves performance
6. **Transfer learning** saves compute and time
7. **Architecture matters** less than data and training
8. **Hybrid approaches** (CNN + Transformer) show promise

## Practical Tips 🛠️

1. **Start with pre-trained models:** Don't train from scratch!
2. **Use HuggingFace:** Easy access to BERT, GPT, T5, ViT
3. **Fine-tune carefully:** Small learning rate, freeze early layers
4. **Monitor attention:** Visualize what the model focuses on
5. **Computational cost:** Transformers are expensive, consider distillation
6. **Data requirements:** Transformers need lots of data or pre-training

## Next Steps 📚

- **[08_Advanced_Topics.md](08_Advanced_Topics.md)** - GANs, RL, Meta-learning
- **[09_NLP_Deep_Dive.md](09_NLP_Deep_Dive.md)** - Advanced NLP techniques
- **[10_Computer_Vision.md](10_Computer_Vision.md)** - Advanced CV topics
- **[11_Production_ML.md](11_Production_ML.md)** - Deploy models to production

---

**You now understand state-of-the-art architectures!** 🎉 These power ChatGPT, Stable Diffusion, and modern AI systems.

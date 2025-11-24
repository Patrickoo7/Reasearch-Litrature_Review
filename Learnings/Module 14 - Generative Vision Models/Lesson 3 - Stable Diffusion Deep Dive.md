# Lesson 3: Stable Diffusion Deep Dive - Production Image Generation 🚀

**Module 14: Generative Vision Models | Lesson 3 of 6**

Master the open-source foundation of modern text-to-image generation!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand Latent Diffusion Models architecture
2. ✅ Master the three components: VAE, U-Net, CLIP
3. ✅ Implement Classifier-Free Guidance (CFG)
4. ✅ Use Hugging Face Diffusers library productively
5. ✅ Apply different samplers (DDIM, DPM-Solver, Euler)
6. ✅ Master prompt engineering for optimal results
7. ✅ Generate production-quality images

---

## Prerequisites

- **Lesson 1**: GANs and VAEs (VAE architecture)
- **Lesson 2**: Diffusion Models (DDPM, DDIM)
- **Module 7**: NLP basics (for CLIP understanding)
- **Python Libraries**: transformers, diffusers, torch, PIL
- **GPU**: Recommended (Stable Diffusion is compute-intensive)

---

## 1. Latent Diffusion Models - The Key Innovation

### Why Compress to Latent Space?

```python
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

"""
LATENT DIFFUSION MODELS (Rombach et al., 2022)

Key Insight: Don't diffuse in pixel space - too expensive!

Instead:
1. Encode to latent space (8x compression)
2. Diffuse in latent space (much cheaper!)
3. Decode back to pixels

Benefits:
✅ 8x faster training/sampling
✅ 8x less memory
✅ Same quality as pixel-space diffusion
✅ Enables high-resolution generation
"""


def compare_pixel_vs_latent_diffusion():
    """
    Compare computational costs.
    """
    print("PIXEL-SPACE vs LATENT-SPACE DIFFUSION\n")

    # Image dimensions
    pixel_h, pixel_w = 512, 512
    latent_h, latent_w = 64, 64  # 8x compression

    pixel_size = pixel_h * pixel_w * 3  # RGB
    latent_size = latent_h * latent_w * 4  # 4 latent channels

    print("Image Size:")
    print(f"  Pixel: {pixel_h}x{pixel_w}x3 = {pixel_size:,} values")
    print(f"  Latent: {latent_h}x{latent_w}x4 = {latent_size:,} values")
    print(f"  Compression: {pixel_size / latent_size:.1f}x smaller!")
    print()

    print("Computation (per diffusion step):")
    print(f"  Pixel-space: Process {pixel_size:,} values")
    print(f"  Latent-space: Process {latent_size:,} values")
    print(f"  Speedup: ~{pixel_size / latent_size:.0f}x faster!")
    print()

    print("Memory:")
    print(f"  Pixel U-Net: ~{pixel_size * 4 / 1e9:.2f} GB (rough estimate)")
    print(f"  Latent U-Net: ~{latent_size * 4 / 1e9:.2f} GB")
    print()

    print("Result: Latent Diffusion is the practical choice! 🚀")


compare_pixel_vs_latent_diffusion()
```

---

## 2. The Three Components of Stable Diffusion

### Architecture Overview

```python
"""
STABLE DIFFUSION ARCHITECTURE

Three neural networks working together:

1. VAE Encoder/Decoder
   - Encoder: Image → Latent (compress 8x)
   - Decoder: Latent → Image (reconstruct)

2. U-Net
   - Denoises latent representations
   - Conditioned on text via cross-attention

3. CLIP Text Encoder
   - Text → Embeddings
   - Provides semantic conditioning

Flow:
  Text → CLIP → Text Embeddings
  Image → VAE Encoder → Latent
  Latent + Text Embeddings → U-Net → Denoised Latent
  Denoised Latent → VAE Decoder → Image
"""


class StableDiffusionComponents:
    """
    Explore the three components individually.
    """
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1-base"):
        """
        Load Stable Diffusion components.

        Args:
            model_id: Hugging Face model identifier
        """
        print(f"Loading Stable Diffusion components from {model_id}...")

        # Load VAE
        self.vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            torch_dtype=torch.float16
        )

        # Load text encoder
        self.text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float16
        )

        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer"
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Move to device
        self.vae.to(self.device)
        self.text_encoder.to(self.device)

        print(f"Loaded on {self.device}")

    def encode_image(self, image):
        """
        Encode image to latent space.

        Args:
            image: PIL Image or tensor [C, H, W]

        Returns:
            Latent representation [4, H/8, W/8]
        """
        # Convert to tensor if needed
        if isinstance(image, Image.Image):
            image = torch.tensor(np.array(image)).float() / 127.5 - 1.0
            image = image.permute(2, 0, 1).unsqueeze(0)

        image = image.to(self.device).half()

        # Encode
        with torch.no_grad():
            latent = self.vae.encode(image).latent_dist.sample()
            latent = latent * 0.18215  # Scaling factor

        return latent

    def decode_latent(self, latent):
        """
        Decode latent to image.

        Args:
            latent: Latent tensor [4, H/8, W/8]

        Returns:
            Image tensor [3, H, W]
        """
        # Unscale
        latent = latent / 0.18215

        # Decode
        with torch.no_grad():
            image = self.vae.decode(latent).sample

        # Convert to [0, 1]
        image = (image + 1.0) / 2.0

        return image

    def encode_text(self, prompt):
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text string

        Returns:
            Text embeddings [1, 77, 768]
        """
        # Tokenize
        text_input = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        # Encode
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device)
            )[0]

        return text_embeddings


# Example usage (requires model download)
print("\nStableDiffusionComponents class defined!")
print("Usage:")
print("  components = StableDiffusionComponents()")
print("  latent = components.encode_image(image)")
print("  text_emb = components.encode_text('a cat')")
```

---

## 3. VAE Encoder and Decoder

### Understanding the Compression

```python
"""
VAE in Stable Diffusion

Architecture:
- Encoder: 512x512x3 → 64x64x4
- Decoder: 64x64x4 → 512x512x3

Key properties:
✅ 8x8 spatial compression
✅ 4 latent channels (learned representation)
✅ Pre-trained on large image dataset
✅ Minimal reconstruction loss
"""


def demonstrate_vae_compression():
    """
    Show VAE encode/decode process.
    """
    print("VAE ENCODE/DECODE PROCESS\n")

    # Simulated shapes
    print("Encoder:")
    print("  Input:  [1, 3, 512, 512]  (RGB image)")
    print("  Output: [1, 4, 64, 64]    (Latent)")
    print("  Compression: 64x smaller!")
    print()

    print("Decoder:")
    print("  Input:  [1, 4, 64, 64]    (Latent)")
    print("  Output: [1, 3, 512, 512]  (RGB image)")
    print("  Reconstruction: Near-perfect quality")
    print()

    print("Latent channels (4 dimensions):")
    print("  - Not RGB! Learned representation")
    print("  - Captures semantic features")
    print("  - Optimized for reconstruction")
    print()

    # Memory calculation
    pixel_mem = 3 * 512 * 512 * 4  # bytes (float32)
    latent_mem = 4 * 64 * 64 * 4

    print(f"Memory usage:")
    print(f"  Pixel space: {pixel_mem / 1e6:.2f} MB")
    print(f"  Latent space: {latent_mem / 1e6:.2f} MB")
    print(f"  Savings: {pixel_mem / latent_mem:.1f}x")


demonstrate_vae_compression()
```

### Visualizing Latent Channels

```python
def visualize_latent_channels(image_path=None):
    """
    Visualize the 4 latent channels.

    Each channel captures different image features.
    """
    if image_path is None:
        print("LATENT CHANNEL VISUALIZATION\n")
        print("Each of the 4 latent channels captures different features:")
        print("  Channel 0: Often edges and structure")
        print("  Channel 1: Color information")
        print("  Channel 2: Texture details")
        print("  Channel 3: High-frequency features")
        print()
        print("These are learned representations, not handcrafted!")
        print()
        print("To visualize with real image:")
        print("  components = StableDiffusionComponents()")
        print("  latent = components.encode_image(image)")
        print("  Plot the 4 channels as heatmaps")
        return

    # If image_path provided, would encode and visualize
    # (Requires loaded model - omitted for brevity)


visualize_latent_channels()
```

---

## 4. U-Net with Cross-Attention

### Text Conditioning Mechanism

```python
"""
U-NET FOR STABLE DIFFUSION

Special features:
1. Cross-Attention layers
   - Query: From image features
   - Key/Value: From text embeddings
   - Allows text to guide denoising!

2. ResNet blocks
   - Time conditioning (as in DDPM)
   - Cross-attention for text
   - Self-attention for spatial coherence

3. Skip connections
   - Preserve fine details
   - Multi-scale features
"""


class CrossAttention(nn.Module):
    """
    Cross-attention layer for text conditioning.

    Allows image features to attend to text embeddings.
    """
    def __init__(self, query_dim, context_dim, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)

        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x, context):
        """
        Args:
            x: Image features [batch, seq_len, query_dim]
            context: Text embeddings [batch, context_len, context_dim]

        Returns:
            Attended features [batch, seq_len, query_dim]
        """
        batch_size = x.shape[0]

        # Project to Q, K, V
        q = self.to_q(x)  # Image query
        k = self.to_k(context)  # Text keys
        v = self.to_v(context)  # Text values

        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.matmul(attn, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        out = self.to_out(out)

        return out


# Test cross-attention
cross_attn = CrossAttention(query_dim=512, context_dim=768, num_heads=8)

# Dummy inputs
image_features = torch.randn(2, 64*64, 512)  # Flattened spatial features
text_embeddings = torch.randn(2, 77, 768)    # CLIP text embeddings

attended_features = cross_attn(image_features, text_embeddings)

print(f"Image features: {image_features.shape}")
print(f"Text embeddings: {text_embeddings.shape}")
print(f"Attended features: {attended_features.shape}")
print()
print("Cross-attention allows text to guide image generation!")
```

---

## 5. CLIP Text Encoder

### From Text to Semantic Embeddings

```python
"""
CLIP (Contrastive Language-Image Pre-training)

Why CLIP for Stable Diffusion?
✅ Trained on 400M image-text pairs
✅ Learns semantic relationships
✅ Robust to various text descriptions
✅ 77 token sequence (max prompt length)

Architecture:
  Text → Tokenizer → Token IDs
  Token IDs → Transformer → Embeddings [77, 768]
"""


def explore_clip_tokenization(tokenizer, text_encoder):
    """
    Demonstrate CLIP tokenization and encoding.
    """
    prompts = [
        "a cat",
        "a photograph of a cat",
        "a highly detailed digital painting of a majestic cat",
    ]

    print("CLIP TEXT ENCODING\n")

    for prompt in prompts:
        # Tokenize
        tokens = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )

        num_tokens = (tokens.input_ids != tokenizer.pad_token_id).sum().item()

        print(f"Prompt: '{prompt}'")
        print(f"  Tokens used: {num_tokens}/77")
        print(f"  Token IDs: {tokens.input_ids[0][:num_tokens].tolist()}")
        print()

    print("Key points:")
    print("✅ Max 77 tokens (longer prompts truncated)")
    print("✅ Padding to 77 for batch processing")
    print("✅ Each token → 768-dim embedding")
    print("✅ More descriptive ≠ always better (saturation)")


print("CLIP tokenization demo ready!")
print("Requires loaded tokenizer and text_encoder")
```

### Prompt Engineering Basics

```python
"""
PROMPT ENGINEERING FOR STABLE DIFFUSION

Effective prompts have:
1. Subject: What to generate
2. Style: Artistic style, medium
3. Quality modifiers: "highly detailed", "4K"
4. Negative prompts: What to avoid

Examples:
Good: "a portrait of a cat, digital art, highly detailed, trending on artstation"
Bad: "cat"
"""


def prompt_engineering_guide():
    """
    Guide to writing effective prompts.
    """
    print("PROMPT ENGINEERING GUIDE\n")

    print("1. SUBJECT (Required)")
    print("   - Be specific: 'a tabby cat' vs 'a cat'")
    print("   - Include key details: 'sitting', 'looking at camera'")
    print()

    print("2. STYLE (Highly recommended)")
    print("   - Medium: 'digital art', 'oil painting', 'photograph'")
    print("   - Artist: 'in the style of Van Gogh'")
    print("   - Art movement: 'impressionism', 'surrealism'")
    print()

    print("3. QUALITY MODIFIERS")
    print("   - Resolution: '4K', '8K', 'high resolution'")
    print("   - Detail: 'highly detailed', 'intricate'")
    print("   - Trending: 'trending on artstation'")
    print()

    print("4. LIGHTING & COMPOSITION")
    print("   - Lighting: 'dramatic lighting', 'golden hour'")
    print("   - Composition: 'centered', 'rule of thirds'")
    print("   - Camera: 'wide angle', 'bokeh', 'depth of field'")
    print()

    print("5. NEGATIVE PROMPTS")
    print("   - What to avoid: 'blurry', 'low quality', 'distorted'")
    print("   - Unwanted elements: 'text', 'watermark'")
    print()

    print("EXAMPLE PROMPTS:")
    print()
    print("Photorealistic:")
    print("  'a professional photograph of a mountain landscape,")
    print("   golden hour lighting, sharp focus, 8K, highly detailed'")
    print()
    print("Artistic:")
    print("  'a digital painting of a fantasy castle,")
    print("   dramatic lighting, intricate details, trending on artstation'")
    print()
    print("Negative:")
    print("  'blurry, low quality, distorted, text, watermark'")


prompt_engineering_guide()
```

---

## 6. Classifier-Free Guidance (CFG)

### The Secret to High-Quality Generation

```python
"""
CLASSIFIER-FREE GUIDANCE

The most important technique for quality!

Training:
  - Randomly drop text conditioning (10% of time)
  - Model learns both conditional and unconditional

Sampling:
  ε_guided = ε_uncond + w * (ε_cond - ε_uncond)
           = (1-w) * ε_uncond + w * ε_cond

Where:
  w = guidance scale (typically 7.5)
  ε_uncond = prediction without text
  ε_cond = prediction with text

Effect:
  w = 1.0: Standard conditional (weak adherence)
  w = 7.5: Strong adherence to prompt (SD default)
  w = 15.0: Very strong (may reduce diversity)
"""


def classifier_free_guidance(noise_pred_cond, noise_pred_uncond, guidance_scale=7.5):
    """
    Apply classifier-free guidance.

    Args:
        noise_pred_cond: Conditional noise prediction
        noise_pred_uncond: Unconditional noise prediction
        guidance_scale: Strength of conditioning (w)

    Returns:
        Guided noise prediction
    """
    # CFG formula
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    return noise_pred


# Example
uncond = torch.randn(1, 4, 64, 64)
cond = torch.randn(1, 4, 64, 64)

# Different guidance scales
scales = [1.0, 3.0, 7.5, 12.0, 20.0]

print("CLASSIFIER-FREE GUIDANCE\n")
print("Effect of different guidance scales:\n")

for scale in scales:
    guided = classifier_free_guidance(cond, uncond, guidance_scale=scale)

    print(f"w = {scale:4.1f}:")
    if scale == 1.0:
        print("  → Standard conditional generation")
        print("  → Weaker prompt adherence")
    elif scale < 7.5:
        print("  → Moderate guidance")
        print("  → More diversity, less prompt adherence")
    elif scale == 7.5:
        print("  → Stable Diffusion default")
        print("  → Good balance of quality and diversity")
    else:
        print("  → Strong guidance")
        print("  → High prompt adherence, may reduce diversity")
    print()

print("Recommendation: Start with 7.5, adjust based on results")
```

### Implementing CFG in Diffusion

```python
def diffusion_step_with_cfg(unet, latent, t, text_embeddings, uncond_embeddings,
                            guidance_scale=7.5):
    """
    Single diffusion step with classifier-free guidance.

    Args:
        unet: U-Net model
        latent: Current latent [batch, 4, h, w]
        t: Timestep
        text_embeddings: Conditional embeddings
        uncond_embeddings: Unconditional embeddings
        guidance_scale: CFG scale

    Returns:
        Noise prediction with CFG applied
    """
    # Concatenate for batch processing
    latent_model_input = torch.cat([latent] * 2)
    text_embeddings_all = torch.cat([uncond_embeddings, text_embeddings])

    # Predict noise
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings_all).sample

    # Split predictions
    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)

    # Apply CFG
    noise_pred_guided = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

    return noise_pred_guided


print("CFG diffusion step function defined!")
print()
print("Key insight: Run U-Net twice per step:")
print("  1. With text embeddings (conditional)")
print("  2. Without text embeddings (unconditional)")
print("  3. Blend predictions based on guidance scale")
```

---

## 7. Using Hugging Face Diffusers

### Basic Text-to-Image Generation

```python
"""
HUGGING FACE DIFFUSERS LIBRARY

Simplifies Stable Diffusion usage!

Installation:
  pip install diffusers transformers accelerate

Models:
  - stabilityai/stable-diffusion-2-1
  - runwayml/stable-diffusion-v1-5
  - stabilityai/stable-diffusion-xl-base-1.0
"""


def generate_image_basic(prompt, negative_prompt="", num_inference_steps=50,
                         guidance_scale=7.5, seed=None):
    """
    Generate image with Stable Diffusion.

    Args:
        prompt: Text description
        negative_prompt: What to avoid
        num_inference_steps: Number of denoising steps
        guidance_scale: CFG strength
        seed: Random seed for reproducibility

    Returns:
        Generated PIL Image
    """
    # Example code (requires model)
    code = f'''
# Load pipeline
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# Set seed
if seed is not None:
    generator = torch.Generator("cuda").manual_seed(seed)
else:
    generator = None

# Generate
image = pipe(
    prompt="{prompt}",
    negative_prompt="{negative_prompt}",
    num_inference_steps={num_inference_steps},
    guidance_scale={guidance_scale},
    generator=generator
).images[0]

# Save
image.save("generated.png")
'''

    print("BASIC IMAGE GENERATION")
    print(code)

    return None  # Would return image if model loaded


# Example usage
generate_image_basic(
    prompt="a beautiful landscape with mountains and a lake, golden hour lighting, 8K",
    negative_prompt="blurry, low quality",
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
)
```

### Advanced Generation Options

```python
"""
ADVANCED GENERATION OPTIONS

Control over:
1. Resolution: width, height
2. Batch size: Generate multiple images
3. Scheduler: Different sampling methods
4. Attention slicing: Reduce memory
5. Half precision: Faster inference
"""


def generate_image_advanced():
    """
    Demonstrate advanced options.
    """
    code = '''
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Load with custom scheduler
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
)

# Use DPM-Solver for faster sampling
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# Enable memory optimizations
pipe.enable_attention_slicing()
pipe = pipe.to("cuda")

# Generate with custom resolution
images = pipe(
    prompt="a cat astronaut in space",
    negative_prompt="blurry, low quality",
    num_inference_steps=25,  # Fewer steps with DPM-Solver!
    guidance_scale=7.5,
    width=768,  # Custom width
    height=512,  # Custom height
    num_images_per_prompt=4  # Generate 4 images
).images

# Save all images
for i, img in enumerate(images):
    img.save(f"image_{i}.png")
'''

    print("ADVANCED GENERATION")
    print(code)


generate_image_advanced()
```

---

## 8. Sampling Methods (Schedulers)

### DDIM, DPM-Solver, Euler

```python
"""
SCHEDULERS: Different sampling strategies

Available in Diffusers:
1. DDIM: Original, deterministic, ~50 steps
2. DPM-Solver: Faster, ~20-25 steps, good quality
3. Euler: Simple, ~30-40 steps
4. Euler Ancestral: Stochastic variant
5. PNDM: Pseudo Numerical Methods
6. LMS: Linear Multi-Step
"""


def compare_schedulers():
    """
    Compare different sampling methods.
    """
    schedulers = {
        "DDIM": {
            "steps": 50,
            "speed": "Medium",
            "quality": "Good",
            "deterministic": True,
            "description": "Original, reliable, well-tested"
        },
        "DPM-Solver": {
            "steps": 20,
            "speed": "Fast",
            "quality": "Excellent",
            "deterministic": True,
            "description": "Best speed/quality tradeoff"
        },
        "Euler": {
            "steps": 30,
            "speed": "Medium",
            "quality": "Good",
            "deterministic": True,
            "description": "Simple, stable"
        },
        "Euler Ancestral": {
            "steps": 30,
            "speed": "Medium",
            "quality": "Good",
            "deterministic": False,
            "description": "Adds randomness, more variety"
        },
        "PNDM": {
            "steps": 50,
            "speed": "Medium",
            "quality": "Good",
            "deterministic": True,
            "description": "Pseudo numerical methods"
        },
    }

    print("SCHEDULER COMPARISON\n")

    for name, info in schedulers.items():
        print(f"{name}:")
        print(f"  Typical steps: {info['steps']}")
        print(f"  Speed: {info['speed']}")
        print(f"  Quality: {info['quality']}")
        print(f"  Deterministic: {info['deterministic']}")
        print(f"  → {info['description']}")
        print()

    print("RECOMMENDATION:")
    print("  Start with: DPM-Solver (fast + high quality)")
    print("  Alternative: DDIM (reliable, well-tested)")
    print("  For variety: Euler Ancestral (adds randomness)")


compare_schedulers()
```

### Changing Schedulers

```python
def scheduler_usage_example():
    """
    Show how to use different schedulers.
    """
    code = '''
from diffusers import (
    StableDiffusionPipeline,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler
)

# Load pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Try different schedulers
schedulers = {
    "DDIM": DDIMScheduler.from_config(pipe.scheduler.config),
    "DPM-Solver": DPMSolverMultistepScheduler.from_config(pipe.scheduler.config),
    "Euler-A": EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config),
}

prompt = "a beautiful sunset over the ocean"

for name, scheduler in schedulers.items():
    pipe.scheduler = scheduler

    image = pipe(
        prompt=prompt,
        num_inference_steps=30,
        guidance_scale=7.5
    ).images[0]

    image.save(f"sunset_{name}.png")
    print(f"Generated with {name}")
'''

    print("SCHEDULER USAGE")
    print(code)


scheduler_usage_example()
```

---

## 9. Negative Prompts

### What to Avoid

```python
"""
NEGATIVE PROMPTS

Tell model what NOT to generate!

Common negative prompts:
- Quality: "blurry, low quality, distorted"
- Artifacts: "watermark, text, signature"
- Anatomy: "bad hands, extra fingers" (for humans)
- Style: "cartoon" (if wanting photorealistic)

How it works:
  Uses unconditional embedding with negative text
  Pushes generation AWAY from negative concepts
"""


def negative_prompt_guide():
    """
    Guide to effective negative prompts.
    """
    print("NEGATIVE PROMPT GUIDE\n")

    categories = {
        "Quality Issues": [
            "blurry", "low quality", "low resolution",
            "pixelated", "jpeg artifacts", "compression artifacts"
        ],
        "Unwanted Elements": [
            "watermark", "text", "signature", "logo",
            "border", "frame", "username"
        ],
        "Anatomy Problems": [
            "bad anatomy", "bad hands", "extra fingers",
            "missing fingers", "deformed", "disfigured",
            "mutation", "ugly"
        ],
        "Style Mismatches": [
            "cartoon", "anime" (if wanting realistic),
            "3D render" (if wanting painterly),
            "painting" (if wanting photo)
        ],
    }

    for category, terms in categories.items():
        print(f"{category}:")
        print(f"  {', '.join(terms)}")
        print()

    print("EXAMPLE USAGE:")
    print()
    print("Prompt:")
    print("  'a portrait of a woman, professional photography, 8K'")
    print()
    print("Negative Prompt:")
    print("  'blurry, low quality, bad anatomy, deformed,")
    print("   watermark, text, signature, disfigured'")


negative_prompt_guide()
```

---

## 10. Complete Generation Pipeline

### Putting It All Together

```python
def stable_diffusion_complete_example():
    """
    Complete example with all best practices.
    """
    code = '''
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# 1. Load pipeline with optimizations
pipe = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16,  # Half precision
    safety_checker=None,  # Optional: disable safety checker
)

# 2. Optimize for memory
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()  # For high resolution

# 3. Use efficient scheduler
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 4. Move to GPU
pipe = pipe.to("cuda")

# 5. Define prompts
prompt = """
a highly detailed digital painting of a fantasy castle on a cliff,
dramatic sunset lighting, volumetric fog, intricate architecture,
by Greg Rutkowski and James Gurney, trending on artstation,
masterpiece, 8K, ultra detailed
"""

negative_prompt = """
blurry, low quality, low resolution, pixelated,
watermark, text, signature, deformed, ugly,
bad composition, amateur
"""

# 6. Set seed for reproducibility
seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

# 7. Generate
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=25,  # DPM-Solver needs fewer!
    guidance_scale=7.5,
    width=768,
    height=512,
    generator=generator
).images[0]

# 8. Save
image.save("fantasy_castle.png")

print(f"Generated image with seed {seed}")
print(f"Prompt: {prompt[:50]}...")
'''

    print("COMPLETE STABLE DIFFUSION PIPELINE")
    print(code)


stable_diffusion_complete_example()
```

---

## 11. Image-to-Image Generation

### Transforming Existing Images

```python
"""
IMAGE-TO-IMAGE (img2img)

Start from an existing image instead of noise!

Process:
1. Encode image to latent
2. Add noise (partial, controlled by strength)
3. Denoise with text guidance
4. Decode to image

Use cases:
✅ Style transfer
✅ Image variations
✅ Upscaling with details
✅ Fixing/enhancing images
"""


def img2img_example():
    """
    Image-to-image generation example.
    """
    code = '''
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

# Load img2img pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1",
    torch_dtype=torch.float16
).to("cuda")

# Load initial image
init_image = Image.open("input.jpg").resize((768, 512))

# Generate variation
prompt = "a beautiful oil painting of the same scene"
negative_prompt = "blurry, low quality"

image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.75,  # How much to change (0=unchanged, 1=completely new)
    guidance_scale=7.5,
    num_inference_steps=50
).images[0]

image.save("output.png")
'''

    print("IMAGE-TO-IMAGE GENERATION")
    print(code)
    print()
    print("Strength parameter:")
    print("  0.0-0.3: Subtle changes (refinement)")
    print("  0.4-0.6: Moderate transformation")
    print("  0.7-0.9: Significant changes")
    print("  1.0: Complete regeneration (like txt2img)")


img2img_example()
```

---

## 12. Inpainting

### Selective Image Editing

```python
"""
INPAINTING

Edit specific regions of an image!

Process:
1. Provide: Original image + Mask (what to edit)
2. Model fills masked region based on prompt
3. Seamlessly blends with surroundings

Use cases:
✅ Remove objects
✅ Add objects
✅ Change backgrounds
✅ Fix artifacts
"""


def inpainting_example():
    """
    Inpainting example.
    """
    code = '''
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

# Load inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("room.jpg").resize((512, 512))
mask = Image.open("mask.jpg").resize((512, 512))  # White = inpaint, Black = keep

# Inpaint
prompt = "a large window with mountain view"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

result.save("inpainted.png")
'''

    print("INPAINTING")
    print(code)
    print()
    print("Tips:")
    print("  - Mask should be slightly larger than target area")
    print("  - Use feathered edges for smoother blending")
    print("  - Describe both the inpainted region AND context")


inpainting_example()
```

---

## Practice Exercises

### Exercise 1: Prompt Engineering Challenge

```python
"""
Generate the same subject with different styles.

Task:
1. Choose subject: "a cat sitting on a chair"
2. Generate with 5 different artistic styles:
   - Photorealistic
   - Oil painting
   - Digital art
   - Watercolor
   - Cyberpunk

3. Compare results
4. Analyze which style keywords work best

Use: Different prompts, same seed for comparison
"""

# Your implementation here
```

### Exercise 2: Guidance Scale Exploration

```python
"""
Explore effect of guidance scale.

Task:
1. Fix prompt: Your choice
2. Generate with guidance scales: [1.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0]
3. Keep same seed
4. Visualize all results
5. Analyze: quality, adherence, diversity

Question: What's the optimal guidance scale for your prompt?
"""

# Your implementation here
```

### Exercise 3: Scheduler Comparison

```python
"""
Benchmark different schedulers.

Task:
1. Choose prompt
2. Generate with: DDIM, DPM-Solver, Euler, Euler-A
3. Measure: Time per image, quality (subjective)
4. Use same seed for fair comparison

Report: Which scheduler is best for your use case?
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Latent Diffusion** 🎯
   - Diffuse in compressed latent space (8x efficiency)
   - VAE encoder/decoder for compression
   - Enables high-resolution generation
   - Foundation of Stable Diffusion

2. **Three Components** 🏗️
   - VAE: Compress images 512×512 → 64×64 latents
   - U-Net: Denoise latents with text conditioning
   - CLIP: Encode text to semantic embeddings
   - All three work together seamlessly

3. **Classifier-Free Guidance** 🚀
   - Most important quality technique!
   - Blend conditional and unconditional predictions
   - Scale 7.5 is standard (adjust per use case)
   - Higher scale = stronger prompt adherence

4. **Prompt Engineering** ✍️
   - Subject + Style + Quality + Lighting
   - Negative prompts prevent unwanted features
   - More words ≠ better (find sweet spot)
   - Experiment and iterate!

5. **Schedulers Matter** ⚡
   - DPM-Solver: Best speed/quality (20-25 steps)
   - DDIM: Reliable default (50 steps)
   - Euler-A: More variety (stochastic)
   - Can dramatically affect generation time

6. **Beyond Text-to-Image** 🎨
   - Img2img: Transform existing images
   - Inpainting: Edit specific regions
   - Variations: Explore alternatives
   - All use same underlying architecture

### Stable Diffusion vs Alternatives

| Model | Open Source | Quality | Speed | Control |
|-------|-------------|---------|-------|---------|
| **Stable Diffusion** | ✅ | Excellent | Good | Excellent |
| DALL-E 2 | ❌ | Excellent | Good | Good |
| Midjourney | ❌ | Excellent | Good | Limited |
| DALL-E 3 | ❌ | Best | Good | Excellent |

**Why Stable Diffusion?**
✅ Open source (can fine-tune, modify)
✅ Runs locally (privacy, no API costs)
✅ Extensive ecosystem (ControlNet, LoRA, etc.)
✅ Active community

### What's Next?

In Lesson 4, we'll explore **Advanced Control**:
- ControlNet: Precise spatial control
- IP-Adapter: Image prompts
- Multi-conditioning strategies
- Production workflows

---

## Additional Resources

### Papers

- Rombach et al. (2022): "High-Resolution Image Synthesis with Latent Diffusion Models" (Stable Diffusion)
- Ho & Salimans (2022): "Classifier-Free Diffusion Guidance"
- Radford et al. (2021): "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)

### Libraries & Tools

- **Hugging Face Diffusers**: https://github.com/huggingface/diffusers
- **Stable Diffusion Web UI**: https://github.com/AUTOMATIC1111/stable-diffusion-webui
- **ComfyUI**: Node-based interface for advanced workflows

### Models

- **SD 2.1**: https://huggingface.co/stabilityai/stable-diffusion-2-1
- **SD XL**: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0
- **Community Models**: https://civitai.com/

### Tutorials

- **Diffusers Docs**: https://huggingface.co/docs/diffusers/
- **Stable Diffusion Guide**: https://stable-diffusion-art.com/
- **Prompt Engineering**: https://promptomania.com/stable-diffusion-prompt-builder/

---

**Next**: [Lesson 4 - Text-to-Image and Control](Lesson%204%20-%20Text-to-Image%20and%20Control.md)

Master **ControlNet** for precise spatial control over generation! 🎮

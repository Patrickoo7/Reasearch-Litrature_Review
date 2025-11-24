# Lesson 5: Fine-Tuning and Personalization 🎯

**Module 14: Generative Vision Models | Lesson 5 of 6**

Master custom model adaptation for specific subjects, styles, and concepts!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand DreamBooth for subject-driven generation
2. ✅ Implement Textual Inversion for new concepts
3. ✅ Master LoRA (Low-Rank Adaptation) training
4. ✅ Apply Hypernetworks and other adaptation methods
5. ✅ Merge and combine multiple trained models
6. ✅ Prevent overfitting and catastrophic forgetting
7. ✅ Build production fine-tuning pipelines

---

## Prerequisites

- **Lesson 3**: Stable Diffusion architecture (U-Net, VAE, CLIP)
- **Lesson 2**: Diffusion training dynamics
- **Module 3**: Deep learning (backprop, optimizers, regularization)
- **Python Libraries**: diffusers, transformers, peft, accelerate
- **GPU**: 8GB+ VRAM recommended for training

---

## 1. The Personalization Problem

### Why Fine-Tune?

```python
import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
import matplotlib.pyplot as plt
import numpy as np

"""
THE PERSONALIZATION PROBLEM

Base Stable Diffusion:
✅ Generates generic concepts well
❌ Can't generate YOUR specific subject
❌ Can't learn new artistic styles
❌ Can't capture unique objects/characters

Examples:
  - "A photo of my dog" → Generic dog
  - "Art in the style of [local artist]" → Generic art
  - "My product on a table" → Generic product

SOLUTION: Fine-tune on custom data!

Methods:
1. DreamBooth: Few-shot subject learning
2. Textual Inversion: New token for concept
3. LoRA: Lightweight adapter
4. Hypernetworks: External network
"""


def personalization_use_cases():
    """
    Demonstrate personalization use cases.
    """
    print("PERSONALIZATION USE CASES\n")

    use_cases = {
        "Personal Photos": {
            "goal": "Generate photos of specific person/pet",
            "method": "DreamBooth",
            "data": "5-10 photos from different angles",
            "example": "My dog playing in various scenes"
        },
        "Brand/Product": {
            "goal": "Product marketing images",
            "method": "DreamBooth or LoRA",
            "data": "10-20 product photos",
            "example": "My sneaker brand in different settings"
        },
        "Artistic Style": {
            "goal": "Generate in specific art style",
            "method": "LoRA or Textual Inversion",
            "data": "20-50 artwork samples",
            "example": "Illustrations in my unique art style"
        },
        "Characters": {
            "goal": "Consistent character generation",
            "method": "DreamBooth + LoRA",
            "data": "15-30 character images",
            "example": "My comic book character in any scene"
        },
        "Interior Design": {
            "goal": "Specific furniture/decor style",
            "method": "LoRA",
            "data": "30-50 room photos",
            "example": "Rooms in my signature design style"
        },
    }

    for category, info in use_cases.items():
        print(f"{category}:")
        print(f"  Goal: {info['goal']}")
        print(f"  Method: {info['method']}")
        print(f"  Data needed: {info['data']}")
        print(f"  Example: {info['example']}")
        print()

    print("Key: Personalization makes SD truly YOUR tool! 🚀")


personalization_use_cases()
```

---

## 2. DreamBooth - Subject-Driven Generation

### Few-Shot Learning for Custom Subjects

```python
"""
DREAMBOOTH (Ruiz et al., 2022)

Key Idea: Fine-tune entire model on 3-5 images of a subject

Process:
1. Choose rare identifier: "sks" (special token)
2. Train on: "a photo of sks [class]" (e.g., "a photo of sks dog")
3. Use prior preservation: Generate and train on class images
4. Result: Model learns your specific subject!

Training objective:
  L = E[||ε - ε_θ(z_t, c_θ("a photo of sks dog"))||²]
      + λ * E[||ε - ε_θ(z_t, c_θ("a photo of dog"))||²]
       ↑                ↑
  Subject loss    Prior preservation loss

Why prior preservation?
  Prevents model from forgetting general "dog" concept
  Maintains diversity and quality
"""


def dreambooth_training_setup():
    """
    Set up DreamBooth training.
    """
    code = '''
from diffusers import DreamBoothTrainingArguments, DDPMScheduler
from accelerate import Accelerator

# Training configuration
training_args = DreamBoothTrainingArguments(
    # Model
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",

    # Instance (your subject)
    instance_data_dir="./my_dog_photos",  # 3-10 photos
    instance_prompt="a photo of sks dog",  # sks = unique identifier

    # Class (prior preservation)
    class_data_dir="./generated_dogs",  # Generated class samples
    class_prompt="a photo of a dog",
    num_class_images=200,  # Generate these for regularization

    # Training
    output_dir="./dreambooth-model",
    train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=5e-6,
    lr_scheduler="constant",
    lr_warmup_steps=0,
    max_train_steps=800,  # ~800 for good results

    # Regularization
    prior_loss_weight=1.0,  # Weight for class loss

    # Memory
    gradient_checkpointing=True,
    use_8bit_adam=True,  # Saves memory

    # Validation
    validation_prompt="a photo of sks dog in a bucket",
    validation_epochs=50,
)

print("DreamBooth training configured!")
print(f"Instance prompt: {training_args.instance_prompt}")
print(f"Class prompt: {training_args.class_prompt}")
print(f"Training steps: {training_args.max_train_steps}")
'''

    print("DREAMBOOTH TRAINING SETUP")
    print(code)


dreambooth_training_setup()
```

### DreamBooth Training Implementation

```python
def train_dreambooth_simplified():
    """
    Simplified DreamBooth training loop.
    """
    code = '''
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class DreamBoothDataset(Dataset):
    """Dataset for DreamBooth training."""

    def __init__(self, instance_data_dir, instance_prompt,
                 class_data_dir=None, class_prompt=None):
        self.instance_prompt = instance_prompt
        self.class_prompt = class_prompt

        # Load instance images
        self.instance_images = [
            os.path.join(instance_data_dir, f)
            for f in os.listdir(instance_data_dir)
            if f.endswith(('.jpg', '.png'))
        ]

        # Load class images (if using prior preservation)
        self.class_images = []
        if class_data_dir:
            self.class_images = [
                os.path.join(class_data_dir, f)
                for f in os.listdir(class_data_dir)
                if f.endswith(('.jpg', '.png'))
            ]

    def __len__(self):
        return max(len(self.instance_images), len(self.class_images))

    def __getitem__(self, idx):
        # Get instance image
        instance_img = Image.open(
            self.instance_images[idx % len(self.instance_images)]
        )

        example = {
            "instance_image": instance_img,
            "instance_prompt": self.instance_prompt,
        }

        # Add class image if using prior preservation
        if self.class_images:
            class_img = Image.open(
                self.class_images[idx % len(self.class_images)]
            )
            example["class_image"] = class_img
            example["class_prompt"] = self.class_prompt

        return example


def train_dreambooth(model_id, instance_data_dir, instance_prompt,
                     class_data_dir=None, class_prompt=None,
                     num_steps=800, lr=5e-6):
    """
    Train DreamBooth model.

    Args:
        model_id: Base Stable Diffusion model
        instance_data_dir: Directory with subject images
        instance_prompt: Prompt for subject (e.g., "a photo of sks dog")
        class_data_dir: Directory with class images (prior preservation)
        class_prompt: Prompt for class (e.g., "a photo of a dog")
        num_steps: Training steps
        lr: Learning rate
    """
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    )

    # Enable gradient checkpointing for memory efficiency
    pipe.unet.enable_gradient_checkpointing()

    # Only train U-Net (freeze VAE and text encoder)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-08,
    )

    # Dataset
    dataset = DreamBoothDataset(
        instance_data_dir, instance_prompt,
        class_data_dir, class_prompt
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Training loop
    pipe.unet.train()

    for step in range(num_steps):
        batch = next(iter(dataloader))

        # Forward pass and compute loss
        # (Simplified - full implementation more complex)

        optimizer.zero_grad()
        # loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")

    # Save
    pipe.save_pretrained("./dreambooth-output")

    print("Training complete!")


# Example usage
train_dreambooth(
    model_id="runwayml/stable-diffusion-v1-5",
    instance_data_dir="./my_subject",
    instance_prompt="a photo of sks person",
    class_data_dir="./class_images",
    class_prompt="a photo of a person",
    num_steps=800
)
'''

    print("DREAMBOOTH TRAINING IMPLEMENTATION")
    print(code)


train_dreambooth_simplified()
```

---

## 3. Textual Inversion - Learning New Concepts

### Embedding Optimization

```python
"""
TEXTUAL INVERSION (Gal et al., 2022)

Key Idea: Learn a new token embedding for your concept

Process:
1. Add new token to vocabulary: "<my-concept>"
2. Freeze entire model
3. Only optimize the new token embedding
4. Result: Lightweight (few KB) concept representation!

Training:
  Optimize embedding e* such that:
  L = E[||ε - ε_θ(z_t, c_θ("<my-concept>"))||²]

Advantages:
✅ Tiny file size (~10KB vs 2GB for DreamBooth)
✅ Easy to share and combine
✅ No model architecture changes

Disadvantages:
❌ Less flexible than DreamBooth
❌ Harder to capture complex subjects
"""


def textual_inversion_concept():
    """
    Explain Textual Inversion concept.
    """
    print("TEXTUAL INVERSION\n")

    print("What gets trained:")
    print("  DreamBooth: Entire U-Net (~3GB)")
    print("  Textual Inversion: One embedding vector (~10KB)")
    print()

    print("Process:")
    print("  1. Add new token '<my-cat>' to vocabulary")
    print("  2. Initialize embedding (copy similar word like 'cat')")
    print("  3. Train: Show images with prompt 'a photo of <my-cat>'")
    print("  4. Optimize only the embedding for '<my-cat>'")
    print("  5. Save: Just the embedding (tiny file!)")
    print()

    print("Usage:")
    print("  Load embedding: model.load_textual_inversion('my-cat.pt')")
    print("  Generate: 'a photo of <my-cat> wearing sunglasses'")
    print()

    print("Perfect for:")
    print("  ✅ Sharing concepts (small file size)")
    print("  ✅ Combining multiple concepts")
    print("  ✅ Learning artistic styles")


textual_inversion_concept()
```

### Training Textual Inversion

```python
def train_textual_inversion():
    """
    Train Textual Inversion embedding.
    """
    code = '''
from diffusers import StableDiffusionPipeline
import torch

def textual_inversion_training(
    model_id="runwayml/stable-diffusion-v1-5",
    concept_images_dir="./my_concept",
    placeholder_token="<my-concept>",
    initializer_token="cat",  # Initialize from similar word
    num_steps=3000,
    lr=5e-4
):
    """
    Train Textual Inversion embedding.

    Args:
        model_id: Base model
        concept_images_dir: Images of your concept
        placeholder_token: New token (e.g., "<my-cat>")
        initializer_token: Word to initialize from
        num_steps: Training steps
        lr: Learning rate
    """
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(model_id)

    # Add new token to tokenizer
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Add token
    num_added_tokens = tokenizer.add_tokens(placeholder_token)
    if num_added_tokens == 0:
        raise ValueError(f"Token {placeholder_token} already exists")

    # Resize token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Get token id
    token_id = tokenizer.convert_tokens_to_ids(placeholder_token)

    # Initialize from similar word
    with torch.no_grad():
        # Get embedding of initializer token
        init_ids = tokenizer.encode(initializer_token, add_special_tokens=False)
        init_embedding = text_encoder.get_input_embeddings().weight[init_ids[0]]

        # Copy to new token
        text_encoder.get_input_embeddings().weight[token_id] = init_embedding

    # Freeze everything except new embedding
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Only optimize new token embedding
    embedding_layer = text_encoder.get_input_embeddings()
    embedding_layer.weight.requires_grad = True

    # Create optimizer for just this embedding
    optimizer = torch.optim.AdamW(
        [embedding_layer.weight[token_id]],
        lr=lr
    )

    # Training loop (simplified)
    for step in range(num_steps):
        # Load image
        # Encode with prompt f"a photo of {placeholder_token}"
        # Compute loss
        # Optimize

        if step % 500 == 0:
            print(f"Step {step}/{num_steps}")

    # Save embedding
    learned_embeds = embedding_layer.weight[token_id].detach().cpu()
    torch.save(
        {"embedding": learned_embeds, "token": placeholder_token},
        f"{placeholder_token}.pt"
    )

    print(f"Saved embedding to {placeholder_token}.pt")


# Example
textual_inversion_training(
    concept_images_dir="./my_style_images",
    placeholder_token="<my-art-style>",
    initializer_token="art",
    num_steps=3000
)
'''

    print("TEXTUAL INVERSION TRAINING")
    print(code)
    print()
    print("Tips:")
    print("  - Use 10-20 images for best results")
    print("  - Initialize from semantically similar word")
    print("  - Train for 2000-5000 steps")
    print("  - Learning rate: 5e-4 to 1e-3")


train_textual_inversion()
```

---

## 4. LoRA - Low-Rank Adaptation

### Efficient Fine-Tuning

```python
"""
LoRA (Hu et al., 2021 - adapted for Stable Diffusion)

Key Idea: Add small trainable matrices to existing layers

For weight matrix W:
  W_new = W + ΔW
  where ΔW = A * B^T

  A: [d, r]  (r << d, e.g., r=4)
  B: [r, d]

Parameters:
  Full fine-tuning: d × d
  LoRA: 2 × d × r  (much smaller!)

Benefits:
✅ ~1000x fewer parameters than full fine-tuning
✅ Fast training (can train on consumer GPU)
✅ Easy to merge and switch
✅ Preserve base model quality

File sizes:
  Full model: ~4GB
  LoRA: ~3-10MB!
"""


def lora_architecture_explained():
    """
    Explain LoRA architecture.
    """
    print("LoRA ARCHITECTURE\n")

    # Example dimensions
    d = 1024  # Original dimension
    r = 4     # LoRA rank

    full_params = d * d
    lora_params = 2 * d * r

    print(f"Example: Adapting a {d}×{d} weight matrix\n")

    print(f"Full fine-tuning:")
    print(f"  Parameters: {d} × {d} = {full_params:,}")
    print(f"  Storage: {full_params * 4 / 1e6:.1f} MB (float32)")
    print()

    print(f"LoRA (rank {r}):")
    print(f"  Matrix A: {d} × {r} = {d * r:,}")
    print(f"  Matrix B: {r} × {d} = {r * d:,}")
    print(f"  Total: {lora_params:,}")
    print(f"  Storage: {lora_params * 4 / 1e6:.2f} MB")
    print()

    print(f"Reduction: {full_params / lora_params:.0f}x fewer parameters!")
    print()

    print("How it works:")
    print("  Original: y = W @ x")
    print("  With LoRA: y = W @ x + (B^T @ (A @ x))")
    print("               ↑         ↑")
    print("            frozen    trainable")


lora_architecture_explained()
```

### Training LoRA

```python
def train_lora_adapter():
    """
    Train LoRA adapter for Stable Diffusion.
    """
    code = '''
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model
import torch

def train_lora(
    model_id="runwayml/stable-diffusion-v1-5",
    dataset_dir="./training_images",
    output_dir="./lora_output",
    rank=4,
    num_steps=1000,
    lr=1e-4
):
    """
    Train LoRA adapter.

    Args:
        model_id: Base model
        dataset_dir: Training images
        output_dir: Where to save LoRA weights
        rank: LoRA rank (4, 8, or 16 typical)
        num_steps: Training steps
        lr: Learning rate
    """
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16
    ).to("cuda")

    # Configure LoRA
    lora_config = LoraConfig(
        r=rank,  # Rank
        lora_alpha=rank,  # Scaling factor (usually same as r)
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0"  # Attention layers
        ],
        lora_dropout=0.0,
        bias="none",
    )

    # Apply LoRA to U-Net
    pipe.unet = get_peft_model(pipe.unet, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in pipe.unet.parameters())

    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    print(f"Percentage: {100 * trainable_params / total_params:.2f}%")

    # Optimizer (only LoRA parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, pipe.unet.parameters()),
        lr=lr
    )

    # Training loop (simplified)
    pipe.unet.train()

    for step in range(num_steps):
        # Load batch
        # Forward pass
        # Compute loss
        # Backward
        optimizer.step()
        optimizer.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}/{num_steps}")

    # Save LoRA weights only
    pipe.unet.save_pretrained(output_dir)

    print(f"LoRA weights saved to {output_dir}")


# Example usage
train_lora(
    dataset_dir="./my_style_images",
    output_dir="./my_lora",
    rank=8,  # Higher rank = more capacity
    num_steps=1000
)
'''

    print("LORA TRAINING")
    print(code)
    print()
    print("Rank selection:")
    print("  rank=4: Fastest, smallest, good for simple styles")
    print("  rank=8: Balanced (recommended)")
    print("  rank=16: More capacity, larger file")
    print("  rank=32+: Approaching full fine-tuning")


train_lora_adapter()
```

### Using Trained LoRA

```python
def use_lora_adapter():
    """
    Load and use LoRA adapter.
    """
    code = '''
from diffusers import StableDiffusionPipeline

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("./my_lora")

# Generate with LoRA applied
prompt = "a beautiful landscape in my style"
image = pipe(prompt, num_inference_steps=30).images[0]

image.save("lora_output.png")

# Can also adjust LoRA strength
pipe.set_lora_scale(0.8)  # 0.0 = off, 1.0 = full strength

# Or unload LoRA
pipe.unload_lora_weights()
'''

    print("USING LORA")
    print(code)
    print()
    print("Multiple LoRAs:")
    print("  - Load multiple LoRAs sequentially")
    print("  - Adjust individual scales")
    print("  - Combine different styles/subjects!")


use_lora_adapter()
```

---

## 5. Comparing Fine-Tuning Methods

### DreamBooth vs Textual Inversion vs LoRA

```python
def compare_fine_tuning_methods():
    """
    Compare different personalization methods.
    """
    methods = {
        "DreamBooth": {
            "trains": "Entire U-Net",
            "size": "~3-4 GB",
            "quality": "Excellent",
            "flexibility": "Very high",
            "training_time": "1-2 hours",
            "gpu_memory": "16GB+",
            "best_for": "Complex subjects, high fidelity",
            "difficulty": "Medium"
        },
        "Textual Inversion": {
            "trains": "One embedding",
            "size": "~10 KB",
            "quality": "Good",
            "flexibility": "Limited",
            "training_time": "30-60 min",
            "gpu_memory": "8GB+",
            "best_for": "Simple concepts, styles",
            "difficulty": "Easy"
        },
        "LoRA": {
            "trains": "Low-rank adapters",
            "size": "~3-10 MB",
            "quality": "Excellent",
            "flexibility": "High",
            "training_time": "30-60 min",
            "gpu_memory": "12GB+",
            "best_for": "Styles, characters, balanced",
            "difficulty": "Easy"
        },
    }

    print("FINE-TUNING METHODS COMPARISON\n")

    for method, info in methods.items():
        print(f"{method}:")
        for key, value in info.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print()

    print("RECOMMENDATIONS:")
    print("  🎯 Best quality: DreamBooth")
    print("  💾 Smallest size: Textual Inversion")
    print("  ⚖️  Best balance: LoRA (recommended!)")
    print("  🚀 Fastest training: Textual Inversion")
    print("  💪 Most flexible: DreamBooth")


compare_fine_tuning_methods()
```

---

## 6. Advanced: Model Merging

### Combining Multiple Fine-Tuned Models

```python
"""
MODEL MERGING

Combine multiple fine-tuned models!

Methods:
1. Weighted Sum: W = α*W1 + (1-α)*W2
2. Add Difference: W = W_base + α*(W_ft - W_base)
3. SLERP: Spherical interpolation

Use cases:
✅ Merge different styles
✅ Combine subjects
✅ Balance capabilities
"""


def merge_models_example():
    """
    Merge two Stable Diffusion models.
    """
    code = '''
import torch
from safetensors.torch import load_file, save_file

def merge_models(model_a_path, model_b_path, output_path, alpha=0.5):
    """
    Merge two models with weighted sum.

    Args:
        model_a_path: Path to first model
        model_b_path: Path to second model
        output_path: Where to save merged model
        alpha: Weight for model A (1-alpha for B)
    """
    # Load state dicts
    state_a = load_file(model_a_path)
    state_b = load_file(model_b_path)

    # Merge
    merged = {}

    for key in state_a.keys():
        merged[key] = alpha * state_a[key] + (1 - alpha) * state_b[key]

    # Save
    save_file(merged, output_path)

    print(f"Merged model saved to {output_path}")
    print(f"Ratio: {alpha:.0%} Model A + {(1-alpha):.0%} Model B")


# Example: Merge two styles
merge_models(
    model_a_path="./anime_style/unet.safetensors",
    model_b_path="./realistic_style/unet.safetensors",
    output_path="./merged_style/unet.safetensors",
    alpha=0.6  # 60% anime, 40% realistic
)
'''

    print("MODEL MERGING")
    print(code)
    print()
    print("Merging strategies:")
    print("  alpha=0.5: Equal mix")
    print("  alpha=0.7: Emphasize first model")
    print("  alpha=0.3: Emphasize second model")


merge_models_example()
```

### Merging LoRAs

```python
def merge_loras():
    """
    Merge multiple LoRA adapters.
    """
    code = '''
from diffusers import StableDiffusionPipeline

# Load base model
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load and merge multiple LoRAs
loras = [
    ("./character_lora", 1.0),  # (path, weight)
    ("./style_lora", 0.7),
    ("./lighting_lora", 0.5),
]

for lora_path, weight in loras:
    pipe.load_lora_weights(lora_path)
    pipe.fuse_lora(lora_scale=weight)  # Merge with weight

# Generate with all LoRAs combined
prompt = "a character portrait with dramatic lighting"
image = pipe(prompt).images[0]
'''

    print("MERGING MULTIPLE LORAS")
    print(code)
    print()
    print("Benefits:")
    print("  ✅ Combine character + style + technique")
    print("  ✅ Adjust individual strengths")
    print("  ✅ Create complex compositions")


merge_loras()
```

---

## 7. Preventing Overfitting

### Regularization Techniques

```python
"""
OVERFITTING IN PERSONALIZATION

Signs of overfitting:
❌ Model only generates training images
❌ Can't generalize to new scenes
❌ Low diversity in outputs

Solutions:
1. Prior preservation (DreamBooth)
2. Regularization losses
3. Data augmentation
4. Early stopping
5. Lower learning rate
"""


def prevent_overfitting():
    """
    Techniques to prevent overfitting.
    """
    print("PREVENTING OVERFITTING\n")

    techniques = {
        "Prior Preservation": {
            "method": "Train on class images alongside subject",
            "implementation": "Generate 200+ class images, add to training",
            "effectiveness": "Very effective for DreamBooth"
        },
        "Data Augmentation": {
            "method": "Augment training images",
            "implementation": "Random crops, flips, color jitter",
            "effectiveness": "Moderate improvement"
        },
        "Early Stopping": {
            "method": "Stop before full convergence",
            "implementation": "Monitor validation loss, stop when diverges",
            "effectiveness": "Critical for all methods"
        },
        "Lower Rank (LoRA)": {
            "method": "Use smaller LoRA rank",
            "implementation": "rank=4 instead of rank=16",
            "effectiveness": "Prevents memorization"
        },
        "Regularization": {
            "method": "Weight decay, dropout",
            "implementation": "weight_decay=0.01 in optimizer",
            "effectiveness": "Helps prevent overfitting"
        },
    }

    for technique, info in techniques.items():
        print(f"{technique}:")
        for key, value in info.items():
            print(f"  {key.title()}: {value}")
        print()

    print("Best practices:")
    print("  ✅ Use 10-20 diverse images (not just 3)")
    print("  ✅ Enable prior preservation")
    print("  ✅ Monitor outputs during training")
    print("  ✅ Stop early if outputs look identical")


prevent_overfitting()
```

---

## 8. Production Fine-Tuning Pipeline

### End-to-End Workflow

```python
class PersonalizationPipeline:
    """
    Production-ready personalization pipeline.
    """
    def __init__(self, base_model="runwayml/stable-diffusion-v1-5",
                 method="lora"):
        """
        Initialize personalization pipeline.

        Args:
            base_model: Base Stable Diffusion model
            method: "dreambooth", "textual_inversion", or "lora"
        """
        self.base_model = base_model
        self.method = method

        print(f"Initialized {method} pipeline with {base_model}")

    def prepare_dataset(self, image_dir, output_dir,
                       min_images=10, max_images=50):
        """
        Prepare and validate training dataset.

        Args:
            image_dir: Directory with training images
            output_dir: Where to save processed images
            min_images: Minimum required images
            max_images: Maximum images to use

        Returns:
            Number of images prepared
        """
        # Check number of images
        import os
        images = [f for f in os.listdir(image_dir)
                 if f.endswith(('.jpg', '.png', '.jpeg'))]

        if len(images) < min_images:
            raise ValueError(f"Need at least {min_images} images, found {len(images)}")

        if len(images) > max_images:
            print(f"Using first {max_images} of {len(images)} images")
            images = images[:max_images]

        print(f"Dataset prepared: {len(images)} images")

        # Would process images here (resize, validate, etc.)

        return len(images)

    def train(self, dataset_dir, output_dir, **training_args):
        """
        Train personalization model.

        Args:
            dataset_dir: Training data directory
            output_dir: Where to save model
            **training_args: Method-specific parameters
        """
        # Validate dataset
        num_images = self.prepare_dataset(dataset_dir, "./processed")

        # Set default parameters based on method
        if self.method == "lora":
            defaults = {
                "rank": 8,
                "learning_rate": 1e-4,
                "num_steps": 1000,
            }
        elif self.method == "dreambooth":
            defaults = {
                "learning_rate": 5e-6,
                "num_steps": 800,
                "prior_preservation": True,
            }
        else:  # textual_inversion
            defaults = {
                "learning_rate": 5e-4,
                "num_steps": 3000,
            }

        # Merge defaults with user args
        config = {**defaults, **training_args}

        print(f"Training with config: {config}")

        # Would train here
        print("Training started...")

    def validate(self, checkpoint_dir, validation_prompts):
        """
        Validate trained model.

        Args:
            checkpoint_dir: Directory with trained weights
            validation_prompts: List of prompts to test

        Returns:
            Validation results
        """
        print(f"Validating with {len(validation_prompts)} prompts...")

        results = {}

        for prompt in validation_prompts:
            # Would generate and evaluate
            results[prompt] = {"quality": "good"}

        return results


# Example usage
pipeline = PersonalizationPipeline(method="lora")

# Prepare dataset
pipeline.prepare_dataset(
    image_dir="./my_subject_photos",
    output_dir="./processed"
)

# Train
pipeline.train(
    dataset_dir="./processed",
    output_dir="./my_lora",
    rank=8,
    num_steps=1000,
    learning_rate=1e-4
)

# Validate
prompts = [
    "a photo of subject in Paris",
    "subject wearing sunglasses",
    "subject as a superhero"
]

results = pipeline.validate("./my_lora", prompts)
```

---

## Practice Exercises

### Exercise 1: Train LoRA for Custom Style

```python
"""
Train LoRA on artistic style.

Tasks:
1. Collect 20-30 images of consistent art style
2. Train LoRA with different ranks (4, 8, 16)
3. Compare results
4. Test with various prompts
5. Experiment with LoRA scales (0.5, 0.8, 1.0)

Report: Which rank gives best style transfer?
"""

# Your implementation here
```

### Exercise 2: DreamBooth with Prior Preservation

```python
"""
Implement DreamBooth with proper regularization.

Requirements:
1. Choose subject (person, pet, or object)
2. Collect 5-10 diverse photos
3. Generate 200 prior preservation images
4. Train with both instance and class losses
5. Validate: Can it generalize to new scenes?

Measure: Diversity vs fidelity trade-off
"""

# Your implementation here
```

### Exercise 3: Model Merging Exploration

```python
"""
Merge two fine-tuned models.

Tasks:
1. Train two LoRAs (different styles/subjects)
2. Merge with different alphas: [0.2, 0.4, 0.6, 0.8]
3. Generate images with each merged model
4. Analyze: How does alpha affect output?

Goal: Find optimal merge ratio for your use case
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **DreamBooth** 🎯
   - Few-shot learning (3-5 images)
   - Fine-tune entire U-Net
   - Prior preservation prevents forgetting
   - Best quality but largest files

2. **Textual Inversion** 💡
   - Learn new token embedding
   - Tiny file size (~10KB)
   - Easy to share and combine
   - Limited flexibility

3. **LoRA** 🚀
   - Low-rank adapter matrices
   - ~1000x fewer parameters
   - Best quality/efficiency trade-off
   - Most recommended method!

4. **Model Merging** 🔀
   - Combine multiple models
   - Weighted interpolation
   - Creative combinations
   - LoRAs especially easy to merge

5. **Overfitting Prevention** ⚠️
   - Prior preservation critical
   - Use diverse training data
   - Early stopping
   - Monitor validation outputs

6. **Production Pipeline** 💼
   - Dataset validation
   - Automated training
   - Quality checks
   - Version control

### Method Comparison

| Method | File Size | Training Time | Quality | Flexibility | Best For |
|--------|-----------|---------------|---------|-------------|----------|
| **DreamBooth** | ~4GB | 1-2 hrs | Excellent | High | Complex subjects |
| **Textual Inv** | ~10KB | 30-60 min | Good | Limited | Simple concepts |
| **LoRA** | ~10MB | 30-60 min | Excellent | High | **Recommended!** |

### What's Next?

In Lesson 6, we'll cover **Production, Ethics, and Evaluation**:
- Inference optimization and deployment
- Safety filters and content moderation
- Watermarking and provenance
- Evaluation metrics (FID, IS, CLIPScore)
- Responsible AI considerations

---

## Additional Resources

### Papers

- Ruiz et al. (2022): "DreamBooth: Fine Tuning Text-to-Image Diffusion Models"
- Gal et al. (2022): "An Image is Worth One Word: Personalizing Text-to-Image Generation"
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"

### Tools & Libraries

- **Diffusers**: https://huggingface.co/docs/diffusers/training/overview
- **PEFT**: https://github.com/huggingface/peft (for LoRA)
- **Kohya_ss**: Popular training GUI
- **Automatic1111**: WebUI with training support

### Pre-trained Models

- **Civitai**: https://civitai.com/ (community LoRAs and models)
- **Hugging Face**: https://huggingface.co/models?pipeline_tag=text-to-image

---

**Next**: [Lesson 6 - Production Ethics and Evaluation](Lesson%206%20-%20Production%20Ethics%20and%20Evaluation.md)

Master **deployment, safety, and responsible AI** for generative models! ⚖️

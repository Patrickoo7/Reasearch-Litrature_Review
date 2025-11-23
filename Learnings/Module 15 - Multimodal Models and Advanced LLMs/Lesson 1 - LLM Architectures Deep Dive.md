# Lesson 1: LLM Architectures Deep Dive - GPT, LLaMA, Mistral, and Mixtral 🏗️

**Module 15: Multimodal Models and Advanced LLMs | Lesson 1 of 7**

Master the architectural innovations that power modern large language models from GPT to Mixtral MoE!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the evolution of LLM architectures from GPT-2 to GPT-4
2. ✅ Implement and analyze LLaMA's architectural innovations
3. ✅ Master Mistral's sliding window attention mechanism
4. ✅ Understand Mixtral's Mixture of Experts (MoE) architecture
5. ✅ Apply grouped-query attention (GQA) for efficient inference
6. ✅ Understand scaling laws and optimal compute allocation
7. ✅ Deep dive into tokenization strategies (BPE, WordPiece, SentencePiece)
8. ✅ Compare positional encodings: Absolute, RoPE, ALiBi

---

## Prerequisites

- **Required**: Module 7 (NLP and Transformers basics)
- **Required**: Understanding of attention mechanisms
- **Helpful**: Module 12 Lesson 9 (RLHF)
- **Libraries**: `transformers`, `torch`, `tiktoken`, `sentencepiece`

```bash
pip install transformers torch tiktoken sentencepiece accelerate bitsandbytes
```

---

## 1. Evolution of LLM Architectures

### The Transformer Foundation

Modern LLMs build on the transformer architecture, but with critical innovations:

```python
import torch
import torch.nn as nn
import math
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Basic Transformer Block (GPT-2 style)
class TransformerBlock(nn.Module):
    """
    Standard transformer block used in GPT-2.

    Components:
    - Multi-head self-attention
    - Layer normalization
    - Feed-forward network
    - Residual connections
    """
    def __init__(self, d_model=768, n_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()

        # Pre-normalization (GPT-2 style)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(self.ln2(x))
        return x


# Compare model sizes across generations
models_evolution = {
    'GPT-2': {'params': '1.5B', 'layers': 48, 'd_model': 1600, 'context': 1024},
    'GPT-3': {'params': '175B', 'layers': 96, 'd_model': 12288, 'context': 2048},
    'GPT-3.5': {'params': '~175B', 'layers': 96, 'd_model': 12288, 'context': 4096},
    'GPT-4': {'params': '~1.8T (MoE)', 'layers': '?', 'd_model': '?', 'context': 32768},
    'LLaMA': {'params': '7-65B', 'layers': '32-80', 'd_model': '4096-8192', 'context': 2048},
    'LLaMA 2': {'params': '7-70B', 'layers': '32-80', 'd_model': '4096-8192', 'context': 4096},
    'Mistral 7B': {'params': '7.3B', 'layers': 32, 'd_model': 4096, 'context': 32768},
    'Mixtral 8x7B': {'params': '46.7B (8 experts)', 'layers': 32, 'd_model': 4096, 'context': 32768},
}

import pandas as pd
df = pd.DataFrame(models_evolution).T
print("Evolution of LLM Architectures:")
print(df)
```

**Key Architectural Evolution:**
- **GPT-2 (2019)**: Standard transformer decoder
- **GPT-3 (2020)**: Massive scale (175B), few-shot learning
- **LLaMA (2023)**: Efficiency innovations, open-source
- **Mistral (2023)**: Sliding window attention
- **Mixtral (2023)**: Sparse MoE for efficiency

---

## 2. LLaMA Architecture Innovations

### Key Innovations in LLaMA

LLaMA introduced several architectural improvements:

1. **Pre-normalization** (RMSNorm instead of LayerNorm)
2. **SwiGLU activation** (instead of ReLU/GELU)
3. **Rotary Positional Embeddings (RoPE)**
4. **Grouped-Query Attention (GQA)** in LLaMA 2

```python
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (LLaMA).

    More efficient than LayerNorm:
    - No mean centering (no bias term)
    - Only rescale by RMS
    - ~10-20% faster than LayerNorm
    """
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        return self.weight * x / rms


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (LLaMA).

    SwiGLU(x) = Swish(xW) ⊙ (xV)
    where Swish(x) = x * sigmoid(x)

    Better performance than GELU for LLMs.
    """
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        # SwiGLU = Swish(xW) * (xV)
        swish_out = F.silu(self.w(x))  # silu = swish
        gated = swish_out * self.v(x)
        return self.w2(gated)


# Example: Compare activations
x = torch.randn(1, 512, 768)

gelu_out = F.gelu(x)
swish_out = F.silu(x)

print(f"GELU output range: [{gelu_out.min():.3f}, {gelu_out.max():.3f}]")
print(f"Swish output range: [{swish_out.min():.3f}, {swish_out.max():.3f}]")
print(f"Swish has smoother gradients and better performance in LLMs")
```

### Rotary Positional Embeddings (RoPE)

RoPE is a key innovation for handling longer contexts:

```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    """
    Precompute rotation frequencies for RoPE.

    Args:
        dim: Embedding dimension (must be even)
        end: Maximum sequence length
        theta: Base for frequency calculation

    Returns:
        Complex numbers representing rotation matrices
    """
    # Frequency for each dimension pair
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))

    # Position indices
    t = torch.arange(end, device=freqs.device)

    # Outer product: (seq_len, dim/2)
    freqs = torch.outer(t, freqs).float()

    # Convert to complex numbers for rotation
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(xq, xk, freqs_cis):
    """
    Apply rotary embeddings to queries and keys.

    This encodes positional information through rotation,
    allowing the model to handle longer sequences at inference.
    """
    # Reshape to complex numbers
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # Reshape freqs to match
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)

    # Apply rotation
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(-2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(-2)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# Example: RoPE in action
batch_size, seq_len, n_heads, head_dim = 2, 128, 8, 64

# Query and key
q = torch.randn(batch_size, seq_len, n_heads, head_dim)
k = torch.randn(batch_size, seq_len, n_heads, head_dim)

# Precompute frequencies
freqs_cis = precompute_freqs_cis(head_dim, seq_len)

# Apply RoPE
q_rot, k_rot = apply_rotary_emb(q, k, freqs_cis)

print(f"Original Q shape: {q.shape}")
print(f"Rotated Q shape: {q_rot.shape}")
print("RoPE encodes position through rotation - no learned parameters!")
print("Advantage: Can extrapolate to longer sequences than seen in training")
```

### Grouped-Query Attention (GQA)

GQA reduces memory and computation by sharing KV across query heads:

```python
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) from LLaMA 2.

    Key innovation:
    - Multiple query heads share the same key/value heads
    - Reduces KV cache size during inference
    - Maintains most of multi-head attention's benefits

    Example:
        n_heads = 32 (query heads)
        n_kv_heads = 8 (key/value heads)
        Each KV head is shared by 4 query heads
    """
    def __init__(self, d_model=4096, n_heads=32, n_kv_heads=8, dropout=0.0):
        super().__init__()

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads  # Repetition factor

        self.head_dim = d_model // n_heads

        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.dropout = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Repeat KV heads to match Q heads
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        output = self.o_proj(attn_output)

        return output


# Example: Memory savings with GQA
def calculate_kv_cache_size(n_layers, n_heads, head_dim, seq_len, batch_size=1):
    """Calculate KV cache size in GB."""
    # Each element is float16 (2 bytes)
    bytes_per_element = 2

    # KV cache: 2 (K and V) * n_layers * n_heads * seq_len * head_dim
    total_elements = 2 * n_layers * n_heads * seq_len * head_dim * batch_size
    size_gb = (total_elements * bytes_per_element) / (1024 ** 3)
    return size_gb

# LLaMA 2 7B configuration
n_layers = 32
seq_len = 4096

# Multi-Head Attention (MHA)
mha_size = calculate_kv_cache_size(n_layers, n_heads=32, head_dim=128, seq_len=seq_len)

# Grouped-Query Attention (GQA)
gqa_size = calculate_kv_cache_size(n_layers, n_heads=8, head_dim=128, seq_len=seq_len)

# Multi-Query Attention (MQA) - extreme case
mqa_size = calculate_kv_cache_size(n_layers, n_heads=1, head_dim=128, seq_len=seq_len)

print(f"KV Cache Size for {seq_len} tokens:")
print(f"  MHA (32 heads):  {mha_size:.2f} GB")
print(f"  GQA (8 heads):   {gqa_size:.2f} GB  (4x smaller)")
print(f"  MQA (1 head):    {mqa_size:.2f} GB  (32x smaller)")
print(f"\nGQA provides the sweet spot: significant memory savings with minimal quality loss")
```

---

## 3. Mistral 7B: Sliding Window Attention

Mistral's key innovation is **sliding window attention** for efficient long-context modeling:

```python
class SlidingWindowAttention(nn.Module):
    """
    Sliding Window Attention (Mistral 7B).

    Key innovation:
    - Each token only attends to W tokens in the past
    - W = window size (e.g., 4096)
    - Enables 32K context with only 4K window
    - Through stacking, effective receptive field grows

    Advantages:
    - Linear memory growth (not quadratic)
    - Faster inference
    - Can handle very long sequences
    """
    def __init__(self, d_model=4096, n_heads=32, window_size=4096, dropout=0.0):
        super().__init__()

        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.shape

        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Create sliding window mask
        # Only attend to W previous tokens
        if seq_len > self.window_size:
            window_mask = self._create_sliding_window_mask(seq_len)
            scores = scores.masked_fill(window_mask == 0, float('-inf'))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        output = self.o_proj(attn_output)

        return output

    def _create_sliding_window_mask(self, seq_len):
        """Create sliding window attention mask."""
        mask = torch.ones(seq_len, seq_len)
        for i in range(seq_len):
            # Token i can attend to tokens [max(0, i-W), i]
            start = max(0, i - self.window_size)
            mask[i, :start] = 0
        return mask.unsqueeze(0).unsqueeze(0)


# Visualize sliding window attention
import matplotlib.pyplot as plt
import numpy as np

def visualize_attention_patterns():
    """Compare full attention vs sliding window."""
    seq_len = 64
    window_size = 16

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Full attention
    full_mask = torch.tril(torch.ones(seq_len, seq_len))
    axes[0].imshow(full_mask, cmap='Blues', aspect='auto')
    axes[0].set_title('Full Causal Attention\n(Quadratic complexity)')
    axes[0].set_xlabel('Key position')
    axes[0].set_ylabel('Query position')

    # Sliding window attention
    sliding_mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - window_size)
        sliding_mask[i, start:i+1] = 1
    axes[1].imshow(sliding_mask, cmap='Greens', aspect='auto')
    axes[1].set_title(f'Sliding Window Attention (W={window_size})\n(Linear complexity)')
    axes[1].set_xlabel('Key position')
    axes[1].set_ylabel('Query position')

    # Effective receptive field through stacking
    # After L layers, token can see L * W tokens back
    n_layers = 4
    effective_window = window_size * n_layers
    effective_mask = torch.zeros(seq_len, seq_len)
    for i in range(seq_len):
        start = max(0, i - effective_window)
        effective_mask[i, start:i+1] = 1
    axes[2].imshow(effective_mask, cmap='Oranges', aspect='auto')
    axes[2].set_title(f'Effective Field (L={n_layers} layers)\n(Window = {effective_window})')
    axes[2].set_xlabel('Key position')
    axes[2].set_ylabel('Query position')

    plt.tight_layout()
    plt.savefig('/tmp/sliding_window_attention.png', dpi=150, bbox_inches='tight')
    print("Visualization saved to /tmp/sliding_window_attention.png")

visualize_attention_patterns()

# Memory comparison
def compare_attention_memory(seq_len, window_size):
    """Compare memory usage."""
    full_attn_elements = seq_len * seq_len
    sliding_attn_elements = seq_len * window_size

    reduction = full_attn_elements / sliding_attn_elements

    print(f"\nMemory Comparison (seq_len={seq_len}, window={window_size}):")
    print(f"  Full attention:    {full_attn_elements:,} elements")
    print(f"  Sliding window:    {sliding_attn_elements:,} elements")
    print(f"  Memory reduction:  {reduction:.1f}x")

compare_attention_memory(seq_len=32768, window_size=4096)
```

---

## 4. Mixtral 8x7B: Mixture of Experts

Mixtral uses **Sparse Mixture of Experts (SMoE)** for efficient scaling:

```python
class MixtureOfExpertsLayer(nn.Module):
    """
    Sparse Mixture of Experts (Mixtral 8x7B).

    Architecture:
    - 8 expert FFN networks
    - Router selects top-2 experts per token
    - Only 2/8 experts activated per token
    - Effective params: 46.7B, active: 12.9B

    Benefits:
    - Higher capacity without proportional compute
    - Better performance than dense models
    - 6x faster than GPT-3.5 equivalent
    """
    def __init__(self, d_model=4096, d_ff=14336, num_experts=8, top_k=2):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # Router: learns which experts to use
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert networks (FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff, bias=False),
                nn.SiLU(),
                nn.Linear(d_ff, d_model, bias=False)
            )
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        # Flatten for routing
        x_flat = x.view(-1, d_model)  # (batch * seq_len, d_model)

        # Router: compute expert weights
        router_logits = self.router(x_flat)  # (batch * seq_len, num_experts)
        router_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)

        # Normalize top-k weights to sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process each expert
        for i in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)

            if expert_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_mask]

                # Apply expert
                expert_output = self.experts[i](expert_input)

                # Get weights for this expert
                expert_weights = top_k_weights[expert_mask]
                expert_weights = expert_weights[top_k_indices[expert_mask] == i].unsqueeze(-1)

                # Add weighted output
                output[expert_mask] += expert_weights * expert_output

        # Reshape back
        output = output.view(batch_size, seq_len, d_model)
        return output


# Simulate MoE routing behavior
def analyze_moe_routing():
    """Analyze how MoE routes tokens to experts."""
    d_model = 256
    num_experts = 8
    top_k = 2

    # Create MoE layer
    moe = MixtureOfExpertsLayer(d_model=d_model, num_experts=num_experts, top_k=top_k)

    # Sample input
    batch_size, seq_len = 2, 16
    x = torch.randn(batch_size, seq_len, d_model)

    # Get router logits
    x_flat = x.view(-1, d_model)
    router_logits = moe.router(x_flat)
    router_probs = F.softmax(router_logits, dim=-1)

    # Analyze routing
    top_k_probs, top_k_experts = torch.topk(router_probs, top_k, dim=-1)

    print("Mixture of Experts Routing Analysis:")
    print(f"  Total tokens: {batch_size * seq_len}")
    print(f"  Experts: {num_experts}")
    print(f"  Top-k: {top_k}")
    print(f"\nExpert utilization (how many tokens use each expert):")

    for i in range(num_experts):
        count = (top_k_experts == i).sum().item()
        percentage = 100 * count / (batch_size * seq_len * top_k)
        print(f"  Expert {i}: {count:2d} selections ({percentage:.1f}%)")

    print(f"\nActive parameters per token:")
    print(f"  Total parameters: {num_experts} experts")
    print(f"  Active per token: {top_k} experts ({100*top_k/num_experts:.1f}%)")
    print(f"  Compute savings: {num_experts/top_k:.1f}x vs dense model")

analyze_moe_routing()


# Calculate Mixtral effective parameters
def mixtral_parameter_count():
    """Calculate Mixtral 8x7B parameter breakdown."""

    config = {
        'n_layers': 32,
        'd_model': 4096,
        'n_heads': 32,
        'd_ff': 14336,
        'num_experts': 8,
        'vocab_size': 32000,
    }

    # Attention parameters (shared, not MoE)
    attn_params = config['n_layers'] * (
        4 * config['d_model'] * config['d_model']  # Q, K, V, O projections
    )

    # MoE FFN parameters
    ffn_per_expert = (
        config['d_model'] * config['d_ff'] +  # Up projection
        config['d_ff'] * config['d_model']     # Down projection
    )
    moe_params = config['n_layers'] * config['num_experts'] * ffn_per_expert

    # Router parameters
    router_params = config['n_layers'] * config['d_model'] * config['num_experts']

    # Embedding
    embed_params = config['vocab_size'] * config['d_model']

    # Total
    total_params = attn_params + moe_params + router_params + embed_params

    # Active params (only 2 experts active)
    active_experts = 2
    active_ffn = config['n_layers'] * active_experts * ffn_per_expert
    active_params = attn_params + active_ffn + router_params + embed_params

    print("Mixtral 8x7B Parameter Breakdown:")
    print(f"  Attention:       {attn_params / 1e9:.2f}B")
    print(f"  MoE FFN:         {moe_params / 1e9:.2f}B")
    print(f"  Router:          {router_params / 1e9:.2f}B")
    print(f"  Embeddings:      {embed_params / 1e9:.2f}B")
    print(f"  ─────────────────────────")
    print(f"  Total params:    {total_params / 1e9:.2f}B")
    print(f"  Active params:   {active_params / 1e9:.2f}B")
    print(f"  Sparsity:        {100 * (1 - active_params/total_params):.1f}%")

mixtral_parameter_count()
```

---

## 5. Scaling Laws and Optimal Compute Allocation

Understanding how to scale models efficiently:

```python
import numpy as np
import matplotlib.pyplot as plt

def chinchilla_scaling_law(compute_budget):
    """
    Chinchilla scaling laws (Hoffmann et al., 2022).

    Key finding: Most LLMs are undertrained!
    Optimal ratio: N (params) ≈ D (tokens) / 20

    Args:
        compute_budget: FLOPs available for training

    Returns:
        optimal_params, optimal_tokens
    """
    # Chinchilla formula (simplified)
    # For compute budget C:
    # N_opt ≈ C^0.5 / k
    # D_opt ≈ 20 * N_opt

    # Constants (approximate)
    k = 1e10

    optimal_params = (compute_budget ** 0.5) / k
    optimal_tokens = 20 * optimal_params

    return optimal_params, optimal_tokens


def analyze_scaling_laws():
    """Analyze scaling laws for different models."""

    models = {
        'GPT-3': {
            'params': 175e9,
            'tokens': 300e9,
            'compute': 3.14e23,
        },
        'Chinchilla': {
            'params': 70e9,
            'tokens': 1.4e12,
            'compute': 5.76e23,
        },
        'LLaMA 1': {
            'params': 65e9,
            'tokens': 1.4e12,
            'compute': 5.4e23,
        },
        'LLaMA 2 70B': {
            'params': 70e9,
            'tokens': 2.0e12,
            'compute': 1.7e24,
        },
    }

    print("Model Training Compute Analysis:")
    print("─" * 70)
    print(f"{'Model':<15} {'Params':<12} {'Tokens':<12} {'Token/Param Ratio':<15}")
    print("─" * 70)

    for name, config in models.items():
        ratio = config['tokens'] / config['params']
        print(f"{name:<15} {config['params']/1e9:>6.0f}B     {config['tokens']/1e9:>8.0f}B     {ratio:>6.1f}x")

    print("\nKey Insights:")
    print("  • GPT-3: Undertrained (1.7x tokens/param vs Chinchilla's 20x)")
    print("  • Chinchilla: Optimally trained (70B model, 1.4T tokens)")
    print("  • LLaMA: Followed Chinchilla's insight (better than GPT-3)")
    print("  • Modern trend: Train longer on more tokens")

analyze_scaling_laws()


def plot_scaling_curves():
    """Visualize scaling laws."""

    # Compute budgets (FLOPs)
    compute_budgets = np.logspace(21, 25, 50)

    optimal_params = []
    optimal_tokens = []

    for C in compute_budgets:
        N, D = chinchilla_scaling_law(C)
        optimal_params.append(N)
        optimal_tokens.append(D)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Optimal parameters vs compute
    axes[0].loglog(compute_budgets, optimal_params, linewidth=2, label='Optimal params')
    axes[0].set_xlabel('Compute Budget (FLOPs)', fontsize=12)
    axes[0].set_ylabel('Model Parameters', fontsize=12)
    axes[0].set_title('Optimal Model Size vs Compute', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Optimal tokens vs parameters
    axes[1].loglog(optimal_params, optimal_tokens, linewidth=2, color='green', label='Optimal tokens')
    axes[1].loglog(optimal_params, np.array(optimal_params) * 20, '--', linewidth=2,
                   color='orange', label='20x line (Chinchilla)')
    axes[1].set_xlabel('Model Parameters', fontsize=12)
    axes[1].set_ylabel('Training Tokens', fontsize=12)
    axes[1].set_title('Optimal Training Tokens vs Model Size', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('/tmp/scaling_laws.png', dpi=150, bbox_inches='tight')
    print("\nScaling laws visualization saved to /tmp/scaling_laws.png")

plot_scaling_curves()
```

---

## 6. Tokenization Deep Dive

Modern LLMs use subword tokenization:

```python
# Install tokenizers
# pip install tiktoken sentencepiece

import tiktoken
from transformers import AutoTokenizer

# GPT-4 tokenizer (BPE with tiktoken)
gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")

# LLaMA tokenizer (SentencePiece)
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Mistral tokenizer
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Test texts
texts = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Reinforcement learning is awesome!",
    "编程很有趣",  # Chinese: "Programming is fun"
    "def factorial(n): return 1 if n == 0 else n * factorial(n-1)",  # Code
]

print("Tokenization Comparison:")
print("=" * 80)

for text in texts:
    gpt4_tokens = gpt4_tokenizer.encode(text)
    llama_tokens = llama_tokenizer.encode(text)
    mistral_tokens = mistral_tokenizer.encode(text)

    print(f"\nText: {text}")
    print(f"  GPT-4:   {len(gpt4_tokens):2d} tokens - {gpt4_tokens}")
    print(f"  LLaMA:   {len(llama_tokens):2d} tokens - {llama_tokens}")
    print(f"  Mistral: {len(mistral_tokens):2d} tokens - {mistral_tokens}")


# Analyze tokenization efficiency
def compare_tokenization_efficiency():
    """Compare tokenizers on different content types."""

    test_cases = {
        'English text': "The quick brown fox jumps over the lazy dog.",
        'Code': "def hello(): print('Hello, world!')",
        'Math': "E = mc², where E is energy, m is mass, and c is speed of light",
        'Chinese': "人工智能正在改变世界",
        'Mixed': "GPT-4 achieves 90% on MMLU benchmark!",
    }

    results = []

    for category, text in test_cases.items():
        gpt4_len = len(gpt4_tokenizer.encode(text))
        llama_len = len(llama_tokenizer.encode(text))
        mistral_len = len(mistral_tokenizer.encode(text))

        results.append({
            'Category': category,
            'GPT-4': gpt4_len,
            'LLaMA': llama_len,
            'Mistral': mistral_len,
        })

    import pandas as pd
    df = pd.DataFrame(results)
    print("\nTokenization Efficiency by Content Type:")
    print(df.to_string(index=False))
    print("\nKey Insights:")
    print("  • GPT-4: Best for multilingual (larger vocabulary)")
    print("  • LLaMA: Optimized for English and code")
    print("  • Mistral: Similar to LLaMA (32K vocab)")

compare_tokenization_efficiency()


# Implement simplified BPE
class SimpleBPE:
    """
    Simplified Byte Pair Encoding (BPE) tokenizer.

    Algorithm:
    1. Start with character-level vocabulary
    2. Find most frequent pair of tokens
    3. Merge this pair into new token
    4. Repeat until desired vocabulary size
    """
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = []

    def get_pairs(self, tokens):
        """Get all adjacent pairs in token sequence."""
        pairs = {}
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            pairs[pair] = pairs.get(pair, 0) + 1
        return pairs

    def train(self, texts):
        """Train BPE on texts."""
        # Start with character-level tokens
        tokens = []
        for text in texts:
            tokens.extend(list(text))

        # Build initial vocabulary
        vocab = {char: i for i, char in enumerate(set(tokens))}
        current_vocab_size = len(vocab)

        print(f"Initial vocabulary size: {current_vocab_size}")

        # Iteratively merge most frequent pairs
        while current_vocab_size < self.vocab_size:
            # Tokenize current texts
            tokenized = [list(text) for text in texts]

            # Get all pairs and their frequencies
            all_pairs = {}
            for token_seq in tokenized:
                pairs = self.get_pairs(token_seq)
                for pair, count in pairs.items():
                    all_pairs[pair] = all_pairs.get(pair, 0) + count

            if not all_pairs:
                break

            # Find most frequent pair
            best_pair = max(all_pairs, key=all_pairs.get)

            # Create new token
            new_token = ''.join(best_pair)
            vocab[new_token] = current_vocab_size
            self.merges.append(best_pair)

            current_vocab_size += 1

            if current_vocab_size % 100 == 0:
                print(f"Vocabulary size: {current_vocab_size}, Latest merge: {best_pair} -> {new_token}")

        self.vocab = vocab
        print(f"Final vocabulary size: {len(vocab)}")

    def encode(self, text):
        """Encode text using learned merges."""
        tokens = list(text)

        # Apply merges in order
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(''.join(pair))
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        # Convert to IDs
        return [self.vocab.get(token, 0) for token in tokens]


# Example: Train simplified BPE
print("\n" + "="*80)
print("Training Simplified BPE Tokenizer")
print("="*80)

training_texts = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the bird sat on the word",
] * 10  # Repeat for more frequency

bpe = SimpleBPE(vocab_size=100)
bpe.train(training_texts)

test_text = "the cat sat"
encoded = bpe.encode(test_text)
print(f"\nEncoded '{test_text}': {encoded}")
```

---

## 7. Positional Encodings Comparison

Different approaches to encoding position information:

```python
import torch
import torch.nn as nn
import math

class AbsolutePositionalEncoding(nn.Module):
    """
    Absolute (sinusoidal) positional encoding.
    Used in original Transformer (Vaswani et al., 2017).

    PE(pos, 2i) = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional embeddings.
    Used in BERT and GPT-2.
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return x + self.pe(positions)


class ALiBi(nn.Module):
    """
    Attention with Linear Biases (ALiBi).
    Used in some recent models.

    Instead of positional embeddings, adds bias to attention scores:
    score(q_i, k_j) = q_i^T k_j - m * |i - j|

    where m is head-specific slope.

    Advantages:
    - No positional embeddings needed
    - Better extrapolation to longer sequences
    - Simpler implementation
    """
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

        # Geometric sequence of slopes
        slopes = torch.tensor([2 ** (-8 * i / n_heads) for i in range(1, n_heads + 1)])
        self.register_buffer('slopes', slopes)

    def get_bias(self, seq_len):
        """Generate ALiBi bias matrix."""
        # Distance matrix
        positions = torch.arange(seq_len).unsqueeze(0)
        distances = positions - positions.T
        distances = torch.abs(distances).float()

        # Apply slopes (one per head)
        bias = -distances.unsqueeze(0) * self.slopes.unsqueeze(-1).unsqueeze(-1)
        return bias

    def forward(self, attention_scores):
        """Add ALiBi bias to attention scores."""
        seq_len = attention_scores.size(-1)
        bias = self.get_bias(seq_len)
        return attention_scores + bias


# Compare positional encodings
def compare_positional_encodings():
    """Visualize different positional encoding schemes."""

    d_model = 64
    max_len = 128
    n_heads = 8

    # Create encodings
    abs_pe = AbsolutePositionalEncoding(d_model, max_len)
    learned_pe = LearnedPositionalEncoding(d_model, max_len)
    alibi = ALiBi(n_heads)

    # Sample input
    x = torch.randn(1, max_len, d_model)

    # Apply encodings
    abs_output = abs_pe(x)
    learned_output = learned_pe(x)

    # For ALiBi, simulate attention scores
    attn_scores = torch.randn(1, n_heads, max_len, max_len)
    alibi_output = alibi(attn_scores)

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Absolute PE
    axes[0, 0].imshow(abs_pe.pe[0, :max_len, :].T, aspect='auto', cmap='coolwarm')
    axes[0, 0].set_title('Absolute (Sinusoidal) PE', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Position')
    axes[0, 0].set_ylabel('Dimension')

    # Learned PE
    with torch.no_grad():
        learned_weights = learned_pe.pe.weight.T
    axes[0, 1].imshow(learned_weights, aspect='auto', cmap='coolwarm')
    axes[0, 1].set_title('Learned PE', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Position')
    axes[0, 1].set_ylabel('Dimension')

    # RoPE (conceptual - rotation in 2D subspaces)
    freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    pos = torch.arange(max_len).float()
    angles = torch.outer(pos, freq)
    rope_vis = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(max_len, -1).T
    axes[1, 0].imshow(rope_vis, aspect='auto', cmap='coolwarm')
    axes[1, 0].set_title('RoPE (Rotary PE)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Position')
    axes[1, 0].set_ylabel('Dimension')

    # ALiBi bias for one head
    alibi_bias = alibi.get_bias(max_len)[0]  # First head
    axes[1, 1].imshow(alibi_bias, aspect='auto', cmap='RdBu_r')
    axes[1, 1].set_title('ALiBi Attention Bias (Head 0)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Key Position')
    axes[1, 1].set_ylabel('Query Position')

    plt.tight_layout()
    plt.savefig('/tmp/positional_encodings.png', dpi=150, bbox_inches='tight')
    print("Positional encodings visualization saved to /tmp/positional_encodings.png")

compare_positional_encodings()


# Comparison table
def print_pe_comparison():
    """Print comparison of positional encoding methods."""

    comparison = {
        'Method': ['Absolute (Sin/Cos)', 'Learned', 'RoPE', 'ALiBi'],
        'Used In': ['Original Transformer', 'BERT, GPT-2', 'LLaMA, Mistral', 'Recent models'],
        'Parameters': ['0 (fixed)', 'max_len × d_model', '0 (computed)', '0 (computed)'],
        'Extrapolation': ['Limited', 'Poor', 'Good', 'Excellent'],
        'Complexity': ['O(n)', 'O(n)', 'O(n)', 'O(n²) attention bias'],
        'Key Advantage': ['No parameters', 'Flexible', 'Rotation property', 'Best extrapolation'],
    }

    import pandas as pd
    df = pd.DataFrame(comparison)

    print("\n" + "="*80)
    print("Positional Encoding Comparison")
    print("="*80)
    print(df.to_string(index=False))
    print("\nTrend: Modern LLMs prefer RoPE or ALiBi for better length extrapolation")

print_pe_comparison()
```

---

## 8. Loading and Analyzing Real Models

Hands-on with actual LLM architectures:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def analyze_model_architecture(model_name):
    """Load and analyze a model's architecture."""

    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}")

    # Load config (no weights)
    config = AutoConfig.from_pretrained(model_name)

    print("\nModel Configuration:")
    print(f"  Architecture type: {config.model_type}")
    print(f"  Number of layers: {config.num_hidden_layers}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Attention heads: {config.num_attention_heads}")

    if hasattr(config, 'num_key_value_heads'):
        print(f"  KV heads (GQA): {config.num_key_value_heads}")
        print(f"  GQA groups: {config.num_attention_heads // config.num_key_value_heads}")

    if hasattr(config, 'sliding_window'):
        print(f"  Sliding window: {config.sliding_window}")

    if hasattr(config, 'num_local_experts'):
        print(f"  Number of experts (MoE): {config.num_local_experts}")
        print(f"  Experts per token: {config.num_experts_per_tok}")

    print(f"  Intermediate size (FFN): {config.intermediate_size}")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Max position embeddings: {config.max_position_embeddings}")

    # Calculate parameters
    def estimate_parameters(config):
        """Rough parameter count estimation."""
        n_layers = config.num_hidden_layers
        d_model = config.hidden_size
        d_ff = config.intermediate_size
        vocab_size = config.vocab_size

        # Attention
        attn_params = n_layers * 4 * d_model * d_model

        # FFN (or MoE)
        if hasattr(config, 'num_local_experts'):
            # MoE
            ffn_params = n_layers * config.num_local_experts * 2 * d_model * d_ff
        else:
            # Standard FFN
            ffn_params = n_layers * 2 * d_model * d_ff

        # Embeddings
        embed_params = vocab_size * d_model

        total = attn_params + ffn_params + embed_params
        return total / 1e9  # Convert to billions

    estimated_params = estimate_parameters(config)
    print(f"\n  Estimated parameters: {estimated_params:.1f}B")


# Analyze different model architectures
models_to_analyze = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    # "mistralai/Mixtral-8x7B-v0.1",  # Uncomment if you have access
]

for model_name in models_to_analyze:
    try:
        analyze_model_architecture(model_name)
    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")


# Load a small model for actual inference
print("\n" + "="*80)
print("Loading Model for Inference")
print("="*80)

model_name = "microsoft/phi-2"  # Smaller model that's easier to load
print(f"Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Test inference
prompt = "The Transformer architecture revolutionized NLP by"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print(f"\nPrompt: {prompt}")
print("Generating...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        top_p=0.9
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nGenerated text:\n{generated_text}")


# Inspect model layers
def inspect_model_layers(model):
    """Inspect model's layer structure."""

    print("\n" + "="*80)
    print("Model Layer Structure")
    print("="*80)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e9:.2f}B")

    print("\nLayer breakdown:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params / 1e6:.2f}M params")
                if params / total_params > 0.01:  # More than 1% of total
                    print(f"    ({100 * params / total_params:.1f}% of total)")

inspect_model_layers(model)
```

---

## 9. Real-World Applications

### Application 1: Model Selection Framework

```python
def select_llm_for_task(
    task_type,
    latency_requirement,
    budget,
    context_length_needed,
    deployment_env
):
    """
    Decision framework for selecting the right LLM.

    Args:
        task_type: 'chat', 'code', 'reasoning', 'summarization', etc.
        latency_requirement: 'real-time', 'interactive', 'batch'
        budget: 'low', 'medium', 'high'
        context_length_needed: int (tokens)
        deployment_env: 'cloud', 'edge', 'mobile'
    """

    recommendations = []

    # Edge deployment
    if deployment_env in ['edge', 'mobile']:
        if budget == 'low':
            recommendations.append({
                'model': 'Phi-2 (2.7B)',
                'reason': 'Smallest with good quality',
                'considerations': 'Quantize to 4-bit for mobile'
            })
        recommendations.append({
            'model': 'Gemma 2B',
            'reason': 'Efficient small model',
            'considerations': 'Good for privacy-critical apps'
        })

    # Cloud deployment
    else:
        # Long context needs
        if context_length_needed > 8192:
            recommendations.append({
                'model': 'Mistral 7B',
                'reason': '32K context with sliding window',
                'considerations': 'Efficient for long documents'
            })
            if budget == 'high':
                recommendations.append({
                    'model': 'GPT-4 Turbo',
                    'reason': '128K context, best quality',
                    'considerations': 'Expensive but most capable'
                })

        # Real-time latency
        if latency_requirement == 'real-time':
            recommendations.append({
                'model': 'Mixtral 8x7B',
                'reason': 'Sparse MoE for speed',
                'considerations': 'Use vLLM for serving'
            })

        # Code tasks
        if task_type == 'code':
            recommendations.append({
                'model': 'CodeLlama 13B',
                'reason': 'Specialized for code',
                'considerations': 'Fine-tuned on code data'
            })

        # General purpose
        if task_type == 'chat' and budget == 'medium':
            recommendations.append({
                'model': 'LLaMA 2 13B',
                'reason': 'Good quality/cost balance',
                'considerations': 'Open-source, customizable'
            })

    print(f"\nModel Selection for:")
    print(f"  Task: {task_type}")
    print(f"  Latency: {latency_requirement}")
    print(f"  Budget: {budget}")
    print(f"  Context: {context_length_needed} tokens")
    print(f"  Environment: {deployment_env}")
    print(f"\nRecommendations:")

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['model']}")
        print(f"   Reason: {rec['reason']}")
        print(f"   Considerations: {rec['considerations']}")


# Example selections
select_llm_for_task(
    task_type='chat',
    latency_requirement='interactive',
    budget='medium',
    context_length_needed=4096,
    deployment_env='cloud'
)

select_llm_for_task(
    task_type='summarization',
    latency_requirement='batch',
    budget='low',
    context_length_needed=16000,
    deployment_env='cloud'
)

select_llm_for_task(
    task_type='chat',
    latency_requirement='real-time',
    budget='low',
    context_length_needed=2048,
    deployment_env='mobile'
)
```

### Application 2: Architecture Analysis Tool

```python
def compare_architectures():
    """Compare different LLM architectures side-by-side."""

    architectures = {
        'GPT-3': {
            'attention': 'Full multi-head',
            'ffn': 'Dense',
            'position': 'Learned',
            'norm': 'LayerNorm',
            'activation': 'GELU',
            'context': 2048,
            'strengths': ['General purpose', 'Well-studied'],
            'weaknesses': ['Expensive', 'Limited context'],
        },
        'LLaMA 2': {
            'attention': 'GQA (grouped-query)',
            'ffn': 'Dense + SwiGLU',
            'position': 'RoPE',
            'norm': 'RMSNorm',
            'activation': 'SwiGLU',
            'context': 4096,
            'strengths': ['Efficient inference', 'Good extrapolation', 'Open-source'],
            'weaknesses': ['Smaller context than Mistral'],
        },
        'Mistral 7B': {
            'attention': 'GQA + Sliding window',
            'ffn': 'Dense + SwiGLU',
            'position': 'RoPE',
            'norm': 'RMSNorm',
            'activation': 'SwiGLU',
            'context': 32768,
            'strengths': ['Long context', 'Efficient', 'Matches larger models'],
            'weaknesses': ['Limited to 7B size'],
        },
        'Mixtral 8x7B': {
            'attention': 'GQA + Sliding window',
            'ffn': 'Sparse MoE (8 experts)',
            'position': 'RoPE',
            'norm': 'RMSNorm',
            'activation': 'SwiGLU',
            'context': 32768,
            'strengths': ['Best quality/speed', 'Sparse computation', 'Long context'],
            'weaknesses': ['Large memory footprint', 'Complex serving'],
        },
    }

    print("="*80)
    print("LLM Architecture Comparison")
    print("="*80)

    for name, specs in architectures.items():
        print(f"\n{name}:")
        print(f"  Attention:   {specs['attention']}")
        print(f"  FFN:         {specs['ffn']}")
        print(f"  Position:    {specs['position']}")
        print(f"  Norm:        {specs['norm']}")
        print(f"  Activation:  {specs['activation']}")
        print(f"  Context:     {specs['context']:,} tokens")
        print(f"  Strengths:   {', '.join(specs['strengths'])}")
        print(f"  Weaknesses:  {', '.join(specs['weaknesses'])}")

compare_architectures()
```

---

## 10. Practice Exercises

### Exercise 1: Implement Multi-Query Attention (MQA)

```python
"""
Exercise: Implement Multi-Query Attention (extreme version of GQA).

In MQA, ALL query heads share a SINGLE key/value head.
This is even more memory-efficient than GQA.

Complete the implementation below:
"""

class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).

    All query heads share the same key and value.
    Maximum memory efficiency.
    """
    def __init__(self, d_model=768, n_heads=12, dropout=0.0):
        super().__init__()

        # TODO: Implement initialization
        # Hint: You need Q projection with n_heads, but K and V with just 1 head
        pass

    def forward(self, x, mask=None):
        # TODO: Implement forward pass
        # Hint: Broadcast single K/V to all query heads
        pass


# Test your implementation
# mqa = MultiQueryAttention(d_model=768, n_heads=12)
# x = torch.randn(2, 128, 768)
# output = mqa(x)
# assert output.shape == x.shape
# print("MQA implementation correct!")
```

### Exercise 2: Analyze Scaling Law Trade-offs

```python
"""
Exercise: Given a fixed compute budget, find optimal model size and training tokens.

Use the Chinchilla scaling laws to determine:
1. Should you train a 7B model or 13B model?
2. How many tokens for each?
3. What's the expected performance?
"""

def optimize_training_budget(compute_budget_flops, options):
    """
    Args:
        compute_budget_flops: Total FLOPs available (e.g., 1e23)
        options: List of model sizes to consider (e.g., [7e9, 13e9, 70e9])

    Returns:
        Best configuration
    """
    # TODO: Implement optimization logic
    # Hint: For each model size, calculate optimal tokens and expected loss
    # Use: compute ≈ 6 * params * tokens (rough approximation)
    pass


# Example usage:
# budget = 1e24  # 1e24 FLOPs
# options = [7e9, 13e9, 70e9]  # 7B, 13B, 70B
# best_config = optimize_training_budget(budget, options)
```

### Exercise 3: Implement Efficient KV Cache

```python
"""
Exercise: Implement KV cache for efficient autoregressive generation.

During generation, we don't need to recompute K and V for past tokens.
Implement caching to speed up generation.
"""

class AttentionWithKVCache:
    """Attention with KV caching for fast generation."""

    def __init__(self, d_model, n_heads):
        # TODO: Initialize
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False):
        """
        Args:
            x: Input tokens (batch, seq_len, d_model)
            use_cache: Whether to use cached K/V

        Returns:
            output, (updated_cache_k, updated_cache_v)
        """
        # TODO: Implement with caching
        # Hint:
        # - If cache exists, only compute K/V for new tokens
        # - Concatenate with cached K/V
        # - Return updated cache
        pass


# Test: Generation should be faster with caching
# Time without cache vs with cache
```

---

## Key Takeaways

1. **Architecture Evolution**:
   - GPT-3 → LLaMA: Efficiency improvements (RMSNorm, RoPE, SwiGLU)
   - LLaMA 2: Added GQA for better inference
   - Mistral: Sliding window for long context
   - Mixtral: Sparse MoE for quality + speed

2. **Critical Innovations**:
   - **GQA**: Reduces KV cache by 4-8x with minimal quality loss
   - **Sliding Window**: Enables long context with linear memory
   - **RoPE**: Better position encoding, extrapolates to longer sequences
   - **MoE**: Higher capacity without proportional compute

3. **Scaling Laws** (Chinchilla):
   - Most models are undertrained
   - Optimal ratio: ~20 tokens per parameter
   - LLaMA and newer models follow this

4. **Positional Encodings**:
   - Absolute (sinusoidal): Original transformer
   - Learned: BERT, GPT-2
   - RoPE: LLaMA, Mistral (best for extrapolation)
   - ALiBi: Excellent extrapolation, no parameters

5. **Model Selection**:
   - Edge: Phi-2, Gemma 2B
   - Efficiency: Mistral 7B, Mixtral 8x7B
   - Quality: GPT-4, Claude
   - Open-source: LLaMA 2, Mistral

6. **Production Considerations**:
   - KV cache size grows with sequence length
   - GQA and MQA reduce memory significantly
   - Sliding window enables longer contexts
   - MoE needs specialized serving infrastructure

---

## Further Reading

### Papers
1. **LLaMA**: "LLaMA: Open and Efficient Foundation Language Models" (Touvron et al., 2023)
2. **LLaMA 2**: "Llama 2: Open Foundation and Fine-Tuned Chat Models" (Touvron et al., 2023)
3. **Mistral**: "Mistral 7B" (Jiang et al., 2023)
4. **Mixtral**: "Mixtral of Experts" (Jiang et al., 2024)
5. **Chinchilla**: "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
6. **RoFormer**: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
7. **ALiBi**: "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation" (Press et al., 2022)
8. **GQA**: "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (Ainslie et al., 2023)

### Resources
- Hugging Face Model Hub: https://huggingface.co/models
- LLaMA 2 Technical Report
- Mistral AI Documentation
- The Transformer Family 2.0 (Lil'Log blog post)

### Related Modules
- **Module 7**: NLP and Transformers (foundation)
- **Module 12 Lesson 9**: RLHF for alignment
- **Module 15 Lesson 2**: Small Language Models (Phi, Gemma)
- **Module 15 Lesson 6**: Efficient Inference (vLLM, TGI)
- **Module 16**: RAG and Agents

---

**Next Lesson**: Small Language Models and Edge Deployment (Phi-3, Gemma, quantization)


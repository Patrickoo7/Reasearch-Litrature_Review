# Lesson 6: Efficient Inference and Serving ⚡

**Module 15: Multimodal Models and Advanced LLMs | Lesson 6 of 7**

Master fast LLM inference - from KV caching to vLLM. Speed matters for production!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand KV cache and its critical importance
2. ✅ Implement dynamic, static, and continuous batching
3. ✅ Master vLLM and PagedAttention
4. ✅ Use TensorRT-LLM for NVIDIA GPUs
5. ✅ Deploy with Text Generation Inference (TGI)
6. ✅ Apply quantization for faster inference
7. ✅ Implement speculative decoding
8. ✅ Design production serving architectures

---

## Prerequisites

- **Required**: Module 15 Lesson 1 (LLM architectures)
- **Required**: Module 15 Lesson 2 (Quantization basics)
- **Helpful**: Understanding of GPU computing
- **Libraries**: `transformers`, `vllm`, `text-generation-inference`

```bash
pip install transformers torch accelerate
pip install vllm  # Efficient inference
pip install triton  # For optimized kernels
```

---

## 1. Why Speed Matters

```python
import time
import numpy as np
from typing import List, Dict

class InferenceSpeedImportance:
    """
    Why inference speed is critical for production.

    Key metrics:
    - Time to First Token (TTFT)
    - Tokens per Second (throughput)
    - Latency per token
    - Request throughput
    """

    def __init__(self):
        print("="*80)
        print("Why Inference Speed Matters")
        print("="*80)

    def latency_impact_on_ux(self):
        """Show how latency impacts user experience."""

        print("\nLatency Impact on User Experience:")

        scenarios = [
            {
                'use_case': 'Chatbot',
                'acceptable_latency': '<500ms first token',
                'why': 'Users expect instant response like human',
                'poor_experience': '>2s feels broken'
            },
            {
                'use_case': 'Code Completion',
                'acceptable_latency': '<100ms',
                'why': 'Must not interrupt typing flow',
                'poor_experience': '>300ms unusable'
            },
            {
                'use_case': 'Search/Summarization',
                'acceptable_latency': '<1s total',
                'why': 'Users have high expectations from Google',
                'poor_experience': '>5s people leave'
            },
            {
                'use_case': 'Batch Processing',
                'acceptable_latency': '<1hr for 10K items',
                'why': 'Cost-effective processing',
                'poor_experience': '>24h missed deadlines'
            },
        ]

        for s in scenarios:
            print(f"\n{s['use_case']}:")
            print(f"  Acceptable: {s['acceptable_latency']}")
            print(f"  Why: {s['why']}")
            print(f"  Poor experience: {s['poor_experience']}")

    def cost_of_latency(self):
        """Calculate cost impact of latency."""

        print("\n" + "="*80)
        print("Cost Impact of Inference Speed")
        print("="*80)

        # Example: GPT-3 scale model
        scenarios = {
            'Baseline (slow)': {
                'throughput_req_per_sec': 10,
                'gpus_needed': 16,
                'gpu_cost_per_hour': 3.0,  # A100
            },
            'Optimized (fast)': {
                'throughput_req_per_sec': 100,  # 10x faster
                'gpus_needed': 2,  # 8x fewer GPUs
                'gpu_cost_per_hour': 3.0,
            },
        }

        print(f"{'Scenario':<20} {'Throughput':<15} {'GPUs':<10} {'$/hour':<10} {'$/month':<12}")
        print("-" * 80)

        for name, config in scenarios.items():
            cost_per_hour = config['gpus_needed'] * config['gpu_cost_per_hour']
            cost_per_month = cost_per_hour * 24 * 30

            print(f"{name:<20} {config['throughput_req_per_sec']:>6} req/s    "
                  f"{config['gpus_needed']:>3} GPUs   "
                  f"${cost_per_hour:>6.0f}     "
                  f"${cost_per_month:>8,.0f}")

        baseline_monthly = scenarios['Baseline (slow)']['gpus_needed'] * \
                          scenarios['Baseline (slow)']['gpu_cost_per_hour'] * 24 * 30
        optimized_monthly = scenarios['Optimized (fast)']['gpus_needed'] * \
                           scenarios['Optimized (fast)']['gpu_cost_per_hour'] * 24 * 30

        savings = baseline_monthly - optimized_monthly
        savings_pct = 100 * savings / baseline_monthly

        print(f"\nMonthly Savings: ${savings:,.0f} ({savings_pct:.0f}%)")
        print("Inference optimization = massive cost savings!")

    def performance_metrics(self):
        """Key performance metrics for LLM serving."""

        print("\n" + "="*80)
        print("Key Performance Metrics")
        print("="*80)

        metrics = [
            ("Time to First Token (TTFT)", "Latency until first token appears", "ms", "Lower = better UX"),
            ("Tokens per Second", "Generation speed", "tokens/s", "Higher = faster completion"),
            ("Throughput", "Requests processed per second", "req/s", "Higher = more users served"),
            ("Batch Size", "Concurrent requests processed", "requests", "Larger = better GPU util"),
            ("Memory Usage", "GPU memory consumed", "GB", "Lower = more concurrent users"),
            ("Cost per 1K tokens", "Inference cost", "$", "Lower = cheaper to run"),
        ]

        for metric, description, unit, goal in metrics:
            print(f"\n{metric}:")
            print(f"  What: {description}")
            print(f"  Unit: {unit}")
            print(f"  Goal: {goal}")


importance = InferenceSpeedImportance()
importance.latency_impact_on_ux()
importance.cost_of_latency()
importance.performance_metrics()
```

---

## 2. KV Cache: The Foundation of Fast Inference

KV cache is essential for efficient autoregressive generation:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCacheExplainer:
    """
    Explain and implement KV caching.

    Key insight: In autoregressive generation, we recompute
    the same K and V for past tokens. Cache them!

    Speedup: ~10-100x for long sequences
    """

    def __init__(self):
        print("="*80)
        print("KV Cache: Foundation of Fast Inference")
        print("="*80)

    def explain_kv_cache(self):
        """Explain why KV cache is critical."""

        print("\nWithout KV Cache (Naive):")
        print("  At each step t:")
        print("    1. Recompute K and V for ALL tokens 1..t")
        print("    2. Compute attention")
        print("    3. Generate token t+1")
        print("  Complexity: O(t²) - quadratic in sequence length!")

        print("\nWith KV Cache (Optimized):")
        print("  At each step t:")
        print("    1. Compute K and V only for NEW token t")
        print("    2. Concatenate with cached K, V from tokens 1..t-1")
        print("    3. Compute attention")
        print("    4. Cache new K, V")
        print("  Complexity: O(t) - linear!")

        print("\nSpeedup:")
        print("  Sequence length 100: ~10x faster")
        print("  Sequence length 1000: ~100x faster")
        print("  Critical for production!")

    def implement_kv_cache(self):
        """Implement attention with KV cache."""

        code = """
class AttentionWithKVCache(nn.Module):
    '''
    Multi-head attention with KV caching.

    During generation:
    - Cache K and V for past tokens
    - Only compute K, V for new token
    - Concatenate and compute attention
    '''

    def __init__(self, d_model=768, n_heads=12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        # KV cache: (batch, n_heads, seq_len, head_dim)
        self.cache_k = None
        self.cache_v = None

    def forward(self, x, use_cache=False, past_kv=None):
        '''
        Args:
            x: Input tokens (batch, seq_len, d_model)
            use_cache: Whether to use/update cache
            past_kv: (cache_k, cache_v) from previous step

        Returns:
            output, (updated_cache_k, updated_cache_v)
        '''
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Transpose for attention: (batch, n_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Use cache if available
        if use_cache and past_kv is not None:
            past_k, past_v = past_kv

            # Concatenate with cached K, V
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        output = self.o_proj(attn_output)

        # Return output and updated cache
        if use_cache:
            return output, (k, v)
        else:
            return output, None


# Example: Generation with KV cache
def generate_with_cache(model, prompt, max_new_tokens=50):
    '''Generate text using KV cache.'''
    generated = prompt
    past_kv = None

    for _ in range(max_new_tokens):
        # Only pass NEW token (last one)
        if past_kv is None:
            # First step: full prompt
            input_ids = generated
        else:
            # Subsequent steps: only last token
            input_ids = generated[:, -1:]

        # Forward pass with cache
        output, past_kv = model(input_ids, use_cache=True, past_kv=past_kv)

        # Sample next token
        next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

        # Append to generated sequence
        generated = torch.cat([generated, next_token], dim=1)

    return generated
"""

        print("\n" + "="*80)
        print("Attention with KV Cache Implementation")
        print("="*80)
        print(code)

    def memory_analysis(self):
        """Analyze KV cache memory usage."""

        print("\n" + "="*80)
        print("KV Cache Memory Analysis")
        print("="*80)

        # LLaMA 2 7B config
        config = {
            'n_layers': 32,
            'n_heads': 32,
            'head_dim': 128,
            'd_model': 4096,
        }

        def calc_kv_cache_size(seq_len, batch_size=1):
            """Calculate KV cache size in bytes."""
            # 2 (K and V) * n_layers * batch * n_heads * seq_len * head_dim * 2 bytes (fp16)
            size_bytes = (2 * config['n_layers'] * batch_size *
                         config['n_heads'] * seq_len * config['head_dim'] * 2)
            size_gb = size_bytes / (1024 ** 3)
            return size_gb

        print(f"\nLLaMA 2 7B KV Cache Size (batch_size=1, fp16):")
        print(f"{'Seq Length':<15} {'KV Cache (GB)':<15} {'% of Model Weights':<20}")
        print("-" * 60)

        model_size_gb = 7 * 2 / 1024  # 7B params * 2 bytes (fp16)

        for seq_len in [128, 512, 2048, 4096, 8192]:
            cache_size = calc_kv_cache_size(seq_len)
            pct = 100 * cache_size / model_size_gb

            print(f"{seq_len:<15} {cache_size:<15.2f} {pct:<20.1f}%")

        print("\nKey Insight:")
        print("  • KV cache grows linearly with sequence length")
        print("  • At 4K context, cache = 36% of model size!")
        print("  • Limits batch size and max concurrent users")
        print("  • PagedAttention (vLLM) solves this!")


kv_cache = KVCacheExplainer()
kv_cache.explain_kv_cache()
kv_cache.implement_kv_cache()
kv_cache.memory_analysis()
```

---

## 3. Batching Strategies

Efficient batching is key to throughput:

```python
class BatchingStrategies:
    """
    Batching strategies for LLM inference.

    Types:
    1. Static batching: Fixed batch, wait for all to complete
    2. Dynamic batching: Variable batch size
    3. Continuous batching: Add/remove requests on the fly
    """

    def __init__(self):
        print("="*80)
        print("Batching Strategies")
        print("="*80)

    def compare_strategies(self):
        """Compare batching strategies."""

        strategies = {
            'Static Batching': {
                'description': 'Wait for N requests, process batch, repeat',
                'pros': 'Simple to implement',
                'cons': 'Poor GPU utilization, high latency',
                'when': 'Batch offline processing'
            },
            'Dynamic Batching': {
                'description': 'Variable batch size based on queue',
                'pros': 'Better than static',
                'cons': 'Still waits for slowest in batch',
                'when': 'Mixed workloads'
            },
            'Continuous Batching': {
                'description': 'Add/remove requests as they arrive/complete',
                'pros': 'Maximum throughput and GPU util',
                'cons': 'Complex to implement',
                'when': 'Production serving (vLLM, TGI)'
            },
        }

        for name, details in strategies.items():
            print(f"\n{name}:")
            print(f"  Description: {details['description']}")
            print(f"  Pros: {details['pros']}")
            print(f"  Cons: {details['cons']}")
            print(f"  When to use: {details['when']}")

    def static_vs_continuous_example(self):
        """Example showing static vs continuous batching."""

        print("\n" + "="*80)
        print("Static vs Continuous Batching Example")
        print("="*80)

        print("\nScenario: 4 requests with varying output lengths")
        print("  Request A: 10 tokens")
        print("  Request B: 50 tokens")
        print("  Request C: 100 tokens")
        print("  Request D: 20 tokens")

        print("\nStatic Batching:")
        print("  Time 0-100: Process all 4 requests")
        print("    - A finishes at step 10, waits 90 steps")
        print("    - B finishes at step 50, waits 50 steps")
        print("    - D finishes at step 20, waits 80 steps")
        print("    - C finishes at step 100")
        print("  Total time: 100 steps")
        print("  GPU utilization: 45% (many slots wasted)")

        print("\nContinuous Batching:")
        print("  Time 0-10: Process A,B,C,D (all 4)")
        print("  Time 10: A done, add new request E")
        print("  Time 10-20: Process B,C,D,E (still 4)")
        print("  Time 20: D done, add new request F")
        print("  Time 20-50: Process B,C,E,F (still 4)")
        print("  ...")
        print("  GPU utilization: ~95% (slots always full)")
        print("  Throughput: 2-3x higher!")

    def implement_continuous_batching(self):
        """Conceptual implementation of continuous batching."""

        code = """
class ContinuousBatchingScheduler:
    '''
    Continuous batching scheduler.

    Key ideas:
    - Maintain running batch
    - When request completes, immediately add new one
    - Maximize GPU utilization
    '''

    def __init__(self, model, max_batch_size=32):
        self.model = model
        self.max_batch_size = max_batch_size

        self.running_requests = []  # Currently processing
        self.waiting_queue = []     # Waiting to start

    def add_request(self, request):
        '''Add new request to queue.'''
        if len(self.running_requests) < self.max_batch_size:
            # Space available, start immediately
            self.running_requests.append(request)
        else:
            # Queue is full, wait
            self.waiting_queue.append(request)

    def step(self):
        '''Execute one generation step.'''
        if not self.running_requests:
            return

        # Prepare batch (all running requests)
        batch = self.prepare_batch(self.running_requests)

        # Forward pass (generates 1 token for each request)
        outputs = self.model(batch)

        # Update requests
        completed = []
        for i, request in enumerate(self.running_requests):
            request.append_token(outputs[i])

            # Check if done
            if request.is_complete():
                completed.append(request)

        # Remove completed requests
        for request in completed:
            self.running_requests.remove(request)
            request.finish()

        # Add waiting requests to fill batch
        while (len(self.running_requests) < self.max_batch_size and
               len(self.waiting_queue) > 0):
            new_request = self.waiting_queue.pop(0)
            self.running_requests.append(new_request)

    def run(self):
        '''Main serving loop.'''
        while True:
            # Check for new requests (from queue, API, etc.)
            new_requests = self.check_for_new_requests()
            for req in new_requests:
                self.add_request(req)

            # Execute one step
            if self.running_requests:
                self.step()
"""

        print("\n" + "="*80)
        print("Continuous Batching Implementation")
        print("="*80)
        print(code)


batching = BatchingStrategies()
batching.compare_strategies()
batching.static_vs_continuous_example()
batching.implement_continuous_batching()
```

---

## 4. vLLM: PagedAttention for Efficient Serving

vLLM revolutionizes LLM serving with PagedAttention:

```python
class VLLMExplainer:
    """
    Explain vLLM and PagedAttention.

    Key innovation: Treat KV cache like virtual memory
    - Page KV cache blocks
    - Share memory across requests
    - Near-zero waste

    Result: 24x higher throughput than HuggingFace!
    """

    def __init__(self):
        print("="*80)
        print("vLLM: Fast Inference with PagedAttention")
        print("="*80)

    def explain_problem(self):
        """Explain the memory fragmentation problem."""

        print("\nProblem: KV Cache Memory Fragmentation")
        print("\n1. Traditional Approach:")
        print("   - Pre-allocate max_length for each request")
        print("   - E.g., 2048 tokens * 32 layers * ... = 2GB per request")
        print("   - Most requests << max_length")
        print("   - Result: 60-80% memory wasted!")

        print("\n2. Example:")
        print("   GPU has 40GB VRAM")
        print("   Each request allocated 2GB")
        print("   Can serve: 40 / 2 = 20 requests")
        print("   But actual usage: Only 30% of allocated memory")
        print("   Effective capacity: ~6 concurrent requests")
        print("   Waste: 70%!")

    def explain_paged_attention(self):
        """Explain PagedAttention solution."""

        print("\n" + "="*80)
        print("PagedAttention Solution")
        print("="*80)

        print("\nKey Ideas:")
        print("  1. Divide KV cache into blocks (pages)")
        print("     - Like virtual memory in OS!")
        print("     - Each block: e.g., 16 tokens")

        print("\n  2. Allocate blocks on demand")
        print("     - Request starts: Allocate 1 block")
        print("     - Generates 16 tokens: Allocate another block")
        print("     - Only pay for what you use!")

        print("\n  3. Share blocks across requests")
        print("     - Multiple requests with same prompt prefix")
        print("     - Share KV cache for prefix")
        print("     - Copy-on-write when they diverge")

        print("\n  4. Non-contiguous storage")
        print("     - Blocks can be anywhere in memory")
        print("     - Page table tracks logical → physical mapping")

    def performance_comparison(self):
        """Compare vLLM vs other frameworks."""

        print("\n" + "="*80)
        print("vLLM Performance Comparison")
        print("="*80)

        # Published benchmarks (LLaMA 7B on A100)
        frameworks = {
            'Framework': ['vLLM', 'Text Generation Inference', 'HuggingFace Transformers', 'FasterTransformer'],
            'Throughput (req/s)': [24.0, 14.5, 1.0, 8.3],
            'Latency (ms)': [125, 180, 2100, 290],
            'Memory Efficiency': ['95%', '75%', '40%', '60%'],
            'Batch Size (max)': [256, 128, 16, 64],
        }

        import pandas as pd
        df = pd.DataFrame(frameworks)
        print(df.to_string(index=False))

        print("\nKey Results:")
        print("  • vLLM: 24x faster than HuggingFace!")
        print("  • 1.7x faster than TGI")
        print("  • 95% memory efficiency (vs 40% baseline)")
        print("  • Enables much larger batch sizes")

    def vllm_usage_example(self):
        """Show vLLM usage example."""

        code = """
from vllm import LLM, SamplingParams

# Initialize vLLM
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=1,  # Number of GPUs
    dtype="half",            # FP16
    max_model_len=4096,      # Context length
)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

# Single request
prompts = ["Explain quantum computing"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# Batch requests (automatically batched!)
prompts = [
    "Explain machine learning",
    "What is deep learning?",
    "How do neural networks work?",
    # ... can handle 100s of prompts
]

outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Output: {output.outputs[0].text}\\n")

# Streaming generation
from vllm import SamplingParams

params = SamplingParams(temperature=0.7, max_tokens=200)

# Stream tokens as they're generated
for output in llm.generate("Write a story:", params, stream=True):
    print(output.outputs[0].text, end='', flush=True)
"""

        print("\n" + "="*80)
        print("vLLM Usage Example")
        print("="*80)
        print(code)

    def when_to_use_vllm(self):
        """When to use vLLM."""

        print("\n" + "="*80)
        print("When to Use vLLM")
        print("="*80)

        scenarios = [
            ("✅ High throughput needed", "Serving many users concurrently"),
            ("✅ Batch processing", "Processing large datasets"),
            ("✅ Cost optimization", "Maximize requests per GPU"),
            ("✅ Production serving", "Battle-tested, widely used"),
            ("❌ Single user", "Overhead not worth it for single requests"),
            ("❌ Very short sequences", "Benefit is smaller"),
            ("❌ Need fine-grained control", "vLLM abstracts details"),
        ]

        for scenario, explanation in scenarios:
            print(f"  {scenario}: {explanation}")


vllm = VLLMExplainer()
vllm.explain_problem()
vllm.explain_paged_attention()
vllm.performance_comparison()
vllm.vllm_usage_example()
vllm.when_to_use_vllm()
```

---

## 5. TensorRT-LLM: NVIDIA GPU Optimization

TensorRT-LLM for maximum performance on NVIDIA GPUs:

```python
class TensorRTLLM:
    """
    TensorRT-LLM: NVIDIA's optimized inference engine.

    Features:
    - Custom CUDA kernels
    - Kernel fusion
    - Mixed precision (FP16, INT8, INT4)
    - Multi-GPU support
    - 2-8x faster than PyTorch
    """

    def __init__(self):
        print("="*80)
        print("TensorRT-LLM: NVIDIA GPU Optimization")
        print("="*80)

    def explain_tensorrt(self):
        """Explain TensorRT optimizations."""

        print("\nTensorRT-LLM Optimizations:")

        optimizations = [
            ("Kernel Fusion", "Fuse multiple ops into single kernel (less memory transfers)"),
            ("Custom CUDA Kernels", "Hand-optimized for specific operations"),
            ("Quantization", "INT8/INT4 inference with minimal quality loss"),
            ("Graph Optimization", "Optimize computation graph"),
            ("Memory Planning", "Minimize memory usage and fragmentation"),
            ("Multi-GPU", "Tensor and pipeline parallelism"),
        ]

        for opt, desc in optimizations:
            print(f"  • {opt}: {desc}")

    def performance_comparison(self):
        """Compare TensorRT-LLM performance."""

        print("\n" + "="*80)
        print("TensorRT-LLM Performance")
        print("="*80)

        # LLaMA 2 70B on 8x A100
        results = {
            'Framework': ['TensorRT-LLM (FP16)', 'TensorRT-LLM (INT8)', 'vLLM', 'HuggingFace'],
            'Throughput (tokens/s)': [15000, 28000, 9000, 2000],
            'Latency (ms/token)': [18, 10, 28, 125],
            'Memory (GB)': [140, 70, 140, 160],
        }

        import pandas as pd
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        print("\nKey Results:")
        print("  • TensorRT-LLM INT8: 14x faster than HuggingFace")
        print("  • 3x faster than vLLM")
        print("  • 50% memory reduction with INT8")
        print("  • Best for: NVIDIA GPUs, maximum performance")

    def usage_example(self):
        """TensorRT-LLM usage example."""

        code = """
# Note: TensorRT-LLM requires model conversion (one-time)

# Step 1: Convert model to TensorRT format
# This is done once, offline

from tensorrt_llm import LLM
from tensorrt_llm.builder import Builder

# Build TensorRT engine
builder = Builder()
builder.load_model("meta-llama/Llama-2-7b-hf")

# Configure optimizations
builder.set_precision("fp16")  # or "int8", "int4"
builder.set_max_batch_size(128)
builder.set_max_input_len(2048)
builder.set_max_output_len(512)

# Build engine (takes a few minutes)
engine = builder.build()
engine.save("llama2_7b_fp16.engine")

# Step 2: Load and run inference
llm = LLM(engine_path="llama2_7b_fp16.engine")

# Generate
outputs = llm.generate(
    prompts=["Explain AI"],
    max_length=100,
    temperature=0.7
)

print(outputs[0])

# Streaming
for token in llm.generate_stream("Write a story:"):
    print(token, end='', flush=True)
"""

        print("\n" + "="*80)
        print("TensorRT-LLM Usage")
        print("="*80)
        print(code)


tensorrt = TensorRTLLM()
tensorrt.explain_tensorrt()
tensorrt.performance_comparison()
tensorrt.usage_example()
```

---

## 6. Quantization for Inference

Quantization significantly speeds up inference:

```python
class InferenceQuantization:
    """
    Quantization techniques for faster inference.

    Methods:
    - Dynamic quantization: Quantize during inference
    - Static quantization: Pre-quantized weights
    - Activation quantization: Quantize activations too
    """

    def __init__(self):
        print("="*80)
        print("Quantization for Inference")
        print("="*80)

    def quantization_impact(self):
        """Impact of quantization on inference speed."""

        print("\nQuantization Impact (LLaMA 2 7B on A100):")

        configs = {
            'Precision': ['FP32', 'FP16', 'INT8', 'INT4'],
            'Memory (GB)': [28.0, 14.0, 7.0, 3.5],
            'Speed (tokens/s)': [25, 50, 90, 150],
            'Quality Loss': ['0%', '<1%', '1-2%', '3-5%'],
        }

        import pandas as pd
        df = pd.DataFrame(configs)
        print(df.to_string(index=False))

        print("\nKey Insights:")
        print("  • INT8: 3.6x faster than FP16, minimal quality loss")
        print("  • INT4: 6x faster, acceptable for many use cases")
        print("  • Memory reduction enables larger batch sizes")

    def implement_dynamic_quantization(self):
        """Implement dynamic quantization."""

        code = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Apply dynamic quantization
# Quantizes weights to INT8 during inference
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize Linear layers
    dtype=torch.qint8   # Use INT8
)

# Compare sizes
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())

print(f"Original size: {original_size / 1e6:.1f} MB")
print(f"Quantized size: {quantized_size / 1e6:.1f} MB")
print(f"Reduction: {100 * (1 - quantized_size/original_size):.0f}%")

# Inference (same API)
inputs = tokenizer("Hello", return_tensors="pt")
outputs = quantized_model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
"""

        print("\n" + "="*80)
        print("Dynamic Quantization Implementation")
        print("="*80)
        print(code)


quant = InferenceQuantization()
quant.quantization_impact()
quant.implement_dynamic_quantization()
```

---

## 7. Speculative Decoding

Speculative decoding for faster generation:

```python
class SpeculativeDecoding:
    """
    Speculative decoding: Generate multiple tokens at once.

    Idea:
    - Use small "draft" model to quickly generate candidates
    - Use large "target" model to verify in parallel
    - Accept if target agrees, reject and retry if not

    Speedup: 2-3x for compatible model pairs
    """

    def __init__(self):
        print("="*80)
        print("Speculative Decoding")
        print("="*80)

    def explain_speculative_decoding(self):
        """Explain speculative decoding."""

        print("\nStandard Autoregressive Generation:")
        print("  For each token:")
        print("    1. Run large model (slow)")
        print("    2. Sample token")
        print("    3. Repeat")
        print("  Bottleneck: Sequential, can't parallelize")

        print("\nSpeculative Decoding:")
        print("  1. Draft model generates K tokens (fast)")
        print("     e.g., K=4: [token1, token2, token3, token4]")
        print("  2. Target model verifies all K tokens in parallel")
        print("     - Computes logits for positions 1..K")
        print("  3. Accept prefix where models agree")
        print("     - If all K match: Accept all, speedup = K")
        print("     - If 2 match: Accept 2, retry from position 3")
        print("  4. Repeat")

        print("\nRequirements:")
        print("  • Draft model: Small, fast (e.g., 1B params)")
        print("  • Target model: Large, accurate (e.g., 70B params)")
        print("  • Models must use same tokenizer")
        print("  • Draft should have reasonable quality (70%+ agreement)")

    def implementation_sketch(self):
        """Implementation sketch."""

        code = """
def speculative_decode(
    draft_model,
    target_model,
    prompt,
    max_new_tokens=100,
    k=4  # Number of speculative tokens
):
    '''
    Speculative decoding implementation.

    Args:
        draft_model: Fast small model
        target_model: Slow large model
        prompt: Input prompt
        max_new_tokens: Max tokens to generate
        k: Speculation depth

    Returns:
        Generated text
    '''

    tokens = prompt
    num_generated = 0

    while num_generated < max_new_tokens:
        # Step 1: Draft model generates k tokens
        draft_tokens = []
        draft_probs = []

        for _ in range(k):
            logits = draft_model(tokens)
            probs = softmax(logits)
            next_token = sample(probs)

            draft_tokens.append(next_token)
            draft_probs.append(probs)

            tokens = torch.cat([tokens, next_token])

        # Step 2: Target model verifies in parallel
        # Run target model once for all k positions
        target_logits = target_model(tokens)  # Returns logits for all positions
        target_probs = softmax(target_logits)

        # Step 3: Acceptance sampling
        accepted = 0
        for i in range(k):
            # Probability target model assigns to draft's choice
            p_target = target_probs[len(prompt) + num_generated + i, draft_tokens[i]]
            p_draft = draft_probs[i][draft_tokens[i]]

            # Accept with probability min(1, p_target / p_draft)
            if random.random() < min(1.0, p_target / p_draft):
                accepted += 1
            else:
                # Rejection: resample and stop
                new_token = sample(target_probs[len(prompt) + num_generated + i])
                tokens[len(prompt) + num_generated + accepted] = new_token
                break

        # Update counts
        num_generated += accepted + 1  # +1 for resampled token if rejected

    return tokens
"""

        print("\n" + "="*80)
        print("Speculative Decoding Implementation")
        print("="*80)
        print(code)

    def performance_analysis(self):
        """Analyze speculative decoding performance."""

        print("\n" + "="*80)
        print("Speculative Decoding Performance")
        print("="*80)

        print("\nExample: LLaMA 2 70B (target) + LLaMA 2 7B (draft)")

        configs = {
            'Config': ['Standard (70B only)', 'Speculative (7B draft + 70B verify)', 'Speedup'],
            'Latency per token (ms)': [80, 30, '2.7x'],
            'Throughput (tokens/s)': [12.5, 33.3, '2.7x'],
            'Quality': ['100%', '100% (same output)', '-'],
        }

        import pandas as pd
        df = pd.DataFrame(configs)
        print(df.to_string(index=False))

        print("\nKey Insights:")
        print("  • 2-3x speedup with no quality loss")
        print("  • Works best when draft model is good (70%+ agreement)")
        print("  • Requires extra GPU memory for draft model")
        print("  • Most benefit for long generations")


spec_decode = SpeculativeDecoding()
spec_decode.explain_speculative_decoding()
spec_decode.implementation_sketch()
spec_decode.performance_analysis()
```

---

## 8. Production Serving Architecture

Design patterns for production LLM serving:

```python
class ProductionArchitecture:
    """
    Production LLM serving architecture.

    Components:
    - Load balancer
    - Model servers (vLLM/TensorRT)
    - Caching layer
    - Monitoring
    - Rate limiting
    """

    def __init__(self):
        print("="*80)
        print("Production LLM Serving Architecture")
        print("="*80)

    def architecture_diagram(self):
        """Show architecture diagram."""

        diagram = """
                          ┌─────────────┐
                          │   Clients   │
                          └──────┬──────┘
                                 │
                          ┌──────▼──────────┐
                          │  Load Balancer  │
                          │   (NGINX/ALB)   │
                          └────────┬────────┘
                                   │
                ┌──────────────────┼──────────────────┐
                │                  │                  │
        ┌───────▼────────┐  ┌──────▼──────┐  ┌──────▼──────┐
        │  Model Server  │  │Model Server │  │Model Server │
        │  (vLLM/TGI)    │  │  (vLLM/TGI) │  │  (vLLM/TGI) │
        │  GPU 1         │  │  GPU 2      │  │  GPU 3      │
        └────────────────┘  └─────────────┘  └─────────────┘
                │                  │                  │
                └──────────────────┼──────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Redis Cache     │
                          │  (Responses)     │
                          └──────────────────┘
                                   │
                          ┌────────▼─────────┐
                          │  Monitoring      │
                          │  (Prometheus)    │
                          └──────────────────┘
"""

        print("\nProduction Architecture:")
        print(diagram)

    def component_overview(self):
        """Overview of components."""

        print("\n" + "="*80)
        print("Architecture Components")
        print("="*80)

        components = {
            'Load Balancer': {
                'role': 'Distribute requests across servers',
                'tech': 'NGINX, AWS ALB, Google Load Balancer',
                'features': 'Health checks, sticky sessions, SSL termination',
            },
            'Model Servers': {
                'role': 'Run inference',
                'tech': 'vLLM, TensorRT-LLM, Text Generation Inference',
                'features': 'Continuous batching, KV caching, streaming',
            },
            'Cache Layer': {
                'role': 'Cache common responses',
                'tech': 'Redis, Memcached',
                'features': 'TTL, LRU eviction, distributed cache',
            },
            'Rate Limiting': {
                'role': 'Prevent abuse, ensure fair usage',
                'tech': 'Redis-based, token bucket algorithm',
                'features': 'Per-user limits, burst allowance',
            },
            'Monitoring': {
                'role': 'Track performance and health',
                'tech': 'Prometheus, Grafana, DataDog',
                'features': 'Metrics, alerting, dashboards',
            },
        }

        for component, details in components.items():
            print(f"\n{component}:")
            print(f"  Role: {details['role']}")
            print(f"  Tech: {details['tech']}")
            print(f"  Features: {details['features']}")

    def scaling_strategy(self):
        """Scaling strategy."""

        print("\n" + "="*80)
        print("Scaling Strategy")
        print("="*80)

        strategies = [
            ("Horizontal Scaling", "Add more model server instances", "Handle more concurrent users"),
            ("Vertical Scaling", "Use larger GPUs (A100 → H100)", "Serve larger models or bigger batches"),
            ("Model Parallelism", "Split model across GPUs", "Serve models too large for one GPU"),
            ("Caching", "Cache popular responses", "Reduce load on model servers"),
            ("Request Queuing", "Queue during peak traffic", "Smooth out bursts"),
        ]

        for strategy, how, benefit in strategies:
            print(f"\n{strategy}:")
            print(f"  How: {how}")
            print(f"  Benefit: {benefit}")


prod_arch = ProductionArchitecture()
prod_arch.architecture_diagram()
prod_arch.component_overview()
prod_arch.scaling_strategy()
```

---

## 9. Practice Exercises

### Exercise 1: Implement KV Cache

```python
"""
Exercise: Implement KV caching from scratch.

1. Create attention layer with KV cache
2. Compare speed with/without cache
3. Measure memory usage
4. Plot speedup vs sequence length
"""

def implement_kv_cache_exercise():
    # TODO: Your implementation
    pass
```

### Exercise 2: Benchmark Batching

```python
"""
Exercise: Compare batching strategies.

1. Implement static and dynamic batching
2. Generate requests with varying lengths
3. Measure throughput and latency
4. Compare GPU utilization
"""

def benchmark_batching():
    # TODO: Your implementation
    pass
```

### Exercise 3: Deploy with vLLM

```python
"""
Exercise: Deploy model with vLLM.

1. Install vLLM
2. Deploy LLaMA 2 7B
3. Benchmark throughput
4. Compare with HuggingFace baseline
5. Measure cost savings
"""

def deploy_with_vllm():
    # TODO: Your implementation
    pass
```

---

## Key Takeaways

1. **Speed is Critical**:
   - Latency impacts UX (<500ms for chatbots)
   - Throughput = more users per GPU = lower cost
   - 10x speedup = 90% cost savings

2. **KV Cache**:
   - Foundation of efficient autoregressive generation
   - 10-100x speedup vs naive approach
   - Memory grows linearly with sequence length
   - PagedAttention solves fragmentation

3. **Batching**:
   - Continuous batching >>> static batching
   - 2-3x higher throughput
   - Maximizes GPU utilization
   - vLLM and TGI implement this

4. **vLLM**:
   - PagedAttention = virtual memory for KV cache
   - 24x faster than HuggingFace
   - 95% memory efficiency
   - Production-ready, widely used

5. **TensorRT-LLM**:
   - Best performance on NVIDIA GPUs
   - 3x faster than vLLM
   - Custom CUDA kernels
   - INT8 = 50% memory reduction

6. **Optimization Techniques**:
   - Quantization: INT8 = 3.6x speedup
   - Speculative decoding: 2-3x speedup
   - Kernel fusion, mixed precision
   - Combine multiple techniques

7. **Production Architecture**:
   - Load balancer + multiple model servers
   - Caching layer for common responses
   - Monitoring and rate limiting
   - Horizontal scaling for growth

---

## Further Reading

### Papers
1. **vLLM**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (Kwon et al., 2023)
2. **Speculative Decoding**: "Fast Inference from Transformers via Speculative Decoding" (Leviathan et al., 2023)
3. **FlashAttention**: "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)

### Tools
- vLLM: https://github.com/vllm-project/vllm
- TensorRT-LLM: https://github.com/NVIDIA/TensorRT-LLM
- Text Generation Inference: https://github.com/huggingface/text-generation-inference
- FastChat: https://github.com/lm-sys/FastChat

### Resources
- vLLM Documentation
- NVIDIA TensorRT-LLM Guide
- Hugging Face TGI Docs

### Related Modules
- **Module 15 Lesson 2**: Quantization (edge deployment)
- **Module 15 Lesson 7**: LLMOps (production monitoring)

---

**Next Lesson**: LLMOps - Production Deployment and Monitoring

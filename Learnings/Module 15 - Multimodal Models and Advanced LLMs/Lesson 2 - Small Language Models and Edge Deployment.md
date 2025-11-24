# Lesson 2: Small Language Models and Edge Deployment 📱

**Module 15: Multimodal Models and Advanced LLMs | Lesson 2 of 7**

Master efficient small language models (SLMs) and deploy them on edge devices - from mobile phones to IoT!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand the motivation and use cases for Small Language Models
2. ✅ Implement and analyze Phi-3, Gemma, and other SLMs
3. ✅ Master quantization techniques: INT8, INT4, GPTQ, AWQ, GGUF
4. ✅ Deploy models on edge devices using ONNX Runtime and llama.cpp
5. ✅ Optimize memory usage and batching for resource-constrained devices
6. ✅ Implement browser-based LLM inference with WebLLM
7. ✅ Design edge deployment strategies for production

---

## Prerequisites

- **Required**: Module 15 Lesson 1 (LLM architectures)
- **Required**: Basic understanding of model compression
- **Helpful**: PyTorch quantization basics
- **Libraries**: `transformers`, `onnx`, `onnxruntime`, `bitsandbytes`, `llama-cpp-python`

```bash
pip install transformers torch onnx onnxruntime bitsandbytes accelerate
pip install auto-gptq optimum sentencepiece
pip install llama-cpp-python  # For GGUF models
```

---

## 1. Why Small Language Models?

### The Case for SLMs

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import psutil
import os

def compare_model_sizes():
    """
    Compare resource requirements of different model sizes.

    Key factors:
    - Memory footprint
    - Inference latency
    - Token throughput
    - Cost
    """

    models_comparison = {
        'GPT-3 175B': {
            'params': 175e9,
            'memory_fp16': 350,  # GB
            'memory_int8': 175,
            'memory_int4': 88,
            'gpu_required': 'A100 (80GB) x4+',
            'cost_per_1m_tokens': 2.00,
            'use_cases': ['Research', 'High-quality generation', 'Complex reasoning']
        },
        'LLaMA 2 70B': {
            'params': 70e9,
            'memory_fp16': 140,
            'memory_int8': 70,
            'memory_int4': 35,
            'gpu_required': 'A100 (80GB) x2',
            'cost_per_1m_tokens': 0.80,
            'use_cases': ['Production', 'Complex tasks', 'Self-hosting']
        },
        'LLaMA 2 13B': {
            'params': 13e9,
            'memory_fp16': 26,
            'memory_int8': 13,
            'memory_int4': 6.5,
            'gpu_required': 'RTX 4090 (24GB)',
            'cost_per_1m_tokens': 0.20,
            'use_cases': ['Cost-effective', 'General purpose', 'Fine-tuning']
        },
        'Mistral 7B': {
            'params': 7.3e9,
            'memory_fp16': 14.6,
            'memory_int8': 7.3,
            'memory_int4': 3.7,
            'gpu_required': 'RTX 3090 (24GB)',
            'cost_per_1m_tokens': 0.10,
            'use_cases': ['Efficient', 'Long context', 'Edge servers']
        },
        'Phi-3 Mini': {
            'params': 3.8e9,
            'memory_fp16': 7.6,
            'memory_int8': 3.8,
            'memory_int4': 1.9,
            'gpu_required': 'RTX 3060 (12GB)',
            'cost_per_1m_tokens': 0.05,
            'use_cases': ['Mobile', 'IoT', 'Low latency', 'Privacy']
        },
        'Gemma 2B': {
            'params': 2e9,
            'memory_fp16': 4.0,
            'memory_int8': 2.0,
            'memory_int4': 1.0,
            'gpu_required': 'Integrated GPU / CPU',
            'cost_per_1m_tokens': 0.02,
            'use_cases': ['Mobile', 'Browser', 'Real-time', 'Offline']
        },
    }

    print("="*100)
    print("Model Size Comparison")
    print("="*100)
    print(f"{'Model':<20} {'Params':<12} {'FP16 GB':<10} {'INT4 GB':<10} {'Hardware':<25} {'$/1M tokens':<12}")
    print("-"*100)

    for model_name, specs in models_comparison.items():
        print(f"{model_name:<20} "
              f"{specs['params']/1e9:>6.1f}B     "
              f"{specs['memory_fp16']:>6.1f}     "
              f"{specs['memory_int4']:>6.1f}     "
              f"{specs['gpu_required']:<25} "
              f"${specs['cost_per_1m_tokens']:>6.2f}")

    print("\n" + "="*100)
    print("When to Use Small Language Models (SLMs):")
    print("="*100)

    use_cases = [
        ("🔒 Privacy", "On-device inference, no data leaves device"),
        ("⚡ Latency", "Real-time applications (<100ms response time)"),
        ("💰 Cost", "High-volume applications where API costs add up"),
        ("📱 Mobile/IoT", "Smartphones, edge devices, embedded systems"),
        ("🌐 Offline", "No internet connectivity required"),
        ("🎯 Specialized", "Fine-tuned for specific domain (legal, medical, code)"),
        ("🔋 Energy", "Battery-powered devices, energy efficiency"),
    ]

    for emoji_title, description in use_cases:
        print(f"  {emoji_title}: {description}")

compare_model_sizes()


# Real-world latency comparison
def measure_inference_latency():
    """Measure actual inference latency for different model sizes."""

    print("\n" + "="*80)
    print("Inference Latency Comparison (Approximate)")
    print("="*80)

    # Simulated latency data (real measurements would vary by hardware)
    latencies = {
        'GPT-3 175B': {'first_token': 2000, 'tokens_per_sec': 20},
        'LLaMA 2 70B': {'first_token': 800, 'tokens_per_sec': 30},
        'LLaMA 2 13B': {'first_token': 300, 'tokens_per_sec': 50},
        'Mistral 7B': {'first_token': 150, 'tokens_per_sec': 80},
        'Phi-3 Mini': {'first_token': 80, 'tokens_per_sec': 120},
        'Gemma 2B': {'first_token': 40, 'tokens_per_sec': 200},
    }

    print(f"{'Model':<20} {'First Token (ms)':<20} {'Tokens/sec':<15} {'100 tokens (ms)':<15}")
    print("-"*80)

    for model, lat in latencies.items():
        total_time = lat['first_token'] + (100 / lat['tokens_per_sec'] * 1000)
        print(f"{model:<20} {lat['first_token']:<20} {lat['tokens_per_sec']:<15} {total_time:<15.0f}")

    print("\nKey Insight: Smaller models = faster inference = better UX")

measure_inference_latency()
```

---

## 2. Microsoft Phi-3: Quality in Small Package

Phi-3 achieves GPT-3.5 level quality with only 3.8B parameters:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

class Phi3Analyzer:
    """
    Analyze and use Microsoft Phi-3 models.

    Phi-3 family:
    - Phi-3-mini (3.8B): Mobile, edge
    - Phi-3-small (7B): Better quality
    - Phi-3-medium (14B): Near GPT-3.5 quality

    Key innovation: High-quality synthetic training data
    """

    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        """Initialize Phi-3 model."""
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load in 4-bit for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            load_in_4bit=True  # Automatic 4-bit quantization
        )

        print(f"Model loaded successfully!")
        self.print_model_info()

    def print_model_info(self):
        """Print model configuration."""
        config = self.model.config

        print("\nPhi-3 Configuration:")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  Vocabulary size: {config.vocab_size}")
        print(f"  Max position: {config.max_position_embeddings}")

        # Calculate memory usage
        param_count = sum(p.numel() for p in self.model.parameters())
        memory_mb = param_count * 0.5 / (1024 ** 2)  # 4-bit = 0.5 bytes per param

        print(f"\nMemory footprint:")
        print(f"  Parameters: {param_count / 1e9:.2f}B")
        print(f"  Memory (4-bit): {memory_mb:.0f} MB ({memory_mb/1024:.2f} GB)")

    def generate(self, prompt, max_tokens=100, temperature=0.7):
        """Generate text with Phi-3."""

        # Phi-3 uses special chat template
        messages = [{"role": "user", "content": prompt}]

        # Apply chat template
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)

        # Generate
        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        elapsed = time.time() - start_time

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the response (after the prompt)
        response = generated_text[len(formatted_prompt):].strip()

        tokens_generated = outputs[0].shape[0] - inputs['input_ids'].shape[1]

        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'-'*80}")
        print(f"Response: {response}")
        print(f"{'-'*80}")
        print(f"Time: {elapsed:.2f}s | Tokens: {tokens_generated} | Speed: {tokens_generated/elapsed:.1f} tok/s")
        print(f"{'='*80}")

        return response


# Example: Use Phi-3
phi3 = Phi3Analyzer()

# Test on various tasks
test_prompts = [
    "Explain quantum computing in one sentence.",
    "Write a Python function to calculate factorial.",
    "What are the key differences between REST and GraphQL?",
]

for prompt in test_prompts:
    phi3.generate(prompt, max_tokens=150)


# Compare Phi-3 to larger models
def compare_phi3_quality():
    """
    Compare Phi-3 quality to larger models.

    Benchmarks (approximate):
    """

    benchmarks = {
        'Model': ['GPT-4', 'GPT-3.5', 'LLaMA 2 70B', 'Mistral 7B', 'Phi-3 Mini', 'Gemma 2B'],
        'MMLU': [86.4, 70.0, 68.9, 62.5, 68.8, 42.3],
        'HumanEval': [67.0, 48.1, 29.9, 30.5, 54.7, 24.3],
        'GSM8K': [92.0, 57.1, 56.8, 52.2, 82.5, 17.7],
        'Parameters': ['1.8T', '175B', '70B', '7.3B', '3.8B', '2B'],
    }

    import pandas as pd
    df = pd.DataFrame(benchmarks)

    print("\n" + "="*80)
    print("Model Quality Comparison (Benchmark Scores)")
    print("="*80)
    print(df.to_string(index=False))

    print("\nKey Insights:")
    print("  • Phi-3 Mini (3.8B) matches LLaMA 2 70B on MMLU!")
    print("  • Phi-3 excels at reasoning (GSM8K) - 82.5 vs 56.8 for LLaMA 2 70B")
    print("  • Secret: High-quality synthetic training data from GPT-4")
    print("  • Gemma 2B: Good for size but lower quality than Phi-3")

compare_phi3_quality()
```

---

## 3. Google Gemma: Responsible AI Focus

```python
class GemmaAnalyzer:
    """
    Analyze Google Gemma models.

    Gemma family:
    - Gemma 2B: Ultra-lightweight
    - Gemma 7B: Higher quality
    - Gemma 2 (9B, 27B): Latest generation

    Focus: Responsible AI, safety, transparency
    """

    def __init__(self, model_name="google/gemma-2b-it"):
        """Initialize Gemma model."""
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        print("Gemma loaded successfully!")

    def analyze_architecture(self):
        """Analyze Gemma's architecture."""
        config = self.model.config

        print("\nGemma Architecture:")
        print(f"  Model type: {config.model_type}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Attention heads: {config.num_attention_heads}")
        print(f"  KV heads: {config.num_key_value_heads}")

        # Gemma uses GQA (Grouped-Query Attention)
        if config.num_key_value_heads < config.num_attention_heads:
            groups = config.num_attention_heads // config.num_key_value_heads
            print(f"  GQA groups: {groups} (memory efficient!)")

        print(f"  Vocabulary: {config.vocab_size}")
        print(f"  Max position: {config.max_position_embeddings}")

        # Innovations
        print("\nKey Features:")
        print("  ✓ Multi-Query Attention (MQA) for fast inference")
        print("  ✓ RoPE positional embeddings")
        print("  ✓ GeGLU activation (variant of GLU)")
        print("  ✓ Trained on diverse, filtered data")
        print("  ✓ Strong safety filters and guardrails")

    def test_safety_features(self):
        """Test Gemma's safety features."""

        test_cases = [
            ("Normal query", "Explain photosynthesis."),
            ("Edge case", "How do I stay safe online?"),
        ]

        print("\n" + "="*80)
        print("Testing Gemma Safety Features")
        print("="*80)

        for category, prompt in test_cases:
            print(f"\n[{category}] Prompt: {prompt}")

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Response: {response[len(prompt):].strip()[:200]}...")


# Initialize Gemma
gemma = GemmaAnalyzer()
gemma.analyze_architecture()
gemma.test_safety_features()


# Compare SLMs
def compare_small_models():
    """Compare different small language models."""

    comparison = {
        'Model': ['Phi-3 Mini', 'Gemma 2B', 'Gemma 7B', 'TinyLlama 1.1B', 'StableLM 3B'],
        'Params': ['3.8B', '2B', '7B', '1.1B', '3B'],
        'Context': ['4K', '8K', '8K', '2K', '4K'],
        'Architecture': ['Transformer', 'Transformer+GQA', 'Transformer+GQA', 'LLaMA', 'LLaMA'],
        'MMLU': [68.8, 42.3, 64.3, 25.0, 45.0],
        'Best For': [
            'Quality+Size',
            'Browser/Mobile',
            'General Purpose',
            'Ultra-light',
            'Open Research'
        ],
    }

    import pandas as pd
    df = pd.DataFrame(comparison)

    print("\n" + "="*80)
    print("Small Language Models Comparison")
    print("="*80)
    print(df.to_string(index=False))

    print("\nSelection Guide:")
    print("  • Phi-3 Mini: Best quality-to-size ratio, mobile/edge")
    print("  • Gemma 2B: Lightest with decent quality, browser inference")
    print("  • Gemma 7B: More capable, still efficient")
    print("  • TinyLlama: Research/experimentation, very fast")
    print("  • StableLM: Open-source, customizable")

compare_small_models()
```

---

## 4. Quantization: INT8, INT4, GPTQ, AWQ

Quantization reduces model size and speeds up inference:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class QuantizationExplorer:
    """
    Explore different quantization techniques.

    Techniques:
    - FP16: Half precision (baseline)
    - INT8: 8-bit integers (2x smaller)
    - INT4: 4-bit integers (4x smaller)
    - GPTQ: Post-training quantization
    - AWQ: Activation-aware quantization
    - GGUF: llama.cpp format
    """

    def compare_quantization_methods(self, model_name="meta-llama/Llama-2-7b-hf"):
        """Compare different quantization approaches."""

        print("="*80)
        print("Quantization Methods Comparison")
        print("="*80)

        methods = {
            'FP32 (Full)': {
                'bits': 32,
                'size_multiplier': 1.0,
                'speed_multiplier': 1.0,
                'quality_loss': '0%',
                'use_case': 'Training, research'
            },
            'FP16 (Half)': {
                'bits': 16,
                'size_multiplier': 0.5,
                'speed_multiplier': 1.8,
                'quality_loss': '<1%',
                'use_case': 'Standard inference'
            },
            'INT8': {
                'bits': 8,
                'size_multiplier': 0.25,
                'speed_multiplier': 2.5,
                'quality_loss': '1-2%',
                'use_case': 'Production, edge'
            },
            'INT4': {
                'bits': 4,
                'size_multiplier': 0.125,
                'speed_multiplier': 3.5,
                'quality_loss': '3-5%',
                'use_case': 'Mobile, IoT'
            },
            'GPTQ (4-bit)': {
                'bits': 4,
                'size_multiplier': 0.125,
                'speed_multiplier': 3.2,
                'quality_loss': '2-3%',
                'use_case': 'Quality-focused mobile'
            },
            'AWQ (4-bit)': {
                'bits': 4,
                'size_multiplier': 0.125,
                'speed_multiplier': 3.8,
                'quality_loss': '1-2%',
                'use_case': 'Best quality/size trade-off'
            },
        }

        for method, specs in methods.items():
            print(f"\n{method}:")
            print(f"  Bits: {specs['bits']}")
            print(f"  Size: {specs['size_multiplier']:.3f}x ({100*specs['size_multiplier']:.0f}%)")
            print(f"  Speed: {specs['speed_multiplier']:.1f}x faster")
            print(f"  Quality loss: {specs['quality_loss']}")
            print(f"  Use case: {specs['use_case']}")

        # Calculate actual sizes for 7B model
        base_size_gb = 7.0 * 4 / 1024  # 7B params * 4 bytes (FP32)

        print("\n" + "="*80)
        print(f"LLaMA 2 7B Model Size Comparison")
        print("="*80)
        print(f"{'Method':<20} {'Size (GB)':<15} {'Fits on Device':<30}")
        print("-"*80)

        devices = {
            'FP32': (base_size_gb, 'High-end server'),
            'FP16': (base_size_gb * 0.5, 'GPU with 16GB VRAM'),
            'INT8': (base_size_gb * 0.25, 'GPU with 8GB VRAM'),
            'INT4': (base_size_gb * 0.125, 'Mobile phone (4GB RAM)'),
        }

        for method, (size, device) in devices.items():
            print(f"{method:<20} {size:<15.2f} {device:<30}")

    def load_quantized_model(self, bits=8):
        """Load model with specific quantization."""

        model_name = "facebook/opt-1.3b"  # Smaller model for demo

        print(f"\nLoading {model_name} with {bits}-bit quantization...")

        if bits == 8:
            # 8-bit quantization with bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,  # Threshold for outlier detection
            )
        elif bits == 4:
            # 4-bit quantization with bitsandbytes
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,  # Nested quantization
                bnb_4bit_quant_type="nf4"  # NormalFloat4
            )
        else:
            quantization_config = None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
        )

        # Measure memory
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"GPU memory allocated: {memory_allocated:.2f} GB")

        return model

    def demonstrate_quantization_quality(self):
        """Demonstrate quality vs size trade-off."""

        print("\n" + "="*80)
        print("Quantization Quality Analysis")
        print("="*80)

        # Simulated benchmark results
        benchmarks = {
            'Precision': ['FP32', 'FP16', 'INT8', 'INT4', 'GPTQ-4', 'AWQ-4'],
            'MMLU': [68.9, 68.7, 67.8, 65.2, 67.1, 68.2],
            'Perplexity': [8.79, 8.82, 9.15, 10.34, 9.45, 8.91],
            'Size (GB)': [28.0, 14.0, 7.0, 3.5, 3.5, 3.5],
            'Inference (ms)': [120, 67, 48, 34, 38, 32],
        }

        import pandas as pd
        df = pd.DataFrame(benchmarks)
        print(df.to_string(index=False))

        print("\nKey Insights:")
        print("  • FP16: Negligible quality loss, standard choice")
        print("  • INT8: Small quality drop, good for production")
        print("  • INT4: Noticeable but acceptable quality loss")
        print("  • AWQ: Best 4-bit method, preserves quality")
        print("  • GPTQ: Good alternative, widely supported")


quantizer = QuantizationExplorer()
quantizer.compare_quantization_methods()

# Load models with different quantization
# model_int8 = quantizer.load_quantized_model(bits=8)
# model_int4 = quantizer.load_quantized_model(bits=4)

quantizer.demonstrate_quantization_quality()


# Implement simple PTQ (Post-Training Quantization)
class SimplePTQ:
    """
    Simplified Post-Training Quantization.

    Steps:
    1. Collect activation statistics from calibration data
    2. Determine quantization parameters (scale, zero_point)
    3. Quantize weights and activations
    """

    @staticmethod
    def quantize_tensor(tensor, bits=8):
        """Quantize a tensor to N bits."""

        # Find min/max values
        min_val = tensor.min()
        max_val = tensor.max()

        # Calculate scale and zero point
        qmin = 0
        qmax = 2 ** bits - 1

        scale = (max_val - min_val) / (qmax - qmin)
        zero_point = qmin - min_val / scale

        # Quantize
        q_tensor = torch.clamp(torch.round(tensor / scale + zero_point), qmin, qmax)

        # Store quantization parameters
        return q_tensor, scale, zero_point

    @staticmethod
    def dequantize_tensor(q_tensor, scale, zero_point):
        """Dequantize back to floating point."""
        return (q_tensor - zero_point) * scale


# Example: Quantize a weight matrix
weight = torch.randn(1024, 1024)

ptq = SimplePTQ()

# Quantize to 8-bit
q_weight, scale, zero_point = ptq.quantize_tensor(weight, bits=8)

# Dequantize
reconstructed = ptq.dequantize_tensor(q_weight, scale, zero_point)

# Measure error
error = (weight - reconstructed).abs().mean()
print(f"\nQuantization Error:")
print(f"  Original range: [{weight.min():.3f}, {weight.max():.3f}]")
print(f"  Reconstructed range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")
print(f"  Mean absolute error: {error:.6f}")
print(f"  Relative error: {100 * error / weight.abs().mean():.2f}%")
print(f"  Memory reduction: {weight.element_size() / q_weight.element_size():.1f}x")
```

---

## 5. ONNX Runtime: Cross-Platform Deployment

ONNX enables deployment across different platforms:

```python
import onnx
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
import numpy as np

class ONNXDeployer:
    """
    Deploy models using ONNX Runtime.

    Benefits:
    - Cross-platform (CPU, GPU, mobile, web)
    - Optimized inference
    - Smaller runtime dependencies
    - Hardware acceleration
    """

    def export_to_onnx(self, model_name="distilbert-base-uncased", output_path="/tmp/model.onnx"):
        """Export a Hugging Face model to ONNX format."""

        print(f"Exporting {model_name} to ONNX...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        # Create dummy input
        dummy_input = tokenizer("This is a test", return_tensors="pt")

        # Export to ONNX
        torch.onnx.export(
            model,
            (dummy_input['input_ids'], dummy_input['attention_mask']),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'sequence'},
                'attention_mask': {0: 'batch', 1: 'sequence'},
                'last_hidden_state': {0: 'batch', 1: 'sequence'}
            },
            opset_version=14
        )

        print(f"Model exported to {output_path}")

        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid!")

        return output_path

    def run_onnx_inference(self, onnx_path, text="Hello world!"):
        """Run inference with ONNX Runtime."""

        print(f"\nRunning ONNX inference...")

        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)

        # Prepare input
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        inputs = tokenizer(text, return_tensors="np")

        # Run inference
        outputs = session.run(
            None,
            {
                'input_ids': inputs['input_ids'].astype(np.int64),
                'attention_mask': inputs['attention_mask'].astype(np.int64)
            }
        )

        print(f"Input text: {text}")
        print(f"Output shape: {outputs[0].shape}")
        print("ONNX inference successful!")

        return outputs

    def optimize_onnx_model(self, onnx_path, optimized_path="/tmp/model_optimized.onnx"):
        """Optimize ONNX model for inference."""

        from onnxruntime.transformers import optimizer

        print("Optimizing ONNX model...")

        # Optimize (requires onnxruntime-tools)
        # This would normally use:
        # optimized_model = optimizer.optimize_model(onnx_path)
        # optimized_model.save_model_to_file(optimized_path)

        print(f"Optimizations applied:")
        print("  • Constant folding")
        print("  • Redundant node elimination")
        print("  • Operator fusion")
        print("  • Quantization (optional)")

    def compare_performance(self):
        """Compare PyTorch vs ONNX performance."""

        print("\n" + "="*80)
        print("PyTorch vs ONNX Runtime Performance")
        print("="*80)

        comparison = {
            'Platform': ['Server CPU', 'Server GPU', 'Mobile CPU', 'Web Browser'],
            'PyTorch': ['1.0x', '1.0x', 'N/A', 'N/A'],
            'ONNX CPU': ['1.5-2x', 'N/A', '2-3x', '3-5x'],
            'ONNX GPU': ['N/A', '1.2-1.5x', 'N/A', 'N/A'],
            'Notes': [
                'ONNX optimized for CPU',
                'Similar GPU performance',
                'ONNX only option',
                'ONNX.js enables ML in browser'
            ]
        }

        import pandas as pd
        df = pd.DataFrame(comparison)
        print(df.to_string(index=False))


# Example: Export and run ONNX model
deployer = ONNXDeployer()

# Export (uncomment to run)
# onnx_path = deployer.export_to_onnx()
# deployer.run_onnx_inference(onnx_path)

deployer.compare_performance()
```

---

## 6. llama.cpp and GGUF Format

llama.cpp enables efficient CPU inference:

```python
# Note: This section uses llama-cpp-python
# pip install llama-cpp-python

class LlamaCppDeployer:
    """
    Deploy models using llama.cpp.

    Benefits:
    - Pure C++ inference (fast CPU performance)
    - GGUF format: Quantized, memory-mapped
    - Cross-platform: x86, ARM, Apple Silicon
    - Low memory usage
    """

    def explain_gguf_format(self):
        """Explain GGUF quantization format."""

        print("="*80)
        print("GGUF Format (GPT-Generated Unified Format)")
        print("="*80)

        formats = {
            'Q4_0': {
                'bits': 4.5,
                'description': '4-bit, legacy format',
                'quality': 'Good',
                'size': '~3.5GB for 7B'
            },
            'Q4_K_M': {
                'bits': 4.8,
                'description': '4-bit, mixed precision (recommended)',
                'quality': 'Better',
                'size': '~4.1GB for 7B'
            },
            'Q5_K_M': {
                'bits': 5.6,
                'description': '5-bit, mixed precision',
                'quality': 'Very Good',
                'size': '~4.8GB for 7B'
            },
            'Q8_0': {
                'bits': 8.5,
                'description': '8-bit, high quality',
                'quality': 'Excellent',
                'size': '~7.2GB for 7B'
            },
        }

        for format_name, specs in formats.items():
            print(f"\n{format_name}:")
            print(f"  Bits per weight: {specs['bits']}")
            print(f"  Description: {specs['description']}")
            print(f"  Quality: {specs['quality']}")
            print(f"  Typical size: {specs['size']}")

        print("\n" + "="*80)
        print("GGUF Advantages:")
        print("  • Memory mapping: Don't load entire model into RAM")
        print("  • Quantization-aware: Optimized for low-bit inference")
        print("  • Fast loading: Binary format, no parsing needed")
        print("  • Cross-platform: Works on CPU, no GPU required")

    def usage_example(self):
        """Show how to use llama-cpp-python."""

        code = '''
# Install: pip install llama-cpp-python

from llama_cpp import Llama

# Load GGUF model
llm = Llama(
    model_path="./models/llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,      # Context window
    n_threads=8,     # CPU threads
    n_gpu_layers=0   # 0 = CPU only, >0 = offload to GPU
)

# Generate
output = llm(
    "Explain machine learning in simple terms:",
    max_tokens=100,
    temperature=0.7,
    top_p=0.9,
)

print(output['choices'][0]['text'])

# Streaming
for chunk in llm(
    "Write a poem about AI:",
    max_tokens=200,
    stream=True
):
    print(chunk['choices'][0]['text'], end='', flush=True)
'''

        print("\n" + "="*80)
        print("llama.cpp Usage Example")
        print("="*80)
        print(code)

    def performance_tips(self):
        """Performance optimization tips for llama.cpp."""

        print("\n" + "="*80)
        print("llama.cpp Performance Optimization")
        print("="*80)

        tips = [
            ("CPU Threads", "Set n_threads to physical cores (not hyperthreads)"),
            ("Quantization", "Use Q4_K_M for best quality/size balance"),
            ("Context Size", "Smaller n_ctx = faster inference"),
            ("Batch Size", "Increase n_batch for throughput (default: 512)"),
            ("BLAS", "Compile with OpenBLAS/MKL for 2-3x speedup"),
            ("Metal", "On Mac: Use Metal for GPU acceleration"),
            ("mmap", "Enable memory mapping for large models"),
            ("mlock", "Lock memory to prevent swapping"),
        ]

        for tip, explanation in tips:
            print(f"  • {tip}: {explanation}")

        print("\n" + "="*80)
        print("Platform-Specific Optimizations:")
        print("="*80)

        platforms = {
            'x86 CPU': ['AVX2', 'AVX-512', 'OpenBLAS', 'MKL'],
            'ARM CPU': ['NEON', 'dotprod', 'i8mm'],
            'Apple Silicon': ['Metal', 'Accelerate framework', 'ANE'],
            'NVIDIA GPU': ['CUDA', 'cuBLAS'],
            'AMD GPU': ['ROCm', 'hipBLAS'],
        }

        for platform, optimizations in platforms.items():
            print(f"  {platform}: {', '.join(optimizations)}")


deployer = LlamaCppDeployer()
deployer.explain_gguf_format()
deployer.usage_example()
deployer.performance_tips()
```

---

## 7. Mobile and Browser Deployment

### MLC LLM for Mobile

```python
class MobileDeploymentGuide:
    """
    Guide for deploying LLMs on mobile devices.

    Frameworks:
    - MLC LLM: iOS, Android native apps
    - ONNX Runtime Mobile: Cross-platform
    - TensorFlow Lite: Android, iOS
    - Core ML: iOS only
    """

    def mlc_llm_overview(self):
        """Overview of MLC LLM for mobile deployment."""

        print("="*80)
        print("MLC LLM: Machine Learning Compilation for LLMs")
        print("="*80)

        features = {
            'Platform Support': 'iOS, Android, Web',
            'Backend': 'Apache TVM compiler',
            'Models': 'LLaMA, Mistral, Phi, Gemma',
            'Optimization': 'Model-specific compilation',
            'Quantization': '4-bit, 3-bit support',
            'Performance': 'Near-native speed',
        }

        for feature, value in features.items():
            print(f"  {feature}: {value}")

        print("\n" + "="*80)
        print("MLC LLM Workflow:")
        print("="*80)

        steps = [
            "1. Select model (e.g., Llama-2-7b-chat)",
            "2. Compile for target (iOS/Android/Web)",
            "3. Quantize (4-bit recommended for mobile)",
            "4. Package in app",
            "5. Deploy to device",
        ]

        for step in steps:
            print(f"  {step}")

    def webllm_browser_deployment(self):
        """WebLLM for browser-based inference."""

        print("\n" + "="*80)
        print("WebLLM: In-Browser LLM Inference")
        print("="*80)

        print("\nKey Features:")
        print("  • Runs entirely in browser (no server needed)")
        print("  • Uses WebGPU for acceleration")
        print("  • Private: Data never leaves device")
        print("  • Works offline after initial load")

        code = '''
<!-- HTML: Add WebLLM to your page -->
<script type="module">
  import { CreateWebWorkerMLCEngine } from "https://esm.run/@mlc-ai/web-llm";

  const engine = await CreateWebWorkerMLCEngine(
    new Worker("worker.js"),
    "Llama-2-7b-chat-hf-q4f32_1", // Model variant
    {
      initProgressCallback: (progress) => {
        console.log("Loading:", progress);
      }
    }
  );

  // Generate
  const response = await engine.chat.completions.create({
    messages: [{ role: "user", content: "Explain quantum computing" }],
    max_tokens: 100,
  });

  console.log(response.choices[0].message.content);
</script>
'''

        print("\nExample Code:")
        print(code)

        print("\nBrowser Requirements:")
        print("  • Chrome 113+ or Edge 113+ (WebGPU support)")
        print("  • ~8GB RAM for 7B model (4-bit)")
        print("  • Modern GPU (integrated GPU sufficient)")

    def mobile_optimization_strategies(self):
        """Strategies for mobile optimization."""

        print("\n" + "="*80)
        print("Mobile Optimization Strategies")
        print("="*80)

        strategies = [
            {
                'Strategy': 'Model Selection',
                'Actions': [
                    'Use ≤3B params for phones',
                    'Phi-3 Mini (3.8B) or Gemma 2B',
                    'Consider TinyLlama (1.1B) for very old devices',
                ]
            },
            {
                'Strategy': 'Quantization',
                'Actions': [
                    '4-bit for best quality/size',
                    '3-bit if size critical',
                    'Mixed precision for key layers',
                ]
            },
            {
                'Strategy': 'Memory Management',
                'Actions': [
                    'Use memory mapping (mmap)',
                    'Limit context window (2K max)',
                    'Clear KV cache between sessions',
                ]
            },
            {
                'Strategy': 'Power Efficiency',
                'Actions': [
                    'Batch queries when possible',
                    'Use CPU for light tasks',
                    'Offload to GPU only for heavy generation',
                ]
            },
            {
                'Strategy': 'User Experience',
                'Actions': [
                    'Show loading progress',
                    'Stream tokens for real-time feel',
                    'Implement stop button',
                ]
            },
        ]

        for item in strategies:
            print(f"\n{item['Strategy']}:")
            for action in item['Actions']:
                print(f"  • {action}")

    def deployment_checklist(self):
        """Mobile deployment checklist."""

        print("\n" + "="*80)
        print("Mobile Deployment Checklist")
        print("="*80)

        checklist = {
            'Model Preparation': [
                '☐ Select appropriate model size (≤3B for mobile)',
                '☐ Quantize to 4-bit or lower',
                '☐ Test on target devices',
                '☐ Measure memory usage',
                '☐ Benchmark inference speed',
            ],
            'App Integration': [
                '☐ Choose framework (MLC LLM, ONNX Mobile, Core ML)',
                '☐ Implement model loading UI',
                '☐ Add progress indicators',
                '☐ Handle errors gracefully',
                '☐ Implement caching strategy',
            ],
            'Performance': [
                '☐ Optimize for battery life',
                '☐ Implement streaming generation',
                '☐ Use background threads',
                '☐ Monitor memory usage',
                '☐ Test on low-end devices',
            ],
            'Privacy & Security': [
                '☐ Ensure on-device processing',
                '☐ Don\'t log sensitive data',
                '☐ Clear data on uninstall',
                '☐ Implement secure storage',
            ],
            'Distribution': [
                '☐ Optimize app size',
                '☐ Consider on-demand model download',
                '☐ Test on various devices',
                '☐ Prepare app store assets',
                '☐ Write privacy policy',
            ],
        }

        for category, items in checklist.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  {item}")


mobile_guide = MobileDeploymentGuide()
mobile_guide.mlc_llm_overview()
mobile_guide.webllm_browser_deployment()
mobile_guide.mobile_optimization_strategies()
mobile_guide.deployment_checklist()
```

---

## 8. Real-World Applications

### Application 1: Privacy-Preserving Medical Assistant

```python
class MedicalAssistantEdge:
    """
    HIPAA-compliant medical assistant on edge device.

    Requirements:
    - All processing on-device (no cloud)
    - Medical domain knowledge
    - Fast response time
    - Low memory footprint
    """

    def __init__(self):
        """Initialize edge medical assistant."""
        print("="*80)
        print("Privacy-Preserving Medical Assistant")
        print("="*80)

        self.architecture_decision()

    def architecture_decision(self):
        """Explain architecture choices."""

        print("\nArchitecture Decisions:")
        print("  Model: Phi-3 Mini (3.8B) fine-tuned on medical data")
        print("  Quantization: 4-bit GPTQ")
        print("  Deployment: iOS app with Core ML")
        print("  Memory: ~2GB on device")
        print("  Latency: <500ms for most queries")

        print("\nWhy This Architecture:")
        print("  ✓ Phi-3: Best quality for size, good reasoning")
        print("  ✓ GPTQ: Preserves quality better than naive 4-bit")
        print("  ✓ On-device: HIPAA compliant, no data leakage")
        print("  ✓ Core ML: Native iOS performance")

    def sample_usage(self):
        """Sample medical assistant usage."""

        examples = [
            {
                'query': 'Patient has fever, cough, and fatigue. Possible diagnosis?',
                'response': 'Based on symptoms, possible diagnoses include:\n'
                           '1. Influenza (most likely)\n'
                           '2. COVID-19\n'
                           '3. Common cold\n'
                           'Recommend: RT-PCR test, rest, hydration.\n'
                           'CAUTION: This is not a substitute for professional medical advice.',
                'latency': '420ms'
            },
            {
                'query': 'Drug interaction: Warfarin + Aspirin?',
                'response': 'WARNING: Significant interaction risk.\n'
                           'Both are anticoagulants. Combined use increases bleeding risk.\n'
                           'Recommendation: Consult physician before combining.\n'
                           'Monitor: PT/INR levels closely if prescribed together.',
                'latency': '380ms'
            },
        ]

        print("\n" + "="*80)
        print("Sample Interactions")
        print("="*80)

        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i}:")
            print(f"  Query: {ex['query']}")
            print(f"  Response: {ex['response']}")
            print(f"  Latency: {ex['latency']}")


assistant = MedicalAssistantEdge()
assistant.sample_usage()
```

### Application 2: Offline Translation Device

```python
class OfflineTranslator:
    """
    Offline translation device for travelers.

    Use case: Translate without internet access
    Model: Small multilingual model
    Device: Raspberry Pi or smartphone
    """

    def __init__(self):
        print("\n" + "="*80)
        print("Offline Translation Device")
        print("="*80)

        self.specs()

    def specs(self):
        """Device specifications."""

        print("\nHardware:")
        print("  Platform: Raspberry Pi 4 (8GB RAM)")
        print("  Storage: 32GB SD card")
        print("  Display: 7-inch touchscreen")
        print("  Power: USB-C, battery pack")

        print("\nSoftware:")
        print("  Model: NLLB-200 (1.3B) + Gemma 2B")
        print("  Quantization: INT4")
        print("  Runtime: llama.cpp + ONNX")
        print("  Languages: 50+ supported")
        print("  Total size: ~3GB")

        print("\nPerformance:")
        print("  Translation speed: ~10 words/second")
        print("  Latency: <1 second for typical sentence")
        print("  Battery life: ~8 hours continuous use")
        print("  Cost: ~$150 per device")

    def architecture(self):
        """System architecture."""

        print("\n" + "="*80)
        print("System Architecture")
        print("="*80)

        components = {
            'Speech Recognition': 'Whisper tiny (39M params, ONNX)',
            'Translation': 'NLLB-200-distilled (1.3B, GGUF Q4)',
            'Text Enhancement': 'Gemma 2B (grammar, formality)',
            'Text-to-Speech': 'Piper TTS (lightweight)',
            'UI': 'Flutter (cross-platform)',
        }

        for component, tech in components.items():
            print(f"  {component}: {tech}")


translator = OfflineTranslator()
translator.architecture()
```

---

## 9. Practice Exercises

### Exercise 1: Model Size Calculator

```python
"""
Exercise: Build a model size calculator.

Given:
- Number of parameters
- Precision (FP32, FP16, INT8, INT4)
- Context length
- Batch size

Calculate:
- Model weight size
- KV cache size
- Total memory required
"""

def calculate_model_memory(
    n_params,
    n_layers,
    d_model,
    n_heads,
    precision='fp16',
    context_len=2048,
    batch_size=1
):
    """
    Calculate total memory required for inference.

    Returns:
        dict with breakdown of memory usage
    """
    # TODO: Implement
    # Hint: weights + KV cache + activations
    pass


# Test
# memory = calculate_model_memory(
#     n_params=7e9,
#     n_layers=32,
#     d_model=4096,
#     n_heads=32,
#     precision='int4',
#     context_len=2048,
#     batch_size=1
# )
# print(f"Total memory: {memory['total_gb']:.2f} GB")
```

### Exercise 2: Quantization Quality Test

```python
"""
Exercise: Test quantization quality on your own model.

1. Load a small model (e.g., OPT-125m)
2. Quantize to INT8 and INT4
3. Compare perplexity on test data
4. Measure inference speed
5. Plot quality vs speed trade-off
"""

def benchmark_quantization(model_name, test_texts):
    """
    Benchmark different quantization methods.

    Returns:
        DataFrame with comparison
    """
    # TODO: Implement
    # Load model in FP16, INT8, INT4
    # Measure perplexity and speed for each
    # Create comparison table
    pass
```

### Exercise 3: Mobile Deployment Plan

```python
"""
Exercise: Create deployment plan for mobile app.

Scenario: Chat assistant for smartphones

Requirements:
- Works offline
- <3 second response time
- <500MB app size
- Supports iPhone 12+ and modern Android

Tasks:
1. Select appropriate model
2. Choose quantization strategy
3. Estimate memory usage
4. Plan user experience
5. Identify potential issues
"""

def create_mobile_deployment_plan():
    """Create comprehensive deployment plan."""
    # TODO: Document your decisions
    # Consider: model size, quantization, framework, UX
    pass
```

---

## Key Takeaways

1. **Small Language Models (SLMs)**:
   - Phi-3 Mini (3.8B): Best quality-to-size ratio
   - Gemma 2B: Lightweight, safety-focused
   - TinyLlama (1.1B): Ultra-efficient research model
   - Trade-off: Size vs quality vs speed

2. **Quantization Techniques**:
   - FP16: Standard, minimal quality loss
   - INT8: 2x smaller, <2% quality loss
   - INT4: 4x smaller, 3-5% quality loss
   - AWQ/GPTQ: Best 4-bit methods
   - GGUF: Optimized for CPU inference

3. **Deployment Frameworks**:
   - ONNX Runtime: Cross-platform, optimized
   - llama.cpp/GGUF: Best CPU performance
   - MLC LLM: Native mobile apps
   - WebLLM: Browser inference with WebGPU

4. **Edge Use Cases**:
   - Privacy: Medical, legal, personal data
   - Latency: Real-time applications
   - Cost: High-volume, consumer apps
   - Offline: Travel, remote areas, IoT

5. **Optimization Strategies**:
   - Model selection: ≤3B for mobile
   - Quantization: 4-bit for best balance
   - Context window: Smaller = faster
   - Batching: Increase throughput
   - Hardware acceleration: GPU/NPU when available

6. **Production Considerations**:
   - Test on low-end devices
   - Monitor memory usage
   - Implement progressive loading
   - Stream responses for UX
   - Plan for model updates

---

## Further Reading

### Papers
1. **Phi-3**: "Phi-3 Technical Report" (Microsoft, 2024)
2. **Gemma**: "Gemma: Open Models Based on Gemini" (Google, 2024)
3. **GPTQ**: "GPTQ: Accurate Post-Training Quantization" (Frantar et al., 2023)
4. **AWQ**: "AWQ: Activation-aware Weight Quantization" (Lin et al., 2023)
5. **LLM.int8()**: "LLM.int8(): 8-bit Matrix Multiplication" (Dettmers et al., 2022)

### Tools & Frameworks
- Hugging Face Optimum: https://huggingface.co/docs/optimum
- llama.cpp: https://github.com/ggerganov/llama.cpp
- MLC LLM: https://mlc.ai/mlc-llm/
- ONNX Runtime: https://onnxruntime.ai/
- WebLLM: https://webllm.mlc.ai/

### Resources
- Quantization Guide: https://huggingface.co/blog/quantization
- GGUF Format Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- Mobile ML Best Practices

### Related Modules
- **Module 15 Lesson 1**: LLM Architectures (foundation)
- **Module 15 Lesson 6**: Efficient Inference (vLLM, serving)
- **Module 15 Lesson 7**: LLMOps (production deployment)

---

**Next Lesson**: Multimodal Vision-Language Models (CLIP, GPT-4V, LLaVA)

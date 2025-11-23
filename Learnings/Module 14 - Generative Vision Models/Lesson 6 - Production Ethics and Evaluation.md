# Lesson 6: Production, Ethics, and Evaluation ⚖️

**Module 14: Generative Vision Models | Lesson 6 of 6**

Master deployment, optimization, safety, and responsible AI for generative vision models!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Optimize inference for production deployment
2. ✅ Implement safety filters and content moderation
3. ✅ Understand watermarking and AI provenance
4. ✅ Master evaluation metrics (FID, IS, CLIPScore)
5. ✅ Navigate copyright and IP considerations
6. ✅ Deploy models with FastAPI and serving infrastructure
7. ✅ Apply responsible AI principles to generative models

---

## Prerequisites

- **Lessons 1-5**: Complete understanding of generative models
- **Module 3**: Deep learning optimization
- **Python Libraries**: torch, diffusers, FastAPI, torchmetrics
- **Concepts**: Ethics, fairness, bias in AI

---

## 1. Inference Optimization

### Speed and Memory Improvements

```python
import torch
import time
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import numpy as np

"""
INFERENCE OPTIMIZATION

Production requirements:
✅ Fast generation (<10s per image)
✅ Low memory footprint
✅ Batch processing
✅ GPU efficiency

Techniques:
1. Half precision (FP16)
2. Attention slicing
3. VAE slicing
4. xFormers memory-efficient attention
5. torch.compile (PyTorch 2.0+)
6. Model distillation
"""


class OptimizedInference:
    """
    Production-optimized inference pipeline.
    """
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1",
                 device="cuda", enable_all_optimizations=True):
        """
        Initialize optimized pipeline.

        Args:
            model_id: Model to load
            device: 'cuda' or 'cpu'
            enable_all_optimizations: Enable all optimizations
        """
        self.device = device

        # Load pipeline with FP16
        print("Loading model with FP16...")
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,  # FP16 for 2x speedup
            safety_checker=None,  # Optional: disable for speed
        ).to(device)

        if enable_all_optimizations:
            self.apply_optimizations()

    def apply_optimizations(self):
        """Apply all optimization techniques."""
        print("Applying optimizations...")

        # 1. Attention slicing (reduce memory)
        self.pipe.enable_attention_slicing()
        print("  ✅ Attention slicing enabled")

        # 2. VAE slicing (for high-res images)
        self.pipe.enable_vae_slicing()
        print("  ✅ VAE slicing enabled")

        # 3. xFormers (if available)
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("  ✅ xFormers enabled")
        except:
            print("  ⚠️  xFormers not available (install for better performance)")

        # 4. Fast scheduler
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        print("  ✅ DPM-Solver scheduler (faster sampling)")

        # 5. Compile (PyTorch 2.0+)
        try:
            self.pipe.unet = torch.compile(
                self.pipe.unet,
                mode="reduce-overhead",
                fullgraph=True
            )
            print("  ✅ torch.compile enabled")
        except:
            print("  ⚠️  torch.compile not available (upgrade PyTorch)")

    def benchmark(self, prompt, num_runs=5):
        """
        Benchmark generation speed.

        Args:
            prompt: Test prompt
            num_runs: Number of runs to average

        Returns:
            Average time per image
        """
        print(f"\nBenchmarking with {num_runs} runs...")

        times = []

        # Warmup
        _ = self.pipe(prompt, num_inference_steps=20).images[0]

        for i in range(num_runs):
            start = time.time()
            _ = self.pipe(prompt, num_inference_steps=20).images[0]
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.2f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)

        print(f"\nAverage: {avg_time:.2f}s ± {std_time:.2f}s")

        return avg_time


# Example usage
optimized = OptimizedInference(enable_all_optimizations=True)

print("\nOptimization checklist:")
print("  ✅ FP16 precision (2x faster)")
print("  ✅ Attention slicing (lower memory)")
print("  ✅ VAE slicing (high-res support)")
print("  ✅ xFormers (if available)")
print("  ✅ Fast scheduler (DPM-Solver)")
print("  ✅ torch.compile (PyTorch 2.0+)")
```

### Batch Processing

```python
def batch_generation_optimized():
    """
    Efficient batch generation.
    """
    code = '''
def generate_batch(pipe, prompts, batch_size=4, **kwargs):
    """
    Generate multiple images efficiently.

    Args:
        pipe: StableDiffusionPipeline
        prompts: List of prompts
        batch_size: Number of images to generate simultaneously

    Returns:
        List of generated images
    """
    all_images = []

    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]

        # Generate batch
        images = pipe(
            batch_prompts,
            num_inference_steps=20,
            **kwargs
        ).images

        all_images.extend(images)

        print(f"Generated batch {i//batch_size + 1}")

    return all_images


# Example: Generate 20 images
prompts = [
    "a beautiful sunset over mountains",
    "a cat wearing sunglasses",
    # ... 18 more prompts
] * 10  # Expand to 20

images = generate_batch(pipe, prompts, batch_size=4)
print(f"Generated {len(images)} images")
'''

    print("BATCH GENERATION")
    print(code)
    print()
    print("Tips:")
    print("  - Larger batch_size = faster total time")
    print("  - Limited by GPU memory")
    print("  - Typical batch_size: 2-8")


batch_generation_optimized()
```

---

## 2. Model Deployment with FastAPI

### REST API for Image Generation

```python
"""
PRODUCTION DEPLOYMENT

Requirements:
1. REST API for generation requests
2. Queue management for concurrent requests
3. Caching and result storage
4. Error handling and logging
5. Rate limiting and authentication
"""


def fastapi_deployment_example():
    """
    Deploy Stable Diffusion with FastAPI.
    """
    code = '''
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, List
import torch
from diffusers import StableDiffusionPipeline
import io
import base64
from PIL import Image
import uuid

app = FastAPI(title="Image Generation API")

# Load model at startup
pipe = None

@app.on_event("startup")
async def load_model():
    """Load model when server starts."""
    global pipe
    pipe = StableDiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        torch_dtype=torch.float16
    ).to("cuda")

    # Apply optimizations
    pipe.enable_attention_slicing()
    pipe.enable_xformers_memory_efficient_attention()

    print("Model loaded and ready!")


class GenerationRequest(BaseModel):
    """Request schema for image generation."""
    prompt: str
    negative_prompt: Optional[str] = ""
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    num_images: int = 1


class GenerationResponse(BaseModel):
    """Response schema."""
    job_id: str
    images: List[str]  # Base64 encoded images
    prompt: str


@app.post("/generate", response_model=GenerationResponse)
async def generate_image(request: GenerationRequest):
    """
    Generate images from text prompt.

    Args:
        request: Generation parameters

    Returns:
        Generated images as base64 strings
    """
    try:
        # Set seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator("cuda").manual_seed(request.seed)

        # Generate
        result = pipe(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            num_images_per_prompt=request.num_images,
            generator=generator
        )

        # Convert to base64
        images_b64 = []
        for img in result.images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            images_b64.append(img_str)

        # Create response
        job_id = str(uuid.uuid4())

        return GenerationResponse(
            job_id=job_id,
            images=images_b64,
            prompt=request.prompt
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": pipe is not None}


# Run with: uvicorn main:app --host 0.0.0.0 --port 8000
'''

    print("FASTAPI DEPLOYMENT")
    print(code)
    print()
    print("Endpoints:")
    print("  POST /generate: Generate images")
    print("  GET /health: Health check")
    print()
    print("Usage:")
    print("  curl -X POST http://localhost:8000/generate \\")
    print("    -H 'Content-Type: application/json' \\")
    print("    -d '{\"prompt\": \"a cat\"}'")


fastapi_deployment_example()
```

---

## 3. Safety Filters and Content Moderation

### NSFW Detection and Filtering

```python
"""
CONTENT SAFETY

Critical for production deployment!

Challenges:
- NSFW content
- Harmful/offensive imagery
- Copyrighted material
- Deepfakes/misinformation

Solutions:
1. Built-in safety checker (CLIP-based)
2. Custom classifiers
3. Human-in-the-loop review
4. User reporting
"""


class SafetyPipeline:
    """
    Image generation with safety checks.
    """
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1"):
        """
        Initialize with safety checker enabled.
        """
        from diffusers import StableDiffusionPipeline

        # Load WITH safety checker
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            # safety_checker is included by default
        ).to("cuda")

        print("Safety checker enabled")

    def generate_safe(self, prompt, **kwargs):
        """
        Generate with safety filtering.

        Returns:
            images: Safe images (unsafe ones filtered)
            has_nsfw: List of booleans indicating filtered images
        """
        result = self.pipe(prompt, **kwargs)

        # Check results
        has_nsfw = result.nsfw_content_detected

        # Filter unsafe images
        safe_images = [
            img for img, is_nsfw in zip(result.images, has_nsfw)
            if not is_nsfw
        ]

        print(f"Generated {len(result.images)} images")
        print(f"Filtered {sum(has_nsfw)} unsafe images")
        print(f"Returned {len(safe_images)} safe images")

        return safe_images, has_nsfw


# Example
safety_pipe = SafetyPipeline()

print("\nSafety Features:")
print("  ✅ NSFW detection (CLIP-based)")
print("  ✅ Automatic filtering")
print("  ✅ Flagged images excluded from results")
```

### Custom Safety Classifier

```python
def custom_safety_classifier():
    """
    Implement custom safety classifier.
    """
    code = '''
from transformers import pipeline

class CustomSafetyChecker:
    """
    Multi-level safety checking.
    """
    def __init__(self):
        # NSFW classifier
        self.nsfw_classifier = pipeline(
            "image-classification",
            model="Falconsai/nsfw_image_detection"
        )

        # Violence detector
        self.violence_classifier = pipeline(
            "image-classification",
            model="your-violence-detector"
        )

    def check_image(self, image):
        """
        Run all safety checks.

        Args:
            image: PIL Image

        Returns:
            is_safe: Boolean
            flags: Dict of triggered filters
        """
        flags = {}

        # Check NSFW
        nsfw_result = self.nsfw_classifier(image)
        if nsfw_result[0]["label"] == "nsfw" and nsfw_result[0]["score"] > 0.7:
            flags["nsfw"] = True

        # Check violence
        violence_result = self.violence_classifier(image)
        if violence_result[0]["label"] == "violent" and violence_result[0]["score"] > 0.7:
            flags["violence"] = True

        # Add more checks...

        is_safe = len(flags) == 0

        return is_safe, flags


# Use in pipeline
safety_checker = CustomSafetyChecker()

def generate_with_custom_safety(prompt):
    images = pipe(prompt).images

    safe_images = []
    for img in images:
        is_safe, flags = safety_checker.check_image(img)

        if is_safe:
            safe_images.append(img)
        else:
            print(f"Filtered image: {flags}")

    return safe_images
'''

    print("CUSTOM SAFETY CLASSIFIER")
    print(code)
    print()
    print("Multi-level checks:")
    print("  1. NSFW content")
    print("  2. Violence/gore")
    print("  3. Hate symbols")
    print("  4. Copyright violations")
    print("  5. Deepfake markers")


custom_safety_classifier()
```

---

## 4. Watermarking and Provenance

### Invisible Watermarks for AI-Generated Content

```python
"""
WATERMARKING

Why watermark AI-generated images?
✅ Identify AI-generated content
✅ Track provenance and attribution
✅ Detect misuse and manipulation
✅ Enable content verification

Methods:
1. Invisible watermarks (embedded in pixels)
2. Metadata (EXIF tags)
3. Blockchain-based tracking
"""


def watermarking_example():
    """
    Add invisible watermark to generated images.
    """
    code = '''
from imwatermark import WatermarkEncoder, WatermarkDecoder
import numpy as np
from PIL import Image

class WatermarkedGenerator:
    """
    Generate images with invisible watermarks.
    """
    def __init__(self, pipe, watermark_text="ai-generated"):
        self.pipe = pipe
        self.watermark = watermark_text

        # Initialize watermark encoder
        self.encoder = WatermarkEncoder()
        self.encoder.set_watermark('bytes', watermark_text.encode('utf-8'))

    def generate_and_watermark(self, prompt, **kwargs):
        """
        Generate image with embedded watermark.

        Args:
            prompt: Text prompt
            **kwargs: Generation parameters

        Returns:
            Watermarked PIL Image
        """
        # Generate image
        image = self.pipe(prompt, **kwargs).images[0]

        # Convert to numpy
        img_array = np.array(image)

        # Add watermark (invisible)
        watermarked = self.encoder.encode(img_array, 'dwtDct')

        # Convert back to PIL
        watermarked_image = Image.fromarray(watermarked)

        return watermarked_image

    def detect_watermark(self, image):
        """
        Detect watermark in image.

        Args:
            image: PIL Image or numpy array

        Returns:
            detected_text: Watermark if found, None otherwise
        """
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Decode watermark
        decoder = WatermarkDecoder('bytes', len(self.watermark))
        detected = decoder.decode(image, 'dwtDct')

        try:
            detected_text = detected.decode('utf-8')
            return detected_text
        except:
            return None


# Example usage
watermarked_gen = WatermarkedGenerator(pipe, watermark_text="MyCompany-AI-2024")

# Generate with watermark
image = watermarked_gen.generate_and_watermark("a beautiful landscape")

# Later: Verify watermark
detected = watermarked_gen.detect_watermark(image)
print(f"Detected watermark: {detected}")
'''

    print("INVISIBLE WATERMARKING")
    print(code)
    print()
    print("Benefits:")
    print("  ✅ Invisible to human eye")
    print("  ✅ Survives compression/resizing")
    print("  ✅ Verifiable provenance")
    print("  ✅ Helps combat misinformation")


watermarking_example()
```

### Metadata and Tracking

```python
def metadata_tracking():
    """
    Add comprehensive metadata to generated images.
    """
    code = '''
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import json
from datetime import datetime

def save_with_metadata(image, filename, generation_params):
    """
    Save image with complete metadata.

    Args:
        image: PIL Image
        filename: Output filename
        generation_params: Dict with generation parameters
    """
    # Create metadata
    metadata = PngInfo()

    # Add generation parameters
    metadata.add_text("prompt", generation_params["prompt"])
    metadata.add_text("model", generation_params.get("model", "unknown"))
    metadata.add_text("timestamp", datetime.now().isoformat())
    metadata.add_text("generator", "Stable Diffusion")

    # Add all parameters as JSON
    params_json = json.dumps(generation_params)
    metadata.add_text("parameters", params_json)

    # Mark as AI-generated
    metadata.add_text("ai_generated", "true")

    # Save with metadata
    image.save(filename, pnginfo=metadata)

    print(f"Saved with metadata: {filename}")


def read_metadata(filename):
    """
    Read metadata from image.

    Args:
        filename: Image filename

    Returns:
        metadata: Dict of metadata
    """
    image = Image.open(filename)

    metadata = {}
    if hasattr(image, 'text'):
        for key, value in image.text.items():
            metadata[key] = value

    return metadata


# Example usage
generation_params = {
    "prompt": "a beautiful sunset",
    "model": "stabilityai/stable-diffusion-2-1",
    "guidance_scale": 7.5,
    "steps": 30,
    "seed": 42,
}

save_with_metadata(image, "output.png", generation_params)

# Later: Read metadata
metadata = read_metadata("output.png")
print(f"Image was generated with prompt: {metadata.get('prompt')}")
'''

    print("METADATA TRACKING")
    print(code)
    print()
    print("Metadata includes:")
    print("  - Original prompt")
    print("  - Model version")
    print("  - Generation parameters")
    print("  - Timestamp")
    print("  - AI-generated flag")


metadata_tracking()
```

---

## 5. Evaluation Metrics

### Quantitative Quality Assessment

```python
"""
EVALUATION METRICS

How to measure image generation quality?

Metrics:
1. FID (Fréchet Inception Distance): Distribution similarity
2. IS (Inception Score): Quality and diversity
3. CLIPScore: Text-image alignment
4. Human evaluation: Gold standard

Each measures different aspects!
"""


def compute_fid_score():
    """
    Compute FID score between real and generated images.
    """
    code = '''
from torchmetrics.image.fid import FrechetInceptionDistance
import torch

def calculate_fid(real_images_dir, generated_images_dir, device="cuda"):
    """
    Calculate FID score.

    Args:
        real_images_dir: Directory with real images
        generated_images_dir: Directory with generated images
        device: 'cuda' or 'cpu'

    Returns:
        FID score (lower is better)
    """
    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=2048).to(device)

    # Load and process real images
    real_images = load_images(real_images_dir)  # Returns tensor [N, 3, H, W]
    real_images = (real_images * 255).to(torch.uint8)

    # Load and process generated images
    gen_images = load_images(generated_images_dir)
    gen_images = (gen_images * 255).to(torch.uint8)

    # Update metric
    fid.update(real_images, real=True)
    fid.update(gen_images, real=False)

    # Compute FID
    fid_score = fid.compute()

    print(f"FID Score: {fid_score:.2f}")
    print(f"  (Lower is better)")

    return fid_score.item()


# Interpret FID:
# FID < 10: Excellent (near-perfect)
# FID 10-50: Good
# FID 50-100: Moderate
# FID > 100: Poor
'''

    print("FID (FRÉCHET INCEPTION DISTANCE)")
    print(code)
    print()
    print("What FID measures:")
    print("  - Distribution similarity between real and generated")
    print("  - Based on Inception-v3 features")
    print("  - Lower = better (0 = identical distributions)")


compute_fid_score()
```

### CLIP Score for Text-Image Alignment

```python
def compute_clip_score():
    """
    Measure text-image alignment with CLIP.
    """
    code = '''
from transformers import CLIPProcessor, CLIPModel
import torch

class CLIPScoreEvaluator:
    """
    Evaluate text-image alignment using CLIP.
    """
    def __init__(self, device="cuda"):
        self.device = device

        # Load CLIP
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def compute_score(self, image, text):
        """
        Compute CLIP score for single image-text pair.

        Args:
            image: PIL Image
            text: Text prompt

        Returns:
            CLIP score (0-100, higher is better)
        """
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # Get embeddings
        outputs = self.model(**inputs)

        # Compute similarity
        logits_per_image = outputs.logits_per_image
        score = logits_per_image.item()

        return score

    def evaluate_batch(self, images, prompts):
        """
        Evaluate multiple images.

        Args:
            images: List of PIL Images
            prompts: List of text prompts

        Returns:
            Average CLIP score
        """
        scores = []

        for img, prompt in zip(images, prompts):
            score = self.compute_score(img, prompt)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        print(f"Average CLIP Score: {avg_score:.2f}")
        print(f"  Min: {min(scores):.2f}")
        print(f"  Max: {max(scores):.2f}")

        return avg_score


# Example usage
evaluator = CLIPScoreEvaluator()

# Generate images
prompts = ["a cat", "a dog", "a bird"]
images = [pipe(p).images[0] for p in prompts]

# Evaluate
avg_score = evaluator.evaluate_batch(images, prompts)
'''

    print("CLIP SCORE")
    print(code)
    print()
    print("What CLIP Score measures:")
    print("  - How well image matches text prompt")
    print("  - Based on CLIP embeddings")
    print("  - Higher = better alignment")
    print("  - Typical range: 20-35 for good matches")


compute_clip_score()
```

### Human Evaluation

```python
def human_evaluation_framework():
    """
    Design human evaluation study.
    """
    print("HUMAN EVALUATION\n")

    print("Metrics to evaluate:")
    criteria = {
        "Photorealism": "How realistic does the image look?",
        "Prompt Adherence": "How well does it match the text?",
        "Composition": "Is the layout/framing good?",
        "Detail Quality": "Are details sharp and coherent?",
        "Artifacts": "Any visible errors or glitches?",
    }

    for criterion, question in criteria.items():
        print(f"  {criterion}: {question}")
        print(f"    Scale: 1-5 (1=poor, 5=excellent)")
        print()

    print("Study design:")
    print("  1. Generate diverse set of images")
    print("  2. Recruit evaluators (10-50 people)")
    print("  3. Random order presentation")
    print("  4. Rate on multiple criteria")
    print("  5. Statistical analysis (inter-rater agreement)")
    print()

    print("Gold standard: Turing test")
    print("  Can humans distinguish real vs generated?")


human_evaluation_framework()
```

---

## 6. Copyright and Legal Considerations

### Navigating IP in Generative AI

```python
"""
COPYRIGHT & INTELLECTUAL PROPERTY

Key Issues:
1. Training data copyright
2. Generated content ownership
3. Style mimicry (living artists)
4. Fair use vs infringement
5. Commercial usage rights

This is an evolving legal landscape!
"""


def copyright_considerations():
    """
    Overview of copyright issues.
    """
    print("COPYRIGHT CONSIDERATIONS\n")

    issues = {
        "Training Data": {
            "question": "Can you train on copyrighted images?",
            "status": "Disputed - ongoing lawsuits",
            "best_practice": "Use public domain or licensed data"
        },
        "Generated Outputs": {
            "question": "Who owns generated images?",
            "status": "Varies by jurisdiction",
            "best_practice": "Check service ToS, may not be copyrightable"
        },
        "Style Mimicry": {
            "question": "Can you generate in artist's style?",
            "status": "Legally unclear, ethically questionable",
            "best_practice": "Avoid mimicking living artists without permission"
        },
        "Commercial Use": {
            "question": "Can you sell generated images?",
            "status": "Depends on model license",
            "best_practice": "Check license (e.g., CreativeML OpenRAIL)"
        },
        "Derivative Works": {
            "question": "Can you modify copyrighted images?",
            "status": "May violate copyright",
            "best_practice": "Only with permission or fair use"
        },
    }

    for topic, info in issues.items():
        print(f"{topic}:")
        for key, value in info.items():
            print(f"  {key.title()}: {value}")
        print()

    print("DISCLAIMER: Not legal advice!")
    print("Consult lawyer for specific situations.")


copyright_considerations()
```

### Responsible AI Principles

```python
def responsible_ai_guidelines():
    """
    Framework for responsible generative AI.
    """
    print("RESPONSIBLE AI PRINCIPLES\n")

    principles = {
        "Transparency": [
            "Disclose when content is AI-generated",
            "Make generation parameters available",
            "Document training data sources"
        ],
        "Consent": [
            "Don't generate images of people without consent",
            "Respect opt-out requests from artists",
            "Honor copyright and attribution"
        ],
        "Safety": [
            "Implement content filtering",
            "Prevent harmful/illegal content",
            "Age-appropriate safeguards"
        ],
        "Fairness": [
            "Mitigate demographic biases",
            "Ensure diverse representation",
            "Monitor for discriminatory outputs"
        ],
        "Accountability": [
            "Enable watermarking and tracking",
            "Provide abuse reporting mechanisms",
            "Maintain audit logs"
        ],
        "Privacy": [
            "Don't memorize training data",
            "Protect user prompts and outputs",
            "Allow data deletion requests"
        ],
    }

    for principle, guidelines in principles.items():
        print(f"{principle}:")
        for guideline in guidelines:
            print(f"  ✅ {guideline}")
        print()

    print("Build AI that benefits society! 🌟")


responsible_ai_guidelines()
```

---

## 7. Cost Optimization and Scaling

### Production Infrastructure

```python
"""
COST & SCALING CONSIDERATIONS

Production costs:
- GPU compute ($1-5 per hour)
- Storage (model weights, outputs)
- Bandwidth (API calls, image delivery)
- Caching and CDN

Optimization strategies:
1. Request batching
2. Result caching
3. Model distillation
4. Edge deployment
5. Queue management
"""


class ProductionInfrastructure:
    """
    Production-ready infrastructure design.
    """
    def __init__(self):
        self.cache = {}  # Simple cache (use Redis in production)

    def estimate_costs(self, requests_per_day, avg_images_per_request=1):
        """
        Estimate daily costs.

        Args:
            requests_per_day: Expected requests
            avg_images_per_request: Images per request

        Returns:
            cost_breakdown: Dict with cost estimates
        """
        # Assumptions
        seconds_per_image = 5  # With optimizations
        gpu_cost_per_hour = 2.0  # USD
        storage_cost_per_gb_month = 0.02  # USD
        bandwidth_cost_per_gb = 0.09  # USD

        # Compute
        total_images = requests_per_day * avg_images_per_request
        compute_hours = (total_images * seconds_per_image) / 3600
        compute_cost = compute_hours * gpu_cost_per_hour

        # Storage (assume 2MB per image)
        daily_storage_gb = (total_images * 2) / 1024
        storage_cost = (daily_storage_gb * 30 * storage_cost_per_gb_month) / 30

        # Bandwidth (assume 2MB per image delivered)
        bandwidth_gb = daily_storage_gb
        bandwidth_cost = bandwidth_gb * bandwidth_cost_per_gb

        # Total
        daily_cost = compute_cost + storage_cost + bandwidth_cost
        monthly_cost = daily_cost * 30

        costs = {
            "daily": {
                "compute": compute_cost,
                "storage": storage_cost,
                "bandwidth": bandwidth_cost,
                "total": daily_cost,
            },
            "monthly": {
                "total": monthly_cost,
            },
            "metrics": {
                "requests_per_day": requests_per_day,
                "images_per_day": total_images,
                "cost_per_image": daily_cost / total_images,
            }
        }

        return costs

    def print_cost_estimate(self, requests_per_day=1000):
        """Print cost breakdown."""
        costs = self.estimate_costs(requests_per_day)

        print(f"COST ESTIMATE ({requests_per_day:,} requests/day)\n")

        print("Daily Costs:")
        print(f"  Compute: ${costs['daily']['compute']:.2f}")
        print(f"  Storage: ${costs['daily']['storage']:.2f}")
        print(f"  Bandwidth: ${costs['daily']['bandwidth']:.2f}")
        print(f"  Total: ${costs['daily']['total']:.2f}")
        print()

        print(f"Monthly: ${costs['monthly']['total']:.2f}")
        print()

        print(f"Cost per image: ${costs['metrics']['cost_per_image']:.4f}")


# Example
infra = ProductionInfrastructure()
infra.print_cost_estimate(requests_per_day=10000)

print("\nOptimization strategies:")
print("  ✅ Cache popular prompts")
print("  ✅ Batch similar requests")
print("  ✅ Use spot instances")
print("  ✅ Implement rate limiting")
print("  ✅ CDN for image delivery")
```

---

## 8. Monitoring and Observability

### Production Monitoring

```python
def monitoring_setup():
    """
    Set up comprehensive monitoring.
    """
    code = '''
import logging
import time
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
generation_requests = Counter(
    'generation_requests_total',
    'Total generation requests',
    ['status']
)

generation_duration = Histogram(
    'generation_duration_seconds',
    'Time to generate images',
    buckets=[1, 5, 10, 30, 60, 120]
)

safety_filter_triggers = Counter(
    'safety_filter_triggers_total',
    'Number of safety filter activations',
    ['filter_type']
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('generation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@app.post("/generate")
async def generate_monitored(request: GenerationRequest):
    """Generate with comprehensive monitoring."""

    start_time = time.time()

    try:
        # Generate image
        result = pipe(request.prompt, **request.dict())

        # Check safety
        if result.nsfw_content_detected[0]:
            safety_filter_triggers.labels(filter_type='nsfw').inc()
            logger.warning(f"NSFW content filtered for prompt: {request.prompt[:50]}")

        # Success metrics
        duration = time.time() - start_time
        generation_duration.observe(duration)
        generation_requests.labels(status='success').inc()

        logger.info(f"Generated image in {duration:.2f}s for prompt: {request.prompt[:50]}")

        return result

    except Exception as e:
        # Error metrics
        generation_requests.labels(status='error').inc()
        logger.error(f"Generation failed: {str(e)}")
        raise


# Start Prometheus metrics server
start_http_server(8001)  # Metrics at :8001/metrics
'''

    print("MONITORING & OBSERVABILITY")
    print(code)
    print()
    print("Key metrics:")
    print("  - Request count and success rate")
    print("  - Generation latency (p50, p95, p99)")
    print("  - Safety filter triggers")
    print("  - GPU utilization")
    print("  - Memory usage")
    print("  - Queue depth")


monitoring_setup()
```

---

## Practice Exercises

### Exercise 1: Production Deployment

```python
"""
Deploy complete production system.

Requirements:
1. FastAPI server with optimizations
2. Safety filtering
3. Watermarking
4. Monitoring and logging
5. Rate limiting
6. Caching for common prompts

Test:
- Load testing (concurrent requests)
- Error handling
- Resource usage

Deliverable: Production-ready API
"""

# Your implementation here
```

### Exercise 2: Evaluation Suite

```python
"""
Build comprehensive evaluation pipeline.

Tasks:
1. Generate 100 images with various prompts
2. Compute FID score (vs real images)
3. Calculate CLIP scores
4. Run safety checks
5. Create evaluation dashboard

Compare:
- Different models
- Different parameters
- Fine-tuned vs base

Report: Which configuration is best?
"""

# Your implementation here
```

### Exercise 3: Responsible AI Audit

```python
"""
Audit model for biases and safety.

Audit criteria:
1. Demographic representation (generate diverse people)
2. Stereotype amplification
3. Safety filter effectiveness
4. Copyright compliance
5. Watermark robustness

Document:
- Findings
- Recommendations
- Mitigation strategies

Goal: Ensure responsible deployment
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **Inference Optimization** ⚡
   - FP16, attention slicing, xFormers
   - torch.compile for PyTorch 2.0+
   - Fast schedulers (DPM-Solver)
   - 5-10x speedup possible!

2. **Production Deployment** 🚀
   - FastAPI for REST APIs
   - Queue management for scaling
   - Monitoring and logging
   - Error handling and retries

3. **Safety & Moderation** 🛡️
   - NSFW detection (critical!)
   - Custom safety classifiers
   - Multi-level filtering
   - Human review for edge cases

4. **Watermarking** 🔍
   - Invisible watermarks for provenance
   - Metadata tracking
   - Blockchain for verification
   - Combat misinformation

5. **Evaluation Metrics** 📊
   - FID: Distribution similarity
   - IS: Quality and diversity
   - CLIP Score: Text alignment
   - Human evaluation: Gold standard

6. **Responsible AI** ⚖️
   - Transparency and disclosure
   - Consent and attribution
   - Fairness and bias mitigation
   - Privacy and accountability

### Production Checklist

✅ **Performance**
  - Inference optimizations applied
  - Batch processing implemented
  - Caching for common requests
  - Load testing completed

✅ **Safety**
  - Content filters enabled
  - Safety classifiers deployed
  - Human review process
  - Abuse reporting mechanism

✅ **Legal & Ethics**
  - Copyright compliance
  - User consent obtained
  - Watermarking implemented
  - Terms of service clear

✅ **Monitoring**
  - Metrics collection (Prometheus)
  - Logging (structured logs)
  - Alerting (error rates, latency)
  - Dashboard (Grafana)

✅ **Cost**
  - Resource utilization optimized
  - Auto-scaling configured
  - Budget alerts set
  - Cost per request tracked

### The Future of Generative Vision

Looking ahead:
- **Better quality**: Continued improvements in realism
- **Faster generation**: Real-time synthesis
- **More control**: Finer-grained spatial control
- **Multimodal**: Integration with text, audio, 3D
- **Efficiency**: Smaller models, edge deployment
- **Responsibility**: Better safety, fairness, transparency

---

## Additional Resources

### Papers

- Ho et al. (2022): "Cascaded Diffusion Models for High Fidelity Image Generation"
- Heusel et al. (2017): "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (FID)
- Radford et al. (2021): "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)

### Tools & Infrastructure

- **FastAPI**: https://fastapi.tiangolo.com/
- **Prometheus**: https://prometheus.io/ (monitoring)
- **Ray Serve**: https://docs.ray.io/en/latest/serve/ (scalable deployment)
- **BentoML**: https://www.bentoml.com/ (ML serving)

### Safety & Ethics

- **Partnership on AI**: https://partnershiponai.org/
- **AI Safety**: https://www.aisafety.com/
- **Responsible AI Practices**: https://ai.google/responsibility/responsible-ai-practices/

### Legal Resources

- **CreativeML OpenRAIL License**: https://huggingface.co/spaces/CompVis/stable-diffusion-license
- **Copyright Office (US)**: https://www.copyright.gov/
- **EU AI Act**: https://artificialintelligenceact.eu/

---

## Conclusion: Building Responsible Generative AI

Congratulations on completing Module 14! 🎉

You've learned:
1. **Fundamentals**: GANs, VAEs, Diffusion Models
2. **State-of-the-Art**: Stable Diffusion architecture
3. **Control**: ControlNet, IP-Adapter, conditioning
4. **Personalization**: DreamBooth, LoRA, Textual Inversion
5. **Production**: Deployment, safety, evaluation
6. **Ethics**: Responsible AI principles

**Remember:**
- With great power comes great responsibility
- Always consider ethical implications
- Prioritize safety and transparency
- Respect copyright and consent
- Build AI that benefits humanity

**Next Steps:**
- Cross-reference **Module 6** for computer vision foundations
- Explore **Module 7** for NLP in multimodal models
- Apply to real projects
- Stay updated on latest research
- Contribute to open-source community

**Keep creating responsibly! 🌟**

---

**End of Module 14: Generative Vision Models**

You now have the knowledge to build, deploy, and maintain production-grade generative vision systems. Use it wisely!

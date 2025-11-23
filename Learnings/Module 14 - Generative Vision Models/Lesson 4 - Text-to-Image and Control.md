# Lesson 4: Text-to-Image and Conditional Generation 🎮

**Module 14: Generative Vision Models | Lesson 4 of 6**

Master precise control over image generation with ControlNet and advanced conditioning!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand ControlNet architecture and conditioning
2. ✅ Use edge detection, depth maps, and pose for spatial control
3. ✅ Implement IP-Adapter for image prompt conditioning
4. ✅ Master inpainting and outpainting techniques
5. ✅ Combine multiple control signals (Multi-ControlNet)
6. ✅ Build production image editing workflows
7. ✅ Apply style transfer with diffusion models

---

## Prerequisites

- **Lesson 3**: Stable Diffusion (pipelines, U-Net, cross-attention)
- **Lesson 2**: Diffusion fundamentals (sampling, guidance)
- **Module 6**: Computer Vision (edge detection, depth estimation)
- **Python Libraries**: diffusers, controlnet_aux, transformers, cv2

---

## 1. The Control Problem

### Why ControlNet?

```python
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)

"""
THE CONTROL PROBLEM

Text-to-Image is powerful, but limited:
❌ Can't specify exact pose of people
❌ Can't control precise composition
❌ Can't match specific layouts
❌ Hard to maintain consistency

Example:
  Prompt: "a woman raising her hand"
  Problem: Which hand? How high? What angle?

SOLUTION: ControlNet (Zhang et al., 2023)

Add spatial conditioning:
✅ Edges (Canny)
✅ Depth maps
✅ Human pose (OpenPose)
✅ Segmentation maps
✅ Scribbles
✅ Normal maps

Result: Precise spatial control!
"""


def demonstrate_control_problem():
    """
    Illustrate why we need spatial conditioning.
    """
    print("THE CONTROL PROBLEM\n")

    scenarios = {
        "Pose Control": {
            "goal": "Person in specific yoga pose",
            "text_only": "Can't specify exact limb positions",
            "with_control": "Use OpenPose skeleton as guide!"
        },
        "Composition": {
            "goal": "Maintain architectural layout",
            "text_only": "Text describes buildings vaguely",
            "with_control": "Use depth map for precise structure!"
        },
        "Style Transfer": {
            "goal": "Keep exact shapes, change style",
            "text_only": "Hard to preserve spatial structure",
            "with_control": "Use Canny edges as constraint!"
        },
        "Product Photography": {
            "goal": "Product in specific lighting/angle",
            "text_only": "Inconsistent camera viewpoints",
            "with_control": "Use depth + normal maps!"
        },
    }

    for scenario, info in scenarios.items():
        print(f"{scenario}:")
        print(f"  Goal: {info['goal']}")
        print(f"  ❌ Text-only: {info['text_only']}")
        print(f"  ✅ ControlNet: {info['with_control']}")
        print()

    print("ControlNet gives you SPATIAL CONTROL! 🎯")


demonstrate_control_problem()
```

---

## 2. ControlNet Architecture

### Additional Conditioning Branch

```python
"""
CONTROLNET ARCHITECTURE

Key idea: Add trainable copy of U-Net encoder

Original U-Net:
  Text → Cross-Attention → Denoising

ControlNet U-Net:
  Text + Control Image → Cross-Attention + Spatial Conditioning → Denoising

Architecture:
1. Trainable Copy: Clone SD U-Net encoder
2. Zero Convolutions: Initialize to zero (preserves pretrained weights)
3. Conditioning Input: Edge/depth/pose maps
4. Combine Features: Add to main U-Net

Training:
  - Freeze original SD weights
  - Only train ControlNet copy
  - Result: Add control without breaking SD!
"""


class ControlNetConcept(torch.nn.Module):
    """
    Conceptual ControlNet architecture.

    Shows the key ideas (simplified).
    """
    def __init__(self, base_unet):
        super().__init__()

        # Main U-Net (frozen)
        self.base_unet = base_unet
        for param in self.base_unet.parameters():
            param.requires_grad = False

        # ControlNet encoder (trainable copy)
        self.control_encoder = self.clone_encoder(base_unet)

        # Zero convolutions (start with zero contribution)
        self.zero_convs = torch.nn.ModuleList([
            torch.nn.Conv2d(channels, channels, 1)
            for channels in [320, 640, 1280, 1280]  # U-Net channel sizes
        ])

        # Initialize zero convs to zero
        for conv in self.zero_convs:
            torch.nn.init.zeros_(conv.weight)
            torch.nn.init.zeros_(conv.bias)

        # Control input projection
        self.control_input_conv = torch.nn.Conv2d(3, 320, 3, padding=1)

    def clone_encoder(self, unet):
        """Create trainable copy of encoder."""
        # Simplified - would copy actual encoder blocks
        return torch.nn.Sequential(
            torch.nn.Conv2d(320, 320, 3, padding=1),
            torch.nn.Conv2d(320, 640, 3, padding=1),
        )

    def forward(self, x, timestep, text_embeddings, control_image):
        """
        Args:
            x: Noisy latent
            timestep: Diffusion timestep
            text_embeddings: Text conditioning
            control_image: Edge/depth/pose map

        Returns:
            Denoised latent
        """
        # Process control image
        control_features = self.control_input_conv(control_image)

        # Extract control features
        control_outputs = []
        for i, zero_conv in enumerate(self.zero_convs):
            # Simplified - would process through encoder layers
            control_feat = control_features
            control_outputs.append(zero_conv(control_feat))

        # Add to base U-Net features
        output = self.base_unet(x, timestep, text_embeddings, additional_features=control_outputs)

        return output


print("ControlNet Architecture Overview:")
print("  1. Frozen base U-Net (preserves SD quality)")
print("  2. Trainable ControlNet copy (adds spatial control)")
print("  3. Zero convolutions (gradual learning)")
print("  4. Control signal injection (edges, depth, etc.)")
print()
print("Result: Spatial control WITHOUT breaking pretrained SD!")
```

---

## 3. Canny Edge Control

### Edge-Based Generation

```python
"""
CANNY EDGE DETECTION

Use case: Preserve exact shapes and outlines

Process:
1. Extract edges from reference image (Canny)
2. Use edges as spatial guide
3. Generate with text prompt + edge control

Perfect for:
✅ Architectural preservation
✅ Object shape consistency
✅ Composition control
"""


def extract_canny_edges(image, low_threshold=100, high_threshold=200):
    """
    Extract Canny edges from image.

    Args:
        image: PIL Image or numpy array
        low_threshold: Lower threshold for edge detection
        high_threshold: Upper threshold for edge detection

    Returns:
        Edge map as PIL Image
    """
    # Convert to numpy if PIL
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)

    # Convert back to PIL
    edges_pil = Image.fromarray(edges)

    return edges_pil


# Example: Create sample edge map
sample_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
edges = extract_canny_edges(sample_image)

print(f"Original image shape: {sample_image.shape}")
print(f"Edge map size: {edges.size}")
print()
print("Canny edges capture structural information!")
```

### Using Canny ControlNet

```python
def canny_controlnet_example():
    """
    Generate images with Canny edge control.
    """
    code = '''
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

# Load ControlNet (Canny)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Load SD pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load reference image
image = Image.open("room.jpg")

# Extract Canny edges
image_np = np.array(image)
gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
edges = cv2.Canny(gray, 100, 200)
edges_pil = Image.fromarray(edges)

# Generate with edge control
prompt = "a modern living room, minimalist design, bright lighting"
result = pipe(
    prompt=prompt,
    image=edges_pil,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

result.save("controlled_room.png")
'''

    print("CANNY CONTROLNET EXAMPLE")
    print(code)
    print()
    print("Key points:")
    print("  - Edges preserve spatial structure")
    print("  - Text controls appearance (style, lighting, etc.)")
    print("  - Great for maintaining composition!")


canny_controlnet_example()
```

---

## 4. Depth Map Control

### 3D Structure Preservation

```python
"""
DEPTH MAP CONTROL

Use case: Preserve 3D spatial structure

Depth map: Grayscale image encoding distance
  - White = close to camera
  - Black = far from camera

Perfect for:
✅ Maintaining perspective
✅ 3D scene consistency
✅ Architectural visualization
"""


def estimate_depth_map(image):
    """
    Estimate depth map from image.

    In practice, use MiDaS or other depth estimation models.
    """
    # Pseudo-code for depth estimation
    code = '''
from transformers import pipeline

# Load depth estimation model
depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")

# Estimate depth
depth = depth_estimator(image)["depth"]

# Convert to PIL
depth_pil = Image.fromarray(depth)
'''

    print("DEPTH ESTIMATION")
    print(code)
    print()
    print("Popular models:")
    print("  - Intel/dpt-large: High accuracy")
    print("  - Intel/dpt-hybrid-midas: Faster")
    print("  - Depth-Anything: Recent SOTA")


estimate_depth_map(None)
```

### Depth ControlNet Usage

```python
def depth_controlnet_example():
    """
    Generate with depth map control.
    """
    code = '''
from controlnet_aux import MidasDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Initialize depth detector
midas = MidasDetector.from_pretrained("lllyasviel/Annotators")

# Load ControlNet (Depth)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load image and estimate depth
image = Image.open("scene.jpg")
depth_map = midas(image)

# Generate with depth control
prompt = "a fantasy landscape with mountains and castles, epic lighting"
result = pipe(
    prompt=prompt,
    image=depth_map,
    num_inference_steps=20,
    controlnet_conditioning_scale=1.0  # Strength of control
).images[0]

result.save("fantasy_scene.png")
'''

    print("DEPTH CONTROLNET EXAMPLE")
    print(code)
    print()
    print("Benefits:")
    print("  ✅ Preserves 3D structure")
    print("  ✅ Maintains perspective")
    print("  ✅ Natural spatial relationships")


depth_controlnet_example()
```

---

## 5. OpenPose - Human Pose Control

### Precise Human Pose Guidance

```python
"""
OPENPOSE CONTROL

Use case: Control human poses precisely

OpenPose detects:
  - Body keypoints (shoulders, elbows, knees, etc.)
  - Hand keypoints (fingers)
  - Face keypoints (eyes, nose, mouth)

Perfect for:
✅ Character consistency
✅ Specific poses/actions
✅ Animation frame generation
"""


def detect_pose(image):
    """
    Detect human pose keypoints.
    """
    code = '''
from controlnet_aux import OpenposeDetector

# Initialize OpenPose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

# Detect pose
pose_image = openpose(image, hand_and_face=True)  # Include hands and face

# Returns image with skeleton overlay
'''

    print("OPENPOSE DETECTION")
    print(code)
    print()
    print("Detected keypoints:")
    print("  - Body: 18 points (neck, shoulders, elbows, etc.)")
    print("  - Hands: 21 points each (optional)")
    print("  - Face: 70 points (optional)")


detect_pose(None)
```

### OpenPose ControlNet

```python
def openpose_controlnet_example():
    """
    Generate images with pose control.
    """
    code = '''
from controlnet_aux import OpenposeDetector
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Initialize pose detector
openpose = OpenposeDetector.from_pretrained("lllyasviel/Annotators")

# Load ControlNet (OpenPose)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16
)

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Detect pose from reference
reference = Image.open("dancer.jpg")
pose_map = openpose(reference)

# Generate different person with same pose
prompt = "a ballet dancer in elegant costume, professional photography, 8K"
result = pipe(
    prompt=prompt,
    image=pose_map,
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

result.save("ballet_dancer.png")
'''

    print("OPENPOSE CONTROLNET EXAMPLE")
    print(code)
    print()
    print("Use cases:")
    print("  - Transfer poses between people")
    print("  - Generate specific actions/gestures")
    print("  - Create consistent character poses")
    print("  - Animation frame generation")


openpose_controlnet_example()
```

---

## 6. Multi-ControlNet - Combining Multiple Conditions

### Layering Control Signals

```python
"""
MULTI-CONTROLNET

Combine multiple control signals simultaneously!

Examples:
  - Canny + Depth: Structure + 3D information
  - OpenPose + Depth: Pose + Spatial layout
  - Canny + Segmentation: Edges + Semantic regions

Benefits:
✅ Fine-grained control
✅ Complementary information
✅ Better quality and consistency
"""


def multi_controlnet_example():
    """
    Use multiple ControlNets together.
    """
    code = '''
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, MultiControlNetModel

# Load multiple ControlNets
controlnet_canny = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

controlnet_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16
)

# Combine into MultiControlNet
controlnet = MultiControlNetModel([controlnet_canny, controlnet_depth])

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Prepare control images
image = Image.open("reference.jpg")
canny_map = get_canny_edges(image)
depth_map = get_depth_map(image)

# Generate with both controls
prompt = "a beautiful architectural interior, modern design"
result = pipe(
    prompt=prompt,
    image=[canny_map, depth_map],  # List of control images
    controlnet_conditioning_scale=[0.8, 0.6],  # Individual strengths
    num_inference_steps=20,
    guidance_scale=7.5
).images[0]

result.save("multi_control.png")
'''

    print("MULTI-CONTROLNET EXAMPLE")
    print(code)
    print()
    print("Conditioning scale tips:")
    print("  - [1.0, 1.0]: Full strength both")
    print("  - [0.8, 0.5]: Prioritize first control")
    print("  - Experiment to find best balance!")


multi_controlnet_example()
```

### Control Strength Tuning

```python
def control_strength_guide():
    """
    Guide to tuning control strength.
    """
    print("CONTROLNET CONDITIONING SCALE\n")

    scales = {
        0.0: "No control (ignore ControlNet)",
        0.3: "Subtle guidance (loose interpretation)",
        0.5: "Moderate control (balanced)",
        0.8: "Strong control (close adherence)",
        1.0: "Maximum control (strict following)",
        1.5: "Very strong (may reduce quality)",
    }

    for scale, description in scales.items():
        print(f"  {scale:.1f}: {description}")

    print()
    print("Recommendations:")
    print("  - Start with 1.0, adjust if needed")
    print("  - Lower for creative freedom")
    print("  - Higher for precise matching")
    print("  - Balance multiple controls carefully")


control_strength_guide()
```

---

## 7. IP-Adapter - Image Prompt Conditioning

### Beyond Text Prompts

```python
"""
IP-ADAPTER (Image Prompt Adapter)

Use images as prompts!

Instead of: "a photo in the style of Vincent van Gogh"
Use: Reference image of Van Gogh painting

Captures:
✅ Style (colors, brushstrokes, mood)
✅ Composition patterns
✅ Aesthetic preferences
✅ Hard-to-describe visual concepts

How it works:
  - CLIP image encoder extracts features
  - Cross-attention with image features
  - Can combine with text prompts!
"""


def ip_adapter_example():
    """
    Use IP-Adapter for image-based conditioning.
    """
    code = '''
from diffusers import StableDiffusionPipeline
from diffusers.models import IPAdapter

# Load base pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")

# Set adapter scale (strength of image conditioning)
pipe.set_ip_adapter_scale(0.6)

# Load reference image
reference_image = Image.open("style_reference.jpg")

# Generate with image prompt
prompt = "a portrait of a woman"  # Can combine text + image!
result = pipe(
    prompt=prompt,
    ip_adapter_image=reference_image,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

result.save("image_prompted.png")
'''

    print("IP-ADAPTER EXAMPLE")
    print(code)
    print()
    print("Use cases:")
    print("  - Style transfer (maintain style from reference)")
    print("  - Composition guidance (similar layout)")
    print("  - Aesthetic matching (color palette, mood)")
    print("  - Multi-image prompting (blend multiple styles)")


ip_adapter_example()
```

### Combining IP-Adapter with ControlNet

```python
def ip_adapter_controlnet_combo():
    """
    Ultimate control: ControlNet + IP-Adapter + Text.
    """
    code = '''
# Load everything
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

# Load IP-Adapter
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.5)

# Prepare controls
structure_image = Image.open("structure.jpg")
canny_map = get_canny_edges(structure_image)
style_reference = Image.open("style.jpg")

# Triple conditioning!
result = pipe(
    prompt="a beautiful landscape",  # Text: What to generate
    image=canny_map,  # ControlNet: Spatial structure
    ip_adapter_image=style_reference,  # IP-Adapter: Visual style
    num_inference_steps=30
).images[0]
'''

    print("IP-ADAPTER + CONTROLNET COMBINATION")
    print(code)
    print()
    print("Three-way control:")
    print("  📝 Text: Semantic content")
    print("  🎮 ControlNet: Spatial structure")
    print("  🎨 IP-Adapter: Visual style")
    print()
    print("Result: Maximum creative control! 🚀")


ip_adapter_controlnet_combo()
```

---

## 8. Inpainting and Outpainting

### Selective Editing

```python
"""
INPAINTING: Edit specific regions

Process:
1. Provide original image
2. Mask area to edit (white = edit, black = keep)
3. Text prompt for masked region
4. Generate seamlessly

OUTPAINTING: Extend image boundaries

Process:
1. Place image on larger canvas
2. Mask extended regions
3. Generate to fill

Both use specialized SD inpainting models!
"""


def inpainting_example():
    """
    Inpainting specific regions.
    """
    code = '''
from diffusers import StableDiffusionInpaintPipeline

# Load inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask
image = Image.open("room.jpg")
mask = Image.open("mask.png")  # White = inpaint area

# Inpaint
prompt = "a large window with ocean view"
result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask,
    num_inference_steps=50,
    guidance_scale=7.5
).images[0]

result.save("inpainted.png")
'''

    print("INPAINTING EXAMPLE")
    print(code)
    print()
    print("Tips:")
    print("  - Feather mask edges for smooth blending")
    print("  - Describe both inpaint area AND context")
    print("  - Use higher num_inference_steps for quality")


inpainting_example()
```

### Outpainting Implementation

```python
def outpainting_example():
    """
    Extend image beyond original boundaries.
    """
    code = '''
from PIL import Image
import numpy as np

def outpaint(image, target_size=(768, 768), direction="all"):
    """
    Extend image by outpainting.

    Args:
        image: Original image
        target_size: Desired output size
        direction: "all", "right", "left", "top", "bottom"
    """
    # Create larger canvas
    canvas = Image.new("RGB", target_size, (255, 255, 255))

    # Paste original in center
    orig_w, orig_h = image.size
    target_w, target_h = target_size

    # Calculate position
    x = (target_w - orig_w) // 2
    y = (target_h - orig_h) // 2
    canvas.paste(image, (x, y))

    # Create mask (white = generate, black = keep)
    mask = Image.new("L", target_size, 255)  # All white
    mask_draw = Image.new("L", (orig_w, orig_h), 0)  # Black rectangle
    mask.paste(mask_draw, (x, y))

    # Inpaint to fill extended regions
    prompt = "seamless continuation of the scene, same style"
    result = inpaint_pipe(
        prompt=prompt,
        image=canvas,
        mask_image=mask,
        num_inference_steps=50
    ).images[0]

    return result

# Use it
image = Image.open("photo.jpg")
extended = outpaint(image, target_size=(1024, 1024))
extended.save("outpainted.png")
'''

    print("OUTPAINTING EXAMPLE")
    print(code)
    print()
    print("Creative uses:")
    print("  - Expand photo backgrounds")
    print("  - Create panoramas from single image")
    print("  - Generate context around subjects")


outpainting_example()
```

---

## 9. Style Transfer with Diffusion

### Advanced Artistic Control

```python
"""
STYLE TRANSFER: Apply artistic styles to images

Methods:
1. IP-Adapter: Use style image as prompt
2. ControlNet + Prompt: Preserve structure, change style
3. Img2Img: Iterative refinement

Combines:
  - Spatial preservation (ControlNet)
  - Style reference (IP-Adapter)
  - Text description (prompts)
"""


def style_transfer_pipeline():
    """
    Complete style transfer workflow.
    """
    code = '''
# 1. Extract structure from content image
content = Image.open("photo.jpg")
canny_edges = get_canny_edges(content)

# 2. Load style reference
style = Image.open("painting.jpg")

# 3. Setup pipeline with ControlNet + IP-Adapter
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipe.set_ip_adapter_scale(0.7)  # Strong style influence

# 4. Generate
prompt = "a masterpiece painting, highly detailed, artistic"
result = pipe(
    prompt=prompt,
    image=canny_edges,  # Preserve structure
    ip_adapter_image=style,  # Apply style
    controlnet_conditioning_scale=0.8,  # Keep composition
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

result.save("stylized.png")
'''

    print("STYLE TRANSFER PIPELINE")
    print(code)
    print()
    print("Control parameters:")
    print("  - controlnet_scale: Structure preservation (0.6-1.0)")
    print("  - ip_adapter_scale: Style strength (0.5-0.9)")
    print("  - guidance_scale: Overall quality (7.5-12.0)")


style_transfer_pipeline()
```

---

## 10. Production Workflows

### Building Robust Image Generation Systems

```python
"""
PRODUCTION CONSIDERATIONS

1. Error Handling
   - Invalid control images
   - Memory overflow
   - Model loading failures

2. Performance
   - Batch processing
   - Caching preprocessors
   - GPU memory management

3. Quality Control
   - Validation of outputs
   - Retry logic
   - Parameter tuning
"""


class ProductionImageGenerator:
    """
    Production-ready image generation class.
    """
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5",
                 controlnet_type=None, device="cuda"):
        """
        Initialize production generator.

        Args:
            model_id: Base Stable Diffusion model
            controlnet_type: "canny", "depth", "openpose", or None
            device: "cuda" or "cpu"
        """
        self.device = device

        # Load ControlNet if specified
        if controlnet_type:
            controlnet_models = {
                "canny": "lllyasviel/sd-controlnet-canny",
                "depth": "lllyasviel/sd-controlnet-depth",
                "openpose": "lllyasviel/sd-controlnet-openpose",
            }

            if controlnet_type not in controlnet_models:
                raise ValueError(f"Unknown ControlNet type: {controlnet_type}")

            # Would load actual model here
            print(f"Loading ControlNet: {controlnet_type}")

        # Would load pipeline here
        print(f"Initialized generator on {device}")

    def generate(self, prompt, negative_prompt="", control_image=None,
                 num_images=1, seed=None, **kwargs):
        """
        Generate images with error handling.

        Args:
            prompt: Text prompt
            negative_prompt: Negative prompt
            control_image: Optional control image
            num_images: Number of images to generate
            seed: Random seed
            **kwargs: Additional generation parameters

        Returns:
            List of generated PIL Images
        """
        try:
            # Validate inputs
            if not prompt or len(prompt) == 0:
                raise ValueError("Prompt cannot be empty")

            # Set seed if provided
            if seed is not None:
                torch.manual_seed(seed)

            # Generate (simplified - would use actual pipeline)
            print(f"Generating {num_images} images with prompt: '{prompt[:50]}...'")

            # Placeholder
            images = []

            return images

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("GPU OOM! Trying with attention slicing...")
                # Would enable memory optimizations and retry
                raise
            else:
                raise

        except Exception as e:
            print(f"Generation failed: {e}")
            raise

    def batch_generate(self, prompts, **kwargs):
        """
        Generate multiple prompts efficiently.

        Args:
            prompts: List of text prompts
            **kwargs: Generation parameters

        Returns:
            List of image lists
        """
        results = []

        for i, prompt in enumerate(prompts):
            print(f"Processing {i+1}/{len(prompts)}: {prompt[:30]}...")

            try:
                images = self.generate(prompt, **kwargs)
                results.append(images)

            except Exception as e:
                print(f"Failed for prompt '{prompt}': {e}")
                results.append([])

        return results


# Example usage
generator = ProductionImageGenerator(controlnet_type="canny")

print("\nProduction features:")
print("✅ Error handling and retry logic")
print("✅ Memory management")
print("✅ Batch processing")
print("✅ Validation and logging")
```

---

## Practice Exercises

### Exercise 1: Multi-ControlNet Experimentation

```python
"""
Explore combining different ControlNets.

Task:
1. Choose base image
2. Extract: Canny edges, Depth map, and Pose (if applicable)
3. Try combinations:
   - Canny alone
   - Depth alone
   - Canny + Depth
4. Compare results
5. Tune conditioning scales for best quality

Report: Which combination works best for your image type?
"""

# Your implementation here
```

### Exercise 2: Style Transfer Challenge

```python
"""
Implement robust style transfer.

Requirements:
1. Load content image and style reference
2. Extract Canny edges from content
3. Use IP-Adapter for style
4. Generate multiple variations (different scales)
5. Create comparison grid

Bonus: Try with different ControlNet types (depth, normal)
"""

# Your implementation here
```

### Exercise 3: Inpainting Application

```python
"""
Build an inpainting tool.

Features:
1. Load image
2. Create mask (manually or programmatically)
3. Inpaint multiple times with different prompts
4. Select best result
5. Optional: Iterative refinement

Use case: Virtual interior design, object replacement
"""

# Your implementation here
```

---

## Key Takeaways

### Essential Concepts

1. **ControlNet Revolution** 🎮
   - Adds spatial conditioning to SD
   - Trainable copy of U-Net encoder
   - Zero convolutions preserve pretrained weights
   - Multiple control types available

2. **Control Types** 🎯
   - **Canny**: Edge preservation, composition
   - **Depth**: 3D structure, perspective
   - **OpenPose**: Human pose control
   - **Segmentation**: Semantic regions
   - **Normal Maps**: Surface details
   - **Scribbles**: Hand-drawn guides

3. **Multi-ControlNet** 🚀
   - Combine multiple signals simultaneously
   - Complementary information
   - Individual strength control
   - More precise generation

4. **IP-Adapter** 🎨
   - Image-based conditioning
   - Captures style and aesthetics
   - Combines with text prompts
   - Hard-to-describe visual concepts

5. **Inpainting & Outpainting** ✏️
   - Selective region editing
   - Seamless blending
   - Image extension
   - Production workflows

6. **Production Ready** 💼
   - Error handling critical
   - Memory management
   - Batch processing
   - Quality validation

### Control Methods Comparison

| Method | Precision | Use Case | Complexity |
|--------|-----------|----------|------------|
| **Text Only** | Low | General generation | Simple |
| **Canny** | High | Composition control | Medium |
| **Depth** | High | 3D structure | Medium |
| **OpenPose** | Very High | Human poses | Medium |
| **IP-Adapter** | Medium | Style transfer | Simple |
| **Multi-ControlNet** | Very High | Complex scenes | High |

### What's Next?

In Lesson 5, we'll explore **Fine-Tuning & Personalization**:
- DreamBooth: Subject-driven generation
- Textual Inversion: New concepts
- LoRA: Efficient adaptation
- Model merging strategies

---

## Additional Resources

### Papers

- Zhang et al. (2023): "Adding Conditional Control to Text-to-Image Diffusion Models" (ControlNet)
- Ye et al. (2023): "IP-Adapter: Text Compatible Image Prompt Adapter"
- Cao et al. (2018): "OpenPose: Realtime Multi-Person 2D Pose Estimation"

### Libraries & Tools

- **ControlNet Models**: https://huggingface.co/lllyasviel
- **controlnet_aux**: Preprocessors for all control types
- **Diffusers Documentation**: https://huggingface.co/docs/diffusers/using-diffusers/controlnet

### Interactive Resources

- **ControlNet Gallery**: https://huggingface.co/spaces/hysts/ControlNet
- **Stable Diffusion WebUI**: Built-in ControlNet support
- **ComfyUI**: Advanced ControlNet workflows

---

**Next**: [Lesson 5 - Fine-Tuning and Personalization](Lesson%205%20-%20Fine-Tuning%20and%20Personalization.md)

Master **DreamBooth and LoRA** for custom model adaptation! 🎯

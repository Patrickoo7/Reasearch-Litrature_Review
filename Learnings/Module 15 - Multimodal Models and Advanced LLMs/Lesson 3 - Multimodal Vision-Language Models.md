# Lesson 3: Multimodal Vision-Language Models 👁️

**Module 15: Multimodal Models and Advanced LLMs | Lesson 3 of 7**

Master vision-language models from CLIP to GPT-4V - enabling AI to see and understand the visual world!

---

## Learning Objectives

By the end of this lesson, you will:

1. ✅ Understand vision-language alignment with CLIP
2. ✅ Implement contrastive learning for image-text pairs
3. ✅ Master zero-shot image classification with CLIP
4. ✅ Explore BLIP/BLIP-2 for vision-language pretraining
5. ✅ Understand GPT-4V and its capabilities
6. ✅ Implement LLaVA (Large Language and Vision Assistant)
7. ✅ Build real-world applications: Image captioning, VQA, visual reasoning

---

## Prerequisites

- **Required**: Module 15 Lesson 1 (LLM architectures)
- **Required**: Module 7 (NLP basics), Module 8 (CNNs)
- **Helpful**: Understanding of transformers and attention
- **Libraries**: `transformers`, `torch`, `clip`, `PIL`

```bash
pip install transformers torch torchvision pillow
pip install git+https://github.com/openai/CLIP.git
pip install open-clip-torch
```

---

## 1. Vision-Language Alignment: CLIP

### Understanding CLIP

CLIP (Contrastive Language-Image Pre-training) learns to align images and text:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

class CLIPExplainer:
    """
    Explain and demonstrate CLIP architecture.

    Key insight: Joint embedding space for images and text
    - Image encoder: Vision Transformer (ViT) or ResNet
    - Text encoder: Transformer
    - Contrastive loss: Align matching pairs
    """

    def __init__(self):
        """Initialize CLIP."""
        import clip

        # Load CLIP model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        print("="*80)
        print("CLIP: Contrastive Language-Image Pre-training")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Model: ViT-B/32 (86M params)")

    def explain_architecture(self):
        """Explain CLIP architecture."""

        print("\nCLIP Architecture:")
        print("  Image Encoder:")
        print("    • Vision Transformer (ViT) or ResNet")
        print("    • Input: 224x224 image")
        print("    • Output: d-dimensional embedding")
        print("\n  Text Encoder:")
        print("    • Transformer (same as GPT)")
        print("    • Input: Text tokens")
        print("    • Output: d-dimensional embedding (same space!)")
        print("\n  Training:")
        print("    • Dataset: 400M image-text pairs from web")
        print("    • Loss: Contrastive loss (InfoNCE)")
        print("    • Goal: max similarity for matching pairs")

    def visualize_embedding_space(self):
        """Visualize how CLIP creates joint embedding space."""

        # Example images and texts
        images_urls = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/1200px-Collage_of_Nine_Dogs.jpg",
        ]

        texts = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a bird",
            "a photo of a car",
        ]

        print("\n" + "="*80)
        print("Zero-Shot Image Classification Demo")
        print("="*80)

        # Load images
        images = []
        for url in images_urls:
            try:
                response = requests.get(url, timeout=5)
                img = Image.open(BytesIO(response.content))
                images.append(img)
            except:
                print(f"Could not load image from {url}")
                return

        # Preprocess
        image_inputs = torch.stack([self.preprocess(img) for img in images]).to(self.device)

        # Tokenize text
        import clip
        text_inputs = clip.tokenize(texts).to(self.device)

        # Get embeddings
        with torch.no_grad():
            image_features = self.model.encode_image(image_inputs)
            text_features = self.model.encode_text(text_inputs)

            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Compute similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Print results
        print("\nSimilarity Matrix (rows=images, cols=texts):")
        print(f"{'':20} " + " ".join([f"{t:20}" for t in texts]))
        print("-" * 100)

        for i, img_url in enumerate(images_urls):
            img_name = f"Image {i+1}"
            probs = similarity[i].cpu().numpy()
            print(f"{img_name:20} " + " ".join([f"{p:20.2%}" for p in probs]))

        print("\nCLIP correctly identifies:")
        for i, img_url in enumerate(images_urls):
            predicted_idx = similarity[i].argmax().item()
            print(f"  Image {i+1}: '{texts[predicted_idx]}' ({similarity[i][predicted_idx]:.1%} confidence)")

    def zero_shot_classification(self, image, candidates):
        """
        Zero-shot classification: Classify image without training!

        Args:
            image: PIL Image
            candidates: List of text descriptions

        Returns:
            Predictions with probabilities
        """
        import clip

        # Preprocess
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize([f"a photo of {c}" for c in candidates]).to(self.device)

        # Encode
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_inputs)

            # Normalize and compute similarity
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        # Get predictions
        probs = similarity[0].cpu().numpy()

        print("\nZero-Shot Predictions:")
        for candidate, prob in zip(candidates, probs):
            print(f"  {candidate:30s}: {prob:6.2%}")

        return list(zip(candidates, probs))


# Initialize and demonstrate
clip_model = CLIPExplainer()
clip_model.explain_architecture()
clip_model.visualize_embedding_space()


# Implement contrastive loss (simplified CLIP training)
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for CLIP training.

    Also called InfoNCE loss.

    For each image-text pair (i, j):
    - Maximize similarity when i == j (positive pair)
    - Minimize similarity when i != j (negative pairs)
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, image_embeddings, text_embeddings):
        """
        Args:
            image_embeddings: (batch, dim)
            text_embeddings: (batch, dim)

        Returns:
            loss: scalar
        """
        batch_size = image_embeddings.shape[0]

        # Normalize
        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        # Cross-entropy loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)  # Image to text
        loss_t2i = F.cross_entropy(logits.T, labels)  # Text to image

        loss = (loss_i2t + loss_t2i) / 2

        return loss


# Demonstrate contrastive loss
def demonstrate_contrastive_loss():
    """Show how contrastive loss works."""

    batch_size = 4
    dim = 512

    # Simulate embeddings
    image_emb = torch.randn(batch_size, dim)
    text_emb = torch.randn(batch_size, dim)

    # Initially, embeddings are random (not aligned)
    print("\n" + "="*80)
    print("Contrastive Loss Demonstration")
    print("="*80)

    # Compute similarity before training
    image_emb_norm = F.normalize(image_emb, dim=-1)
    text_emb_norm = F.normalize(text_emb, dim=-1)
    similarity_before = (image_emb_norm @ text_emb_norm.T).numpy()

    print("\nSimilarity matrix BEFORE training (random):")
    print("(Diagonal should be high after training)")
    print(similarity_before)
    print(f"Diagonal mean: {np.diag(similarity_before).mean():.3f}")

    # Compute loss
    criterion = ContrastiveLoss()
    loss = criterion(image_emb, text_emb)

    print(f"\nContrastive loss: {loss.item():.4f}")
    print("Training would minimize this loss, aligning matching pairs!")

demonstrate_contrastive_loss()
```

---

## 2. BLIP and BLIP-2: Bootstrapping Vision-Language

BLIP improves vision-language pretraining with better data curation:

```python
from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

class BLIPAnalyzer:
    """
    Analyze BLIP models.

    BLIP (Bootstrapping Language-Image Pre-training):
    - Multimodal mixture of encoder-decoder
    - Captioning, VQA, retrieval
    - Bootstraps own training data (filters noisy web data)

    BLIP-2:
    - Querying Transformer (Q-Former)
    - Bridges frozen vision encoder and frozen LLM
    - More efficient, better performance
    """

    def __init__(self):
        """Initialize BLIP models."""
        print("="*80)
        print("BLIP: Bootstrapping Language-Image Pre-training")
        print("="*80)

        # BLIP for image captioning
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

        # BLIP for VQA
        self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
        self.vqa_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

        print("BLIP models loaded!")

    def explain_architecture(self):
        """Explain BLIP architecture."""

        print("\n" + "="*80)
        print("BLIP Architecture")
        print("="*80)

        print("\nBLIP (v1):")
        print("  Components:")
        print("    • Vision encoder: ViT")
        print("    • Text encoder: BERT-like")
        print("    • Text decoder: GPT-like")
        print("  Training objectives:")
        print("    1. Image-Text Contrastive (ITC)")
        print("    2. Image-Text Matching (ITM)")
        print("    3. Language Modeling (LM)")
        print("  Innovation: CapFilt - Bootstraps training data")
        print("    • Captioner generates synthetic captions")
        print("    • Filter removes noisy image-text pairs")

        print("\nBLIP-2:")
        print("  Key innovation: Q-Former (Querying Transformer)")
        print("    • Frozen vision encoder (from CLIP)")
        print("    • Frozen LLM (OPT, FlanT5, etc.)")
        print("    • Q-Former bridges them efficiently")
        print("  Advantages:")
        print("    • Much cheaper to train (only Q-Former trained)")
        print("    • Can use any LLM backend")
        print("    • Better performance than BLIP v1")

    def image_captioning(self, image_url):
        """Generate caption for image."""

        print("\n" + "="*80)
        print("Image Captioning with BLIP")
        print("="*80)

        # Load image
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content))
        except:
            print("Could not load image")
            return

        # Process
        inputs = self.caption_processor(image, return_tensors="pt")

        # Generate caption
        with torch.no_grad():
            out = self.caption_model.generate(**inputs, max_length=50)

        caption = self.caption_processor.decode(out[0], skip_special_tokens=True)

        print(f"\nImage URL: {image_url}")
        print(f"Generated Caption: {caption}")

        # Generate with beam search for better quality
        with torch.no_grad():
            out_beam = self.caption_model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                early_stopping=True
            )

        caption_beam = self.caption_processor.decode(out_beam[0], skip_special_tokens=True)
        print(f"Caption (beam search): {caption_beam}")

        return caption

    def visual_question_answering(self, image_url, question):
        """Answer questions about an image."""

        print("\n" + "="*80)
        print("Visual Question Answering (VQA)")
        print("="*80)

        # Load image
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content))
        except:
            print("Could not load image")
            return

        # Process
        inputs = self.vqa_processor(image, question, return_tensors="pt")

        # Generate answer
        with torch.no_grad():
            out = self.vqa_model.generate(**inputs, max_length=20)

        answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)

        print(f"\nImage URL: {image_url}")
        print(f"Question: {question}")
        print(f"Answer: {answer}")

        return answer

    def compare_blip_versions(self):
        """Compare BLIP v1 vs BLIP-2."""

        comparison = {
            'Feature': [
                'Architecture',
                'Vision Encoder',
                'LLM Backend',
                'Training Cost',
                'Parameters Trained',
                'Zero-shot Perf',
                'VQA Accuracy',
                'Image Captioning',
            ],
            'BLIP v1': [
                'End-to-end',
                'ViT (trained)',
                'BERT/GPT (trained)',
                'High',
                'All (~400M)',
                'Good',
                '78.3%',
                'COCO 140.4 CIDEr',
            ],
            'BLIP-2': [
                'Modular with Q-Former',
                'Frozen (from CLIP)',
                'Frozen (any LLM)',
                'Low (54x cheaper)',
                'Q-Former only (188M)',
                'Better',
                '82.4%',
                'COCO 144.5 CIDEr',
            ],
        }

        import pandas as pd
        df = pd.DataFrame(comparison)

        print("\n" + "="*80)
        print("BLIP v1 vs BLIP-2 Comparison")
        print("="*80)
        print(df.to_string(index=False))


# Demonstrate BLIP
blip = BLIPAnalyzer()
blip.explain_architecture()

# Example usage (with sample images)
sample_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"
blip.image_captioning(sample_image)
blip.visual_question_answering(sample_image, "What animal is this?")
blip.visual_question_answering(sample_image, "What color is it?")

blip.compare_blip_versions()
```

---

## 3. GPT-4V: Vision Capabilities

GPT-4V extends GPT-4 with vision understanding:

```python
class GPT4VisionOverview:
    """
    Overview of GPT-4V (GPT-4 with Vision).

    Note: This is an overview and API examples.
    Actual model weights are not available.
    """

    def __init__(self):
        print("="*80)
        print("GPT-4V: GPT-4 with Vision")
        print("="*80)

    def explain_capabilities(self):
        """Explain GPT-4V capabilities."""

        print("\nGPT-4V Capabilities:")

        capabilities = [
            ("Image Understanding", "Recognizes objects, scenes, actions, text in images"),
            ("OCR", "Reads text from images (handwriting, printed, signs)"),
            ("Chart Analysis", "Interprets graphs, charts, diagrams"),
            ("Visual Reasoning", "Solves visual puzzles, math problems from images"),
            ("Multi-image Understanding", "Compares and relates multiple images"),
            ("Creative Vision", "Generates creative descriptions, stories from images"),
            ("Safety", "Refuses harmful requests, identifies sensitive content"),
        ]

        for capability, description in capabilities:
            print(f"\n  • {capability}:")
            print(f"    {description}")

    def architecture_speculation(self):
        """Speculate on GPT-4V architecture (not publicly disclosed)."""

        print("\n" + "="*80)
        print("GPT-4V Architecture (Speculative)")
        print("="*80)

        print("\nLikely approach (based on public research):")
        print("  1. Vision encoder: CLIP-like or custom ViT")
        print("  2. Vision-language adapter: Projects image features to LLM space")
        print("  3. Multimodal LLM: Processes both text and image tokens")
        print("  4. Training: Mixture of text-only and vision-language data")

        print("\nPossible architecture patterns:")
        print("  • Flamingo-style: Cross-attention between frozen LLM and vision")
        print("  • BLIP-2-style: Q-Former to bridge vision and language")
        print("  • LLaVA-style: Vision features as additional tokens")

    def api_usage_example(self):
        """Show GPT-4V API usage example."""

        code = '''
# Using OpenAI GPT-4V API

import openai
import base64

# Load image
with open("image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Call API
response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                    }
                }
            ]
        }
    ],
    max_tokens=300
)

print(response.choices[0].message.content)

# Multi-image understanding
response = openai.ChatCompletion.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the differences between these images?"},
                {"type": "image_url", "image_url": {"url": "image1.jpg"}},
                {"type": "image_url", "image_url": {"url": "image2.jpg"}},
            ]
        }
    ],
    max_tokens=500
)
'''

        print("\n" + "="*80)
        print("GPT-4V API Usage Example")
        print("="*80)
        print(code)

    def use_cases(self):
        """Real-world use cases for GPT-4V."""

        print("\n" + "="*80)
        print("GPT-4V Real-World Use Cases")
        print("="*80)

        use_cases = {
            'E-commerce': [
                'Product description generation from photos',
                'Visual search and similarity',
                'Quality control from images',
            ],
            'Education': [
                'Math problem solving from handwritten work',
                'Diagram explanation',
                'Visual learning assistance',
            ],
            'Accessibility': [
                'Describe surroundings for visually impaired',
                'Read signs and documents',
                'Navigate environments',
            ],
            'Healthcare': [
                'Medical image analysis (with proper training)',
                'Chart/report interpretation',
                'Patient education with visual aids',
            ],
            'Content Moderation': [
                'Detect inappropriate content',
                'Identify misinformation',
                'Flag safety concerns',
            ],
        }

        for category, cases in use_cases.items():
            print(f"\n{category}:")
            for case in cases:
                print(f"  • {case}")


gpt4v = GPT4VisionOverview()
gpt4v.explain_capabilities()
gpt4v.architecture_speculation()
gpt4v.api_usage_example()
gpt4v.use_cases()
```

---

## 4. LLaVA: Open-Source Multimodal Model

LLaVA is an open-source alternative to GPT-4V:

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

class LLaVAImplementation:
    """
    LLaVA: Large Language and Vision Assistant.

    Architecture:
    - Vision encoder: CLIP ViT-L/14
    - Projection: Linear layer
    - Language model: Vicuna (LLaMA fine-tuned)
    - Training: Visual instruction tuning
    """

    def __init__(self, model_name="llava-hf/llava-1.5-7b-hf"):
        """Initialize LLaVA."""
        print("="*80)
        print("LLaVA: Large Language and Vision Assistant")
        print("="*80)
        print(f"Loading {model_name}...")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        print("LLaVA loaded successfully!")

    def explain_architecture(self):
        """Explain LLaVA architecture."""

        print("\n" + "="*80)
        print("LLaVA Architecture")
        print("="*80)

        print("\nComponents:")
        print("  1. Vision Encoder (CLIP ViT-L/14):")
        print("     • Input: 224×224 image")
        print("     • Output: Grid of visual tokens (e.g., 24×24)")
        print("\n  2. Vision-Language Connector:")
        print("     • Simple linear projection")
        print("     • Maps vision features to LLM input space")
        print("     • Trainable, lightweight")
        print("\n  3. Language Model (Vicuna/LLaMA):")
        print("     • Processes visual tokens + text tokens")
        print("     • Generates text response")

        print("\nTraining Process:")
        print("  Stage 1: Feature alignment")
        print("    • Train projection layer only")
        print("    • Dataset: Image-caption pairs")
        print("    • Goal: Align vision and language features")
        print("\n  Stage 2: Visual instruction tuning")
        print("    • Fine-tune connector + LLM")
        print("    • Dataset: GPT-4 generated instruction-following data")
        print("    • Goal: Follow visual instructions")

    def generate_response(self, image_url, prompt):
        """Generate response for image + text prompt."""

        print("\n" + "="*80)
        print("LLaVA Generation")
        print("="*80)

        # Load image
        try:
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content))
        except:
            print("Could not load image")
            return

        # Format conversation
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # Apply chat template
        prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        # Process inputs
        inputs = self.processor(images=image, text=prompt_text, return_tensors="pt").to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True
            )

        # Decode
        generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

        print(f"\nImage: {image_url}")
        print(f"Prompt: {prompt}")
        print(f"Response: {generated_text}")

        return generated_text

    def benchmark_comparison(self):
        """Compare LLaVA to other models."""

        benchmarks = {
            'Model': ['GPT-4V', 'Gemini Pro Vision', 'LLaVA-1.5 (13B)', 'LLaVA-1.5 (7B)', 'BLIP-2'],
            'VQAv2': [77.2, 71.2, 80.0, 78.5, 65.0],
            'GQA': [63.8, 62.2, 63.3, 62.0, 44.7],
            'MMBench': [75.1, 73.6, 70.7, 66.5, 53.8],
            'Open Source': ['No', 'No', 'Yes', 'Yes', 'Yes'],
            'Size': ['?', '?', '13B', '7B', '2.7B'],
        }

        import pandas as pd
        df = pd.DataFrame(benchmarks)

        print("\n" + "="*80)
        print("Multimodal Model Benchmarks")
        print("="*80)
        print(df.to_string(index=False))

        print("\nKey Insights:")
        print("  • LLaVA-1.5 matches or exceeds GPT-4V on some benchmarks!")
        print("  • Open-source, can be fine-tuned and deployed")
        print("  • 7B version is efficient enough for edge deployment")
        print("  • Training approach: Visual instruction tuning with GPT-4")


# Example usage
llava = LLaVAImplementation()
llava.explain_architecture()

# Generate responses (uncomment to run with actual model)
# llava.generate_response(
#     "https://example.com/image.jpg",
#     "Describe this image in detail."
# )

llava.benchmark_comparison()


# Implement simplified LLaVA architecture
class SimplifiedLLaVA(nn.Module):
    """
    Simplified LLaVA architecture for educational purposes.

    Shows how vision and language models are combined.
    """

    def __init__(self, vision_encoder, language_model, hidden_size=4096):
        super().__init__()

        self.vision_encoder = vision_encoder  # e.g., CLIP ViT
        self.language_model = language_model  # e.g., LLaMA

        # Vision-language projection
        # Maps from vision feature dim to LLM embedding dim
        vision_dim = 1024  # CLIP ViT-L output dim
        self.vision_projection = nn.Linear(vision_dim, hidden_size)

    def forward(self, images, input_ids, attention_mask):
        """
        Forward pass.

        Args:
            images: (batch, 3, H, W)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Process:
        1. Encode image to visual tokens
        2. Project to LLM space
        3. Concatenate with text tokens
        4. Forward through LLM
        """

        # Encode images
        with torch.no_grad():
            vision_features = self.vision_encoder(images)  # (batch, num_patches, vision_dim)

        # Project to LLM space
        vision_tokens = self.vision_projection(vision_features)  # (batch, num_patches, hidden_size)

        # Get text embeddings from LLM
        text_embeddings = self.language_model.get_input_embeddings()(input_ids)

        # Concatenate vision and text tokens
        # In practice, you'd insert vision tokens at specific positions
        combined_embeddings = torch.cat([vision_tokens, text_embeddings], dim=1)

        # Adjust attention mask
        vision_attention = torch.ones(
            vision_tokens.shape[0],
            vision_tokens.shape[1],
            device=attention_mask.device
        )
        combined_attention = torch.cat([vision_attention, attention_mask], dim=1)

        # Forward through LLM
        outputs = self.language_model(
            inputs_embeds=combined_embeddings,
            attention_mask=combined_attention
        )

        return outputs


print("\n" + "="*80)
print("Simplified LLaVA Architecture (Conceptual)")
print("="*80)
print("Key insight: Visual tokens are treated as additional input tokens!")
print("  1. Encode image → visual features")
print("  2. Project features → same dimension as word embeddings")
print("  3. Concatenate vision tokens + text tokens")
print("  4. Process combined sequence with LLM")
print("\nThis allows LLM to 'see' by treating images as special tokens!")
```

---

## 5. Qwen-VL and Other Multimodal Models

```python
class MultimodalModelsSurvey:
    """
    Survey of multimodal vision-language models.
    """

    def __init__(self):
        print("="*80)
        print("Multimodal Models Landscape")
        print("="*80)

    def model_comparison(self):
        """Compare different multimodal models."""

        models = {
            'Model': [
                'GPT-4V',
                'Gemini Pro Vision',
                'Claude 3 Opus',
                'LLaVA-1.5 (13B)',
                'Qwen-VL-Chat',
                'CogVLM',
                'MiniGPT-4',
                'InstructBLIP',
            ],
            'Organization': [
                'OpenAI',
                'Google',
                'Anthropic',
                'Open',
                'Alibaba',
                'Tsinghua',
                'Open',
                'Salesforce',
            ],
            'Open Source': [
                'No',
                'No',
                'No',
                'Yes',
                'Yes',
                'Yes',
                'Yes',
                'Yes',
            ],
            'Size': [
                '?',
                '?',
                '?',
                '13B',
                '9.6B',
                '17B',
                '13B',
                '11B',
            ],
            'Context Length': [
                '~32K',
                '~32K',
                '~200K',
                '4K',
                '32K',
                '2K',
                '2K',
                '2K',
            ],
            'Special Features': [
                'Best quality, reasoning',
                'Multi-image, video',
                'Long context, safety',
                'Efficient, instruction-tuned',
                'High-resolution, Chinese',
                'Grounding, referring',
                'Conversational',
                'Instruction-following',
            ],
        }

        import pandas as pd
        df = pd.DataFrame(models)

        print("\nMultimodal Models Comparison:")
        print(df.to_string(index=False))

    def architecture_patterns(self):
        """Common architectural patterns in multimodal models."""

        print("\n" + "="*80)
        print("Architectural Patterns")
        print("="*80)

        patterns = {
            'Early Fusion': {
                'description': 'Combine vision and text features early',
                'examples': ['VisualBERT', 'ViLBERT'],
                'pros': ['Rich cross-modal interactions'],
                'cons': ['Computationally expensive'],
            },
            'Late Fusion': {
                'description': 'Separate encoders, combine at end',
                'examples': ['CLIP', 'ALIGN'],
                'pros': ['Modular, efficient'],
                'cons': ['Limited cross-modal reasoning'],
            },
            'Adapter-based': {
                'description': 'Lightweight adapter between frozen encoders',
                'examples': ['BLIP-2 Q-Former', 'Flamingo Perceiver'],
                'pros': ['Cheap to train, flexible'],
                'cons': ['Bottleneck in information flow'],
            },
            'Token Concatenation': {
                'description': 'Treat vision as additional tokens',
                'examples': ['LLaVA', 'Qwen-VL', 'CogVLM'],
                'pros': ['Simple, leverages LLM power'],
                'cons': ['Long sequence length'],
            },
        }

        for pattern, details in patterns.items():
            print(f"\n{pattern}:")
            print(f"  Description: {details['description']}")
            print(f"  Examples: {', '.join(details['examples'])}")
            print(f"  Pros: {', '.join(details['pros'])}")
            print(f"  Cons: {', '.join(details['cons'])}")

    def training_approaches(self):
        """Different training approaches for multimodal models."""

        print("\n" + "="*80)
        print("Training Approaches")
        print("="*80)

        approaches = [
            {
                'name': 'From Scratch',
                'description': 'Train entire model end-to-end',
                'cost': 'Very High ($$$$$)',
                'examples': ['Gemini', 'GPT-4V'],
                'when': 'Unlimited budget, best performance',
            },
            {
                'name': 'Frozen Vision + Frozen LLM',
                'description': 'Only train connector',
                'cost': 'Low ($)',
                'examples': ['BLIP-2', 'Flamingo'],
                'when': 'Budget constraints, quick experiments',
            },
            {
                'name': 'Frozen Vision + Fine-tune LLM',
                'description': 'Train connector and LLM',
                'cost': 'Medium ($$)',
                'examples': ['LLaVA', 'InstructBLIP'],
                'when': 'Good quality, reasonable cost',
            },
            {
                'name': 'Visual Instruction Tuning',
                'description': 'Fine-tune on instruction-following data',
                'cost': 'Low ($)',
                'examples': ['LLaVA', 'MiniGPT-4'],
                'when': 'Improve instruction-following',
            },
        ]

        for approach in approaches:
            print(f"\n{approach['name']}:")
            print(f"  Description: {approach['description']}")
            print(f"  Cost: {approach['cost']}")
            print(f"  Examples: {approach['examples']}")
            print(f"  When to use: {approach['when']}")


survey = MultimodalModelsSurvey()
survey.model_comparison()
survey.architecture_patterns()
survey.training_approaches()
```

---

## 6. Real-World Applications

### Application 1: E-commerce Product Understanding

```python
class ProductUnderstandingSystem:
    """
    E-commerce product understanding with multimodal models.

    Tasks:
    - Generate product descriptions from images
    - Extract attributes (color, style, material)
    - Visual search and similarity
    - Quality assessment
    """

    def __init__(self):
        print("="*80)
        print("E-commerce Product Understanding System")
        print("="*80)

    def product_description_generation(self, product_image):
        """
        Generate product description from image.

        Uses: BLIP or LLaVA
        """

        prompt = """
        Analyze this product image and generate a compelling product description.

        Include:
        - Product type and category
        - Key features and materials
        - Style and design elements
        - Suggested use cases
        - Target audience

        Format as an e-commerce product listing.
        """

        print("\nProduct Description Generation:")
        print(f"  Input: Product image")
        print(f"  Prompt: {prompt.strip()}")
        print(f"  Model: BLIP-2 or LLaVA")
        print(f"  Output: SEO-optimized product description")

    def attribute_extraction(self, product_image):
        """Extract structured attributes from product image."""

        attributes_to_extract = [
            "Color",
            "Material",
            "Style",
            "Pattern",
            "Season",
            "Occasion",
            "Size category",
        ]

        print("\nAttribute Extraction:")
        print(f"  Attributes to extract: {', '.join(attributes_to_extract)}")
        print(f"  Method: Few-shot prompting with vision model")
        print(f"  Output: Structured JSON")

        # Example output
        example_output = {
            "color": "Navy blue",
            "material": "Cotton",
            "style": "Casual",
            "pattern": "Solid",
            "season": "All season",
            "occasion": "Casual wear",
            "size_category": "Regular fit"
        }

        print(f"  Example: {example_output}")

    def visual_search(self):
        """Visual search implementation."""

        print("\n" + "="*80)
        print("Visual Search System")
        print("="*80)

        print("\nArchitecture:")
        print("  1. Encode all products: CLIP image encoder → embeddings")
        print("  2. Store embeddings: Vector database (Pinecone, Weaviate)")
        print("  3. Query time:")
        print("     a. Encode query image")
        print("     b. Similarity search in vector DB")
        print("     c. Return top-k similar products")

        print("\nPerformance:")
        print("  • Index 1M products: ~10 minutes (one-time)")
        print("  • Query latency: <50ms")
        print("  • Accuracy: ~95% for exact matches")
        print("  • Accuracy: ~80% for similar items")


product_system = ProductUnderstandingSystem()
product_system.product_description_generation(None)
product_system.attribute_extraction(None)
product_system.visual_search()
```

### Application 2: Accessibility - Image Description for Visually Impaired

```python
class AccessibilityAssistant:
    """
    Accessibility assistant for visually impaired users.

    Features:
    - Detailed image descriptions
    - Scene understanding
    - Text reading (OCR)
    - Object identification
    - Navigation assistance
    """

    def __init__(self):
        print("\n" + "="*80)
        print("Accessibility Assistant")
        print("="*80)

    def describe_scene(self, image):
        """Provide detailed scene description."""

        prompt = """
        Provide a detailed description of this image for a visually impaired person.

        Include:
        1. Overall scene (location, setting, time of day)
        2. Main objects and people (positions, appearances)
        3. Actions and activities
        4. Text visible in the image
        5. Colors and lighting
        6. Spatial relationships
        7. Potential hazards or important safety information

        Be clear, specific, and comprehensive.
        """

        print("\nScene Description:")
        print("  Purpose: Enable visually impaired users to understand surroundings")
        print("  Model: GPT-4V or LLaVA")
        print("  Output: Detailed audio description")
        print("  Latency requirement: <2 seconds")

    def read_text_in_image(self, image):
        """Read and speak text from image (signs, menus, etc.)."""

        print("\nText Reading (OCR + TTS):")
        print("  1. OCR: Extract text from image")
        print("  2. Orientation: Determine reading order")
        print("  3. TTS: Convert to speech")
        print("  4. Context: Explain what the text is (e.g., 'menu', 'sign')")

    def navigation_assistance(self, image):
        """Help navigate environment."""

        print("\nNavigation Assistance:")
        print("  • Identify obstacles and hazards")
        print("  • Describe doorways and pathways")
        print("  • Read street signs and directions")
        print("  • Identify landmarks")
        print("  • Real-time guidance")

    def implementation_considerations(self):
        """Key considerations for accessibility app."""

        print("\n" + "="*80)
        print("Implementation Considerations")
        print("="*80)

        considerations = [
            ("Latency", "Must be near real-time (<2s for safety)"),
            ("Privacy", "On-device inference preferred (sensitive locations)"),
            ("Reliability", "High accuracy critical for safety"),
            ("Battery", "Optimize for extended mobile use"),
            ("Connectivity", "Work offline when possible"),
            ("Audio", "Clear text-to-speech output"),
            ("Continuous", "Process video stream, not just photos"),
        ]

        for consideration, detail in considerations:
            print(f"  • {consideration}: {detail}")


accessibility_app = AccessibilityAssistant()
accessibility_app.describe_scene(None)
accessibility_app.read_text_in_image(None)
accessibility_app.navigation_assistance(None)
accessibility_app.implementation_considerations()
```

---

## 7. Practice Exercises

### Exercise 1: Build a Visual QA System

```python
"""
Exercise: Build a visual question answering system.

Requirements:
1. Load BLIP or LLaVA model
2. Create interface for image + question input
3. Generate answer
4. Evaluate on VQAv2 dataset (optional)

Bonus:
- Add support for follow-up questions
- Implement confidence scoring
- Handle multiple images
"""

def build_vqa_system():
    """Implement VQA system."""
    # TODO: Your implementation
    pass
```

### Exercise 2: Implement Image Captioning with Beam Search

```python
"""
Exercise: Implement image captioning with beam search.

1. Load BLIP captioning model
2. Implement beam search decoding
3. Compare greedy vs beam search quality
4. Visualize beam search tree

Hint: Beam search explores multiple hypotheses simultaneously
"""

def beam_search_captioning(model, image, beam_size=5):
    """Generate captions with beam search."""
    # TODO: Implement beam search
    # Keep top-k hypotheses at each step
    # Return best caption
    pass
```

### Exercise 3: Visual Similarity Search

```python
"""
Exercise: Build visual similarity search.

1. Use CLIP to encode product images
2. Store embeddings in vector database
3. Implement similarity search
4. Evaluate retrieval quality

Dataset: Use a small product dataset or create synthetic one
"""

class VisualSearchEngine:
    def __init__(self):
        # TODO: Initialize CLIP model
        pass

    def index_images(self, image_paths):
        # TODO: Encode and store embeddings
        pass

    def search(self, query_image, top_k=10):
        # TODO: Find similar images
        pass
```

---

## Key Takeaways

1. **CLIP** revolutionized vision-language learning:
   - Joint embedding space for images and text
   - Contrastive learning on 400M pairs
   - Enables zero-shot classification
   - Foundation for many downstream models

2. **BLIP/BLIP-2** improved vision-language pretraining:
   - BLIP: Bootstraps own training data (CapFilt)
   - BLIP-2: Q-Former bridges frozen vision and frozen LLM
   - 54x cheaper to train than end-to-end
   - Modular design allows using any LLM backend

3. **GPT-4V** set new SOTA:
   - Multimodal understanding at GPT-4 quality
   - OCR, chart analysis, visual reasoning
   - Multi-image understanding
   - Proprietary, API-only access

4. **LLaVA** is the open-source alternative:
   - Simple architecture: Vision encoder + Projection + LLM
   - Visual instruction tuning with GPT-4 data
   - Matches or exceeds GPT-4V on some benchmarks
   - Efficient: 7B version can run on consumer GPUs

5. **Architectural patterns**:
   - Early fusion: Rich interactions, expensive
   - Late fusion: Efficient, limited reasoning
   - Adapter-based: Cheap training, bottleneck
   - Token concatenation: Simple, leverages LLM

6. **Real-world applications**:
   - E-commerce: Product understanding, visual search
   - Accessibility: Scene description, navigation
   - Education: Visual learning, problem solving
   - Content moderation: Image understanding

---

## Further Reading

### Papers
1. **CLIP**: "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
2. **BLIP**: "BLIP: Bootstrapping Language-Image Pre-training" (Li et al., 2022)
3. **BLIP-2**: "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models" (Li et al., 2023)
4. **LLaVA**: "Visual Instruction Tuning" (Liu et al., 2023)
5. **Flamingo**: "Flamingo: a Visual Language Model for Few-Shot Learning" (Alayrac et al., 2022)
6. **Qwen-VL**: "Qwen-VL: A Versatile Vision-Language Model" (Bai et al., 2023)

### Resources
- Hugging Face Multimodal Models: https://huggingface.co/models?pipeline_tag=visual-question-answering
- CLIP GitHub: https://github.com/openai/CLIP
- LLaVA Project: https://llava-vl.github.io/
- Awesome Multimodal LLM: https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models

### Related Modules
- **Module 7**: NLP and Transformers
- **Module 8**: Computer Vision
- **Module 15 Lesson 1**: LLM Architectures
- **Module 15 Lesson 4**: Advanced Prompting (vision prompts)

---

**Next Lesson**: Advanced Prompting and In-Context Learning

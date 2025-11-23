"""
Multimodal Model Architecture Visualization
Shows vision-language model structure (CLIP, GPT-4V style)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def visualize_multimodal():
    """Visualize multimodal vision-language architecture."""

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.suptitle('Multimodal Vision-Language Model Architecture', fontsize=18, fontweight='bold')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Image Input (left)
    img_input = FancyBboxPatch((0.5, 9), 3, 2, boxstyle="round,pad=0.1",
                                edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=3)
    ax.add_patch(img_input)
    ax.text(2, 10.3, 'Image Input', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 9.7, '224 × 224 × 3', ha='center', fontsize=10, family='monospace')

    # Vision Encoder (left)
    vision_enc = FancyBboxPatch((0.5, 5.5), 3, 3, boxstyle="round,pad=0.15",
                                 edgecolor='#E74C3C', facecolor='#F5B7B1', linewidth=3)
    ax.add_patch(vision_enc)
    ax.text(2, 7.8, 'Vision Encoder', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 7.3, '(ViT/ResNet)', ha='center', fontsize=10)

    # Vision encoder internals
    ax.text(2, 6.8, 'Patch Embed', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, 6.4, '↓', ha='center', fontsize=10)
    ax.text(2, 6.2, 'Transformer', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(2, 5.8, 'Blocks × 12', ha='center', fontsize=7, style='italic')

    # Arrow image to encoder
    arrow = FancyArrowPatch((2, 9), (2, 8.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#E74C3C')
    ax.add_patch(arrow)

    # Text Input (right)
    text_input = FancyBboxPatch((16.5, 9), 3, 2, boxstyle="round,pad=0.1",
                                 edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=3)
    ax.add_patch(text_input)
    ax.text(18, 10.3, 'Text Input', ha='center', fontsize=12, fontweight='bold')
    ax.text(18, 9.7, '"A cat on beach"', ha='center', fontsize=10, style='italic')

    # Text Encoder (right)
    text_enc = FancyBboxPatch((16.5, 5.5), 3, 3, boxstyle="round,pad=0.15",
                               edgecolor='#3498DB', facecolor='#AED6F1', linewidth=3)
    ax.add_patch(text_enc)
    ax.text(18, 7.8, 'Text Encoder', ha='center', fontsize=12, fontweight='bold')
    ax.text(18, 7.3, '(Transformer)', ha='center', fontsize=10)

    # Text encoder internals
    ax.text(18, 6.8, 'Token Embed', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(18, 6.4, '↓', ha='center', fontsize=10)
    ax.text(18, 6.2, 'Transformer', ha='center', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(18, 5.8, 'Blocks × 12', ha='center', fontsize=7, style='italic')

    # Arrow text to encoder
    arrow = FancyArrowPatch((18, 9), (18, 8.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#3498DB')
    ax.add_patch(arrow)

    # Vision embeddings
    vision_emb = FancyBboxPatch((0.5, 3), 3, 2, boxstyle="round,pad=0.1",
                                 edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=2)
    ax.add_patch(vision_emb)
    ax.text(2, 4.3, 'Image', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 3.9, 'Embeddings', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 3.4, '512-dim vector', ha='center', fontsize=9, family='monospace')

    # Arrow encoder to embeddings
    arrow = FancyArrowPatch((2, 5.5), (2, 5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#E74C3C')
    ax.add_patch(arrow)

    # Text embeddings
    text_emb = FancyBboxPatch((16.5, 3), 3, 2, boxstyle="round,pad=0.1",
                               edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=2)
    ax.add_patch(text_emb)
    ax.text(18, 4.3, 'Text', ha='center', fontsize=11, fontweight='bold')
    ax.text(18, 3.9, 'Embeddings', ha='center', fontsize=11, fontweight='bold')
    ax.text(18, 3.4, '512-dim vector', ha='center', fontsize=9, family='monospace')

    # Arrow encoder to embeddings
    arrow = FancyArrowPatch((18, 5.5), (18, 5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#3498DB')
    ax.add_patch(arrow)

    # Shared embedding space (center)
    shared_space = FancyBboxPatch((7, 2.5), 6, 3, boxstyle="round,pad=0.2",
                                   edgecolor='#9B59B6', facecolor='#E8DAEF', linewidth=4)
    ax.add_patch(shared_space)
    ax.text(10, 5, 'Shared Embedding Space', ha='center', fontsize=13, fontweight='bold',
            color='#6C3483')
    ax.text(10, 4.5, 'Images and text mapped to same space', ha='center', fontsize=10,
            style='italic')

    # Projection layers
    ax.text(10, 3.9, 'Linear Projection + Normalization', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # Similarity/dot product
    ax.text(10, 3.3, 'Cosine Similarity:', ha='center', fontsize=10, fontweight='bold')
    ax.text(10, 2.9, r'$\text{sim}(I, T) = \frac{I \cdot T}{||I|| \cdot ||T||}$',
            ha='center', fontsize=11, family='monospace')

    # Arrows to shared space
    arrow = FancyArrowPatch((3.5, 4), (7, 4), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#E74C3C')
    ax.add_patch(arrow)
    ax.text(5.25, 4.3, 'Project', ha='center', fontsize=9, color='#C0392B')

    arrow = FancyArrowPatch((16.5, 4), (13, 4), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#3498DB')
    ax.add_patch(arrow)
    ax.text(14.75, 4.3, 'Project', ha='center', fontsize=9, color='#2874A6')

    # Applications (bottom)
    app_y = 0.8
    apps = [
        ('Image-Text\nRetrieval', '#27AE60', 2),
        ('Zero-Shot\nClassification', '#E67E22', 6),
        ('Visual Q&A', '#C0392B', 10),
        ('Image\nGeneration', '#8E44AD', 14),
        ('Vision\nLLMs', '#16A085', 18)
    ]

    for app_name, color, x_pos in apps:
        app_box = FancyBboxPatch((x_pos-1, app_y-0.3), 2, 0.8, boxstyle="round,pad=0.05",
                                  edgecolor=color, facecolor=color, alpha=0.3, linewidth=2)
        ax.add_patch(app_box)
        ax.text(x_pos, app_y, app_name, ha='center', fontsize=8, fontweight='bold')

        # Arrow from shared space to apps
        arrow = FancyArrowPatch((10, 2.5), (x_pos, 1.2), arrowstyle='->', mutation_scale=15,
                                 linewidth=1.5, color=color, alpha=0.6)
        ax.add_patch(arrow)

    # Training objective box
    train_box = FancyBboxPatch((4.5, 6.5), 11, 1.5, boxstyle="round,pad=0.15",
                                edgecolor='#2C3E50', facecolor='#F8F9F9', linewidth=2)
    ax.add_patch(train_box)
    ax.text(10, 7.6, 'Contrastive Learning (CLIP-style)', ha='center', fontsize=11,
            fontweight='bold', color='#2C3E50')
    ax.text(10, 7.2, 'Match paired (image, text), separate unpaired',
            ha='center', fontsize=10)
    ax.text(10, 6.8, r'$\mathcal{L} = -\log \frac{\exp(sim(I_i, T_i) / \tau)}{\sum_j \exp(sim(I_i, T_j) / \tau)}$',
            ha='center', fontsize=9, family='monospace')

    plt.tight_layout()
    plt.savefig('Learnings/Module 15 - Multimodal Models and Advanced LLMs/visualizations/01_multimodal_architecture.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 01_multimodal_architecture.png")
    plt.close()

if __name__ == "__main__":
    visualize_multimodal()

"""
Stable Diffusion Architecture Visualization
Shows the complete pipeline with latent space
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def visualize_stable_diffusion():
    """Visualize Stable Diffusion architecture."""

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))
    fig.suptitle('Stable Diffusion Architecture', fontsize=18, fontweight='bold')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Text Encoder (left side)
    text_box = FancyBboxPatch((0.5, 8), 3, 3, boxstyle="round,pad=0.15",
                               edgecolor='#9B59B6', facecolor='#E8DAEF', linewidth=3)
    ax.add_patch(text_box)
    ax.text(2, 10.5, 'Text Encoder', ha='center', fontsize=12, fontweight='bold')
    ax.text(2, 9.7, '(CLIP/T5)', ha='center', fontsize=10)
    ax.text(2, 9.2, 'Converts text', ha='center', fontsize=9)
    ax.text(2, 8.8, 'to embeddings', ha='center', fontsize=9)
    ax.text(2, 8.3, '77 × 768', ha='center', fontsize=9,
            family='monospace', style='italic')

    # Text input
    text_input_box = FancyBboxPatch((0.5, 11.2), 3, 0.6, boxstyle="round,pad=0.05",
                                     edgecolor='#8E44AD', facecolor='#F4ECF7', linewidth=2)
    ax.add_patch(text_input_box)
    ax.text(2, 11.5, '"A cat wearing sunglasses"', ha='center', fontsize=9,
            style='italic')

    # Arrow from text to encoder
    arrow = FancyArrowPatch((2, 11.2), (2, 11), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#9B59B6')
    ax.add_patch(arrow)

    # Arrow from encoder to U-Net
    arrow = FancyArrowPatch((3.5, 9.5), (6.5, 9.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#9B59B6')
    ax.add_patch(arrow)
    ax.text(5, 10, 'Text', ha='center', fontsize=9, color='#8E44AD')
    ax.text(5, 9.7, 'Embeddings', ha='center', fontsize=9, color='#8E44AD')

    # U-Net (center)
    unet_box = FancyBboxPatch((6.5, 5.5), 7, 6, boxstyle="round,pad=0.2",
                               edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=4)
    ax.add_patch(unet_box)
    ax.text(10, 10.8, 'U-Net', ha='center', fontsize=14, fontweight='bold')
    ax.text(10, 10.2, 'Iterative Denoising in Latent Space', ha='center', fontsize=10)

    # U-Net internal structure
    # Downsampling
    down_y = 9
    for i in range(3):
        box_h = 0.8 - i*0.15
        box = FancyBboxPatch((7 + i*0.8, down_y - i*0.9), 1.2, box_h,
                              boxstyle="round,pad=0.05",
                              edgecolor='#2874A6', facecolor='#AED6F1', linewidth=2)
        ax.add_patch(box)
        ax.text(7.6 + i*0.8, down_y - i*0.9 + box_h/2, f'{64*(2**i)}ch',
                ha='center', fontsize=7)

    # Bottleneck
    bottle_box = FancyBboxPatch((9.2, 6.2), 1.6, 0.5, boxstyle="round,pad=0.05",
                                 edgecolor='#1A5490', facecolor='#85C1E9', linewidth=2)
    ax.add_patch(bottle_box)
    ax.text(10, 6.45, 'Bottleneck', ha='center', fontsize=7, fontweight='bold')

    # Upsampling
    for i in range(3):
        box_h = 0.5 + i*0.15
        box = FancyBboxPatch((10.6 + i*0.8, 7 + i*0.9), 1.2, box_h,
                              boxstyle="round,pad=0.05",
                              edgecolor='#2874A6', facecolor='#AED6F1', linewidth=2)
        ax.add_patch(box)
        ax.text(11.2 + i*0.8, 7 + i*0.9 + box_h/2, f'{256/(2**i):.0f}ch',
                ha='center', fontsize=7)

    # Cross-attention
    for i in range(3):
        attn_box = FancyBboxPatch((7.5 + i*2, 7.5 + i*0.5), 1, 0.4,
                                   boxstyle="round,pad=0.03",
                                   edgecolor='#9B59B6', facecolor='#D7BDE2', linewidth=1.5)
        ax.add_patch(attn_box)
        ax.text(8 + i*2, 7.7 + i*0.5, 'Attn', ha='center', fontsize=6)

    # Time embedding
    time_box = FancyBboxPatch((8.5, 11.5), 3, 0.6, boxstyle="round,pad=0.05",
                               edgecolor='#E67E22', facecolor='#FAE5D3', linewidth=2)
    ax.add_patch(time_box)
    ax.text(10, 11.8, 'Time Step Embedding (t)', ha='center', fontsize=9,
            fontweight='bold', color='#D35400')

    # Arrow from time to U-Net
    arrow = FancyArrowPatch((10, 11.5), (10, 11.1), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#E67E22')
    ax.add_patch(arrow)

    # Latent space (bottom middle)
    latent_box = FancyBboxPatch((6.5, 3.5), 7, 1.5, boxstyle="round,pad=0.15",
                                 edgecolor='#16A085', facecolor='#A9DFBF', linewidth=3)
    ax.add_patch(latent_box)
    ax.text(10, 4.7, 'Latent Space Representation', ha='center', fontsize=11,
            fontweight='bold')
    ax.text(10, 4.2, 'z ~ 64 × 64 × 4 (8× compression)', ha='center', fontsize=9,
            family='monospace')
    ax.text(10, 3.8, 'Much smaller & faster than pixel space!', ha='center',
            fontsize=9, style='italic', color='#138D75')

    # VAE Encoder (left bottom)
    vae_enc_box = FancyBboxPatch((0.5, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                                  edgecolor='#16A085', facecolor='#D5F4E6', linewidth=2)
    ax.add_patch(vae_enc_box)
    ax.text(2, 4.6, 'VAE Encoder', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 4.1, '512×512×3', ha='center', fontsize=9)
    ax.text(2, 3.8, '↓', ha='center', fontsize=12)
    ax.text(2, 3.7, '64×64×4', ha='center', fontsize=9)

    # Arrow encoder to latent
    arrow = FancyArrowPatch((3.5, 4.25), (6.5, 4.25), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#16A085', linestyle='--')
    ax.add_patch(arrow)
    ax.text(5, 4.5, 'Encode', ha='center', fontsize=8, color='#138D75')

    # VAE Decoder (right bottom)
    vae_dec_box = FancyBboxPatch((16.5, 3.5), 3, 1.5, boxstyle="round,pad=0.1",
                                  edgecolor='#D35400', facecolor='#FAE5D3', linewidth=2)
    ax.add_patch(vae_dec_box)
    ax.text(18, 4.6, 'VAE Decoder', ha='center', fontsize=11, fontweight='bold')
    ax.text(18, 4.1, '64×64×4', ha='center', fontsize=9)
    ax.text(18, 3.8, '↓', ha='center', fontsize=12)
    ax.text(18, 3.7, '512×512×3', ha='center', fontsize=9)

    # Arrow latent to decoder
    arrow = FancyArrowPatch((13.5, 4.25), (16.5, 4.25), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#D35400')
    ax.add_patch(arrow)
    ax.text(15, 4.5, 'Decode', ha='center', fontsize=8, color='#C0392B')

    # Noise input
    noise_box = FancyBboxPatch((6.5, 1.5), 3, 1, boxstyle="round,pad=0.1",
                                edgecolor='#7F8C8D', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(noise_box)
    ax.text(8, 2.2, 'Random Noise', ha='center', fontsize=10, fontweight='bold')
    ax.text(8, 1.85, 'z_T ~ N(0, I)', ha='center', fontsize=9, family='monospace')

    # Arrow noise to latent
    arrow = FancyArrowPatch((8, 2.5), (8, 3.5), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#7F8C8D', linestyle='--')
    ax.add_patch(arrow)
    ax.text(8.5, 3, 't=1000', ha='center', fontsize=8, color='#5D6D7E')

    # Iterative loop
    loop_arrow1 = FancyArrowPatch((10, 5.5), (10, 3.5), arrowstyle='->', mutation_scale=20,
                                   linewidth=3, color='#3498DB')
    ax.add_patch(loop_arrow1)
    ax.text(10.5, 4.5, 'Denoise', ha='center', fontsize=9,
            color='#2874A6', fontweight='bold')

    loop_arrow2 = FancyArrowPatch((12, 3.5), (12, 5.5), arrowstyle='->', mutation_scale=20,
                                   linewidth=3, color='#3498DB', linestyle='--')
    ax.add_patch(loop_arrow2)
    ax.text(12.7, 4.5, '50-100', ha='center', fontsize=8, color='#2874A6')
    ax.text(12.7, 4.2, 'steps', ha='center', fontsize=8, color='#2874A6')

    # Final output
    output_box = FancyBboxPatch((16.5, 1.5), 3, 1, boxstyle="round,pad=0.1",
                                 edgecolor='#27AE60', facecolor='#ABEBC6', linewidth=3)
    ax.add_patch(output_box)
    ax.text(18, 2.2, 'Generated Image', ha='center', fontsize=11, fontweight='bold',
            color='#1E8449')
    ax.text(18, 1.85, '512 × 512 × 3', ha='center', fontsize=9, family='monospace')

    # Arrow decoder to output
    arrow = FancyArrowPatch((18, 3.5), (18, 2.5), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#27AE60')
    ax.add_patch(arrow)

    # Key advantages box
    adv_box = FancyBboxPatch((0.5, 0.2), 9, 1, boxstyle="round,pad=0.1",
                              edgecolor='#2C3E50', facecolor='#F8F9F9', linewidth=2)
    ax.add_patch(adv_box)
    ax.text(5, 0.8, '✓ Works in compressed latent space (8× faster than pixel-space diffusion)',
            ha='center', fontsize=9)
    ax.text(5, 0.4, '✓ Cross-attention for precise text conditioning ✓ Open source!',
            ha='center', fontsize=9)

    # Components box
    comp_box = FancyBboxPatch((10.5, 0.2), 9, 1, boxstyle="round,pad=0.1",
                               edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(comp_box)
    ax.text(15, 0.8, 'Components: CLIP Text Encoder + U-Net + VAE',
            ha='center', fontsize=9, fontweight='bold')
    ax.text(15, 0.4, 'Parameters: ~860M (U-Net: 860M, VAE: 83M, CLIP: 123M)',
            ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('Learnings/Module 14 - Generative Vision Models/visualizations/02_stable_diffusion_architecture.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 02_stable_diffusion_architecture.png")
    plt.close()

if __name__ == "__main__":
    visualize_stable_diffusion()

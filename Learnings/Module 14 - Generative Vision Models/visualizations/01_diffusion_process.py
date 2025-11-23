"""
Diffusion Model Process Visualization
Shows forward and reverse diffusion process
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def visualize_diffusion():
    """Visualize diffusion model forward and reverse process."""

    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    fig.suptitle('Diffusion Models: Forward and Reverse Process', fontsize=18, fontweight='bold')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Forward process (top)
    forward_y = 7
    ax.text(10, 9, 'Forward Diffusion (Add Noise)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#E74C3C')

    # Original image
    img_box = FancyBboxPatch((0.5, forward_y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                              edgecolor='#2ECC71', facecolor='#D5F4E6', linewidth=3)
    ax.add_patch(img_box)
    ax.text(1.5, forward_y, 'Original\nImage', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(1.5, forward_y-1.3, 't = 0', ha='center', fontsize=9, style='italic')

    # Noise steps
    steps = 4
    for i in range(1, steps + 1):
        x_pos = 0.5 + i * 4
        noise_level = i / steps

        # Arrow
        arrow = FancyArrowPatch((x_pos - 1.3, forward_y), (x_pos + 0.3, forward_y),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                 color='#E74C3C')
        ax.add_patch(arrow)
        ax.text(x_pos - 0.5, forward_y + 0.6, f'+ noise', ha='center',
                fontsize=9, color='#E74C3C')

        # Noisy image box
        if i < steps:
            box_color = '#FCE4EC' if noise_level < 0.7 else '#FFCDD2'
        else:
            box_color = '#FFEBEE'

        img_box = FancyBboxPatch((x_pos, forward_y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                                  edgecolor='#E74C3C', facecolor=box_color, linewidth=2,
                                  alpha=0.7 + noise_level*0.3)
        ax.add_patch(img_box)

        if i == steps:
            ax.text(x_pos + 1, forward_y, 'Pure\nNoise', ha='center', va='center',
                    fontsize=11, fontweight='bold')
        else:
            ax.text(x_pos + 1, forward_y, f'Noisy\nImage', ha='center', va='center',
                    fontsize=10)

        ax.text(x_pos + 1, forward_y-1.3, f't = {i*25}', ha='center',
                fontsize=9, style='italic')

    # Reverse process (bottom)
    reverse_y = 3
    ax.text(10, 4.7, 'Reverse Diffusion (Denoise with Neural Network)', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#3498DB')

    # Pure noise (starting point for generation)
    img_box = FancyBboxPatch((16.5, reverse_y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                              edgecolor='#9B59B6', facecolor='#E8DAEF', linewidth=3)
    ax.add_patch(img_box)
    ax.text(17.5, reverse_y, 'Random\nNoise', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(17.5, reverse_y-1.3, 't = 100', ha='center', fontsize=9, style='italic')

    # Denoising steps
    for i in range(steps -1, -1, -1):
        x_pos = 16.5 - (steps - i) * 4
        noise_level = i / steps

        # Arrow with NN
        arrow = FancyArrowPatch((x_pos + 2.3, reverse_y), (x_pos - 0.3, reverse_y),
                                 arrowstyle='->', mutation_scale=20, linewidth=2.5,
                                 color='#3498DB')
        ax.add_patch(arrow)

        # Neural network box
        nn_box = FancyBboxPatch((x_pos + 0.5, reverse_y - 0.3), 1.3, 0.6,
                                 boxstyle="round,pad=0.05",
                                 edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=1.5)
        ax.add_patch(nn_box)
        ax.text(x_pos + 1.15, reverse_y, 'U-Net', ha='center', va='center',
                fontsize=8, fontweight='bold', color='#2C3E50')

        # Denoised image box
        if i == 0:
            box_color = '#D5F4E6'
            edge_color = '#2ECC71'
            linewidth = 3
        else:
            box_color = '#FCE4EC' if noise_level < 0.5 else '#FFCDD2'
            edge_color = '#3498DB'
            linewidth = 2

        img_box = FancyBboxPatch((x_pos, reverse_y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                                  edgecolor=edge_color, facecolor=box_color,
                                  linewidth=linewidth, alpha=0.8)
        ax.add_patch(img_box)

        if i == 0:
            ax.text(x_pos + 1, reverse_y, 'Generated\nImage', ha='center', va='center',
                    fontsize=11, fontweight='bold')
        else:
            ax.text(x_pos + 1, reverse_y, f'Less Noisy\nImage', ha='center', va='center',
                    fontsize=10)

        ax.text(x_pos + 1, reverse_y-1.3, f't = {i*25}', ha='center',
                fontsize=9, style='italic')

    # Key equations
    eq_box = FancyBboxPatch((0.5, 0.2), 8, 1.3, boxstyle="round,pad=0.1",
                             edgecolor='#2C3E50', facecolor='#F8F9F9', linewidth=2)
    ax.add_patch(eq_box)
    ax.text(4.5, 1.2, 'Forward: ' + r'$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$',
            ha='center', fontsize=10, family='monospace')
    ax.text(4.5, 0.6, 'Reverse: ' + r'$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$',
            ha='center', fontsize=10, family='monospace')

    # Legend
    legend_box = FancyBboxPatch((11, 0.2), 8, 1.3, boxstyle="round,pad=0.1",
                                 edgecolor='#34495E', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(legend_box)
    ax.text(15, 1.2, 'Training: Learn to predict noise at each step',
            ha='center', fontsize=10, fontweight='bold')
    ax.text(15, 0.6, 'Generation: Start from noise, iteratively denoise',
            ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Learnings/Module 14 - Generative Vision Models/visualizations/01_diffusion_process.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 01_diffusion_process.png")
    plt.close()

if __name__ == "__main__":
    visualize_diffusion()

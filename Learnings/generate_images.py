#!/usr/bin/env python3
"""
Generate all visualization images for ML learning modules
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.patches import ConnectionPatch
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def ensure_dir(path):
    """Ensure directory exists"""
    import os
    os.makedirs(path, exist_ok=True)

# ============================================================================
# MODULE 5: Neural Networks
# ============================================================================

def generate_module5_images():
    """Generate Neural Networks visualizations"""
    print("Generating Module 5 images...")

    # 1. Neural Network Architecture
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Input layer
    for i in range(4):
        circle = Circle((1, 2 + i*2), 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(0.3, 2 + i*2, f'x{i+1}', fontsize=12, ha='center', va='center')

    # Hidden layer 1
    for i in range(5):
        circle = Circle((4, 1.5 + i*1.5), 0.3, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(circle)
        # Draw connections from input
        for j in range(4):
            ax.plot([1.3, 3.7], [2 + j*2, 1.5 + i*1.5], 'gray', alpha=0.3, linewidth=0.5)

    # Hidden layer 2
    for i in range(3):
        circle = Circle((7, 2.5 + i*2), 0.3, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(circle)
        # Draw connections from hidden layer 1
        for j in range(5):
            ax.plot([4.3, 6.7], [1.5 + j*1.5, 2.5 + i*2], 'gray', alpha=0.3, linewidth=0.5)

    # Output layer
    circle = Circle((9.5, 5), 0.3, color='gold', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(10.2, 5, 'Output', fontsize=12, ha='left', va='center')
    # Draw connections from hidden layer 2
    for i in range(3):
        ax.plot([7.3, 9.2], [2.5 + i*2, 5], 'gray', alpha=0.3, linewidth=0.5)

    # Labels
    ax.text(1, 0.5, 'Input Layer\n(4 neurons)', fontsize=12, ha='center', weight='bold')
    ax.text(4, 0.2, 'Hidden Layer 1\n(5 neurons)', fontsize=12, ha='center', weight='bold')
    ax.text(7, 0.5, 'Hidden Layer 2\n(3 neurons)', fontsize=12, ha='center', weight='bold')
    ax.text(9.5, 0.5, 'Output Layer\n(1 neuron)', fontsize=12, ha='center', weight='bold')

    ax.set_title('Neural Network Architecture', fontsize=16, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('images/module5/neural_network_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Activation Functions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    x = np.linspace(-5, 5, 100)

    # Sigmoid
    y = 1 / (1 + np.exp(-x))
    axes[0, 0].plot(x, y, linewidth=3, color='blue')
    axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e^-x)', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[0, 0].set_ylabel('Output', fontsize=10)

    # Tanh
    y = np.tanh(x)
    axes[0, 1].plot(x, y, linewidth=3, color='green')
    axes[0, 1].set_title('Tanh: tanh(x)', fontsize=12, weight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 1].axvline(x=0, color='k', linewidth=0.5)

    # ReLU
    y = np.maximum(0, x)
    axes[0, 2].plot(x, y, linewidth=3, color='red')
    axes[0, 2].set_title('ReLU: max(0, x)', fontsize=12, weight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axhline(y=0, color='k', linewidth=0.5)
    axes[0, 2].axvline(x=0, color='k', linewidth=0.5)

    # Leaky ReLU
    y = np.where(x > 0, x, 0.1 * x)
    axes[1, 0].plot(x, y, linewidth=3, color='purple')
    axes[1, 0].set_title('Leaky ReLU: max(0.1x, x)', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 0].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 0].set_xlabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Output', fontsize=10)

    # ELU
    alpha = 1.0
    y = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    axes[1, 1].plot(x, y, linewidth=3, color='orange')
    axes[1, 1].set_title('ELU: x if x>0 else α(e^x-1)', fontsize=12, weight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=0, color='k', linewidth=0.5)
    axes[1, 1].axvline(x=0, color='k', linewidth=0.5)
    axes[1, 1].set_xlabel('Input', fontsize=10)

    # Softmax visualization (different)
    x_soft = np.array([1, 2, 3, 4, 5])
    y_soft = np.exp(x_soft) / np.sum(np.exp(x_soft))
    axes[1, 2].bar(range(len(x_soft)), y_soft, color='teal', alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Softmax: Probability Distribution', fontsize=12, weight='bold')
    axes[1, 2].set_xlabel('Class', fontsize=10)
    axes[1, 2].set_ylabel('Probability', fontsize=10)
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Common Activation Functions', fontsize=16, weight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig('images/module5/activation_functions.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Gradient Descent Visualization
    fig = plt.figure(figsize=(14, 5))

    # Loss surface
    ax1 = fig.add_subplot(121, projection='3d')
    w1 = np.linspace(-3, 3, 100)
    w2 = np.linspace(-3, 3, 100)
    W1, W2 = np.meshgrid(w1, w2)
    Loss = W1**2 + W2**2  # Simple quadratic loss

    ax1.plot_surface(W1, W2, Loss, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Weight 1', fontsize=10)
    ax1.set_ylabel('Weight 2', fontsize=10)
    ax1.set_zlabel('Loss', fontsize=10)
    ax1.set_title('Loss Surface', fontsize=12, weight='bold')

    # Gradient descent path
    ax2 = fig.add_subplot(122)
    w1_path = [2.5, 2.0, 1.5, 1.0, 0.5, 0.1]
    w2_path = [2.5, 2.0, 1.5, 1.0, 0.5, 0.1]
    loss_path = [w1**2 + w2**2 for w1, w2 in zip(w1_path, w2_path)]

    # Contour plot
    contour = ax2.contour(W1, W2, Loss, levels=20, cmap='viridis', alpha=0.6)
    ax2.clabel(contour, inline=True, fontsize=8)

    # Plot path
    ax2.plot(w1_path, w2_path, 'r.-', linewidth=2, markersize=10, label='Gradient Descent Path')
    ax2.plot(w1_path[0], w2_path[0], 'go', markersize=12, label='Start')
    ax2.plot(w1_path[-1], w2_path[-1], 'r*', markersize=15, label='Minimum')

    ax2.set_xlabel('Weight 1', fontsize=10)
    ax2.set_ylabel('Weight 2', fontsize=10)
    ax2.set_title('Gradient Descent Path', fontsize=12, weight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Gradient Descent Optimization', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module5/gradient_descent.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Training Progress
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = np.arange(1, 51)
    train_loss = 2.5 * np.exp(-epochs/10) + 0.1
    val_loss = 2.5 * np.exp(-epochs/10) + 0.1 + 0.05 * np.random.randn(50).cumsum() * 0.01

    axes[0].plot(epochs, train_loss, linewidth=2, label='Training Loss', color='blue')
    axes[0].plot(epochs, val_loss, linewidth=2, label='Validation Loss', color='red')
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss', fontsize=11)
    axes[0].set_title('Training vs Validation Loss', fontsize=12, weight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    train_acc = 100 * (1 - np.exp(-epochs/15))
    val_acc = 100 * (1 - np.exp(-epochs/15)) - 2

    axes[1].plot(epochs, train_acc, linewidth=2, label='Training Accuracy', color='green')
    axes[1].plot(epochs, val_acc, linewidth=2, label='Validation Accuracy', color='orange')
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Accuracy (%)', fontsize=11)
    axes[1].set_title('Training vs Validation Accuracy', fontsize=12, weight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Neural Network Training Progress', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module5/training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 5 images generated")


# ============================================================================
# MODULE 6: Computer Vision
# ============================================================================

def generate_module6_images():
    """Generate Computer Vision visualizations"""
    print("Generating Module 6 images...")

    # 1. Convolution Operation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Input image (5x5)
    input_img = np.random.rand(5, 5)
    im1 = axes[0].imshow(input_img, cmap='gray', interpolation='nearest')
    axes[0].set_title('Input Image (5×5)', fontsize=12, weight='bold')
    axes[0].grid(True, which='both', color='black', linewidth=1)
    axes[0].set_xticks(np.arange(-0.5, 5, 1), minor=False)
    axes[0].set_yticks(np.arange(-0.5, 5, 1), minor=False)
    axes[0].tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Kernel (3x3)
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # Edge detection
    im2 = axes[1].imshow(kernel, cmap='RdBu', interpolation='nearest', vmin=-1, vmax=1)
    axes[1].set_title('Kernel/Filter (3×3)\nEdge Detection', fontsize=12, weight='bold')
    axes[1].grid(True, which='both', color='black', linewidth=1)
    axes[1].set_xticks(np.arange(-0.5, 3, 1), minor=False)
    axes[1].set_yticks(np.arange(-0.5, 3, 1), minor=False)
    axes[1].tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    for i in range(3):
        for j in range(3):
            axes[1].text(j, i, f'{kernel[i, j]:.0f}', ha='center', va='center', fontsize=11, weight='bold')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    # Output feature map (3x3)
    output = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            output[i, j] = np.sum(input_img[i:i+3, j:j+3] * kernel)

    im3 = axes[2].imshow(output, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Output Feature Map (3×3)\n(After Convolution)', fontsize=12, weight='bold')
    axes[2].grid(True, which='both', color='black', linewidth=1)
    axes[2].set_xticks(np.arange(-0.5, 3, 1), minor=False)
    axes[2].set_yticks(np.arange(-0.5, 3, 1), minor=False)
    axes[2].tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    plt.colorbar(im3, ax=axes[2], fraction=0.046)

    plt.suptitle('Convolution Operation: Input ⊗ Kernel = Output', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module6/convolution_operation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. CNN Architecture
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Input
    rect = FancyBboxPatch((0.5, 2), 1.5, 4, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.25, 0.8, 'Input\n32×32×3', ha='center', fontsize=10, weight='bold')

    # Conv1
    rect = FancyBboxPatch((2.5, 2.2), 1.3, 3.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(3.15, 0.8, 'Conv1\n30×30×32', ha='center', fontsize=9, weight='bold')

    # Pool1
    rect = FancyBboxPatch((4.2, 2.5), 1.0, 3.0, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='yellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.7, 0.8, 'Pool1\n15×15×32', ha='center', fontsize=9, weight='bold')

    # Conv2
    rect = FancyBboxPatch((5.6, 2.7), 0.9, 2.6, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(6.05, 0.8, 'Conv2\n13×13×64', ha='center', fontsize=9, weight='bold')

    # Pool2
    rect = FancyBboxPatch((6.9, 3.0), 0.7, 2.0, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='yellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.25, 0.8, 'Pool2\n6×6×64', ha='center', fontsize=9, weight='bold')

    # Flatten
    rect = FancyBboxPatch((8.0, 3.5), 0.3, 1.0, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='orange', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.15, 0.8, 'Flatten\n2304', ha='center', fontsize=9, weight='bold')

    # FC1
    for i in range(6):
        circle = Circle((9.5, 2.5 + i*0.6), 0.15, color='lightcoral', ec='black', linewidth=1.5)
        ax.add_patch(circle)
    ax.text(9.5, 0.8, 'FC1\n128', ha='center', fontsize=9, weight='bold')

    # FC2
    for i in range(6):
        circle = Circle((11, 2.5 + i*0.6), 0.15, color='lightcoral', ec='black', linewidth=1.5)
        ax.add_patch(circle)
    ax.text(11, 0.8, 'FC2\n64', ha='center', fontsize=9, weight='bold')

    # Output
    for i in range(3):
        circle = Circle((12.5, 3 + i*1), 0.2, color='gold', ec='black', linewidth=2)
        ax.add_patch(circle)
    ax.text(12.5, 0.8, 'Output\n10 classes', ha='center', fontsize=9, weight='bold')

    # Arrows
    for i in range(7):
        if i < 5:
            ax.arrow(2 + i*1.4, 4, 0.3, 0, head_width=0.3, head_length=0.15, fc='gray', ec='gray')
        else:
            ax.arrow(8.5 + (i-5)*1.3, 4, 0.3, 0, head_width=0.3, head_length=0.15, fc='gray', ec='gray')

    ax.text(8, 7, 'Convolutional Neural Network Architecture', ha='center', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('images/module6/cnn_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Pooling Operations
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Create sample feature map
    feature_map = np.array([[1, 3, 2, 4],
                            [5, 6, 1, 2],
                            [7, 2, 8, 3],
                            [1, 4, 5, 6]])

    # Max pooling
    max_pooled = np.array([[6, 4],
                           [7, 8]])

    im1 = axes[0].imshow(feature_map, cmap='Blues', interpolation='nearest')
    axes[0].set_title('Feature Map (4×4)', fontsize=12, weight='bold')
    axes[0].grid(True, which='both', color='black', linewidth=2)
    axes[0].set_xticks(np.arange(-0.5, 4, 1))
    axes[0].set_yticks(np.arange(-0.5, 4, 1))
    axes[0].tick_params(which='both', size=0, labelbottom=False, labelleft=False)

    # Add grid lines for pooling regions
    axes[0].plot([1.5, 1.5], [-0.5, 3.5], 'r-', linewidth=3)
    axes[0].plot([-0.5, 3.5], [1.5, 1.5], 'r-', linewidth=3)

    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f'{feature_map[i, j]:.0f}', ha='center', va='center',
                        fontsize=14, weight='bold')

    # Max pooled result
    im2 = axes[1].imshow(max_pooled, cmap='Greens', interpolation='nearest')
    axes[1].set_title('After Max Pooling (2×2)\nStride=2', fontsize=12, weight='bold')
    axes[1].grid(True, which='both', color='black', linewidth=2)
    axes[1].set_xticks(np.arange(-0.5, 2, 1))
    axes[1].set_yticks(np.arange(-0.5, 2, 1))
    axes[1].tick_params(which='both', size=0, labelbottom=False, labelleft=False)

    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, f'{max_pooled[i, j]:.0f}', ha='center', va='center',
                        fontsize=16, weight='bold', color='darkgreen')

    plt.suptitle('Max Pooling Operation (Takes Maximum Value from Each Region)',
                 fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module6/pooling_operations.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. ResNet Skip Connections
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Input
    rect = FancyBboxPatch((1, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.75, 7.75, 'Input\nx', ha='center', va='center', fontsize=11, weight='bold')

    # Conv1
    rect = FancyBboxPatch((4, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 7.75, 'Conv\n+ReLU', ha='center', va='center', fontsize=10, weight='bold')

    # Conv2
    rect = FancyBboxPatch((4, 4.5), 1.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 5.25, 'Conv\n+ReLU', ha='center', va='center', fontsize=10, weight='bold')

    # Addition
    circle = Circle((7.5, 5.25), 0.4, color='yellow', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(7.5, 5.25, '+', ha='center', va='center', fontsize=20, weight='bold')

    # Output
    rect = FancyBboxPatch((8.5, 4.5), 1, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='gold', linewidth=2)
    ax.add_patch(rect)
    ax.text(9, 5.25, 'Output\nF(x)+x', ha='center', va='center', fontsize=9, weight='bold')

    # Main path arrows
    ax.arrow(2.6, 7.75, 1.2, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    ax.arrow(4.75, 6.9, 0, -1.0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    ax.arrow(5.6, 5.25, 1.5, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    ax.arrow(7.95, 5.25, 0.4, 0, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)

    # Skip connection (curved arrow)
    arrow = FancyArrowPatch((2.5, 7.5), (7.1, 5.5),
                           connectionstyle="arc3,rad=.5", arrowstyle='->',
                           mutation_scale=30, linewidth=3, color='red')
    ax.add_patch(arrow)
    ax.text(4, 8.7, 'Skip Connection (Identity)', ha='center', fontsize=12,
            weight='bold', color='red')

    ax.text(5, 9.5, 'ResNet Block: F(x) + x', ha='center', fontsize=16, weight='bold')
    ax.text(5, 0.5, 'Allows gradient to flow directly, solving vanishing gradient problem',
            ha='center', fontsize=11, style='italic')

    plt.tight_layout()
    plt.savefig('images/module6/resnet_skip_connection.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 6 images generated")


# ============================================================================
# MODULE 7: Time Series
# ============================================================================

def generate_module7_images():
    """Generate Time Series visualizations"""
    print("Generating Module 7 images...")

    np.random.seed(42)

    # 1. Time Series Components
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    # Generate time series with components
    n = 365
    time = np.arange(n)

    # Trend
    trend = 0.05 * time + 10
    axes[0].plot(time, trend, linewidth=2, color='blue')
    axes[0].set_title('Trend Component', fontsize=12, weight='bold')
    axes[0].set_ylabel('Value', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Seasonality
    seasonality = 5 * np.sin(2 * np.pi * time / 365)
    axes[1].plot(time, seasonality, linewidth=2, color='green')
    axes[1].set_title('Seasonal Component (Yearly)', fontsize=12, weight='bold')
    axes[1].set_ylabel('Value', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    # Noise
    noise = np.random.randn(n) * 1.5
    axes[2].plot(time, noise, linewidth=1, color='gray', alpha=0.7)
    axes[2].set_title('Random Noise', fontsize=12, weight='bold')
    axes[2].set_ylabel('Value', fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # Combined
    combined = trend + seasonality + noise
    axes[3].plot(time, combined, linewidth=1.5, color='purple')
    axes[3].set_title('Combined Time Series = Trend + Seasonality + Noise', fontsize=12, weight='bold')
    axes[3].set_ylabel('Value', fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # Decomposition overlay
    axes[4].plot(time, trend, linewidth=2, label='Trend', alpha=0.7)
    axes[4].plot(time, combined, linewidth=1, label='Original', alpha=0.5, color='black')
    axes[4].fill_between(time, trend - 2, trend + 2, alpha=0.2, label='Seasonal Range')
    axes[4].set_title('Decomposition Overlay', fontsize=12, weight='bold')
    axes[4].set_xlabel('Time (Days)', fontsize=10)
    axes[4].set_ylabel('Value', fontsize=10)
    axes[4].legend(fontsize=9)
    axes[4].grid(True, alpha=0.3)

    plt.suptitle('Time Series Decomposition', fontsize=16, weight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('images/module7/time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. ACF and PACF
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Simulate ACF
    lags = np.arange(0, 30)
    acf_values = 0.7 ** lags + np.random.randn(30) * 0.05
    acf_values[0] = 1.0

    # ACF plot
    axes[0].bar(lags, acf_values, width=0.3, color='blue', alpha=0.7, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=1)
    axes[0].axhline(y=0.2, color='red', linestyle='--', linewidth=1, label='Confidence Interval')
    axes[0].axhline(y=-0.2, color='red', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Lag', fontsize=11)
    axes[0].set_ylabel('ACF', fontsize=11)
    axes[0].set_title('Autocorrelation Function (ACF)', fontsize=12, weight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].set_ylim(-0.5, 1.1)

    # Simulate PACF
    pacf_values = np.zeros(30)
    pacf_values[0] = 1.0
    pacf_values[1] = 0.7
    pacf_values[2] = 0.15
    pacf_values[3:] = np.random.randn(27) * 0.05

    # PACF plot
    axes[1].bar(lags, pacf_values, width=0.3, color='green', alpha=0.7, edgecolor='black')
    axes[1].axhline(y=0, color='black', linewidth=1)
    axes[1].axhline(y=0.2, color='red', linestyle='--', linewidth=1, label='Confidence Interval')
    axes[1].axhline(y=-0.2, color='red', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Lag', fontsize=11)
    axes[1].set_ylabel('PACF', fontsize=11)
    axes[1].set_title('Partial Autocorrelation Function (PACF)', fontsize=12, weight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(-0.5, 1.1)

    plt.suptitle('ACF and PACF for ARIMA Model Selection', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module7/acf_pacf.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. ARIMA Forecast
    fig, ax = plt.subplots(figsize=(14, 6))

    # Historical data
    n_hist = 100
    n_forecast = 30
    time_hist = np.arange(n_hist)
    time_forecast = np.arange(n_hist, n_hist + n_forecast)

    # Generate data
    historical = 50 + 0.3 * time_hist + 5 * np.sin(2 * np.pi * time_hist / 20) + np.random.randn(n_hist) * 2
    forecast = 50 + 0.3 * time_forecast + 5 * np.sin(2 * np.pi * time_forecast / 20)

    # Confidence intervals
    lower_bound = forecast - 2 * (1 + 0.1 * np.arange(n_forecast))
    upper_bound = forecast + 2 * (1 + 0.1 * np.arange(n_forecast))

    # Plot
    ax.plot(time_hist, historical, linewidth=2, label='Historical Data', color='blue')
    ax.plot(time_forecast, forecast, linewidth=2, label='ARIMA Forecast', color='red', linestyle='--')
    ax.fill_between(time_forecast, lower_bound, upper_bound, alpha=0.3, color='red', label='95% Confidence Interval')

    # Add vertical line at forecast start
    ax.axvline(x=n_hist, color='black', linestyle=':', linewidth=2, label='Forecast Start')

    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('ARIMA Time Series Forecast with Confidence Intervals', fontsize=14, weight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('images/module7/arima_forecast.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. LSTM Architecture
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Time steps
    for t in range(4):
        x_pos = 2 + t * 3

        # LSTM cell
        rect = FancyBboxPatch((x_pos - 0.6, 4), 1.2, 2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightblue', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_pos, 5, f'LSTM\nt={t}', ha='center', va='center', fontsize=10, weight='bold')

        # Input
        ax.arrow(x_pos, 2.5, 0, 1.3, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
        ax.text(x_pos, 2, f'x_{t}', ha='center', fontsize=10, weight='bold')

        # Output
        ax.arrow(x_pos, 6.1, 0, 1.3, head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
        ax.text(x_pos, 7.8, f'h_{t}', ha='center', fontsize=10, weight='bold')

        # Hidden state connection
        if t < 3:
            ax.arrow(x_pos + 0.7, 5, 1.8, 0, head_width=0.15, head_length=0.1,
                    fc='red', ec='red', linewidth=2)

        # Cell state connection (curved)
        if t < 3:
            arrow = FancyArrowPatch((x_pos + 0.6, 6.2), (x_pos + 2.4, 6.2),
                                   connectionstyle="arc3,rad=0", arrowstyle='->',
                                   mutation_scale=20, linewidth=2, color='purple')
            ax.add_patch(arrow)

    # Labels
    ax.text(7, 9, 'LSTM for Time Series', ha='center', fontsize=16, weight='bold')
    ax.text(7, 0.5, 'Sequential processing with memory of past states', ha='center', fontsize=11, style='italic')

    # Legend
    ax.text(12, 7.5, 'Hidden State', color='red', fontsize=10, weight='bold')
    ax.text(12, 6.8, 'Cell State', color='purple', fontsize=10, weight='bold')
    ax.text(12, 6.1, 'Input', color='green', fontsize=10, weight='bold')
    ax.text(12, 5.4, 'Output', color='blue', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig('images/module7/lstm_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 7 images generated")


# ============================================================================
# MODULE 8: Recommender Systems
# ============================================================================

def generate_module8_images():
    """Generate Recommender Systems visualizations"""
    print("Generating Module 8 images...")

    # 1. User-Item Matrix
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create sample user-item rating matrix
    np.random.seed(42)
    n_users, n_items = 8, 10
    ratings = np.random.choice([0, 1, 2, 3, 4, 5], size=(n_users, n_items), p=[0.6, 0.05, 0.05, 0.1, 0.1, 0.1])

    # Mask zeros
    masked_ratings = np.ma.masked_where(ratings == 0, ratings)

    # Plot
    im = ax.imshow(masked_ratings, cmap='YlOrRd', interpolation='nearest', vmin=1, vmax=5)

    # Add ratings text
    for i in range(n_users):
        for j in range(n_items):
            if ratings[i, j] > 0:
                ax.text(j, i, f'{ratings[i, j]}', ha='center', va='center',
                       fontsize=11, weight='bold')
            else:
                ax.text(j, i, '?', ha='center', va='center',
                       fontsize=11, color='gray', style='italic')

    # Labels
    ax.set_xticks(np.arange(n_items))
    ax.set_yticks(np.arange(n_users))
    ax.set_xticklabels([f'Item {i+1}' for i in range(n_items)], fontsize=9)
    ax.set_yticklabels([f'User {i+1}' for i in range(n_users)], fontsize=9)
    ax.set_xlabel('Items (Movies, Products, etc.)', fontsize=11, weight='bold')
    ax.set_ylabel('Users', fontsize=11, weight='bold')

    # Grid
    ax.set_xticks(np.arange(n_items) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_users) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linewidth=1.5)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Rating', fontsize=11, weight='bold')

    ax.set_title('User-Item Rating Matrix\n(? = Missing ratings to predict)', fontsize=14, weight='bold', pad=15)

    plt.tight_layout()
    plt.savefig('images/module8/user_item_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Collaborative Filtering
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # User-based CF
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Target user
    circle = Circle((5, 8), 0.6, color='red', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(5, 8, 'Target\nUser', ha='center', va='center', fontsize=9, weight='bold', color='white')

    # Similar users
    similar_positions = [(3, 6), (7, 6), (4, 4), (6, 4)]
    for pos in similar_positions:
        circle = Circle(pos, 0.5, color='lightblue', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], 'Similar\nUser', ha='center', va='center', fontsize=8)
        # Draw connection
        ax.plot([5, pos[0]], [8, pos[1]], 'b--', linewidth=1.5, alpha=0.5)

    # Items
    item_positions = [(2, 1), (5, 1), (8, 1)]
    for i, pos in enumerate(item_positions):
        rect = Rectangle((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6,
                        edgecolor='black', facecolor='gold', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f'Item {i+1}', ha='center', va='center', fontsize=8, weight='bold')

    # Arrows from similar users to items
    for user_pos in similar_positions:
        for item_pos in item_positions[:2]:  # They rated some items
            ax.arrow(user_pos[0], user_pos[1]-0.6, item_pos[0]-user_pos[0], item_pos[1]+0.5-user_pos[1]+0.6,
                    head_width=0.1, head_length=0.1, fc='green', ec='green', linewidth=1, alpha=0.3)

    ax.text(5, 9.5, 'User-Based Collaborative Filtering', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.2, '"Users similar to you also liked..."', ha='center', fontsize=9, style='italic')

    # Item-based CF
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # User
    circle = Circle((5, 8), 0.6, color='red', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(5, 8, 'User', ha='center', va='center', fontsize=10, weight='bold', color='white')

    # Rated items
    rated_positions = [(2, 5), (4, 5), (6, 5), (8, 5)]
    for i, pos in enumerate(rated_positions):
        rect = Rectangle((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6,
                        edgecolor='black', facecolor='lightgreen', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f'Rated\n{i+1}', ha='center', va='center', fontsize=8, weight='bold')
        # Connect to user
        ax.plot([5, pos[0]], [8, pos[1]], 'g-', linewidth=1.5, alpha=0.5)

    # Similar items (recommendations)
    similar_items = [(3, 2), (5, 2), (7, 2)]
    for i, pos in enumerate(similar_items):
        rect = Rectangle((pos[0]-0.4, pos[1]-0.3), 0.8, 0.6,
                        edgecolor='black', facecolor='gold', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], f'Rec\n{i+1}', ha='center', va='center', fontsize=8, weight='bold')
        # Connect similar items
        ax.plot([rated_positions[i][0], pos[0]], [rated_positions[i][1], pos[1]],
               'b--', linewidth=1.5, alpha=0.5)

    ax.text(5, 9.5, 'Item-Based Collaborative Filtering', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.2, '"Items similar to what you liked..."', ha='center', fontsize=9, style='italic')

    plt.suptitle('Collaborative Filtering Approaches', fontsize=16, weight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('images/module8/collaborative_filtering.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Matrix Factorization
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Rating matrix R
    rect = FancyBboxPatch((0.5, 3), 2, 2.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 4.25, 'R', ha='center', va='center', fontsize=24, weight='bold')
    ax.text(1.5, 2.3, 'm × n', ha='center', fontsize=10, weight='bold')
    ax.text(1.5, 6.2, 'Rating Matrix', ha='center', fontsize=11, weight='bold')

    # Equals sign
    ax.text(3.2, 4.25, '≈', ha='center', va='center', fontsize=32)

    # User matrix U
    rect = FancyBboxPatch((4, 3.5), 1.5, 2.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 4.75, 'U', ha='center', va='center', fontsize=20, weight='bold')
    ax.text(4.75, 2.8, 'm × k', ha='center', fontsize=9, weight='bold')
    ax.text(4.75, 6.7, 'User\nFeatures', ha='center', fontsize=10, weight='bold')

    # Multiplication sign
    ax.text(6, 4.75, '×', ha='center', va='center', fontsize=28)

    # Item matrix V
    rect = FancyBboxPatch((6.8, 4), 2.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.05, 4.75, 'V^T', ha='center', va='center', fontsize=20, weight='bold')
    ax.text(8.05, 3.3, 'k × n', ha='center', fontsize=9, weight='bold')
    ax.text(8.05, 6.2, 'Item Features', ha='center', fontsize=10, weight='bold')

    # Latent factors explanation
    ax.text(7, 1.5, 'k = latent factors (e.g., genres for movies)',
            ha='center', fontsize=10, style='italic')
    ax.text(7, 0.8, 'Decompose large sparse matrix into smaller dense matrices',
            ha='center', fontsize=10, style='italic')

    # Example visualization
    ax.text(11.5, 7, 'Example:\nUser i, Item j', ha='center', fontsize=10, weight='bold')

    # Small matrices
    rect = FancyBboxPatch((10.3, 5), 0.6, 1, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightblue', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(10.6, 5.5, 'u_i', ha='center', va='center', fontsize=9, weight='bold')

    ax.text(11.2, 5.5, '·', ha='center', va='center', fontsize=16)

    rect = FancyBboxPatch((11.5, 5.3), 1, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightgreen', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(12, 5.6, 'v_j', ha='center', va='center', fontsize=9, weight='bold')

    ax.text(11.5, 4.5, 'Dot product\n= Predicted Rating', ha='center', fontsize=9, style='italic')

    ax.text(7, 7.5, 'Matrix Factorization for Recommender Systems',
            ha='center', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('images/module8/matrix_factorization.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Neural Recommender (Two-Tower)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # User Tower
    ax.text(3, 11, 'User Tower', ha='center', fontsize=13, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black', linewidth=2))

    # User input
    rect = FancyBboxPatch((2, 9.5), 2, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, 9.8, 'User ID, Age, Location', ha='center', fontsize=9)

    # User embedding
    rect = FancyBboxPatch((2.3, 8.3), 1.4, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightblue', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(3, 8.7, 'Embedding', ha='center', fontsize=10, weight='bold')

    # User Dense layers
    for i, y in enumerate([7.2, 6.2, 5.2]):
        width = 1.6 - i * 0.2
        rect = FancyBboxPatch((3 - width/2, y), width, 0.6, boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor='lightblue', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(3, y+0.3, f'Dense {i+1}', ha='center', fontsize=9)

    # User vector
    circle = Circle((3, 4), 0.5, color='blue', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(3, 4, 'u', ha='center', va='center', fontsize=14, weight='bold', color='white')

    # Item Tower
    ax.text(9, 11, 'Item Tower', ha='center', fontsize=13, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black', linewidth=2))

    # Item input
    rect = FancyBboxPatch((8, 9.5), 2, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='white', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9, 9.8, 'Item ID, Category, Price', ha='center', fontsize=9)

    # Item embedding
    rect = FancyBboxPatch((8.3, 8.3), 1.4, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='lightgreen', linewidth=1.5)
    ax.add_patch(rect)
    ax.text(9, 8.7, 'Embedding', ha='center', fontsize=10, weight='bold')

    # Item Dense layers
    for i, y in enumerate([7.2, 6.2, 5.2]):
        width = 1.6 - i * 0.2
        rect = FancyBboxPatch((9 - width/2, y), width, 0.6, boxstyle="round,pad=0.05",
                              edgecolor='black', facecolor='lightgreen', linewidth=1.5)
        ax.add_patch(rect)
        ax.text(9, y+0.3, f'Dense {i+1}', ha='center', fontsize=9)

    # Item vector
    circle = Circle((9, 4), 0.5, color='green', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(9, 4, 'v', ha='center', va='center', fontsize=14, weight='bold', color='white')

    # Dot product
    circle = Circle((6, 2), 0.6, color='yellow', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.text(6, 2, 'u · v', ha='center', va='center', fontsize=12, weight='bold')

    # Arrows
    ax.arrow(3, 3.4, 2.5, -1.2, head_width=0.2, head_length=0.15, fc='blue', ec='blue', linewidth=2)
    ax.arrow(9, 3.4, -2.5, -1.2, head_width=0.2, head_length=0.15, fc='green', ec='green', linewidth=2)

    # Output
    rect = FancyBboxPatch((5.2, 0.5), 1.6, 0.6, boxstyle="round,pad=0.05",
                          edgecolor='black', facecolor='gold', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 0.8, 'Predicted Rating', ha='center', fontsize=11, weight='bold')

    ax.arrow(6, 1.35, 0, -0.5, head_width=0.2, head_length=0.1, fc='black', ec='black', linewidth=2)

    ax.text(6, 11.5, 'Two-Tower Neural Recommender', ha='center', fontsize=16, weight='bold')

    plt.tight_layout()
    plt.savefig('images/module8/neural_recommender.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 8 images generated")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating ML Learning Module Images")
    print("=" * 60)

    ensure_dir('images/module5')
    ensure_dir('images/module6')
    ensure_dir('images/module7')
    ensure_dir('images/module8')

    generate_module5_images()
    generate_module6_images()
    generate_module7_images()
    generate_module8_images()

    print("\n" + "=" * 60)
    print("✓ All Module 5-8 images generated successfully!")
    print("=" * 60)

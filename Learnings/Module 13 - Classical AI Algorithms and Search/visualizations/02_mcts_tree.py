"""
Monte Carlo Tree Search (MCTS) Visualization
Shows the four phases of MCTS
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np

def visualize_mcts():
    """Visualize MCTS algorithm phases."""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Monte Carlo Tree Search (MCTS)', fontsize=18, fontweight='bold')

    # Create 4 subplots for each phase
    phases = ['Selection', 'Expansion', 'Simulation', 'Backpropagation']
    colors = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']

    for idx, (phase, color) in enumerate(zip(phases, colors)):
        ax = plt.subplot(2, 2, idx + 1)
        ax.set_title(f'Phase {idx+1}: {phase}', fontsize=14, fontweight='bold', color=color)
        ax.set_xlim(-1, 8)
        ax.set_ylim(-1, 6)
        ax.axis('off')

        # Draw tree structure
        draw_game_tree(ax, phase, color)

    plt.tight_layout()
    plt.savefig('Learnings/Module 13 - Classical AI Algorithms and Search/visualizations/02_mcts_tree.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 02_mcts_tree.png")
    plt.close()

def draw_game_tree(ax, phase, color):
    """Draw game tree for each MCTS phase."""

    # Node positions
    root = (3.5, 5)
    level1 = [(1.5, 3.5), (3.5, 3.5), (5.5, 3.5)]
    level2_1 = [(0.5, 2), (2.5, 2)]
    level2_2 = [(2.5, 2), (4.5, 2)]
    level2_3 = [(4.5, 2), (6.5, 2)]

    # Draw edges
    for node in level1:
        ax.plot([root[0], node[0]], [root[1], node[1]], 'k-', linewidth=2, alpha=0.3)

    for node in level2_1:
        ax.plot([level1[0][0], node[0]], [level1[0][1], node[1]], 'k-', linewidth=2, alpha=0.3)

    for node in level2_2:
        ax.plot([level1[1][0], node[0]], [level1[1][1], node[1]], 'k-', linewidth=2, alpha=0.3)

    # Root node (always visited)
    circle = Circle(root, 0.35, color='#2C3E50', zorder=5)
    ax.add_patch(circle)
    ax.text(root[0], root[1], '10/15', ha='center', va='center',
            color='white', fontsize=9, fontweight='bold')

    # Phase-specific highlighting
    if phase == 'Selection':
        # Highlight selection path
        path = [root, level1[1], level2_2[0]]
        for i in range(len(path)-1):
            arrow = FancyArrowPatch(path[i], path[i+1], arrowstyle='->',
                                     mutation_scale=20, linewidth=3,
                                     color=color, zorder=4)
            ax.add_patch(arrow)

        # Draw nodes
        for i, node in enumerate(level1):
            node_color = color if i == 1 else '#95A5A6'
            circle = Circle(node, 0.35, color=node_color, zorder=5)
            ax.add_patch(circle)
            visits = ['3/5', '5/6', '2/4'][i]
            ax.text(node[0], node[1], visits, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')

        circle = Circle(level2_2[0], 0.35, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(level2_2[0][0], level2_2[0][1], '2/3', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        ax.text(3.5, 0.5, 'Select node with best UCB1 score', ha='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    elif phase == 'Expansion':
        # Draw all nodes
        for i, node in enumerate(level1):
            circle = Circle(node, 0.35, color='#2C3E50', zorder=5)
            ax.add_patch(circle)
            visits = ['3/5', '5/6', '2/4'][i]
            ax.text(node[0], node[1], visits, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')

        # Highlight expanded node
        circle = Circle(level2_2[0], 0.35, color=color, zorder=5, linewidth=3,
                        edgecolor='yellow')
        ax.add_patch(circle)
        ax.text(level2_2[0][0], level2_2[0][1], '0/0', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        # New child
        new_child = (3.5, 0.5)
        ax.plot([level2_2[0][0], new_child[0]], [level2_2[0][1], new_child[1]],
                'k-', linewidth=2, alpha=0.3)
        circle = Circle(new_child, 0.35, color=color, zorder=5, linewidth=3,
                        edgecolor='yellow')
        ax.add_patch(circle)
        ax.text(new_child[0], new_child[1], 'NEW', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        ax.text(3.5, -0.5, 'Add new child node to tree', ha='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    elif phase == 'Simulation':
        # Draw nodes
        for i, node in enumerate(level1):
            circle = Circle(node, 0.35, color='#2C3E50', zorder=5)
            ax.add_patch(circle)

        circle = Circle(level2_2[0], 0.35, color='#2C3E50', zorder=5)
        ax.add_patch(circle)

        # Simulation path
        sim_nodes = [(3.5, 0.5), (4.5, -0.5), (5.5, -1)]
        ax.plot([level2_2[0][0], sim_nodes[0][0]], [level2_2[0][1], sim_nodes[0][1]],
                'k-', linewidth=2, alpha=0.3)

        for i, node in enumerate(sim_nodes):
            if i > 0:
                ax.plot([sim_nodes[i-1][0], node[0]], [sim_nodes[i-1][1], node[1]],
                        '--', linewidth=3, color=color, alpha=0.7)
            circle = Circle(node, 0.35, color=color, alpha=0.7, zorder=5)
            ax.add_patch(circle)

        # Result
        result_box = FancyBboxPatch((4.8, -1.3), 1.4, 0.6, boxstyle="round,pad=0.05",
                                     edgecolor=color, facecolor=color, linewidth=2)
        ax.add_patch(result_box)
        ax.text(5.5, -1, 'WIN!', ha='center', va='center',
                color='white', fontsize=10, fontweight='bold')

        ax.text(3.5, -1.8, 'Simulate random game to terminal state', ha='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    elif phase == 'Backpropagation':
        # Draw nodes with updated values
        for i, node in enumerate(level1):
            node_color = color if i == 1 else '#2C3E50'
            circle = Circle(node, 0.35, color=node_color, zorder=5)
            ax.add_patch(circle)
            visits = ['3/5', '6/7', '2/4'][i]  # Updated
            ax.text(node[0], node[1], visits, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold')

        circle = Circle(level2_2[0], 0.35, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(level2_2[0][0], level2_2[0][1], '3/4', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        new_child = (3.5, 0.5)
        circle = Circle(new_child, 0.35, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(new_child[0], new_child[1], '1/1', ha='center', va='center',
                color='white', fontsize=8, fontweight='bold')

        # Backprop arrows
        arrow1 = FancyArrowPatch(new_child, level2_2[0], arrowstyle='->',
                                  mutation_scale=15, linewidth=2,
                                  color=color, linestyle='--', zorder=4)
        arrow2 = FancyArrowPatch(level2_2[0], level1[1], arrowstyle='->',
                                  mutation_scale=15, linewidth=2,
                                  color=color, linestyle='--', zorder=4)
        arrow3 = FancyArrowPatch(level1[1], root, arrowstyle='->',
                                  mutation_scale=15, linewidth=2,
                                  color=color, linestyle='--', zorder=4)
        ax.add_patch(arrow1)
        ax.add_patch(arrow2)
        ax.add_patch(arrow3)

        # Updated root
        circle = Circle(root, 0.35, color=color, zorder=5)
        ax.add_patch(circle)
        ax.text(root[0], root[1], '11/16', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')

        ax.text(3.5, -0.5, 'Update win/visit counts up the tree', ha='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Legend
    ax.text(0, 5.5, 'Format: wins/visits', ha='left', fontsize=9,
            style='italic', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

if __name__ == "__main__":
    visualize_mcts()

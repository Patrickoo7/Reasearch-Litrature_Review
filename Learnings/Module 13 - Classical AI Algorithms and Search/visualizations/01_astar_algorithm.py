"""
A* Algorithm Visualization
Demonstrates pathfinding with heuristic search
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

def visualize_astar():
    """Visualize A* algorithm concept and execution."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('A* Algorithm: Heuristic Pathfinding', fontsize=16, fontweight='bold')

    # Left plot: Grid with path
    ax1.set_title('A* Pathfinding on Grid', fontsize=14, fontweight='bold')
    ax1.set_xlim(-0.5, 7.5)
    ax1.set_ylim(-0.5, 7.5)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)

    # Draw grid
    for i in range(8):
        for j in range(8):
            rect = mpatches.Rectangle((i-0.4, j-0.4), 0.8, 0.8,
                                       linewidth=1, edgecolor='gray',
                                       facecolor='white', alpha=0.3)
            ax1.add_patch(rect)

    # Obstacles
    obstacles = [(2, 3), (2, 4), (2, 5), (3, 5), (4, 5), (5, 2), (5, 3), (5, 4)]
    for obs in obstacles:
        rect = mpatches.Rectangle((obs[0]-0.4, obs[1]-0.4), 0.8, 0.8,
                                   linewidth=2, edgecolor='black',
                                   facecolor='#2C3E50', alpha=0.8)
        ax1.add_patch(rect)

    # Explored nodes
    explored = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3),
                (4, 1), (4, 2), (4, 3), (4, 4), (5, 5), (5, 6), (6, 6)]
    for node in explored:
        circle = mpatches.Circle(node, 0.25, color='#3498DB', alpha=0.5)
        ax1.add_patch(circle)

    # Optimal path
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)]
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax1.plot(path_x, path_y, 'o-', color='#E74C3C', linewidth=3,
             markersize=8, label='Optimal Path', zorder=5)

    # Start and goal
    start_circle = mpatches.Circle((0, 0), 0.35, color='#2ECC71',
                                    edgecolor='black', linewidth=2, zorder=6)
    goal_circle = mpatches.Circle((7, 7), 0.35, color='#F39C12',
                                   edgecolor='black', linewidth=2, zorder=6)
    ax1.add_patch(start_circle)
    ax1.add_patch(goal_circle)
    ax1.text(0, -0.8, 'START', ha='center', fontsize=10, fontweight='bold', color='#2ECC71')
    ax1.text(7, 7.8, 'GOAL', ha='center', fontsize=10, fontweight='bold', color='#F39C12')

    ax1.legend(loc='upper left', fontsize=10)

    # Right plot: Algorithm explanation
    ax2.set_title('A* Algorithm Formula', fontsize=14, fontweight='bold')
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    # Main formula box
    formula_box = FancyBboxPatch((0.5, 7), 9, 2.5, boxstyle="round,pad=0.1",
                                  edgecolor='#2C3E50', facecolor='#ECF0F1', linewidth=2)
    ax2.add_patch(formula_box)
    ax2.text(5, 8.5, r'$f(n) = g(n) + h(n)$', ha='center', va='center',
             fontsize=24, fontweight='bold', family='monospace')
    ax2.text(5, 7.5, 'Total Cost = Actual Cost + Heuristic Estimate',
             ha='center', va='center', fontsize=11, style='italic')

    # g(n) explanation
    g_box = FancyBboxPatch((0.5, 4.5), 4, 2, boxstyle="round,pad=0.1",
                            edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=2)
    ax2.add_patch(g_box)
    ax2.text(2.5, 5.8, r'$g(n)$', ha='center', fontsize=18, fontweight='bold', color='#2C3E50')
    ax2.text(2.5, 5.2, 'Actual cost from', ha='center', fontsize=10)
    ax2.text(2.5, 4.9, 'start to node n', ha='center', fontsize=10)

    # h(n) explanation
    h_box = FancyBboxPatch((5.5, 4.5), 4, 2, boxstyle="round,pad=0.1",
                            edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=2)
    ax2.add_patch(h_box)
    ax2.text(7.5, 5.8, r'$h(n)$', ha='center', fontsize=18, fontweight='bold', color='#2C3E50')
    ax2.text(7.5, 5.2, 'Heuristic estimate', ha='center', fontsize=10)
    ax2.text(7.5, 4.9, 'from n to goal', ha='center', fontsize=10)

    # Algorithm steps
    steps_text = [
        '1. Start with open list = {start node}',
        '2. Pick node with lowest f(n) from open list',
        '3. Move node to closed list',
        '4. Expand neighbors, calculate f(n)',
        '5. Add neighbors to open list',
        '6. Repeat until goal found'
    ]

    y_pos = 3.5
    ax2.text(5, y_pos + 0.5, 'Algorithm Steps:', ha='center',
             fontsize=12, fontweight='bold')
    for i, step in enumerate(steps_text):
        ax2.text(0.7, y_pos - i*0.5, step, ha='left', fontsize=9,
                 family='monospace', bbox=dict(boxstyle='round',
                 facecolor='#F8F9F9', alpha=0.8))

    plt.tight_layout()
    plt.savefig('Learnings/Module 13 - Classical AI Algorithms and Search/visualizations/01_astar_algorithm.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 01_astar_algorithm.png")
    plt.close()

if __name__ == "__main__":
    visualize_astar()

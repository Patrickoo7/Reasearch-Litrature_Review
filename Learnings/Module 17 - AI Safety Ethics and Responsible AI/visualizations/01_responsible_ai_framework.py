"""
Responsible AI Framework Visualization
Shows key principles and deployment lifecycle
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
import numpy as np

def visualize_responsible_ai():
    """Visualize Responsible AI framework and principles."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Responsible AI Framework', fontsize=18, fontweight='bold')

    # LEFT PLOT: Six Principles (hexagon)
    ax1.set_title('Six Pillars of Responsible AI', fontsize=14, fontweight='bold')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.axis('off')

    # Center circle
    center_circle = Circle((0, 0), 0.6, color='#2C3E50', zorder=10)
    ax1.add_patch(center_circle)
    ax1.text(0, 0, 'Responsible\nAI', ha='center', va='center', color='white',
             fontsize=12, fontweight='bold')

    # Six principles in hexagon
    principles = [
        ('Fairness', '#3498DB', 0, '• No discrimination\n• Equal treatment\n• Bias mitigation'),
        ('Transparency', '#9B59B6', 60, '• Explainability\n• Documentation\n• Open disclosure'),
        ('Privacy', '#16A085', 120, '• Data protection\n• Consent\n• Anonymization'),
        ('Accountability', '#E67E22', 180, '• Clear ownership\n• Audit trails\n• Governance'),
        ('Safety', '#E74C3C', 240, '• Robustness\n• Reliability\n• Fail-safe design'),
        ('Inclusiveness', '#27AE60', 300, '• Accessibility\n• Diverse teams\n• Broad benefit')
    ]

    radius = 2
    for principle, color, angle, details in principles:
        angle_rad = np.radians(angle)
        x = radius * np.cos(angle_rad)
        y = radius * np.sin(angle_rad)

        # Principle box
        box = FancyBboxPatch((x-0.55, y-0.35), 1.1, 0.7, boxstyle="round,pad=0.1",
                              edgecolor=color, facecolor=color, alpha=0.3, linewidth=3)
        ax1.add_patch(box)
        ax1.text(x, y, principle, ha='center', va='center', fontsize=10,
                 fontweight='bold', color=color)

        # Line to center
        ax1.plot([0, x*0.65], [0, y*0.65], color=color, linewidth=2, alpha=0.5)

        # Details in outer area
        detail_x = x * 1.35
        detail_y = y * 1.35
        ax1.text(detail_x, detail_y, details, ha='center', va='center',
                 fontsize=7, bbox=dict(boxstyle='round', facecolor='white',
                 alpha=0.8, edgecolor=color, linewidth=1))

    # RIGHT PLOT: Deployment Lifecycle
    ax2.set_title('Responsible AI Deployment Lifecycle', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 12)
    ax2.axis('off')

    # Phases
    phases = [
        ('1. Design', 10, '#3498DB', ['• Stakeholder analysis', '• Impact assessment', '• Risk evaluation']),
        ('2. Data', 8.5, '#9B59B6', ['• Data quality audit', '• Bias detection', '• Privacy review']),
        ('3. Development', 7, '#16A085', ['• Fairness metrics', '• Testing diverse cases', '• Documentation']),
        ('4. Validation', 5.5, '#E67E22', ['• External audit', '• Bias testing', '• Safety checks']),
        ('5. Deployment', 4, '#E74C3C', ['• Gradual rollout', '• Monitoring setup', '• Incident plan']),
        ('6. Monitoring', 2.5, '#27AE60', ['• Performance tracking', '• Bias monitoring', '• User feedback']),
        ('7. Governance', 1, '#8E44AD', ['• Regular audits', '• Updates', '• Decommissioning'])
    ]

    for phase, y_pos, color, checklist in phases:
        # Phase box
        phase_box = FancyBboxPatch((0.3, y_pos-0.3), 3, 0.6, boxstyle="round,pad=0.1",
                                    edgecolor=color, facecolor=color, alpha=0.7, linewidth=2)
        ax2.add_patch(phase_box)
        ax2.text(1.8, y_pos, phase, ha='center', va='center', fontsize=10,
                 fontweight='bold', color='white')

        # Checklist
        for i, item in enumerate(checklist):
            ax2.text(4, y_pos + 0.2 - i*0.25, item, ha='left', fontsize=7,
                     family='monospace')

        # Arrow to next phase
        if y_pos > 1:
            arrow = FancyArrowPatch((1.8, y_pos-0.4), (1.8, y_pos-1.1), arrowstyle='->',
                                     mutation_scale=15, linewidth=2.5, color=color)
            ax2.add_patch(arrow)

    # Feedback loop
    feedback_arrow = FancyArrowPatch((3.5, 1), (9, 10), arrowstyle='->', mutation_scale=20,
                                      linewidth=3, color='#F39C12', linestyle='--',
                                      connectionstyle="arc3,rad=.3")
    ax2.add_patch(feedback_arrow)
    ax2.text(7, 6, 'Continuous\nImprovement', ha='center', fontsize=9,
             fontweight='bold', color='#D68910', rotation=60)

    # Key metrics box
    metrics_box = FancyBboxPatch((0.3, 0.1), 9.4, 0.6, boxstyle="round,pad=0.1",
                                  edgecolor='#2C3E50', facecolor='#ECF0F1', linewidth=2)
    ax2.add_patch(metrics_box)
    ax2.text(5, 0.4, 'Key Metrics: Fairness (Demographic Parity, Equalized Odds) • Performance Across Groups • Privacy Compliance • Carbon Footprint',
             ha='center', fontsize=7)

    plt.tight_layout()
    plt.savefig('Learnings/Module 17 - AI Safety Ethics and Responsible AI/visualizations/01_responsible_ai_framework.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 01_responsible_ai_framework.png")
    plt.close()

if __name__ == "__main__":
    visualize_responsible_ai()

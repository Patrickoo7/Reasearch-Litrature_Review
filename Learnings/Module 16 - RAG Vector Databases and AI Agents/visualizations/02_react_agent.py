"""
ReAct Agent Pattern Visualization
Shows Reasoning + Acting loop
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def visualize_react():
    """Visualize ReAct (Reasoning + Acting) agent pattern."""

    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    fig.suptitle('ReAct: Reasoning + Acting Agent Pattern', fontsize=18, fontweight='bold')

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title boxes
    thought_title = FancyBboxPatch((1, 12.5), 5, 0.8, boxstyle="round,pad=0.1",
                                    edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=2)
    ax.add_patch(thought_title)
    ax.text(3.5, 12.9, 'THOUGHT (Reasoning)', ha='center', fontsize=12, fontweight='bold')

    action_title = FancyBboxPatch((10, 12.5), 5, 0.8, boxstyle="round,pad=0.1",
                                   edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=2)
    ax.add_patch(action_title)
    ax.text(12.5, 12.9, 'ACTION (Acting)', ha='center', fontsize=12, fontweight='bold')

    # Example: "What's the weather in Paris?"
    query_box = FancyBboxPatch((4, 11), 8, 1, boxstyle="round,pad=0.15",
                                edgecolor='#F39C12', facecolor='#FCF3CF', linewidth=3)
    ax.add_patch(query_box)
    ax.text(8, 11.6, 'User Query:', ha='center', fontsize=11, fontweight='bold')
    ax.text(8, 11.2, '"What is the weather in Paris and should I bring an umbrella?"',
            ha='center', fontsize=10, style='italic')

    # Cycle 1
    y_pos = 9
    step_num = 1

    # Thought 1
    thought1 = FancyBboxPatch((0.5, y_pos-0.7), 6.5, 1.4, boxstyle="round,pad=0.1",
                               edgecolor='#3498DB', facecolor='#EBF5FB', linewidth=2)
    ax.add_patch(thought1)
    ax.text(3.75, y_pos + 0.3, f'Thought {step_num}:', ha='center', fontsize=10, fontweight='bold',
            color='#2874A6')
    ax.text(3.75, y_pos - 0.1, 'I need to get current weather', ha='center', fontsize=9)
    ax.text(3.75, y_pos - 0.4, 'data for Paris. I will use the', ha='center', fontsize=9)
    ax.text(3.75, y_pos - 0.7, 'weather API tool.', ha='center', fontsize=9)

    # Arrow
    arrow = FancyArrowPatch((7, y_pos), (9, y_pos), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#16A085')
    ax.add_patch(arrow)

    # Action 1
    action1 = FancyBboxPatch((9, y_pos-0.7), 6.5, 1.4, boxstyle="round,pad=0.1",
                              edgecolor='#E74C3C', facecolor='#FADBD8', linewidth=2)
    ax.add_patch(action1)
    ax.text(12.25, y_pos + 0.3, f'Action {step_num}:', ha='center', fontsize=10, fontweight='bold',
            color='#C0392B')
    ax.text(12.25, y_pos - 0.2, 'get_weather(city="Paris")', ha='center', fontsize=9,
            family='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # Observation 1
    obs1 = FancyBboxPatch((9, y_pos-1.7), 6.5, 0.8, boxstyle="round,pad=0.05",
                           edgecolor='#27AE60', facecolor='#D5F4E6', linewidth=1.5)
    ax.add_patch(obs1)
    ax.text(12.25, y_pos - 1.3, 'Observation: 18°C, Rainy, 80% chance', ha='center',
            fontsize=8, family='monospace')

    # Cycle 2
    y_pos = 6
    step_num = 2

    # Arrow down
    arrow = FancyArrowPatch((12.25, 7.5), (3.75, 6.7), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#27AE60')
    ax.add_patch(arrow)

    # Thought 2
    thought2 = FancyBboxPatch((0.5, y_pos-0.7), 6.5, 1.4, boxstyle="round,pad=0.1",
                               edgecolor='#3498DB', facecolor='#EBF5FB', linewidth=2)
    ax.add_patch(thought2)
    ax.text(3.75, y_pos + 0.3, f'Thought {step_num}:', ha='center', fontsize=10, fontweight='bold',
            color='#2874A6')
    ax.text(3.75, y_pos - 0.1, 'Now I have the weather data.', ha='center', fontsize=9)
    ax.text(3.75, y_pos - 0.4, 'It\'s rainy with 80% chance.', ha='center', fontsize=9)
    ax.text(3.75, y_pos - 0.7, 'I can answer the question.', ha='center', fontsize=9)

    # Arrow
    arrow = FancyArrowPatch((7, y_pos), (9, y_pos), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#16A085')
    ax.add_patch(arrow)

    # Action 2 (Final answer)
    action2 = FancyBboxPatch((9, y_pos-0.7), 6.5, 1.4, boxstyle="round,pad=0.1",
                              edgecolor='#8E44AD', facecolor='#E8DAEF', linewidth=3)
    ax.add_patch(action2)
    ax.text(12.25, y_pos + 0.3, f'Action {step_num}: FINISH', ha='center', fontsize=10,
            fontweight='bold', color='#6C3483')
    ax.text(12.25, y_pos - 0.15, '"The weather in Paris is 18°C', ha='center', fontsize=9)
    ax.text(12.25, y_pos - 0.45, 'and rainy. Yes, you should', ha='center', fontsize=9)
    ax.text(12.25, y_pos - 0.75, 'bring an umbrella!"', ha='center', fontsize=9)

    # ReAct loop diagram (bottom)
    loop_y = 3
    ax.text(8, 4, 'ReAct Loop Structure', ha='center', fontsize=13, fontweight='bold')

    # Loop boxes
    loop_boxes = [
        ('Thought\n(LLM)', 2, loop_y, '#3498DB', '#D6EAF8'),
        ('Action\n(Tool)', 5.5, loop_y, '#E74C3C', '#FADBD8'),
        ('Observation\n(Result)', 9, loop_y, '#27AE60', '#D5F4E6'),
        ('Thought\n(LLM)', 12.5, loop_y, '#3498DB', '#D6EAF8')
    ]

    for i, (label, x, y, edge_col, face_col) in enumerate(loop_boxes):
        box = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, boxstyle="round,pad=0.1",
                              edgecolor=edge_col, facecolor=face_col, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', fontsize=9, fontweight='bold')

        if i < len(loop_boxes) - 1:
            next_x = loop_boxes[i+1][1]
            arrow = FancyArrowPatch((x+0.9, y), (next_x-0.9, y), arrowstyle='->',
                                     mutation_scale=15, linewidth=2, color='#16A085')
            ax.add_patch(arrow)

    # Return arrow
    return_arrow = FancyArrowPatch((13.4, loop_y-0.5), (13.4, loop_y-1.2), arrowstyle='->',
                                    mutation_scale=15, linewidth=2, color='#16A085',
                                    connectionstyle="arc3,rad=.5")
    ax.add_patch(return_arrow)
    return_arrow2 = FancyArrowPatch((13.4, loop_y-1.2), (2.9, loop_y-1.2), arrowstyle='->',
                                     mutation_scale=15, linewidth=2, color='#16A085')
    ax.add_patch(return_arrow2)
    return_arrow3 = FancyArrowPatch((2.9, loop_y-1.2), (2.9, loop_y-0.5), arrowstyle='->',
                                     mutation_scale=15, linewidth=2, color='#16A085')
    ax.add_patch(return_arrow3)
    ax.text(8, loop_y-1.5, 'Repeat until task complete', ha='center', fontsize=8,
            style='italic', color='#138D75')

    # Available tools box
    tools_box = FancyBboxPatch((0.5, 0.2), 7, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#34495E', facecolor='#F8F9F9', linewidth=2)
    ax.add_patch(tools_box)
    ax.text(4, 1.1, 'Available Tools:', ha='center', fontsize=10, fontweight='bold')
    ax.text(4, 0.75, '• get_weather(city) • search_web(query)', ha='center', fontsize=8,
            family='monospace')
    ax.text(4, 0.45, '• calculator(expression) • get_stock_price(symbol)', ha='center',
            fontsize=8, family='monospace')

    # Key benefits box
    benefits_box = FancyBboxPatch((8.5, 0.2), 7, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor='#27AE60', facecolor='#EAFAF1', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(12, 1.1, 'Benefits of ReAct:', ha='center', fontsize=10, fontweight='bold',
            color='#1E8449')
    ax.text(12, 0.75, '✓ Interpretable reasoning steps', ha='center', fontsize=9)
    ax.text(12, 0.45, '✓ Combines LLM reasoning with tool use', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('Learnings/Module 16 - RAG Vector Databases and AI Agents/visualizations/02_react_agent.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 02_react_agent.png")
    plt.close()

if __name__ == "__main__":
    visualize_react()

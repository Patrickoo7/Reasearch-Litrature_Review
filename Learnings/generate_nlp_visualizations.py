#!/usr/bin/env python3
"""
Generate comprehensive NLP visualizations for Module 7
Creates 16 detailed visualizations covering all NLP concepts
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
import numpy as np
import seaborn as sns
from pathlib import Path

# Create output directory
output_dir = Path("images/module7")
output_dir.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 70)
print("Generating NLP Visualizations for Module 7")
print("=" * 70)


def save_figure(filename, dpi=300):
    """Save figure with consistent settings"""
    filepath = output_dir / filename
    plt.tight_layout()
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"  ✓ Saved: {filename}")
    plt.close()


# 1. Text Preprocessing Pipeline
print("\n1. Text Preprocessing Pipeline")
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

steps = [
    ("Raw Text", "The AI's making progress!\nVisit https://example.com", 8.5),
    ("Lowercasing", "the ai's making progress!\nvisit https://example.com", 7.0),
    ("URL/Email Removal", "the ai's making progress!\nvisit", 5.5),
    ("Tokenization", "['the', 'ai', \"'s\", 'making', 'progress', 'visit']", 4.0),
    ("Stopword Removal", "['ai', 'making', 'progress', 'visit']", 2.5),
    ("Lemmatization", "['ai', 'make', 'progress', 'visit']", 1.0)
]

for i, (step, text, y) in enumerate(steps):
    # Step box
    box = FancyBboxPatch((0.2, y - 0.3), 3, 0.6,
                         boxstyle="round,pad=0.1",
                         edgecolor='steelblue',
                         facecolor='lightblue',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(1.7, y, step, fontsize=11, weight='bold', ha='center', va='center')

    # Result box
    result_box = FancyBboxPatch((4, y - 0.3), 5.8, 0.6,
                                boxstyle="round,pad=0.1",
                                edgecolor='darkgreen',
                                facecolor='lightgreen',
                                linewidth=1)
    ax.add_patch(result_box)
    ax.text(7, y, text, fontsize=9, ha='center', va='center', family='monospace')

    # Arrow
    if i < len(steps) - 1:
        arrow = FancyArrowPatch((1.7, y - 0.4), (1.7, y - 1.1),
                               arrowstyle='->', mutation_scale=20,
                               color='gray', linewidth=2)
        ax.add_patch(arrow)

ax.set_title('Text Preprocessing Pipeline', fontsize=16, weight='bold', pad=20)
save_figure('text_preprocessing_pipeline.png')


# 2. BPE Tokenization Algorithm
print("2. BPE Tokenization Merge Algorithm")
fig, ax = plt.subplots(figsize=(12, 8))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Initial vocabulary
ax.text(5, 9.5, 'BPE Tokenization: Iterative Merge Process',
        fontsize=14, weight='bold', ha='center')

iterations = [
    ("Initial", ["l", "o", "w", " ", "l", "o", "w", "e", "r"], 8.0),
    ("Merge 1: 'l'+'o' → 'lo'", ["lo", "w", " ", "lo", "w", "e", "r"], 6.5),
    ("Merge 2: 'lo'+'w' → 'low'", ["low", " ", "low", "e", "r"], 5.0),
    ("Merge 3: 'e'+'r' → 'er'", ["low", " ", "low", "er"], 3.5),
    ("Merge 4: 'low'+' ' → 'low '", ["low ", "low", "er"], 2.0)
]

for i, (desc, tokens, y) in enumerate(iterations):
    # Description
    ax.text(0.5, y, desc, fontsize=10, weight='bold', va='center')

    # Tokens
    x_start = 4.5
    for j, token in enumerate(tokens):
        color = 'lightcoral' if i > 0 and j < 2 else 'lightblue'
        box = FancyBboxPatch((x_start + j * 0.5, y - 0.2), 0.45, 0.4,
                             boxstyle="round,pad=0.05",
                             edgecolor='black',
                             facecolor=color,
                             linewidth=1.5)
        ax.add_patch(box)
        ax.text(x_start + j * 0.5 + 0.225, y, token,
               fontsize=9, ha='center', va='center', family='monospace')

    # Arrow to next step
    if i < len(iterations) - 1:
        arrow = FancyArrowPatch((2.5, y - 0.3), (2.5, y - 1.2),
                               arrowstyle='->', mutation_scale=15,
                               color='gray', linewidth=2)
        ax.add_patch(arrow)

# Frequency table
ax.text(0.5, 0.5, 'Most Frequent Pair → Merge', fontsize=9,
       style='italic', ha='left')

save_figure('bpe_tokenization_algorithm.png')


# 3. Tokenization Comparison
print("3. Tokenization Methods Comparison")
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

text = "unhappiness"
methods = [
    ("Word-Level", ["unhappiness"], "Simple, large vocab", 7.5),
    ("Character-Level", ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"],
     "Small vocab, long sequences", 5.5),
    ("Subword (BPE)", ["un", "happiness"], "Balanced, handles OOV", 3.5),
    ("WordPiece", ["un", "##happiness"], "BERT-style, ## for continuations", 1.5)
]

ax.text(5, 9.2, f'Tokenizing: "{text}"', fontsize=14, weight='bold', ha='center')

for method, tokens, desc, y in methods:
    # Method name
    box = FancyBboxPatch((0.3, y - 0.25), 2, 0.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='navy',
                         facecolor='lightsteelblue',
                         linewidth=2)
    ax.add_patch(box)
    ax.text(1.3, y, method, fontsize=11, weight='bold', ha='center', va='center')

    # Tokens
    x_start = 3
    for i, token in enumerate(tokens):
        token_box = FancyBboxPatch((x_start + i * 0.8, y - 0.2), 0.75, 0.4,
                                   boxstyle="round,pad=0.05",
                                   edgecolor='darkgreen',
                                   facecolor='lightgreen',
                                   linewidth=1)
        ax.add_patch(token_box)
        ax.text(x_start + i * 0.8 + 0.375, y, token,
               fontsize=9, ha='center', va='center', family='monospace')

    # Description
    ax.text(8.5, y, desc, fontsize=9, style='italic', va='center')

save_figure('tokenization_comparison.png')


# 4. Word Embedding Space (t-SNE)
print("4. Word Embedding Space Visualization")
np.random.seed(42)

# Simulate word embeddings
words_categories = {
    'Animals': ['dog', 'cat', 'bird', 'fish', 'lion'],
    'Fruits': ['apple', 'banana', 'orange', 'grape', 'mango'],
    'Countries': ['USA', 'China', 'India', 'Brazil', 'France'],
    'Sports': ['soccer', 'tennis', 'basketball', 'cricket', 'swimming']
}

fig, ax = plt.subplots(figsize=(12, 10))

colors = ['red', 'green', 'blue', 'orange']
for idx, (category, words) in enumerate(words_categories.items()):
    # Generate clustered points
    center = np.random.randn(2) * 3
    points = center + np.random.randn(len(words), 2) * 0.5

    ax.scatter(points[:, 0], points[:, 1],
              s=200, alpha=0.6, c=colors[idx], label=category)

    for i, word in enumerate(words):
        ax.annotate(word, (points[i, 0], points[i, 1]),
                   fontsize=10, weight='bold',
                   ha='center', va='center')

ax.set_title('Word Embedding Space (t-SNE Projection)',
            fontsize=14, weight='bold', pad=20)
ax.set_xlabel('Dimension 1', fontsize=12)
ax.set_ylabel('Dimension 2', fontsize=12)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

save_figure('word_embedding_space.png')


# 5. Self-Attention Mechanism Heatmap
print("5. Self-Attention Mechanism Heatmap")
sentence = ["The", "cat", "sat", "on", "the", "mat"]

# Simulate attention scores
np.random.seed(42)
attention = np.random.rand(len(sentence), len(sentence))
# Make it more diagonal (words attend to themselves)
attention = attention + np.eye(len(sentence)) * 2
attention = attention / attention.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(attention, cmap='YlOrRd', aspect='auto')

# Set ticks
ax.set_xticks(np.arange(len(sentence)))
ax.set_yticks(np.arange(len(sentence)))
ax.set_xticklabels(sentence, fontsize=11)
ax.set_yticklabels(sentence, fontsize=11)

# Rotate x labels
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Add values
for i in range(len(sentence)):
    for j in range(len(sentence)):
        text = ax.text(j, i, f'{attention[i, j]:.2f}',
                      ha="center", va="center", color="black", fontsize=9)

ax.set_title('Self-Attention Scores: Each word attends to all words',
            fontsize=13, weight='bold', pad=15)
ax.set_xlabel('Keys', fontsize=12)
ax.set_ylabel('Queries', fontsize=12)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Attention Weight', fontsize=11)

save_figure('self_attention_heatmap.png')


# 6. Transformer Architecture
print("6. Transformer Encoder Architecture")
fig, ax = plt.subplots(figsize=(10, 14))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

# Input
input_box = FancyBboxPatch((3, 0.5), 4, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black',
                          facecolor='lightgray',
                          linewidth=2)
ax.add_patch(input_box)
ax.text(5, 0.9, 'Input Embeddings + Positional Encoding',
       fontsize=10, weight='bold', ha='center', va='center')

# Encoder blocks (stack of 3 for visualization)
y_positions = [2.5, 6.0, 9.5]

for block_num, y_base in enumerate(y_positions, 1):
    # Multi-Head Attention
    mha_box = FancyBboxPatch((2, y_base), 6, 1.0,
                            boxstyle="round,pad=0.1",
                            edgecolor='steelblue',
                            facecolor='lightblue',
                            linewidth=2)
    ax.add_patch(mha_box)
    ax.text(5, y_base + 0.5, 'Multi-Head Attention',
           fontsize=10, weight='bold', ha='center', va='center')

    # Add & Norm
    norm1_box = FancyBboxPatch((2.5, y_base + 1.3), 5, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor='green',
                              facecolor='lightgreen',
                              linewidth=1.5)
    ax.add_patch(norm1_box)
    ax.text(5, y_base + 1.5, 'Add & Normalize',
           fontsize=9, ha='center', va='center')

    # Feed Forward
    ff_box = FancyBboxPatch((2, y_base + 2.0), 6, 1.0,
                           boxstyle="round,pad=0.1",
                           edgecolor='coral',
                           facecolor='lightsalmon',
                           linewidth=2)
    ax.add_patch(ff_box)
    ax.text(5, y_base + 2.5, 'Feed Forward Network',
           fontsize=10, weight='bold', ha='center', va='center')

    # Add & Norm
    norm2_box = FancyBboxPatch((2.5, y_base + 3.3), 5, 0.4,
                              boxstyle="round,pad=0.05",
                              edgecolor='green',
                              facecolor='lightgreen',
                              linewidth=1.5)
    ax.add_patch(norm2_box)
    ax.text(5, y_base + 3.5, 'Add & Normalize',
           fontsize=9, ha='center', va='center')

    # Block label
    ax.text(0.5, y_base + 1.75, f'Block {block_num}',
           fontsize=11, weight='bold', rotation=90, va='center')

# Output
output_box = FancyBboxPatch((3, 13), 4, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='black',
                           facecolor='gold',
                           linewidth=2)
ax.add_patch(output_box)
ax.text(5, 13.4, 'Output Representations',
       fontsize=10, weight='bold', ha='center', va='center')

# Arrows
arrow_positions = [1.5, 5.0, 8.5, 12.5]
for y in arrow_positions:
    arrow = FancyArrowPatch((5, y), (5, y + 0.8),
                           arrowstyle='->', mutation_scale=20,
                           color='black', linewidth=2)
    ax.add_patch(arrow)

ax.set_title('Transformer Encoder Architecture', fontsize=14, weight='bold', pad=20)
save_figure('transformer_architecture.png')


# 7. Dependency Parsing Tree
print("7. Dependency Parsing Tree")
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)

# Sentence
sentence = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
pos_tags = ["DET", "ADJ", "ADJ", "NOUN", "VERB", "ADP", "DET", "ADJ", "NOUN"]

# Draw words
word_positions = {}
for i, (word, pos) in enumerate(zip(sentence, pos_tags)):
    x = 1.5 + i * 1.3
    y = 1.5

    word_positions[i] = (x, y)

    # POS tag
    ax.text(x, y + 0.8, pos, fontsize=9, ha='center',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

    # Word
    ax.text(x, y, word, fontsize=11, weight='bold', ha='center',
           bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue',
                    edgecolor='navy', linewidth=1.5))

# Dependencies (fox <- root, fox <- The, fox <- quick, fox <- brown, etc.)
dependencies = [
    (4, 3, "nsubj", "purple"),    # fox -> jumps
    (3, 0, "det", "green"),       # The -> fox
    (3, 1, "amod", "orange"),     # quick -> fox
    (3, 2, "amod", "orange"),     # brown -> fox
    (4, 5, "prep", "blue"),       # jumps -> over
    (5, 8, "pobj", "red"),        # over -> dog
    (8, 6, "det", "green"),       # the -> dog
    (8, 7, "amod", "orange"),     # lazy -> dog
]

for head, dep, label, color in dependencies:
    x1, y1 = word_positions[dep]
    x2, y2 = word_positions[head]

    # Arc
    arc_height = 0.8 + abs(head - dep) * 0.3
    arrow = FancyArrowPatch((x1, y1 + 0.5), (x2, y2 + 0.5),
                           connectionstyle=f"arc3,rad=.{int(arc_height*10)}",
                           arrowstyle='->', mutation_scale=15,
                           color=color, linewidth=2, alpha=0.7)
    ax.add_patch(arrow)

    # Label
    mid_x = (x1 + x2) / 2
    mid_y = y1 + arc_height + 0.5
    ax.text(mid_x, mid_y, label, fontsize=8, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                    edgecolor=color, linewidth=1))

ax.set_title('Dependency Parsing: Syntactic Structure',
            fontsize=14, weight='bold', pad=20)

# Legend
legend_elements = [
    mpatches.Patch(color='purple', label='nsubj (nominal subject)'),
    mpatches.Patch(color='green', label='det (determiner)'),
    mpatches.Patch(color='orange', label='amod (adjectival modifier)'),
    mpatches.Patch(color='blue', label='prep (prepositional modifier)'),
    mpatches.Patch(color='red', label='pobj (object of preposition)')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

save_figure('dependency_parsing_tree.png')


# 8. BERT Layer-wise Linguistic Knowledge
print("8. BERT Layer-wise Linguistic Knowledge")
layers = list(range(1, 13))
linguistic_features = {
    'Surface Features\n(POS, Word Order)': [0.9, 0.85, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05],
    'Syntactic Features\n(Dependency, Constituency)': [0.3, 0.5, 0.8, 0.9, 0.85, 0.7, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1],
    'Semantic Features\n(Entities, Relations)': [0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.85, 0.7, 0.5, 0.4, 0.3],
    'Task-Specific Features': [0.05, 0.1, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 0.9, 0.95, 0.95]
}

fig, ax = plt.subplots(figsize=(12, 8))

for feature, values in linguistic_features.items():
    ax.plot(layers, values, marker='o', linewidth=2.5,
           markersize=8, label=feature, alpha=0.8)

ax.set_xlabel('BERT Layer', fontsize=12, weight='bold')
ax.set_ylabel('Feature Importance', fontsize=12, weight='bold')
ax.set_title('BERT Layers Capture Different Linguistic Information',
            fontsize=14, weight='bold', pad=20)
ax.set_xticks(layers)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=10, loc='right')
ax.grid(True, alpha=0.3, linestyle='--')

# Add annotations
ax.annotate('Surface\nFeatures', xy=(2, 0.85), xytext=(3, 0.95),
           arrowprops=dict(arrowstyle='->', color='gray'),
           fontsize=9, ha='center')
ax.annotate('Syntactic\nFeatures', xy=(5, 0.85), xytext=(6, 0.95),
           arrowprops=dict(arrowstyle='->', color='gray'),
           fontsize=9, ha='center')
ax.annotate('Semantic\nFeatures', xy=(7, 0.9), xytext=(8, 0.98),
           arrowprops=dict(arrowstyle='->', color='gray'),
           fontsize=9, ha='center')

save_figure('bert_layer_linguistic_knowledge.png')


# 9. Chain-of-Thought Reasoning Flow
print("9. Chain-of-Thought Reasoning Flow")
fig, ax = plt.subplots(figsize=(12, 10))
ax.axis('off')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)

# Problem
problem_box = FancyBboxPatch((1, 8.5), 8, 1,
                            boxstyle="round,pad=0.15",
                            edgecolor='darkred',
                            facecolor='mistyrose',
                            linewidth=2.5)
ax.add_patch(problem_box)
ax.text(5, 9.2, 'Problem: Roger has 5 tennis balls. He buys 2 cans,',
       fontsize=10, ha='center', weight='bold')
ax.text(5, 8.8, 'each with 3 balls. How many balls does he have now?',
       fontsize=10, ha='center', weight='bold')

# Reasoning steps
steps = [
    ("Step 1: Identify initial count", "Roger starts with 5 tennis balls", 7.0),
    ("Step 2: Calculate balls per purchase", "Each can has 3 balls", 5.5),
    ("Step 3: Calculate total new balls", "2 cans × 3 balls = 6 balls", 4.0),
    ("Step 4: Add to initial count", "5 + 6 = 11 balls", 2.5),
]

for i, (step, reasoning, y) in enumerate(steps):
    # Step label
    step_circle = Circle((1.5, y), 0.3, color='steelblue', zorder=10)
    ax.add_patch(step_circle)
    ax.text(1.5, y, str(i+1), fontsize=12, weight='bold',
           ha='center', va='center', color='white', zorder=11)

    # Reasoning box
    reason_box = FancyBboxPatch((2.5, y - 0.35), 6.5, 0.7,
                               boxstyle="round,pad=0.1",
                               edgecolor='navy',
                               facecolor='lightblue',
                               linewidth=1.5)
    ax.add_patch(reason_box)
    ax.text(3, y + 0.15, step, fontsize=9, weight='bold', va='center')
    ax.text(3, y - 0.15, reasoning, fontsize=9, va='center', style='italic')

    # Arrow
    if i < len(steps) - 1:
        arrow = FancyArrowPatch((5, y - 0.5), (5, y - 1.0),
                               arrowstyle='->', mutation_scale=20,
                               color='gray', linewidth=2.5)
        ax.add_patch(arrow)

# Answer
answer_box = FancyBboxPatch((2, 0.8), 6, 0.8,
                           boxstyle="round,pad=0.15",
                           edgecolor='darkgreen',
                           facecolor='lightgreen',
                           linewidth=2.5)
ax.add_patch(answer_box)
ax.text(5, 1.2, 'Final Answer: Roger has 11 tennis balls',
       fontsize=11, weight='bold', ha='center', va='center')

ax.set_title('Chain-of-Thought: Step-by-Step Reasoning',
            fontsize=14, weight='bold', pad=20)
save_figure('chain_of_thought_reasoning.png')


# 10. LoRA Architecture
print("10. LoRA Architecture Diagram")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# Title
ax.text(7, 9.5, 'LoRA: Low-Rank Adaptation of Large Language Models',
       fontsize=14, weight='bold', ha='center')

# Left: Full Fine-tuning
ax.text(3.5, 8.5, 'Full Fine-tuning', fontsize=12, weight='bold', ha='center')

# Pre-trained weight
w_box = FancyBboxPatch((2, 6.5), 3, 1.5,
                      boxstyle="round,pad=0.1",
                      edgecolor='red',
                      facecolor='lightcoral',
                      linewidth=2)
ax.add_patch(w_box)
ax.text(3.5, 7.25, 'W (Pre-trained)', fontsize=10, weight='bold', ha='center')
ax.text(3.5, 6.9, '768 × 768', fontsize=9, ha='center', style='italic')

# Update
ax.text(3.5, 5.8, '↓ Update ALL', fontsize=9, weight='bold', ha='center', color='red')

# Updated weight
w_updated = FancyBboxPatch((2, 4.5), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='darkred',
                          facecolor='salmon',
                          linewidth=2)
ax.add_patch(w_updated)
ax.text(3.5, 5.25, "W' (Fine-tuned)", fontsize=10, weight='bold', ha='center')
ax.text(3.5, 4.9, '768 × 768', fontsize=9, ha='center', style='italic')

# Right: LoRA
ax.text(10.5, 8.5, 'LoRA', fontsize=12, weight='bold', ha='center')

# Pre-trained weight (frozen)
w_frozen = FancyBboxPatch((9, 6.5), 3, 1.5,
                         boxstyle="round,pad=0.1",
                         edgecolor='blue',
                         facecolor='lightblue',
                         linewidth=2)
ax.add_patch(w_frozen)
ax.text(10.5, 7.4, 'W (Frozen)', fontsize=10, weight='bold', ha='center')
ax.text(10.5, 7.0, '768 × 768', fontsize=9, ha='center', style='italic')
ax.text(10.5, 6.6, '❄️ No updates', fontsize=8, ha='center')

# LoRA matrices A and B
a_box = FancyBboxPatch((8.5, 4.5), 1.5, 1,
                      boxstyle="round,pad=0.1",
                      edgecolor='green',
                      facecolor='lightgreen',
                      linewidth=2)
ax.add_patch(a_box)
ax.text(9.25, 5.1, 'A', fontsize=10, weight='bold', ha='center')
ax.text(9.25, 4.8, '768 × 8', fontsize=8, ha='center', style='italic')

b_box = FancyBboxPatch((10.5, 4.5), 1.5, 1,
                      boxstyle="round,pad=0.1",
                      edgecolor='green',
                      facecolor='lightgreen',
                      linewidth=2)
ax.add_patch(b_box)
ax.text(11.25, 5.1, 'B', fontsize=10, weight='bold', ha='center')
ax.text(11.25, 4.8, '8 × 768', fontsize=8, ha='center', style='italic')

# Arrows and operations
ax.text(10.5, 5.8, '↓', fontsize=16, ha='center', color='green')
ax.text(10.5, 5.9, 'Train only', fontsize=8, weight='bold', ha='center', color='green')

ax.text(10.1, 4.8, '×', fontsize=14, weight='bold', ha='center')

# Result
ax.text(10.5, 3.9, 'W + BA', fontsize=11, weight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

# Stats
stats_text = '''LoRA Benefits:
• Trainable params: 0.1% of full model
• Memory: 3x less
• Same performance as full fine-tuning
• Multiple adapters can be swapped'''

ax.text(7, 2.5, stats_text, fontsize=9, ha='center',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                edgecolor='gold', linewidth=2))

save_figure('lora_architecture.png')


# 11. Memory Comparison: Full vs LoRA vs QLoRA
print("11. Memory Comparison: Full vs LoRA vs QLoRA")
methods = ['Full Fine-tuning\n(FP32)', 'Full Fine-tuning\n(FP16)',
          'LoRA\n(FP16)', 'QLoRA\n(4-bit)']
memory_gb = [28, 14, 5, 3.5]
trainable_params = [7000, 7000, 8, 8]  # in millions

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Memory usage
colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
bars1 = ax1.bar(methods, memory_gb, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, mem in zip(bars1, memory_gb):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{mem} GB', ha='center', va='bottom', fontsize=10, weight='bold')

ax1.set_ylabel('GPU Memory (GB)', fontsize=12, weight='bold')
ax1.set_title('Memory Usage for 7B Model', fontsize=13, weight='bold', pad=15)
ax1.set_ylim(0, 32)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Trainable parameters
bars2 = ax2.bar(methods, trainable_params, color=colors, alpha=0.7,
               edgecolor='black', linewidth=1.5)

# Add value labels
for bar, params in zip(bars2, trainable_params):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{params}M', ha='center', va='bottom', fontsize=10, weight='bold')

ax2.set_ylabel('Trainable Parameters (Millions)', fontsize=12, weight='bold')
ax2.set_title('Trainable Parameters', fontsize=13, weight='bold', pad=15)
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('QLoRA: 8x less memory, 99.9% fewer trainable parameters!',
            fontsize=14, weight='bold', y=1.02)

save_figure('memory_comparison_lora_qlora.png')


# 12. RAG Pipeline Architecture
print("12. RAG Pipeline Architecture")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# User query
query_box = FancyBboxPatch((1, 8.5), 3, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='purple',
                          facecolor='lavender',
                          linewidth=2)
ax.add_patch(query_box)
ax.text(2.5, 8.9, 'User Query', fontsize=10, weight='bold', ha='center')

# Step 1: Embedding
arrow1 = FancyArrowPatch((4.2, 8.9), (5.8, 8.9),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow1)
ax.text(5, 9.3, '1. Embed', fontsize=9, weight='bold', ha='center')

embed_box = FancyBboxPatch((6, 8.5), 2, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='blue',
                          facecolor='lightblue',
                          linewidth=2)
ax.add_patch(embed_box)
ax.text(7, 8.9, 'Embedding\nModel', fontsize=9, weight='bold', ha='center')

# Step 2: Vector search
arrow2 = FancyArrowPatch((7, 8.3), (7, 7.2),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow2)
ax.text(7.7, 7.8, '2. Search', fontsize=9, weight='bold', ha='left')

vector_db = FancyBboxPatch((5.5, 5.5), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          edgecolor='green',
                          facecolor='lightgreen',
                          linewidth=2)
ax.add_patch(vector_db)
ax.text(7, 6.7, 'Vector Database', fontsize=10, weight='bold', ha='center')
ax.text(7, 6.3, '(FAISS/Pinecone)', fontsize=9, ha='center', style='italic')
ax.text(7, 5.9, '1M+ documents', fontsize=8, ha='center')

# Retrieved docs
arrow3 = FancyArrowPatch((7, 5.3), (7, 4.2),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow3)
ax.text(7.7, 4.8, '3. Retrieve', fontsize=9, weight='bold', ha='left')

docs_box = FancyBboxPatch((5, 3), 4, 1,
                         boxstyle="round,pad=0.1",
                         edgecolor='orange',
                         facecolor='lightyellow',
                         linewidth=2)
ax.add_patch(docs_box)
ax.text(7, 3.7, 'Top-k Documents', fontsize=10, weight='bold', ha='center')
ax.text(7, 3.3, '(k=3-5)', fontsize=9, ha='center', style='italic')

# Step 4: Augment prompt
arrow4 = FancyArrowPatch((9.2, 3.5), (10.8, 3.5),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow4)
ax.text(10, 3.9, '4. Augment', fontsize=9, weight='bold', ha='center')

prompt_box = FancyBboxPatch((11, 2.5), 2.5, 2,
                           boxstyle="round,pad=0.1",
                           edgecolor='red',
                           facecolor='mistyrose',
                           linewidth=2)
ax.add_patch(prompt_box)
ax.text(12.25, 3.8, 'Context +', fontsize=9, weight='bold', ha='center')
ax.text(12.25, 3.4, 'Query →', fontsize=9, weight='bold', ha='center')
ax.text(12.25, 3.0, 'Prompt', fontsize=9, weight='bold', ha='center')

# Step 5: Generate
arrow5 = FancyArrowPatch((12.25, 2.3), (12.25, 1.2),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow5)
ax.text(12.9, 1.8, '5. Generate', fontsize=9, weight='bold', ha='left')

llm_box = FancyBboxPatch((10.75, 0.2), 3, 0.8,
                        boxstyle="round,pad=0.1",
                        edgecolor='darkblue',
                        facecolor='lightsteelblue',
                        linewidth=2)
ax.add_patch(llm_box)
ax.text(12.25, 0.6, 'LLM (GPT-4)', fontsize=10, weight='bold', ha='center')

# Answer
arrow6 = FancyArrowPatch((10.5, 0.6), (8.5, 0.6),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow6)

answer_box = FancyBboxPatch((5, 0.2), 3, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='darkgreen',
                           facecolor='lightgreen',
                           linewidth=2.5)
ax.add_patch(answer_box)
ax.text(6.5, 0.6, 'Generated Answer', fontsize=10, weight='bold', ha='center')

# Documents storage (left side)
doc_storage = FancyBboxPatch((0.5, 5.5), 3.5, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='gray',
                            facecolor='lightgray',
                            linewidth=1.5)
ax.add_patch(doc_storage)
ax.text(2.25, 6.7, 'Document Store', fontsize=9, weight='bold', ha='center')
ax.text(2.25, 6.3, '📄 PDFs', fontsize=8, ha='center')
ax.text(2.25, 6.0, '📄 Web Pages', fontsize=8, ha='center')
ax.text(2.25, 5.7, '📄 Databases', fontsize=8, ha='center')

# Arrow from docs to vector DB
arrow_docs = FancyArrowPatch((4.2, 6.2), (5.3, 6.2),
                            arrowstyle='->', mutation_scale=15,
                            color='gray', linewidth=1.5, linestyle='dashed')
ax.add_patch(arrow_docs)
ax.text(4.75, 6.5, 'Index', fontsize=8, ha='center', style='italic')

ax.set_title('RAG (Retrieval-Augmented Generation) Pipeline',
            fontsize=14, weight='bold', pad=20)

save_figure('rag_pipeline_architecture.png')


# 13. Vector Search Visualization
print("13. Vector Search Visualization")
fig, ax = plt.subplots(figsize=(12, 10))

# Generate document embeddings (2D projection)
np.random.seed(42)
n_docs = 50
doc_embeddings = np.random.randn(n_docs, 2) * 2

# Query embedding
query_emb = np.array([2.5, 1.5])

# Calculate distances
distances = np.sqrt(((doc_embeddings - query_emb) ** 2).sum(axis=1))
top_k_indices = distances.argsort()[:5]

# Plot all documents
ax.scatter(doc_embeddings[:, 0], doc_embeddings[:, 1],
          s=100, c='lightblue', alpha=0.6,
          edgecolors='navy', linewidth=1, label='Documents')

# Highlight top-k
ax.scatter(doc_embeddings[top_k_indices, 0],
          doc_embeddings[top_k_indices, 1],
          s=200, c='lightgreen', alpha=0.8,
          edgecolors='darkgreen', linewidth=2,
          label='Top-5 Retrieved', zorder=5)

# Plot query
ax.scatter(query_emb[0], query_emb[1],
          s=300, c='red', marker='*',
          edgecolors='darkred', linewidth=2,
          label='Query', zorder=10)

# Draw lines to top-k
for idx in top_k_indices:
    ax.plot([query_emb[0], doc_embeddings[idx, 0]],
           [query_emb[1], doc_embeddings[idx, 1]],
           'g--', alpha=0.4, linewidth=1.5)

# Add distance labels for top-k
for i, idx in enumerate(top_k_indices[:3]):
    dist = distances[idx]
    ax.text(doc_embeddings[idx, 0], doc_embeddings[idx, 1] - 0.4,
           f'd={dist:.2f}', fontsize=8, ha='center',
           bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Embedding Dimension 1', fontsize=12, weight='bold')
ax.set_ylabel('Embedding Dimension 2', fontsize=12, weight='bold')
ax.set_title('Vector Similarity Search: Finding Nearest Neighbors',
            fontsize=14, weight='bold', pad=20)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--')

# Add info box
info_text = '''Vector Search:
1. Embed query
2. Compute similarity (cosine/L2)
3. Return top-k closest docs'''

ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                edgecolor='gold', linewidth=2))

save_figure('vector_search_visualization.png')


# 14. Decoding Strategies Comparison
print("14. Decoding Strategies Comparison")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

strategies = [
    ('Greedy', 0.0),
    ('Temperature=0.7', 0.7),
    ('Top-k=50', 1.0),
    ('Top-p=0.9', 1.0)
]

for idx, (ax, (strategy, temp)) in enumerate(zip(axes.flat, strategies)):
    # Simulate token probabilities
    np.random.seed(42 + idx)
    tokens = ['great', 'good', 'excellent', 'fine', 'nice', 'okay',
             'decent', 'fair', 'amazing', 'superb']

    if 'Greedy' in strategy:
        probs = np.array([0.5, 0.2, 0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.005])
    elif 'Temperature' in strategy:
        probs = np.array([0.3, 0.2, 0.15, 0.1, 0.08, 0.06, 0.05, 0.03, 0.02, 0.01])
    elif 'Top-k' in strategy:
        probs = np.array([0.25, 0.2, 0.15, 0.1, 0.08, 0.05, 0.05, 0.04, 0.04, 0.04])
    else:  # Top-p
        probs = np.array([0.35, 0.25, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.005, 0.005])

    # Highlight selected tokens
    colors = ['red' if i == 0 else 'lightblue' for i in range(len(tokens))]
    if 'Top-k' in strategy or 'Top-p' in strategy:
        colors = ['lightcoral' if i < 5 else 'lightgray' for i in range(len(tokens))]

    bars = ax.barh(tokens, probs, color=colors, edgecolor='black', linewidth=1)

    ax.set_xlabel('Probability', fontsize=10, weight='bold')
    ax.set_title(strategy, fontsize=12, weight='bold', pad=10)
    ax.set_xlim(0, 0.6)

    # Add value labels
    for bar, prob in zip(bars, probs):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{prob:.3f}', ha='left', va='center', fontsize=8)

    # Add strategy description
    if 'Greedy' in strategy:
        ax.text(0.98, 0.02, 'Always pick highest\nprobability token',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
    elif 'Temperature' in strategy:
        ax.text(0.98, 0.02, 'Softens probabilities\nfor more diversity',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
    elif 'Top-k' in strategy:
        ax.text(0.98, 0.02, 'Sample from\ntop-k tokens only',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))
    else:
        ax.text(0.98, 0.02, 'Sample from tokens\nwith cumulative p≥0.9',
               transform=ax.transAxes, fontsize=9,
               verticalalignment='bottom', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

plt.suptitle('Text Generation Decoding Strategies', fontsize=14, weight='bold', y=1.00)
save_figure('decoding_strategies_comparison.png')


# 15. NLP Production Architecture
print("15. NLP Production Architecture")
fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)

# Title
ax.text(7, 9.5, 'Production NLP System Architecture',
       fontsize=14, weight='bold', ha='center')

# Client layer
client_box = FancyBboxPatch((5.5, 8), 3, 0.8,
                           boxstyle="round,pad=0.1",
                           edgecolor='purple',
                           facecolor='lavender',
                           linewidth=2)
ax.add_patch(client_box)
ax.text(7, 8.4, 'Client App / Web UI', fontsize=10, weight='bold', ha='center')

# API Gateway
arrow1 = FancyArrowPatch((7, 7.8), (7, 7.2),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow1)

gateway_box = FancyBboxPatch((5.5, 6.3), 3, 0.8,
                            boxstyle="round,pad=0.1",
                            edgecolor='blue',
                            facecolor='lightblue',
                            linewidth=2)
ax.add_patch(gateway_box)
ax.text(7, 6.7, 'API Gateway (FastAPI)', fontsize=10, weight='bold', ha='center')

# Cache layer
cache_box = FancyBboxPatch((10, 6.3), 2, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='orange',
                          facecolor='lightyellow',
                          linewidth=1.5)
ax.add_patch(cache_box)
ax.text(11, 6.7, 'Redis Cache', fontsize=9, weight='bold', ha='center')

# Arrow to cache
arrow_cache = FancyArrowPatch((8.7, 6.7), (9.8, 6.7),
                             arrowstyle='<->', mutation_scale=15,
                             color='orange', linewidth=1.5, linestyle='dashed')
ax.add_patch(arrow_cache)

# Model serving layer
arrow2 = FancyArrowPatch((7, 6.1), (7, 5.5),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow2)

# Multiple model instances
for i, x_pos in enumerate([3, 6, 9]):
    model_box = FancyBboxPatch((x_pos - 0.8, 4.2), 1.6, 1.0,
                               boxstyle="round,pad=0.1",
                               edgecolor='green',
                               facecolor='lightgreen',
                               linewidth=2)
    ax.add_patch(model_box)
    ax.text(x_pos, 4.9, f'Model {i+1}', fontsize=9, weight='bold', ha='center')
    ax.text(x_pos, 4.5, '(ONNX)', fontsize=8, ha='center', style='italic')

ax.text(7, 5.9, 'Load Balancer', fontsize=9, weight='bold', ha='center',
       bbox=dict(boxstyle='round,pad=0.2', facecolor='wheat', alpha=0.7))

# Monitoring
monitor_box = FancyBboxPatch((0.5, 4.2), 1.8, 1.0,
                            boxstyle="round,pad=0.1",
                            edgecolor='red',
                            facecolor='mistyrose',
                            linewidth=1.5)
ax.add_patch(monitor_box)
ax.text(1.4, 4.9, 'Prometheus', fontsize=9, weight='bold', ha='center')
ax.text(1.4, 4.5, 'Monitoring', fontsize=8, ha='center')

# Arrow to monitoring
arrow_mon = FancyArrowPatch((2.4, 4.7), (3.0, 4.7),
                           arrowstyle='->', mutation_scale=15,
                           color='red', linewidth=1.5, linestyle='dotted')
ax.add_patch(arrow_mon)

# Database layer
arrow3 = FancyArrowPatch((7, 4.0), (7, 3.4),
                        arrowstyle='->', mutation_scale=20,
                        color='black', linewidth=2)
ax.add_patch(arrow3)

# Vector DB
vector_db_box = FancyBboxPatch((3.5, 2.2), 2.5, 1.0,
                               boxstyle="round,pad=0.1",
                               edgecolor='teal',
                               facecolor='lightcyan',
                               linewidth=2)
ax.add_patch(vector_db_box)
ax.text(4.75, 2.9, 'Vector DB', fontsize=9, weight='bold', ha='center')
ax.text(4.75, 2.5, '(Pinecone)', fontsize=8, ha='center', style='italic')

# Regular DB
db_box = FancyBboxPatch((8, 2.2), 2.5, 1.0,
                       boxstyle="round,pad=0.1",
                       edgecolor='darkblue',
                       facecolor='lightsteelblue',
                       linewidth=2)
ax.add_patch(db_box)
ax.text(9.25, 2.9, 'PostgreSQL', fontsize=9, weight='bold', ha='center')
ax.text(9.25, 2.5, '(Metadata)', fontsize=8, ha='center', style='italic')

# Logging
log_box = FancyBboxPatch((11, 2.2), 2, 1.0,
                        boxstyle="round,pad=0.1",
                        edgecolor='brown',
                        facecolor='wheat',
                        linewidth=1.5)
ax.add_patch(log_box)
ax.text(12, 2.9, 'ELK Stack', fontsize=9, weight='bold', ha='center')
ax.text(12, 2.5, '(Logging)', fontsize=8, ha='center', style='italic')

# MLOps layer
mlops_box = FancyBboxPatch((3, 0.5), 8, 1.2,
                          boxstyle="round,pad=0.1",
                          edgecolor='darkviolet',
                          facecolor='thistle',
                          linewidth=2)
ax.add_patch(mlops_box)
ax.text(7, 1.5, 'MLOps Layer', fontsize=10, weight='bold', ha='center')
ax.text(7, 1.1, 'Model Registry • Experiment Tracking • CI/CD • A/B Testing',
       fontsize=8, ha='center', style='italic')
ax.text(7, 0.7, '(Weights & Biases, MLflow, Kubeflow)',
       fontsize=8, ha='center', style='italic')

save_figure('nlp_production_architecture.png')


# 16. Evaluation Metrics Comparison
print("16. Evaluation Metrics Comparison")
fig, ax = plt.subplots(figsize=(12, 8))

metrics = ['BLEU', 'ROUGE-L', 'METEOR', 'BERTScore']
use_cases = ['Translation', 'Summarization', 'Paraphrase', 'QA']

# Suitability scores (0-10)
scores = np.array([
    [9, 6, 7, 5],    # BLEU
    [5, 9, 6, 7],    # ROUGE-L
    [8, 7, 8, 6],    # METEOR
    [7, 8, 9, 9]     # BERTScore
])

im = ax.imshow(scores, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

# Set ticks
ax.set_xticks(np.arange(len(use_cases)))
ax.set_yticks(np.arange(len(metrics)))
ax.set_xticklabels(use_cases, fontsize=11)
ax.set_yticklabels(metrics, fontsize=11)

# Add values
for i in range(len(metrics)):
    for j in range(len(use_cases)):
        text = ax.text(j, i, f'{scores[i, j]}/10',
                      ha="center", va="center",
                      color="black" if scores[i, j] < 7 else "white",
                      fontsize=10, weight='bold')

ax.set_title('NLP Evaluation Metrics: Suitability by Task',
            fontsize=14, weight='bold', pad=20)

# Colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Suitability Score', fontsize=11)

# Add legend
legend_text = '''Score Guide:
9-10: Excellent
7-8: Good
5-6: Fair
<5: Poor'''

ax.text(1.15, 0.3, legend_text, transform=ax.transAxes,
       fontsize=9, verticalalignment='top',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                edgecolor='gold', linewidth=2))

save_figure('evaluation_metrics_comparison.png')


print("\n" + "=" * 70)
print(f"✓ Successfully generated 16 NLP visualizations!")
print(f"✓ All images saved to: {output_dir}/")
print("=" * 70)

# List all generated files
print("\nGenerated files:")
for i, filename in enumerate(sorted(output_dir.glob("*.png")), 1):
    print(f"  {i:2d}. {filename.name}")

print("\n" + "=" * 70)
print("Visualization generation complete!")
print("=" * 70)

"""
RAG (Retrieval-Augmented Generation) Architecture
Shows end-to-end RAG pipeline
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def visualize_rag():
    """Visualize RAG architecture."""

    fig, ax = plt.subplots(1, 1, figsize=(18, 11))
    fig.suptitle('RAG: Retrieval-Augmented Generation Architecture', fontsize=18, fontweight='bold')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # === INDEXING PHASE (TOP) ===
    ax.text(10, 12.5, 'INDEXING PHASE (Offline)', ha='center', fontsize=14,
            fontweight='bold', color='#7F8C8D')

    # Document collection
    docs_box = FancyBboxPatch((0.5, 10.5), 3, 1.5, boxstyle="round,pad=0.1",
                               edgecolor='#95A5A6', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(docs_box)
    ax.text(2, 11.5, 'Documents', ha='center', fontsize=11, fontweight='bold')
    ax.text(2, 11.1, '📄 PDFs', ha='center', fontsize=9)
    ax.text(2, 10.8, '📝 Text files', ha='center', fontsize=9)

    # Chunking
    chunk_box = FancyBboxPatch((4.5, 10.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#3498DB', facecolor='#D6EAF8', linewidth=2)
    ax.add_patch(chunk_box)
    ax.text(5.75, 11.5, 'Chunking', ha='center', fontsize=11, fontweight='bold')
    ax.text(5.75, 11, '256-512', ha='center', fontsize=9)
    ax.text(5.75, 10.7, 'tokens', ha='center', fontsize=9)

    arrow = FancyArrowPatch((3.5, 11.25), (4.5, 11.25), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#3498DB')
    ax.add_patch(arrow)

    # Embedding
    embed_box = FancyBboxPatch((8, 10.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                                edgecolor='#9B59B6', facecolor='#E8DAEF', linewidth=2)
    ax.add_patch(embed_box)
    ax.text(9.25, 11.5, 'Embed', ha='center', fontsize=11, fontweight='bold')
    ax.text(9.25, 11.1, 'sentence-', ha='center', fontsize=8)
    ax.text(9.25, 10.8, 'transformers', ha='center', fontsize=8)

    arrow = FancyArrowPatch((7, 11.25), (8, 11.25), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#9B59B6')
    ax.add_patch(arrow)

    # Vector DB
    vdb_box = FancyBboxPatch((11.5, 10.5), 3, 1.5, boxstyle="round,pad=0.1",
                              edgecolor='#16A085', facecolor='#A9DFBF', linewidth=3)
    ax.add_patch(vdb_box)
    ax.text(13, 11.5, 'Vector DB', ha='center', fontsize=11, fontweight='bold')
    ax.text(13, 11.1, 'FAISS', ha='center', fontsize=9)
    ax.text(13, 10.8, 'Pinecone', ha='center', fontsize=9)

    arrow = FancyArrowPatch((10.5, 11.25), (11.5, 11.25), arrowstyle='->', mutation_scale=15,
                             linewidth=2.5, color='#16A085')
    ax.add_patch(arrow)
    ax.text(11, 11.55, 'Store', ha='center', fontsize=8, color='#138D75')

    # === RETRIEVAL PHASE (MIDDLE) ===
    ax.text(10, 9.3, 'RETRIEVAL PHASE (Online)', ha='center', fontsize=14,
            fontweight='bold', color='#E67E22')

    # User query
    query_box = FancyBboxPatch((0.5, 7.3), 3.5, 1.2, boxstyle="round,pad=0.1",
                                edgecolor='#E67E22', facecolor='#FAE5D3', linewidth=3)
    ax.add_patch(query_box)
    ax.text(2.25, 8.2, 'User Query', ha='center', fontsize=11, fontweight='bold')
    ax.text(2.25, 7.7, '"How does RAG work?"', ha='center', fontsize=9, style='italic')

    # Query embedding
    q_embed_box = FancyBboxPatch((5, 7.3), 2.5, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor='#9B59B6', facecolor='#E8DAEF', linewidth=2)
    ax.add_patch(q_embed_box)
    ax.text(6.25, 8.2, 'Embed', ha='center', fontsize=11, fontweight='bold')
    ax.text(6.25, 7.6, 'Query', ha='center', fontsize=9)

    arrow = FancyArrowPatch((4, 7.9), (5, 7.9), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#9B59B6')
    ax.add_patch(arrow)

    # Similarity search
    search_box = FancyBboxPatch((8.5, 7.3), 3, 1.2, boxstyle="round,pad=0.1",
                                 edgecolor='#16A085', facecolor='#A9DFBF', linewidth=3)
    ax.add_patch(search_box)
    ax.text(10, 8.2, 'Similarity', ha='center', fontsize=11, fontweight='bold')
    ax.text(10, 7.75, 'Search', ha='center', fontsize=11, fontweight='bold')

    arrow = FancyArrowPatch((7.5, 7.9), (8.5, 7.9), arrowstyle='->', mutation_scale=15,
                             linewidth=2.5, color='#16A085')
    ax.add_patch(arrow)

    # Arrow to VDB and back
    arrow_up = FancyArrowPatch((11, 8.5), (13, 10.5), arrowstyle='->', mutation_scale=15,
                                linewidth=2, color='#16A085', linestyle='--')
    ax.add_patch(arrow_up)
    ax.text(12.5, 9.5, 'Query', ha='center', fontsize=8, color='#138D75')

    # Retrieved docs
    ret_docs_box = FancyBboxPatch((12.5, 7.3), 3.5, 1.2, boxstyle="round,pad=0.1",
                                   edgecolor='#27AE60', facecolor='#ABEBC6', linewidth=2)
    ax.add_patch(ret_docs_box)
    ax.text(14.25, 8.2, 'Top-k Docs', ha='center', fontsize=11, fontweight='bold')
    ax.text(14.25, 7.7, 'k=3-5 chunks', ha='center', fontsize=9, style='italic')

    arrow = FancyArrowPatch((11.5, 7.9), (12.5, 7.9), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#27AE60')
    ax.add_patch(arrow)

    # === GENERATION PHASE (BOTTOM) ===
    ax.text(10, 6.1, 'GENERATION PHASE', ha='center', fontsize=14,
            fontweight='bold', color='#C0392B')

    # Context assembly
    context_box = FancyBboxPatch((1, 4.2), 7, 1.5, boxstyle="round,pad=0.15",
                                  edgecolor='#F39C12', facecolor='#FCF3CF', linewidth=2)
    ax.add_patch(context_box)
    ax.text(4.5, 5.3, 'Assemble Context', ha='center', fontsize=11, fontweight='bold')
    ax.text(4.5, 4.9, 'Retrieved Docs + User Query', ha='center', fontsize=9)
    ax.text(4.5, 4.5, 'Format as prompt for LLM', ha='center', fontsize=9, style='italic')

    # Arrows to context
    arrow = FancyArrowPatch((2.25, 7.3), (2.5, 5.7), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#E67E22')
    ax.add_patch(arrow)
    ax.text(2, 6.5, 'Query', ha='center', fontsize=8, color='#D68910')

    arrow = FancyArrowPatch((14.25, 7.3), (6.5, 5.7), arrowstyle='->', mutation_scale=15,
                             linewidth=2, color='#27AE60')
    ax.add_patch(arrow)
    ax.text(10.5, 6.5, 'Context', ha='center', fontsize=8, color='#1E8449')

    # LLM
    llm_box = FancyBboxPatch((10, 4.2), 5, 1.5, boxstyle="round,pad=0.15",
                              edgecolor='#8E44AD', facecolor='#E8DAEF', linewidth=3)
    ax.add_patch(llm_box)
    ax.text(12.5, 5.3, 'Large Language Model', ha='center', fontsize=11, fontweight='bold')
    ax.text(12.5, 4.9, 'GPT-4, Claude, Llama', ha='center', fontsize=9)
    ax.text(12.5, 4.5, 'Generate answer using context', ha='center', fontsize=9, style='italic')

    arrow = FancyArrowPatch((8, 4.95), (10, 4.95), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#8E44AD')
    ax.add_patch(arrow)

    # Final response
    response_box = FancyBboxPatch((16, 4.2), 3.5, 1.5, boxstyle="round,pad=0.1",
                                   edgecolor='#27AE60', facecolor='#D5F4E6', linewidth=3)
    ax.add_patch(response_box)
    ax.text(17.75, 5.3, 'Final Answer', ha='center', fontsize=11, fontweight='bold',
            color='#1E8449')
    ax.text(17.75, 4.8, 'Grounded in', ha='center', fontsize=9)
    ax.text(17.75, 4.5, 'retrieved docs', ha='center', fontsize=9)

    arrow = FancyArrowPatch((15, 4.95), (16, 4.95), arrowstyle='->', mutation_scale=20,
                             linewidth=2.5, color='#27AE60')
    ax.add_patch(arrow)

    # Example prompt box
    prompt_box = FancyBboxPatch((1, 2.2), 14, 1.7, boxstyle="round,pad=0.15",
                                 edgecolor='#34495E', facecolor='#F8F9F9', linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(8, 3.5, 'Example Prompt to LLM:', ha='center', fontsize=10,
            fontweight='bold', color='#2C3E50')
    ax.text(8, 3.1, 'Context: [Retrieved Document 1] [Retrieved Document 2] ...',
            ha='center', fontsize=8, family='monospace')
    ax.text(8, 2.75, 'Question: How does RAG work?',
            ha='center', fontsize=8, family='monospace')
    ax.text(8, 2.4, 'Answer based only on the context provided above:',
            ha='center', fontsize=8, family='monospace')

    # Benefits box
    benefits_box = FancyBboxPatch((1, 0.3), 18, 1.5, boxstyle="round,pad=0.15",
                                   edgecolor='#27AE60', facecolor='#EAFAF1', linewidth=2)
    ax.add_patch(benefits_box)
    ax.text(10, 1.45, '✓ Benefits of RAG:', ha='center', fontsize=11,
            fontweight='bold', color='#1E8449')
    ax.text(10, 1.1, '1. Reduces hallucinations   2. Up-to-date information   3. Source attribution   4. Domain-specific knowledge without retraining',
            ha='center', fontsize=9)
    ax.text(10, 0.7, '5. Cost-effective scaling   6. Privacy (keep data local)   7. Easy to update knowledge base',
            ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('Learnings/Module 16 - RAG Vector Databases and AI Agents/visualizations/01_rag_architecture.png',
                dpi=300, bbox_inches='tight')
    print("✓ Created: 01_rag_architecture.png")
    plt.close()

if __name__ == "__main__":
    visualize_rag()

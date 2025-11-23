#!/usr/bin/env python3
"""
Generate visualization images for ML learning modules 9-11
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch, Wedge
from matplotlib.patches import ConnectionPatch, Arrow
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
# MODULE 9: Explainable AI
# ============================================================================

def generate_module9_images():
    """Generate Explainable AI visualizations"""
    print("Generating Module 9 images...")

    # 1. Feature Importance
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Tree-based feature importance
    features = ['Age', 'Income', 'Credit Score', 'Loan Amount', 'Employment',
                'Debt Ratio', 'Assets', 'Location']
    importance = np.array([0.25, 0.22, 0.18, 0.12, 0.10, 0.08, 0.03, 0.02])

    colors = plt.cm.viridis(importance / importance.max())
    axes[0].barh(features, importance, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlabel('Importance Score', fontsize=11, weight='bold')
    axes[0].set_title('Tree-Based Feature Importance', fontsize=12, weight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Add values on bars
    for i, (feat, imp) in enumerate(zip(features, importance)):
        axes[0].text(imp + 0.005, i, f'{imp:.2f}', va='center', fontsize=9, weight='bold')

    # Permutation importance
    perm_importance = np.array([0.23, 0.25, 0.20, 0.11, 0.09, 0.07, 0.03, 0.02])
    perm_std = np.array([0.02, 0.03, 0.025, 0.015, 0.01, 0.01, 0.005, 0.003])

    colors = plt.cm.plasma(perm_importance / perm_importance.max())
    axes[1].barh(features, perm_importance, xerr=perm_std, color=colors,
                edgecolor='black', linewidth=1.5, error_kw={'linewidth': 2, 'ecolor': 'black'})
    axes[1].set_xlabel('Importance Score', fontsize=11, weight='bold')
    axes[1].set_title('Permutation Feature Importance\n(with std dev)', fontsize=12, weight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.suptitle('Feature Importance Methods', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module9/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. SHAP Summary Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # SHAP summary plot (beeswarm style)
    np.random.seed(42)
    n_samples = 100
    features_shap = ['Credit Score', 'Income', 'Age', 'Loan Amount', 'Employment Years',
                     'Debt Ratio', 'Previous Loans', 'Education']

    ax = axes[0]
    for i, feat in enumerate(features_shap):
        # Generate SHAP values
        shap_values = np.random.randn(n_samples) * (len(features_shap) - i) * 0.3
        feature_values = np.random.rand(n_samples)

        # Scatter with color based on feature value
        scatter = ax.scatter(shap_values, [i] * n_samples, c=feature_values,
                           cmap='coolwarm', alpha=0.6, s=20, edgecolors='black', linewidth=0.3)

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=2)
    ax.set_yticks(range(len(features_shap)))
    ax.set_yticklabels(features_shap, fontsize=10)
    ax.set_xlabel('SHAP Value (impact on model output)', fontsize=11, weight='bold')
    ax.set_title('SHAP Summary Plot\n(Each dot = one prediction)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Feature Value', fontsize=10, weight='bold')
    cbar.ax.set_yticklabels(['Low', '', '', '', 'High'])

    # SHAP waterfall plot for single prediction
    ax = axes[1]

    base_value = 0.5
    shap_values_single = np.array([0.15, 0.12, -0.08, 0.06, -0.04, 0.03, -0.02, 0.01])
    features_single = features_shap

    # Calculate cumulative sum
    cumsum = np.cumsum([base_value] + list(shap_values_single))

    # Plot bars
    colors_waterfall = ['red' if val < 0 else 'green' for val in shap_values_single]

    for i, (feat, shap_val) in enumerate(zip(features_single, shap_values_single)):
        start = cumsum[i]
        ax.barh(i, shap_val, left=start, color=colors_waterfall[i],
               alpha=0.7, edgecolor='black', linewidth=1.5)
        # Add connector lines
        if i < len(features_single) - 1:
            ax.plot([cumsum[i+1], cumsum[i+1]], [i+0.4, i+0.6], 'k-', linewidth=1)

    # Base value line
    ax.axvline(x=base_value, color='gray', linestyle='--', linewidth=2, label='Base Value')

    # Final prediction line
    final = cumsum[-1]
    ax.axvline(x=final, color='purple', linestyle='--', linewidth=2, label='Prediction')

    ax.set_yticks(range(len(features_single)))
    ax.set_yticklabels(features_single, fontsize=10)
    ax.set_xlabel('Model Output Value', fontsize=11, weight='bold')
    ax.set_title('SHAP Waterfall Plot\n(Single Prediction Explanation)', fontsize=12, weight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='x')

    plt.suptitle('SHAP Visualizations', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module9/shap_plots.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. LIME Explanation
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # LIME local explanation
    ax = axes[0]

    features_lime = ['Credit Score', 'Income', 'Age', 'Loan Amount', 'Employment']
    lime_weights = np.array([0.35, 0.28, -0.15, 0.12, 0.08])
    colors_lime = ['green' if w > 0 else 'red' for w in lime_weights]

    ax.barh(features_lime, lime_weights, color=colors_lime, alpha=0.7,
           edgecolor='black', linewidth=1.5)
    ax.axvline(x=0, color='black', linewidth=2)
    ax.set_xlabel('Weight (Impact on Prediction)', fontsize=11, weight='bold')
    ax.set_title('LIME: Local Feature Weights\n(For Single Instance)', fontsize=12, weight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # Add annotations
    ax.text(0.18, 4.3, 'Positive →\nApprove Loan', fontsize=9, color='green',
           weight='bold', ha='center')
    ax.text(-0.08, 4.3, '← Negative\nReject Loan', fontsize=9, color='red',
           weight='bold', ha='center')

    # Prediction probabilities
    ax = axes[1]

    classes = ['Reject', 'Approve']
    probabilities = [0.25, 0.75]
    colors_pred = ['red', 'green']

    bars = ax.bar(classes, probabilities, color=colors_pred, alpha=0.7,
                 edgecolor='black', linewidth=2)
    ax.set_ylabel('Probability', fontsize=11, weight='bold')
    ax.set_title('LIME: Prediction Probabilities', fontsize=12, weight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{prob:.1%}', ha='center', va='bottom', fontsize=14, weight='bold')

    plt.suptitle('LIME (Local Interpretable Model-agnostic Explanations)', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module9/lime_explanation.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Partial Dependence Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # PDP for continuous feature
    x_income = np.linspace(20000, 200000, 100)
    y_income = 1 / (1 + np.exp(-(x_income - 80000) / 20000))  # Sigmoid

    axes[0, 0].plot(x_income, y_income, linewidth=3, color='blue')
    axes[0, 0].fill_between(x_income, y_income - 0.05, y_income + 0.05, alpha=0.3)
    axes[0, 0].set_xlabel('Income ($)', fontsize=11, weight='bold')
    axes[0, 0].set_ylabel('Predicted Probability', fontsize=11, weight='bold')
    axes[0, 0].set_title('Partial Dependence: Income', fontsize=12, weight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # PDP for continuous feature 2
    x_age = np.linspace(18, 80, 100)
    y_age = 0.3 + 0.4 * np.exp(-(x_age - 40)**2 / 500)  # Gaussian bump

    axes[0, 1].plot(x_age, y_age, linewidth=3, color='green')
    axes[0, 1].fill_between(x_age, y_age - 0.03, y_age + 0.03, alpha=0.3, color='green')
    axes[0, 1].set_xlabel('Age (years)', fontsize=11, weight='bold')
    axes[0, 1].set_ylabel('Predicted Probability', fontsize=11, weight='bold')
    axes[0, 1].set_title('Partial Dependence: Age', fontsize=12, weight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # PDP for categorical
    categories = ['No College', 'Bachelors', 'Masters', 'PhD']
    y_education = [0.35, 0.55, 0.68, 0.75]
    colors_cat = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))

    axes[1, 0].bar(categories, y_education, color=colors_cat, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1, 0].set_ylabel('Predicted Probability', fontsize=11, weight='bold')
    axes[1, 0].set_title('Partial Dependence: Education', fontsize=12, weight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # 2D PDP (heatmap)
    x_income_2d = np.linspace(30000, 150000, 50)
    x_credit_2d = np.linspace(300, 850, 50)
    X_income, X_credit = np.meshgrid(x_income_2d, x_credit_2d)

    # Create interaction effect
    Y_pred = 1 / (1 + np.exp(-((X_income - 70000) / 30000 + (X_credit - 600) / 100)))

    im = axes[1, 1].contourf(X_income, X_credit, Y_pred, levels=15, cmap='RdYlGn', alpha=0.8)
    axes[1, 1].set_xlabel('Income ($)', fontsize=11, weight='bold')
    axes[1, 1].set_ylabel('Credit Score', fontsize=11, weight='bold')
    axes[1, 1].set_title('2D Partial Dependence:\nIncome × Credit Score', fontsize=12, weight='bold')
    plt.colorbar(im, ax=axes[1, 1], label='Predicted Probability')

    plt.suptitle('Partial Dependence Plots (PDP)', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module9/partial_dependence.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 9 images generated")


# ============================================================================
# MODULE 10: Production ML & MLOps
# ============================================================================

def generate_module10_images():
    """Generate MLOps visualizations"""
    print("Generating Module 10 images...")

    # 1. ML Pipeline Architecture
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(8, 11.5, 'End-to-End ML Pipeline Architecture', ha='center',
           fontsize=18, weight='bold')

    # Data Ingestion
    rect = FancyBboxPatch((0.5, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 9.75, 'Data\nIngestion', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(1.5, 8.5, 'APIs, DBs,\nFiles', ha='center', fontsize=8, style='italic')

    # Data Validation
    rect = FancyBboxPatch((3, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(4, 9.75, 'Data\nValidation', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(4, 8.5, 'Schema,\nQuality', ha='center', fontsize=8, style='italic')

    # Feature Engineering
    rect = FancyBboxPatch((5.5, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(6.5, 9.75, 'Feature\nEngineering', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(6.5, 8.5, 'Transform,\nScale', ha='center', fontsize=8, style='italic')

    # Model Training
    rect = FancyBboxPatch((8, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(9, 9.75, 'Model\nTraining', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(9, 8.5, 'XGBoost,\nNN, etc.', ha='center', fontsize=8, style='italic')

    # Model Validation
    rect = FancyBboxPatch((10.5, 9), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='orange', linewidth=2)
    ax.add_patch(rect)
    ax.text(11.5, 9.75, 'Model\nValidation', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(11.5, 8.5, 'Metrics,\nTests', ha='center', fontsize=8, style='italic')

    # Model Deployment
    rect = FancyBboxPatch((13, 9), 2.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='gold', linewidth=2)
    ax.add_patch(rect)
    ax.text(14.25, 9.75, 'Model\nDeployment', ha='center', va='center', fontsize=10, weight='bold')
    ax.text(14.25, 8.5, 'API,\nDocker', ha='center', fontsize=8, style='italic')

    # Arrows
    for i in range(5):
        ax.arrow(2.6 + i*2.5, 9.75, 0.3, 0, head_width=0.2, head_length=0.15,
                fc='black', ec='black', linewidth=2)

    # Supporting components
    # Feature Store
    rect = FancyBboxPatch((0.5, 6), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(2, 6.6, 'Feature Store', ha='center', va='center', fontsize=10, weight='bold')

    # Model Registry
    rect = FancyBboxPatch((4.5, 6), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(6, 6.6, 'Model Registry', ha='center', va='center', fontsize=10, weight='bold')

    # Metadata Store
    rect = FancyBboxPatch((8.5, 6), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(10, 6.6, 'Metadata Store', ha='center', va='center', fontsize=10, weight='bold')

    # Monitoring
    rect = FancyBboxPatch((12.5, 6), 3, 1.2, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(14, 6.6, 'Monitoring &\nAlerting', ha='center', va='center', fontsize=10, weight='bold')

    # Orchestration layer
    rect = FancyBboxPatch((0.5, 3.5), 15, 1, boxstyle="round,pad=0.1",
                          edgecolor='darkblue', facecolor='lightsteelblue', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(8, 4, 'Orchestration Layer (Airflow, Kubeflow, MLflow)', ha='center',
           va='center', fontsize=11, weight='bold')

    # CI/CD
    rect = FancyBboxPatch((0.5, 2), 15, 1, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(8, 2.5, 'CI/CD Pipeline (GitHub Actions, Jenkins, GitLab CI)', ha='center',
           va='center', fontsize=11, weight='bold')

    # Infrastructure
    rect = FancyBboxPatch((0.5, 0.5), 15, 1, boxstyle="round,pad=0.1",
                          edgecolor='brown', facecolor='wheat', linewidth=2, linestyle='--')
    ax.add_patch(rect)
    ax.text(8, 1, 'Infrastructure (AWS, GCP, Azure, Kubernetes)', ha='center',
           va='center', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig('images/module10/ml_pipeline_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Model Deployment Patterns
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Batch Prediction
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Database
    rect = FancyBboxPatch((0.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.5, 7.75, 'Input\nData', ha='center', va='center', fontsize=10, weight='bold')

    # Model
    rect = FancyBboxPatch((4, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(5, 7.75, 'ML\nModel', ha='center', va='center', fontsize=10, weight='bold')

    # Predictions
    rect = FancyBboxPatch((7.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.5, 7.75, 'Output\nPredictions', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    ax.arrow(2.6, 7.75, 1.2, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)
    ax.arrow(6.1, 7.75, 1.2, 0, head_width=0.2, head_length=0.15, fc='black', ec='black', linewidth=2)

    # Schedule
    rect = FancyBboxPatch((3, 5), 4, 0.8, boxstyle="round,pad=0.05",
                          edgecolor='purple', facecolor='lavender', linewidth=1.5, linestyle='--')
    ax.add_patch(rect)
    ax.text(5, 5.4, 'Scheduled (Hourly/Daily)', ha='center', fontsize=9, style='italic')

    ax.text(5, 9.5, 'Batch Prediction', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.5, 'Use: Reports, Analytics', ha='center', fontsize=9, style='italic')

    # Real-time API
    ax = axes[0, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Client
    rect = FancyBboxPatch((0.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.25, 7.75, 'Client\nApp', ha='center', va='center', fontsize=10, weight='bold')

    # API
    rect = FancyBboxPatch((3.5, 7), 2.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='gold', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 7.75, 'API\n(FastAPI)', ha='center', va='center', fontsize=10, weight='bold')

    # Model
    rect = FancyBboxPatch((7.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(8.5, 7.75, 'ML\nModel', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows (bidirectional)
    ax.arrow(2.1, 8, 1.2, 0, head_width=0.15, head_length=0.1, fc='green', ec='green', linewidth=2)
    ax.arrow(3.4, 7.5, -1.2, 0, head_width=0.15, head_length=0.1, fc='blue', ec='blue', linewidth=2)
    ax.arrow(6.1, 7.75, 1.2, 0, head_width=0.15, head_length=0.1, fc='black', ec='black', linewidth=2)

    ax.text(2.6, 8.5, 'Request', ha='center', fontsize=8, color='green', weight='bold')
    ax.text(2.6, 7, 'Response', ha='center', fontsize=8, color='blue', weight='bold')

    ax.text(5, 9.5, 'Real-time API', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.5, 'Use: Web apps, Mobile, <100ms', ha='center', fontsize=9, style='italic')

    # Streaming
    ax = axes[1, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Stream source
    rect = FancyBboxPatch((0.5, 7), 1.8, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.4, 7.75, 'Stream\n(Kafka)', ha='center', va='center', fontsize=10, weight='bold')

    # Processing
    rect = FancyBboxPatch((3.5, 7), 2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.5, 7.75, 'Stream\nProcessor', ha='center', va='center', fontsize=10, weight='bold')

    # Model
    rect = FancyBboxPatch((6.5, 7), 1.5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.25, 7.75, 'ML\nModel', ha='center', va='center', fontsize=10, weight='bold')

    # Output
    rect = FancyBboxPatch((8.5, 7), 1.2, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(9.1, 7.75, 'Sink', ha='center', va='center', fontsize=10, weight='bold')

    # Arrows
    for start_x in [2.4, 5.6, 8.1]:
        ax.arrow(start_x, 7.75, 0.7, 0, head_width=0.15, head_length=0.1,
                fc='black', ec='black', linewidth=2)

    ax.text(5, 9.5, 'Streaming Prediction', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.5, 'Use: Real-time analytics, IoT', ha='center', fontsize=9, style='italic')

    # Edge Deployment
    ax = axes[1, 1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Devices
    devices = [(2, 7.5), (5, 7.5), (8, 7.5)]
    device_names = ['Phone', 'IoT', 'Car']

    for pos, name in zip(devices, device_names):
        rect = FancyBboxPatch((pos[0]-0.8, pos[1]-0.6), 1.6, 1.2, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor='lightblue', linewidth=2)
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=9, weight='bold')

        # Model on device
        circle = Circle((pos[0], pos[1]-0.2), 0.15, color='red', ec='black', linewidth=1)
        ax.add_patch(circle)

    # Cloud (optional sync)
    rect = FancyBboxPatch((3.5, 4.5), 3, 1, boxstyle="round,pad=0.1",
                          edgecolor='gray', facecolor='lightgray', linewidth=1.5, linestyle='--')
    ax.add_patch(rect)
    ax.text(5, 5, 'Cloud (Model Updates)', ha='center', va='center', fontsize=9, style='italic')

    # Sync arrows
    for pos in devices:
        ax.arrow(pos[0], pos[1]-0.8, 0, -0.5, head_width=0.15, head_length=0.1,
                fc='gray', ec='gray', linewidth=1, linestyle='--', alpha=0.5)

    ax.text(5, 9.5, 'Edge Deployment', ha='center', fontsize=12, weight='bold')
    ax.text(5, 0.5, 'Use: Low latency, Offline, Privacy', ha='center', fontsize=9, style='italic')

    plt.suptitle('Model Deployment Patterns', fontsize=16, weight='bold')
    plt.tight_layout()
    plt.savefig('images/module10/deployment_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Monitoring Dashboard
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Model Performance Over Time
    ax1 = fig.add_subplot(gs[0, :2])
    days = np.arange(1, 31)
    accuracy = 0.95 - 0.001 * days + np.random.randn(30) * 0.005
    threshold = 0.93

    ax1.plot(days, accuracy, linewidth=2, marker='o', markersize=4, label='Daily Accuracy')
    ax1.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.fill_between(days, threshold, accuracy, where=(accuracy < threshold), alpha=0.3, color='red')
    ax1.set_xlabel('Day', fontsize=10, weight='bold')
    ax1.set_ylabel('Accuracy', fontsize=10, weight='bold')
    ax1.set_title('Model Performance Over Time', fontsize=12, weight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Prediction Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    predictions = np.random.beta(5, 2, 1000)
    ax2.hist(predictions, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Predicted Probability', fontsize=10, weight='bold')
    ax2.set_ylabel('Count', fontsize=10, weight='bold')
    ax2.set_title('Prediction Distribution', fontsize=12, weight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Feature Drift
    ax3 = fig.add_subplot(gs[1, :2])
    features_drift = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4', 'Feature 5']
    drift_scores = [0.02, 0.15, 0.08, 0.25, 0.05]
    colors_drift = ['green' if score < 0.1 else 'orange' if score < 0.2 else 'red' for score in drift_scores]

    ax3.barh(features_drift, drift_scores, color=colors_drift, edgecolor='black', linewidth=1.5)
    ax3.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Warning')
    ax3.axvline(x=0.2, color='red', linestyle='--', linewidth=2, label='Alert')
    ax3.set_xlabel('Drift Score', fontsize=10, weight='bold')
    ax3.set_title('Feature Drift Detection', fontsize=12, weight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='x')

    # Latency
    ax4 = fig.add_subplot(gs[1, 2])
    latencies = np.random.gamma(2, 10, 500)
    ax4.hist(latencies, bins=40, color='lightcoral', edgecolor='black', alpha=0.7)
    ax4.axvline(x=50, color='red', linestyle='--', linewidth=2, label='SLA: 50ms')
    ax4.set_xlabel('Latency (ms)', fontsize=10, weight='bold')
    ax4.set_ylabel('Count', fontsize=10, weight='bold')
    ax4.set_title('Prediction Latency', fontsize=12, weight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Requests per second
    ax5 = fig.add_subplot(gs[2, :])
    hours = np.arange(24)
    rps = 100 + 50 * np.sin(2 * np.pi * hours / 24) + np.random.randn(24) * 10

    ax5.plot(hours, rps, linewidth=2, marker='s', markersize=5, color='purple')
    ax5.fill_between(hours, 0, rps, alpha=0.3, color='purple')
    ax5.set_xlabel('Hour of Day', fontsize=10, weight='bold')
    ax5.set_ylabel('Requests/Second', fontsize=10, weight='bold')
    ax5.set_title('Traffic Pattern (24 hours)', fontsize=12, weight='bold')
    ax5.grid(True, alpha=0.3)

    plt.suptitle('ML Model Monitoring Dashboard', fontsize=16, weight='bold')
    plt.savefig('images/module10/monitoring_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 10 images generated")


# ============================================================================
# MODULE 11: Career
# ============================================================================

def generate_module11_images():
    """Generate Career visualizations"""
    print("Generating Module 11 images...")

    # 1. ML Career Paths
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(7, 11.5, 'Machine Learning Career Paths', ha='center', fontsize=18, weight='bold')

    # Entry Level
    entry_roles = [
        ('ML Engineer\nJunior', 2, 9, 'lightblue'),
        ('Data Scientist\nJunior', 6, 9, 'lightgreen'),
        ('Research\nIntern', 10, 9, 'lightcoral')
    ]

    for role, x, y, color in entry_roles:
        rect = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, role, ha='center', va='center', fontsize=9, weight='bold')

    ax.text(0.5, 9, 'Entry\nLevel\n(0-2 yrs)', fontsize=10, weight='bold', style='italic')

    # Mid Level
    mid_roles = [
        ('ML Engineer', 2, 6.5, 'skyblue'),
        ('Data Scientist', 6, 6.5, 'lightgreen'),
        ('Applied Scientist', 10, 6.5, 'lightcoral')
    ]

    for role, x, y, color in mid_roles:
        rect = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, role, ha='center', va='center', fontsize=9, weight='bold')
        # Arrow from entry to mid
        if x == 2:
            ax.arrow(2, 8.4, 0, -1.3, head_width=0.2, head_length=0.1, fc='gray', ec='gray', linewidth=2)
        elif x == 6:
            ax.arrow(6, 8.4, 0, -1.3, head_width=0.2, head_length=0.1, fc='gray', ec='gray', linewidth=2)
        else:
            ax.arrow(10, 8.4, 0, -1.3, head_width=0.2, head_length=0.1, fc='gray', ec='gray', linewidth=2)

    ax.text(0.5, 6.5, 'Mid\nLevel\n(2-5 yrs)', fontsize=10, weight='bold', style='italic')

    # Senior Level
    senior_roles = [
        ('Senior ML\nEngineer', 1.5, 4, 'cornflowerblue'),
        ('Senior DS', 4.5, 4, 'mediumseagreen'),
        ('ML Platform\nEngineer', 7.5, 4, 'gold'),
        ('Research\nScientist', 10.5, 4, 'salmon')
    ]

    for role, x, y, color in senior_roles:
        rect = FancyBboxPatch((x-0.8, y-0.5), 1.6, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, role, ha='center', va='center', fontsize=9, weight='bold')

    # Arrows from mid to senior
    ax.arrow(2, 5.9, -0.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(2, 5.9, 2.5, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(6, 5.9, -1.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(10, 5.9, 0.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)

    ax.text(0.5, 4, 'Senior\nLevel\n(5-10 yrs)', fontsize=10, weight='bold', style='italic')

    # Lead Level
    lead_roles = [
        ('Staff ML Engineer', 3, 1.5, 'royalblue'),
        ('Principal Scientist', 7, 1.5, 'crimson'),
        ('ML Manager', 11, 1.5, 'orange')
    ]

    for role, x, y, color in lead_roles:
        rect = FancyBboxPatch((x-1.2, y-0.5), 2.4, 1, boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=color, linewidth=2.5)
        ax.add_patch(rect)
        ax.text(x, y, role, ha='center', va='center', fontsize=10, weight='bold', color='white')

    # Arrows from senior to lead
    ax.arrow(1.5, 3.4, 1.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(4.5, 3.4, -1.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(7.5, 3.4, -0.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)
    ax.arrow(10.5, 3.4, 0.3, -1.3, head_width=0.15, head_length=0.1, fc='gray', ec='gray', linewidth=1.5)

    ax.text(0.5, 1.5, 'Lead\nLevel\n(10+ yrs)', fontsize=10, weight='bold', style='italic')

    # Legend
    ax.text(7, 0.3, 'Multiple paths available | Specialize based on interests | Continuous learning',
           ha='center', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig('images/module11/career_paths.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Skills Progression
    fig, ax = plt.subplots(figsize=(12, 8))

    # Skill categories
    skills = {
        'Entry (0-2 yrs)': ['Python', 'ML Basics', 'Git', 'SQL', 'Pandas'],
        'Mid (2-5 yrs)': ['PyTorch', 'MLOps', 'Docker', 'Cloud', 'System Design'],
        'Senior (5-10 yrs)': ['Architecture', 'Mentoring', 'Strategy', 'Leadership', 'Business'],
        'Lead (10+ yrs)': ['Org Impact', 'Vision', 'Team Building', 'Technical Direction', 'External']
    }

    y_positions = [0.7, 0.5, 0.3, 0.1]
    colors_skills = ['lightblue', 'lightgreen', 'lightcoral', 'gold']

    for (level, level_skills), y, color in zip(skills.items(), y_positions, colors_skills):
        # Level box
        rect = FancyBboxPatch((0.02, y-0.05), 0.18, 0.08, boxstyle="round,pad=0.005",
                              edgecolor='black', facecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(0.11, y, level, ha='center', va='center', fontsize=10, weight='bold',
               transform=ax.transAxes)

        # Skills
        for i, skill in enumerate(level_skills):
            x = 0.25 + i * 0.15
            circle = Circle((x, y), 0.04, color=color, ec='black', linewidth=1.5,
                          transform=ax.transAxes)
            ax.add_patch(circle)
            ax.text(x, y, skill, ha='center', va='center', fontsize=8, weight='bold',
                   transform=ax.transAxes)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.text(0.5, 0.95, 'Skills Development Roadmap', ha='center', fontsize=16, weight='bold',
           transform=ax.transAxes)
    ax.text(0.5, 0.02, 'Technical skills → Business skills | Depth → Breadth',
           ha='center', fontsize=10, style='italic', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('images/module11/skills_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Salary Ranges
    fig, ax = plt.subplots(figsize=(12, 8))

    levels = ['Entry\n(0-2 yrs)', 'Mid\n(2-5 yrs)', 'Senior\n(5-10 yrs)', 'Staff/Principal\n(10+ yrs)']
    min_salary = np.array([90, 120, 160, 250])
    max_salary = np.array([120, 160, 250, 450])
    avg_salary = (min_salary + max_salary) / 2

    y_pos = np.arange(len(levels))

    # Plot ranges
    for i, (level, min_sal, max_sal, avg_sal) in enumerate(zip(levels, min_salary, max_salary, avg_salary)):
        ax.barh(i, max_sal - min_sal, left=min_sal, height=0.5, color='lightblue',
               edgecolor='black', linewidth=2, alpha=0.7)
        ax.plot(avg_sal, i, 'ro', markersize=12, label='Average' if i == 0 else '')
        ax.text(max_sal + 10, i, f'${max_sal}K', va='center', fontsize=10, weight='bold')
        ax.text(min_sal - 10, i, f'${min_sal}K', va='center', fontsize=10, weight='bold', ha='right')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(levels, fontsize=11)
    ax.set_xlabel('Total Compensation (USD, thousands)', fontsize=12, weight='bold')
    ax.set_title('ML Engineering Salary Ranges (US, 2024)\nBase + Equity + Bonus', fontsize=14, weight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(50, 500)

    plt.tight_layout()
    plt.savefig('images/module11/salary_ranges.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Learning Path
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 12)
    ax.axis('off')

    # Title
    ax.text(6, 11.5, 'ML Learning Journey (12-18 months)', ha='center', fontsize=16, weight='bold')

    # Month 0-3: Foundations
    rect = FancyBboxPatch((0.5, 8.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightblue', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.75, 10.2, 'Months 0-3', ha='center', fontsize=11, weight='bold')
    ax.text(1.75, 9.7, 'Foundations', ha='center', fontsize=10, weight='bold')
    ax.text(1.75, 9.2, '• Python', ha='left', fontsize=8)
    ax.text(1.75, 8.9, '• Math basics', ha='left', fontsize=8)
    ax.text(1.75, 8.6, '• ML algorithms', ha='left', fontsize=8)

    # Month 4-6: Practice
    rect = FancyBboxPatch((3.5, 8.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightgreen', linewidth=2)
    ax.add_patch(rect)
    ax.text(4.75, 10.2, 'Months 4-6', ha='center', fontsize=11, weight='bold')
    ax.text(4.75, 9.7, 'Deep Learning', ha='center', fontsize=10, weight='bold')
    ax.text(4.75, 9.2, '• PyTorch', ha='left', fontsize=8)
    ax.text(4.75, 8.9, '• CNNs, RNNs', ha='left', fontsize=8)
    ax.text(4.75, 8.6, '• 2-3 projects', ha='left', fontsize=8)

    # Month 7-9: MLOps
    rect = FancyBboxPatch((6.5, 8.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightyellow', linewidth=2)
    ax.add_patch(rect)
    ax.text(7.75, 10.2, 'Months 7-9', ha='center', fontsize=11, weight='bold')
    ax.text(7.75, 9.7, 'MLOps', ha='center', fontsize=10, weight='bold')
    ax.text(7.75, 9.2, '• Docker', ha='left', fontsize=8)
    ax.text(7.75, 8.9, '• Cloud (AWS)', ha='left', fontsize=8)
    ax.text(7.75, 8.6, '• Deploy models', ha='left', fontsize=8)

    # Month 10-12: Advanced
    rect = FancyBboxPatch((9.5, 8.5), 2.5, 2, boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor='lightcoral', linewidth=2)
    ax.add_patch(rect)
    ax.text(10.75, 10.2, 'Months 10-12', ha='center', fontsize=11, weight='bold')
    ax.text(10.75, 9.7, 'Specialize', ha='center', fontsize=10, weight='bold')
    ax.text(10.75, 9.2, '• Pick domain', ha='left', fontsize=8)
    ax.text(10.75, 8.9, '• Advanced projects', ha='left', fontsize=8)
    ax.text(10.75, 8.6, '• Portfolio', ha='left', fontsize=8)

    # Arrows
    for i in range(3):
        ax.arrow(3.1 + i*3, 9.5, 0.3, 0, head_width=0.3, head_length=0.15,
                fc='black', ec='black', linewidth=2)

    # Job prep
    rect = FancyBboxPatch((3.5, 5.5), 5, 1.8, boxstyle="round,pad=0.1",
                          edgecolor='purple', facecolor='lavender', linewidth=2)
    ax.add_patch(rect)
    ax.text(6, 7, 'Job Preparation (Parallel)', ha='center', fontsize=11, weight='bold')
    ax.text(6, 6.5, '• LeetCode: 100+ problems', ha='center', fontsize=9)
    ax.text(6, 6.15, '• Portfolio: 4-6 projects', ha='center', fontsize=9)
    ax.text(6, 5.8, '• Mock interviews', ha='center', fontsize=9)

    # Outcome
    rect = FancyBboxPatch((3.5, 3), 5, 1.5, boxstyle="round,pad=0.1",
                          edgecolor='darkgreen', facecolor='lightgreen', linewidth=3)
    ax.add_patch(rect)
    ax.text(6, 4.2, '🎯 Land ML Job', ha='center', fontsize=14, weight='bold')
    ax.text(6, 3.7, 'Junior ML Engineer / Data Scientist', ha='center', fontsize=10)
    ax.text(6, 3.3, '$90K - $120K', ha='center', fontsize=10, style='italic')

    # Arrow to outcome
    ax.arrow(6, 5.4, 0, -0.7, head_width=0.4, head_length=0.15,
            fc='darkgreen', ec='darkgreen', linewidth=3)

    # Timeline
    for month in [0, 3, 6, 9, 12]:
        x = 0.5 + month / 12 * 11.5
        ax.plot([x, x], [1.5, 2], 'k-', linewidth=1.5)
        ax.text(x, 1.2, f'Month {month}', ha='center', fontsize=9)

    ax.plot([0.5, 12], [1.75, 1.75], 'k-', linewidth=2)

    plt.tight_layout()
    plt.savefig('images/module11/learning_path.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("✓ Module 11 images generated")


# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Generating ML Learning Module Images (Part 2)")
    print("=" * 60)

    ensure_dir('images/module9')
    ensure_dir('images/module10')
    ensure_dir('images/module11')

    generate_module9_images()
    generate_module10_images()
    generate_module11_images()

    print("\n" + "=" * 60)
    print("✓ All Module 9-11 images generated successfully!")
    print("=" * 60)

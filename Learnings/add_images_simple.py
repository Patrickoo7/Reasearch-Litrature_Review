#!/usr/bin/env python3
"""
Add visual guides section to lesson markdown files
"""

import os

# Simpler approach: add images right after the first separator (---)
LESSON_IMAGES = {
    "Module 5 - Neural Networks Foundations/Lesson 1 - Neural Network Basics.md": [
        ("../images/module5/neural_network_architecture.png", "Multi-layer neural network architecture"),
        ("../images/module5/activation_functions.png", "Common activation functions (Sigmoid, Tanh, ReLU, etc.)"),
    ],
    "Module 5 - Neural Networks Foundations/Lesson 2 - Backpropagation and Training.md": [
        ("../images/module5/gradient_descent.png", "Gradient descent optimization on loss surface"),
    ],
    "Module 5 - Neural Networks Foundations/Lesson 4 - PyTorch Fundamentals.md": [
        ("../images/module5/training_progress.png", "Training and validation curves over epochs"),
    ],

    "Module 6 - Deep Learning for Computer Vision/Lesson 1 - CNNs Architecture.md": [
        ("../images/module6/convolution_operation.png", "Convolution operation - filter applied to input"),
        ("../images/module6/pooling_operations.png", "Max pooling reduces spatial dimensions"),
        ("../images/module6/cnn_architecture.png", "Complete CNN architecture"),
    ],
    "Module 6 - Deep Learning for Computer Vision/Lesson 2 - Advanced CNN Architectures.md": [
        ("../images/module6/resnet_skip_connection.png", "ResNet skip connections solving vanishing gradient"),
    ],

    "Module 7 - Time Series and Forecasting/Lesson 1 - Time Series Fundamentals.md": [
        ("../images/module7/time_series_decomposition.png", "Time series decomposition: trend + seasonality + noise"),
        ("../images/module7/acf_pacf.png", "ACF and PACF plots for ARIMA model selection"),
    ],
    "Module 7 - Time Series and Forecasting/Lesson 2 - ARIMA and Statistical Methods.md": [
        ("../images/module7/arima_forecast.png", "ARIMA forecast with confidence intervals"),
    ],
    "Module 7 - Time Series and Forecasting/Lesson 4 - Deep Learning for Time Series.md": [
        ("../images/module7/lstm_architecture.png", "LSTM architecture for sequential data"),
    ],

    "Module 8 - Recommender Systems/Lesson 1 - Collaborative Filtering.md": [
        ("../images/module8/user_item_matrix.png", "User-item rating matrix with missing values"),
        ("../images/module8/collaborative_filtering.png", "User-based vs Item-based collaborative filtering"),
    ],
    "Module 8 - Recommender Systems/Lesson 3 - Matrix Factorization.md": [
        ("../images/module8/matrix_factorization.png", "Matrix factorization into user and item features"),
    ],
    "Module 8 - Recommender Systems/Lesson 4 - Neural Recommenders.md": [
        ("../images/module8/neural_recommender.png", "Two-tower neural recommender architecture"),
    ],

    "Module 9 - Explainable AI/Lesson 1 - Feature Importance.md": [
        ("../images/module9/feature_importance.png", "Feature importance methods comparison"),
        ("../images/module9/partial_dependence.png", "Partial dependence plots"),
    ],
    "Module 9 - Explainable AI/Lesson 2 - SHAP Values.md": [
        ("../images/module9/shap_plots.png", "SHAP summary and waterfall plots"),
    ],
    "Module 9 - Explainable AI/Lesson 3 - LIME.md": [
        ("../images/module9/lime_explanation.png", "LIME local explanation"),
    ],

    "Module 10 - Production ML and MLOps/Lesson 1 - Model Deployment.md": [
        ("../images/module10/deployment_patterns.png", "Common deployment patterns: batch, real-time, streaming, edge"),
    ],
    "Module 10 - Production ML and MLOps/Lesson 2 - MLOps Basics.md": [
        ("../images/module10/ml_pipeline_architecture.png", "End-to-end ML pipeline architecture"),
    ],
    "Module 10 - Production ML and MLOps/Lesson 3 - Monitoring and Maintenance.md": [
        ("../images/module10/monitoring_dashboard.png", "ML model monitoring dashboard"),
    ],

    "Module 11 - Capstone and Career/Lesson 2 - Interview Preparation.md": [
        # Already has images
    ],
}


def add_visual_section(file_path, images):
    """Add visual guides section after first separator"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if visual section already exists
        if "## Visual Guides" in content or "Visual Guides" in content:
            print(f"  ⊘ Visual section already exists in {os.path.basename(file_path)}")
            return False

        # Find first separator (---)
        first_sep = content.find('---')
        if first_sep == -1:
            print(f"  ⚠ No separator found in {os.path.basename(file_path)}")
            return False

        # Find the end of the first separator line
        next_newline = content.find('\n', first_sep + 3)
        if next_newline == -1:
            next_newline = len(content)

        # Create visual section
        visual_section = "\n\n## Visual Guides 📊\n\n"

        for img_path, caption in images:
            visual_section += f"![{caption}]({img_path})\n"
            visual_section += f"*{caption}*\n\n"

        visual_section += "---\n"

        # Insert visual section
        new_content = content[:next_newline+1] + visual_section + content[next_newline+1:]

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  ✓ Added visual section to {os.path.basename(file_path)} ({len(images)} images)")
        return True

    except Exception as e:
        print(f"  ✗ Error with {os.path.basename(file_path)}: {e}")
        return False


def main():
    """Main function"""
    print("=" * 70)
    print("Adding Visual Guides to Lessons")
    print("=" * 70)

    total_added = 0
    total_images = 0

    for lesson_file, images in LESSON_IMAGES.items():
        if not images:
            continue

        print(f"\n📄 {os.path.basename(lesson_file)}")

        if not os.path.exists(lesson_file):
            print(f"  ⚠ File not found")
            continue

        if add_visual_section(lesson_file, images):
            total_added += 1
            total_images += len(images)

    print("\n" + "=" * 70)
    print(f"✓ Added visual sections to {total_added} lessons")
    print(f"✓ Total images added: {total_images}")
    print("=" * 70)


if __name__ == "__main__":
    main()

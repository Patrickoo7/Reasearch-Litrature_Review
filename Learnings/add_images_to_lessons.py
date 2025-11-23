#!/usr/bin/env python3
"""
Add image references to lesson markdown files
"""

import os
import re

# Image mappings: lesson file -> [(position_keyword, image_path, caption)]
IMAGE_MAPPINGS = {
    # Module 5
    "Module 5 - Neural Networks Foundations/Lesson 1 - Neural Network Basics.md": [
        ("## 2. Multi-Layer Neural Network", "../images/module5/neural_network_architecture.png", "**Figure:** Multi-layer neural network architecture with input, hidden, and output layers"),
        ("## Activation Functions", "../images/module5/activation_functions.png", "**Figure:** Common activation functions used in neural networks"),
    ],
    "Module 5 - Neural Networks Foundations/Lesson 2 - Backpropagation and Training.md": [
        ("## Gradient Descent Visualization", "../images/module5/gradient_descent.png", "**Figure:** Gradient descent optimization on a loss surface"),
    ],
    "Module 5 - Neural Networks Foundations/Lesson 4 - PyTorch Fundamentals.md": [
        ("## Training Loop", "../images/module5/training_progress.png", "**Figure:** Training and validation curves over epochs"),
    ],

    # Module 6
    "Module 6 - Deep Learning for Computer Vision/Lesson 1 - CNNs Architecture.md": [
        ("## Convolution Operation", "../images/module6/convolution_operation.png", "**Figure:** Convolution operation - applying a filter to extract features"),
        ("## Pooling Layers", "../images/module6/pooling_operations.png", "**Figure:** Max pooling operation reduces spatial dimensions"),
        ("## Complete CNN Architecture", "../images/module6/cnn_architecture.png", "**Figure:** Complete CNN architecture from input to output"),
    ],
    "Module 6 - Deep Learning for Computer Vision/Lesson 2 - Advanced CNN Architectures.md": [
        ("## ResNet (Residual Networks)", "../images/module6/resnet_skip_connection.png", "**Figure:** ResNet block with skip connections solving vanishing gradient"),
    ],

    # Module 7
    "Module 7 - Time Series and Forecasting/Lesson 1 - Time Series Fundamentals.md": [
        ("## Time Series Components", "../images/module7/time_series_decomposition.png", "**Figure:** Time series decomposition into trend, seasonality, and noise"),
        ("## Autocorrelation", "../images/module7/acf_pacf.png", "**Figure:** ACF and PACF plots for identifying ARIMA parameters"),
    ],
    "Module 7 - Time Series and Forecasting/Lesson 2 - ARIMA and Statistical Methods.md": [
        ("## ARIMA Forecasting", "../images/module7/arima_forecast.png", "**Figure:** ARIMA forecast with confidence intervals"),
    ],
    "Module 7 - Time Series and Forecasting/Lesson 4 - Deep Learning for Time Series.md": [
        ("## LSTM Architecture", "../images/module7/lstm_architecture.png", "**Figure:** LSTM architecture for sequential time series data"),
    ],

    # Module 8
    "Module 8 - Recommender Systems/Lesson 1 - Collaborative Filtering.md": [
        ("## User-Item Matrix", "../images/module8/user_item_matrix.png", "**Figure:** User-item rating matrix with missing values to predict"),
        ("## Collaborative Filtering Approaches", "../images/module8/collaborative_filtering.png", "**Figure:** User-based vs Item-based collaborative filtering"),
    ],
    "Module 8 - Recommender Systems/Lesson 3 - Matrix Factorization.md": [
        ("## Matrix Factorization", "../images/module8/matrix_factorization.png", "**Figure:** Matrix factorization decomposes ratings into user and item features"),
    ],
    "Module 8 - Recommender Systems/Lesson 4 - Neural Recommenders.md": [
        ("## Two-Tower Architecture", "../images/module8/neural_recommender.png", "**Figure:** Two-tower neural recommender architecture"),
    ],

    # Module 9
    "Module 9 - Explainable AI/Lesson 1 - Feature Importance.md": [
        ("## Feature Importance Methods", "../images/module9/feature_importance.png", "**Figure:** Tree-based and permutation feature importance"),
        ("## Partial Dependence Plots", "../images/module9/partial_dependence.png", "**Figure:** Partial dependence plots showing feature effects"),
    ],
    "Module 9 - Explainable AI/Lesson 2 - SHAP Values.md": [
        ("## SHAP Visualizations", "../images/module9/shap_plots.png", "**Figure:** SHAP summary and waterfall plots for model interpretation"),
    ],
    "Module 9 - Explainable AI/Lesson 3 - LIME.md": [
        ("## LIME Explanations", "../images/module9/lime_explanation.png", "**Figure:** LIME local explanation for a single prediction"),
    ],

    # Module 10
    "Module 10 - Production ML and MLOps/Lesson 1 - Model Deployment.md": [
        ("## Deployment Patterns", "../images/module10/deployment_patterns.png", "**Figure:** Common ML model deployment patterns"),
    ],
    "Module 10 - Production ML and MLOps/Lesson 2 - MLOps Basics.md": [
        ("## ML Pipeline", "../images/module10/ml_pipeline_architecture.png", "**Figure:** End-to-end ML pipeline architecture"),
    ],
    "Module 10 - Production ML and MLOps/Lesson 3 - Monitoring and Maintenance.md": [
        ("## Monitoring Dashboard", "../images/module10/monitoring_dashboard.png", "**Figure:** ML model monitoring dashboard with key metrics"),
    ],

    # Module 11
    "Module 11 - Capstone and Career/Lesson 2 - Interview Preparation.md": [
        # Career lesson has specific sections
    ],
    "Module 11 - Capstone and Career/Lesson 4 - Career Paths.md": [
        ("## Common ML Roles", "../images/module11/career_paths.png", "**Figure:** ML career progression paths from entry to lead level"),
        ("## Skills Development Roadmap", "../images/module11/skills_progression.png", "**Figure:** Skills development from technical to leadership"),
        ("## Salary Negotiation", "../images/module11/salary_ranges.png", "**Figure:** ML engineering salary ranges by experience level (US, 2024)"),
        ("## Continuous Learning", "../images/module11/learning_path.png", "**Figure:** 12-18 month ML learning journey roadmap"),
    ],
}


def add_image_to_file(file_path, position_keyword, image_path, caption):
    """Add an image reference after a specific keyword in the file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if image already exists
        if image_path in content:
            print(f"  ⊘ Image already exists in {os.path.basename(file_path)}")
            return False

        # Find the position
        if position_keyword not in content:
            print(f"  ⚠ Position keyword not found in {os.path.basename(file_path)}: {position_keyword}")
            return False

        # Create image markdown
        image_md = f"\n\n![{caption}]({image_path})\n\n{caption}\n"

        # Insert after the keyword (after the next newline)
        pos = content.find(position_keyword)
        next_newline = content.find('\n', pos + len(position_keyword))

        if next_newline == -1:
            next_newline = len(content)

        new_content = content[:next_newline+1] + image_md + content[next_newline+1:]

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)

        print(f"  ✓ Added image to {os.path.basename(file_path)}")
        return True

    except Exception as e:
        print(f"  ✗ Error with {os.path.basename(file_path)}: {e}")
        return False


def main():
    """Main function to add all images"""
    print("=" * 70)
    print("Adding Images to Lesson Files")
    print("=" * 70)

    total_added = 0

    for lesson_file, image_configs in IMAGE_MAPPINGS.items():
        if not image_configs:
            continue

        full_path = lesson_file
        print(f"\n📄 Processing: {os.path.basename(lesson_file)}")

        if not os.path.exists(full_path):
            print(f"  ⚠ File not found: {full_path}")
            continue

        for position_keyword, image_path, caption in image_configs:
            if add_image_to_file(full_path, position_keyword, image_path, caption):
                total_added += 1

    print("\n" + "=" * 70)
    print(f"✓ Added {total_added} images to lessons successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

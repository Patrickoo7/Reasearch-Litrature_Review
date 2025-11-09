"""
Hugging Face Spaces / Gradio App Entry Point

This file is used when deploying to Hugging Face Spaces or running the web interface
"""

import os
from src.research_reproducer.web_interface import launch_web_interface

# Get GitHub token from environment (set in Hugging Face Spaces secrets)
github_token = os.getenv('GITHUB_TOKEN')

# Launch the interface
if __name__ == "__main__":
    launch_web_interface(
        github_token=github_token,
        share=False,  # Hugging Face Spaces handles sharing
        port=7860
    )

"""
Notebook Generator
Auto-generate Colab/Jupyter notebooks for reproducing papers
"""

import json
import logging
from typing import Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


class NotebookGenerator:
    """Generate executable notebooks for paper reproduction"""

    def __init__(self):
        pass

    def generate_colab_notebook(self, analysis: Dict, repo_url: str) -> Dict:
        """
        Generate Google Colab notebook

        Args:
            analysis: Repository analysis from RepositoryAnalyzer
            repo_url: GitHub repository URL

        Returns:
            Notebook dict (can be saved as .ipynb)
        """
        cells = []

        # Title cell
        cells.append(self._markdown_cell(
            f"# Reproducing: {analysis.get('paper_title', 'Research Paper')}\n\n"
            f"Auto-generated notebook for reproducing this research paper.\n\n"
            f"Repository: [{repo_url}]({repo_url})"
        ))

        # Setup cell
        setup_code = self._generate_setup_code(analysis)
        cells.append(self._code_cell(setup_code, "Setup Environment"))

        # Clone repository
        cells.append(self._code_cell(
            f"# Clone repository\n"
            f"!git clone {repo_url} repo\n"
            f"%cd repo",
            "Clone Repository"
        ))

        # Install dependencies
        install_code = self._generate_install_code(analysis)
        if install_code:
            cells.append(self._code_cell(install_code, "Install Dependencies"))

        # GPU check
        if analysis.get('gpu_required'):
            cells.append(self._markdown_cell(
                "## ⚠️ GPU Required\n\n"
                "This code requires GPU. Make sure you have enabled GPU in Colab:\n"
                "- Runtime → Change runtime type → GPU"
            ))
            cells.append(self._code_cell(
                "# Check GPU\n"
                "!nvidia-smi",
                "Verify GPU"
            ))

        # Data download
        if analysis.get('data_requirements'):
            cells.append(self._markdown_cell(
                "## 📁 Data Requirements\n\n"
                "This paper requires external data:\n" +
                "\n".join(f"- {req}" for req in analysis['data_requirements'])
            ))

        # Run code
        run_code = self._generate_run_code(analysis)
        cells.append(self._code_cell(run_code, "Run Reproduction"))

        # Create notebook
        notebook = {
            "nbformat": 4,
            "nbformat_minor": 0,
            "metadata": {
                "colab": {
                    "provenance": [],
                    "gpuType": "T4" if analysis.get('gpu_required') else None,
                },
                "kernelspec": {
                    "name": "python3",
                    "display_name": "Python 3"
                },
                "language_info": {
                    "name": "python"
                },
                "accelerator": "GPU" if analysis.get('gpu_required') else None,
            },
            "cells": cells
        }

        return notebook

    def _markdown_cell(self, content: str) -> Dict:
        """Create markdown cell"""
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [content]
        }

    def _code_cell(self, code: str, title: str = None) -> Dict:
        """Create code cell"""
        if title:
            code = f"# {title}\n\n{code}"

        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [code]
        }

    def _generate_setup_code(self, analysis: Dict) -> str:
        """Generate setup code"""
        code = "# Install system dependencies\n"

        if 'Python' in analysis.get('languages', []):
            code += "# Upgrade pip\n"
            code += "!pip install --upgrade pip\n\n"

        return code

    def _generate_install_code(self, analysis: Dict) -> str:
        """Generate dependency installation code"""
        code = ""

        deps = analysis.get('dependencies', {})

        # Python dependencies
        if 'python' in deps:
            python_deps = deps['python']

            if python_deps.get('requirements_files'):
                for req_file in python_deps['requirements_files']:
                    code += f"# Install from {req_file}\n"
                    code += f"!pip install -r {req_file}\n\n"

            if python_deps.get('setup_py'):
                code += "# Install from setup.py\n"
                code += "!pip install -e .\n\n"

        # Node dependencies
        if 'node' in deps and deps['node'].get('package_json'):
            code += "# Install Node.js dependencies\n"
            code += "!npm install\n\n"

        return code

    def _generate_run_code(self, analysis: Dict) -> str:
        """Generate code execution commands"""
        code = "# Run reproduction\n\n"

        entry_points = analysis.get('entry_points', [])

        if entry_points:
            # Use first entry point
            ep = entry_points[0]
            command = ep.get('command', ep.get('file', 'python main.py'))

            code += f"# Execute: {command}\n"
            code += f"!{command}\n"
        else:
            code += "# No clear entry point found\n"
            code += "# Check the repository README for run instructions\n"

        return code

    def save_notebook(self, notebook: Dict, output_path: str):
        """Save notebook to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(notebook, f, indent=2)

        logger.info(f"Notebook saved to {output_path}")

    def generate_colab_url(self, github_notebook_path: str) -> str:
        """
        Generate Colab open URL

        Args:
            github_notebook_path: Path to notebook on GitHub
                e.g., 'username/repo/blob/main/notebook.ipynb'

        Returns:
            Colab URL that opens the notebook
        """
        return f"https://colab.research.google.com/github/{github_notebook_path}"

    def generate_binder_url(self, github_repo: str) -> str:
        """Generate Binder URL for repository"""
        # Remove https://github.com/ prefix
        repo_path = github_repo.replace('https://github.com/', '')

        return f"https://mybinder.org/v2/gh/{repo_path}/main"

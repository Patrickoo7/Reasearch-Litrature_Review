# Research Reproducer

**A tool for early and solo researchers to automatically reproduce research papers**

Research Reproducer helps you find, setup, and run code from research papers automatically. Simply provide a paper (PDF, arXiv ID, or URL) and let the tool handle the rest!

## Features

- 📄 **Paper Ingestion**: Extract metadata and GitHub links from PDFs, arXiv papers, or URLs
- 🔍 **Repository Discovery**: Automatically find associated code using multiple sources (Papers with Code, direct extraction)
- 🔬 **Code Analysis**: Detect dependencies, languages, entry points, and complexity
- 🐳 **Environment Setup**: Automatically set up Docker, Conda, or Python virtual environments
- 💬 **Interactive Assistant**: Ask intelligent questions to configure your reproduction
- ⚡ **Code Execution**: Run code with monitoring, timeout handling, and error analysis
- 📊 **Detailed Reports**: Generate comprehensive reports of the reproduction process

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Patrickoo7/Reasearch-Litrature_Review.git
cd Reasearch-Litrature_Review

# Install dependencies
pip install -e .
```

### Basic Usage

```bash
# Reproduce from arXiv paper
research-reproduce reproduce 2301.12345

# Reproduce from PDF file
research-reproduce reproduce paper.pdf

# Reproduce from URL
research-reproduce reproduce https://arxiv.org/abs/2301.12345

# Non-interactive mode with custom timeout
research-reproduce reproduce 2301.12345 --no-interactive --timeout 3600
```

### Analysis Without Execution

Just want to see if a paper has code? Use the analyze command:

```bash
# Analyze a paper without running code
research-reproduce analyze 2301.12345

# Save analysis to file
research-reproduce analyze paper.pdf -o report.json
```

### Inspect a Repository

Already have a repository cloned? Inspect its structure:

```bash
research-reproduce inspect ./path/to/repo
```

## How It Works

### The Reproduction Pipeline

```
1. Paper Ingestion
   ↓ Extract metadata, GitHub URLs, arXiv ID
2. Repository Finding
   ↓ Search Papers with Code, parse paper text
3. Interactive Configuration (optional)
   ↓ Ask about environment, data, hardware
4. Repository Analysis
   ↓ Detect languages, dependencies, entry points
5. Environment Setup
   ↓ Create Docker/Conda/venv environment
6. Code Execution
   ↓ Run with monitoring and timeout
7. Report Generation
   ↓ Save detailed reproduction report
```

### What Gets Analyzed

- **Languages**: Python, JavaScript, TypeScript, R, Julia, C++, Java, Go, Rust
- **Dependencies**: requirements.txt, setup.py, pyproject.toml, environment.yml, package.json
- **Entry Points**: main.py, train.py, run.py, npm scripts, etc.
- **Configuration**: config files, .env files
- **Data Requirements**: data directories, download scripts
- **Hardware**: GPU requirements, computational complexity
- **Docker Support**: Dockerfiles, docker-compose files

## Interactive Mode

When running in interactive mode (default), the tool will ask you questions to customize the reproduction:

```bash
$ research-reproduce reproduce 2301.12345

Configuration Assistant
I'll help you configure the environment to run this research code.

? Select repository:
  1. https://github.com/author/paper-code (1234 stars) [paper_text]

? Preferred environment:
  • Docker (recommended)
  • Conda
  • Python venv

? Do you have the required data available? No

? Should I attempt to run data download scripts? Yes

? Do you have GPU access? No

⚠ Warning: Code may fail or run very slowly without GPU
? Continue anyway? Yes
```

## Configuration

### Environment Variables

```bash
# GitHub token for higher API rate limits
export GITHUB_TOKEN=your_github_token_here

# Custom work directory
export WORK_DIR=./my_reproductions
```

### GitHub Token

To avoid rate limiting when searching for repositories, set a GitHub token:

1. Create a token at https://github.com/settings/tokens
2. Set the `GITHUB_TOKEN` environment variable:
   ```bash
   export GITHUB_TOKEN=ghp_xxxxxxxxxxxxx
   ```

## Examples

### Example 1: Quick arXiv Reproduction

```bash
# Reproduce a popular paper with default settings
research-reproduce reproduce 1706.03762  # Attention Is All You Need
```

### Example 2: Custom Configuration

```bash
# Non-interactive with custom settings
research-reproduce reproduce paper.pdf \
  --no-interactive \
  --timeout 7200 \
  --work-dir ./my_experiments \
  --verbose
```

### Example 3: Batch Analysis

```bash
# Analyze multiple papers
for arxiv_id in 2301.12345 2302.54321 2303.98765; do
  research-reproduce analyze $arxiv_id -o "analysis_${arxiv_id}.json"
done
```

## Output Structure

Each reproduction creates a session directory:

```
reproductions/
└── Paper_Title_20240109_143022/
    ├── repo/              # Cloned repository
    ├── envs/              # Created environments
    ├── logs/              # Execution logs
    └── report.json        # Detailed report
```

### Report Format

```json
{
  "paper": {
    "title": "Paper Title",
    "authors": ["Author 1", "Author 2"],
    "arxiv_id": "2301.12345",
    "github_urls": ["https://github.com/..."]
  },
  "repositories": [...],
  "analysis": {
    "languages": ["Python"],
    "dependencies": {...},
    "entry_points": [...],
    "complexity": "medium"
  },
  "environment": {
    "type": "docker",
    "success": true
  },
  "execution": {
    "success": true,
    "exit_code": 0,
    "execution_time": 123.45,
    "errors": [],
    "warnings": []
  },
  "success": true,
  "timestamp": "2024-01-09T14:30:22"
}
```

## Supported Environments

### Docker (Recommended)
- ✅ Best isolation
- ✅ Handles complex dependencies
- ✅ GPU support via nvidia-docker
- ⚠️ Requires Docker installed

### Conda
- ✅ Good for scientific computing
- ✅ Multi-language support
- ⚠️ Requires Conda/Miniconda

### Python venv
- ✅ Lightweight and fast
- ✅ No extra dependencies
- ⚠️ Python-only

## Known Limitations

- **~30-40% of papers don't share code** - We can only reproduce papers with public code
- **Broken dependencies** - Old papers may have outdated/conflicting dependencies
- **Missing data** - Many papers require datasets that aren't publicly available
- **Computational resources** - Some papers require GPUs or extensive compute time
- **Incomplete documentation** - Not all research code is well-documented

## Troubleshooting

### Common Issues

**No repositories found**
```bash
# Try specifying the paper URL directly
research-reproduce reproduce https://arxiv.org/abs/YOUR_ID

# Or analyze manually to see what was detected
research-reproduce analyze YOUR_SOURCE --verbose
```

**Environment setup fails**
```bash
# Try a different environment type
research-reproduce reproduce YOUR_SOURCE --no-interactive
# Then manually select environment in interactive prompt

# Or use Docker if available
docker pull python:3.9
research-reproduce reproduce YOUR_SOURCE
```

**Execution timeout**
```bash
# Increase timeout (in seconds)
research-reproduce reproduce YOUR_SOURCE --timeout 7200
```

**GPU required but not available**
```bash
# Some papers require GPU - check the analysis first
research-reproduce analyze YOUR_SOURCE

# If GPU required, consider:
# - Using cloud GPU (Colab, AWS, etc.)
# - Modifying code to run on CPU
# - Using smaller model/dataset
```

## Contributing

Contributions welcome! Areas for improvement:

- Additional paper sources (OpenReview, PapersWithCode direct API)
- Better dependency resolution
- More language support (MATLAB, Julia, etc.)
- Improved error diagnosis
- Cloud compute integration
- Result validation against paper claims

## Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/Patrickoo7/Reasearch-Litrature_Review.git
cd Reasearch-Litrature_Review
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

Built with:
- [Rich](https://github.com/Textualize/rich) - Beautiful terminal output
- [Click](https://click.palletsprojects.com/) - CLI framework
- [GitPython](https://github.com/gitpython-developers/GitPython) - Git operations
- [PyGithub](https://github.com/PyGithub/PyGithub) - GitHub API
- [arxiv](https://github.com/lukasschwab/arxiv.py) - arXiv API

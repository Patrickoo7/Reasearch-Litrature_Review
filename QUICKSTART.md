# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -e .

# Or install with web interface support
pip install -e ".[web]"
```

## Basic Usage

### 1. Reproduce a Paper (CLI)

```bash
# From arXiv ID
research-reproduce reproduce 2301.12345

# From PDF
research-reproduce reproduce paper.pdf

# From URL
research-reproduce reproduce https://arxiv.org/abs/2301.12345
```

### 2. Launch Web Interface (Easiest!)

```bash
# Start web UI
research-reproduce web

# Then open http://localhost:7860 in your browser
```

### 3. Quick Analysis (No Execution)

```bash
# Check if paper has code
research-reproduce analyze 2301.12345

# Save results
research-reproduce analyze 2301.12345 -o report.json
```

### 4. Check GPU Status

```bash
research-reproduce gpu
```

### 5. Manage Cache

```bash
# View cache stats
research-reproduce cache stats

# Clear cache
research-reproduce cache clear
```

## Common Workflows

### Workflow 1: Quick Check
```bash
# See if paper has code
research-reproduce analyze 2301.12345
```

### Workflow 2: Full Reproduction
```bash
# Full reproduction with custom timeout
research-reproduce reproduce 2301.12345 --timeout 3600
```

### Workflow 3: Batch Analysis
```bash
# Analyze multiple papers
for id in 2301.12345 2302.54321 2303.98765; do
  research-reproduce analyze $id -o "results/$id.json"
done
```

### Workflow 4: Web Interface
```bash
# Launch web UI for easy browser-based access
research-reproduce web
```

## New Features You Should Try

### 🌐 Web Interface
The easiest way to use the tool:
```bash
research-reproduce web
```
- No CLI knowledge needed
- Visual progress indicators
- Easy repository selection
- Built-in GPU checker

### ⚡ Caching
Automatically speeds up repeated operations by 10-100x:
```bash
# First run: ~3 seconds
research-reproduce analyze 2301.12345

# Second run: ~0.1 seconds (cached!)
research-reproduce analyze 2301.12345
```

### 🎮 GPU Detection
Automatically detects and reports GPU availability:
```bash
research-reproduce gpu
```

### 🔍 Smart Error Diagnosis
Errors now include helpful suggestions:
```
❌ Error: ModuleNotFoundError: No module named 'torch'

🔧 Suggested Fixes:
  1. Install the missing package: pip install torch
  2. Try installing with conda: conda install torch
  3. Check if package name has changed
```

### 🔄 Auto-Retry
Network failures automatically retry with exponential backoff:
- Failed GitHub API call? → Automatic retry
- Papers with Code timeout? → Automatic retry
- Connection error? → Automatic retry

## Configuration

### Set GitHub Token (Recommended)
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### Custom Work Directory
```bash
research-reproduce reproduce 2301.12345 --work-dir ./my_experiments
```

### Verbose Output
```bash
research-reproduce reproduce 2301.12345 --verbose
```

## Tips for Success

1. **Start with Analysis**
   ```bash
   research-reproduce analyze <paper>
   ```
   Check if code exists before attempting full reproduction

2. **Use the Web Interface for Demos**
   ```bash
   research-reproduce web --share
   ```
   Creates a public link you can share

3. **Check GPU First for ML Papers**
   ```bash
   research-reproduce gpu
   ```
   Verify GPU availability before running ML papers

4. **Enable Caching for Batch Jobs**
   Already enabled by default! Just run multiple times.

5. **Read Error Suggestions**
   Error messages now include specific fixes - read them!

## Troubleshooting

### "No repositories found"
```bash
# Try the web interface - it might find more sources
research-reproduce web
```

### Slow operations
```bash
# Check if cache is working
research-reproduce cache stats

# Clear and rebuild cache if needed
research-reproduce cache clear
```

### GPU not detected
```bash
# Verify nvidia-smi works
nvidia-smi

# Check GPU status
research-reproduce gpu
```

### Web interface won't start
```bash
# Install gradio
pip install gradio

# Try different port
research-reproduce web --port 8080
```

## Examples

### Example 1: Famous Paper
```bash
# Reproduce "Attention Is All You Need"
research-reproduce reproduce 1706.03762
```

### Example 2: Recent Paper
```bash
# Analyze a recent paper
research-reproduce analyze 2301.12345
```

### Example 3: Custom Config
```bash
# Full control
research-reproduce reproduce 2301.12345 \
  --timeout 7200 \
  --work-dir ./experiments \
  --verbose
```

### Example 4: Web Demo
```bash
# Launch web UI with public sharing
research-reproduce web --share
```

## What's Next?

- Read [FEATURES.md](FEATURES.md) for detailed feature overview
- Check [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for future enhancements
- See [CONTRIBUTING.md](CONTRIBUTING.md) to contribute

## Help

```bash
# General help
research-reproduce --help

# Command-specific help
research-reproduce reproduce --help
research-reproduce analyze --help
research-reproduce web --help
```

---

Happy reproducing! 🔬✨

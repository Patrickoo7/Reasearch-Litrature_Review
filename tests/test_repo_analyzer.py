"""
Test suite for repository analyzer
"""

import pytest
from pathlib import Path
import tempfile
import os
from research_reproducer.repo_analyzer import RepositoryAnalyzer


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create sample Python files
        (repo_path / "main.py").write_text("""
import torch
import numpy as np

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

if __name__ == "__main__":
    main()
""")

        # Create requirements.txt
        (repo_path / "requirements.txt").write_text("""
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
""")

        # Create README
        (repo_path / "README.md").write_text("""
# Test Project

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Testing
```bash
pytest tests/
```
""")

        yield repo_path


class TestRepositoryAnalyzer:

    def test_detect_languages(self, temp_repo):
        """Test language detection"""
        analyzer = RepositoryAnalyzer(temp_repo)
        languages = analyzer._detect_languages()

        assert 'Python' in languages

    def test_analyze_python_deps(self, temp_repo):
        """Test Python dependency analysis"""
        analyzer = RepositoryAnalyzer(temp_repo)
        deps = analyzer._analyze_python_deps()

        assert 'requirements.txt' in str(deps['requirements_files'][0])
        assert 'torch' in deps['packages']
        assert 'numpy' in deps['packages']
        assert 'pandas' in deps['packages']

    def test_find_python_entry_points(self, temp_repo):
        """Test finding Python entry points"""
        analyzer = RepositoryAnalyzer(temp_repo)
        entry_points = analyzer._find_python_entry_points()

        assert len(entry_points) > 0
        assert any('main.py' in ep['file'] for ep in entry_points)

    def test_check_gpu_requirement(self, temp_repo):
        """Test GPU requirement detection"""
        analyzer = RepositoryAnalyzer(temp_repo)
        requires_gpu = analyzer._check_gpu_requirement()

        # main.py contains torch.cuda
        assert requires_gpu is True

    def test_extract_commands_from_readme(self, temp_repo):
        """Test command extraction from README"""
        analyzer = RepositoryAnalyzer(temp_repo)
        commands = analyzer._extract_commands_from_readme('README.md')

        assert len(commands['run']) > 0
        assert len(commands['test']) > 0

    def test_full_analysis(self, temp_repo):
        """Test complete analysis"""
        analyzer = RepositoryAnalyzer(temp_repo)
        analysis = analyzer.analyze()

        assert analysis['languages'] == ['Python']
        assert analysis['dependencies']['python']['packages']
        assert analysis['gpu_required'] is True
        assert analysis['entry_points']
        assert analysis['readme'] == 'README.md'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

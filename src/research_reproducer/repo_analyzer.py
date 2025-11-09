"""
Repository Analyzer Module
Analyzes cloned repositories to detect dependencies, entry points, and structure
"""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set
import yaml

logger = logging.getLogger(__name__)


class RepositoryAnalyzer:
    """Analyze repository structure and dependencies"""

    def __init__(self, repo_path: Path):
        """
        Args:
            repo_path: Path to cloned repository
        """
        self.repo_path = Path(repo_path)
        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist: {repo_path}")

    def analyze(self) -> Dict:
        """
        Perform full repository analysis

        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'repo_path': str(self.repo_path),
            'languages': self._detect_languages(),
            'dependencies': {},
            'entry_points': [],
            'config_files': [],
            'readme': self._find_readme(),
            'docker_support': False,
            'container_files': [],
            'test_commands': [],
            'build_commands': [],
            'run_commands': [],
            'data_requirements': [],
            'gpu_required': False,
            'estimated_complexity': 'unknown',
        }

        # Detect Python dependencies
        if 'Python' in analysis['languages']:
            analysis['dependencies']['python'] = self._analyze_python_deps()
            analysis['entry_points'].extend(self._find_python_entry_points())
            analysis['gpu_required'] = self._check_gpu_requirement()

        # Detect JavaScript/Node dependencies
        if any(lang in analysis['languages'] for lang in ['JavaScript', 'TypeScript']):
            analysis['dependencies']['node'] = self._analyze_node_deps()
            analysis['entry_points'].extend(self._find_node_entry_points())

        # Detect R dependencies
        if 'R' in analysis['languages']:
            analysis['dependencies']['r'] = self._analyze_r_deps()

        # Check for Docker support
        dockerfiles = list(self.repo_path.glob('**/Dockerfile*'))
        docker_compose = list(self.repo_path.glob('**/docker-compose*.yml'))
        if dockerfiles or docker_compose:
            analysis['docker_support'] = True
            analysis['container_files'] = [str(f.relative_to(self.repo_path)) for f in dockerfiles + docker_compose]

        # Find config files
        analysis['config_files'] = self._find_config_files()

        # Extract commands from README
        if analysis['readme']:
            commands = self._extract_commands_from_readme(analysis['readme'])
            analysis['test_commands'] = commands.get('test', [])
            analysis['build_commands'] = commands.get('build', [])
            analysis['run_commands'] = commands.get('run', [])

        # Check for data requirements
        analysis['data_requirements'] = self._detect_data_requirements()

        # Estimate complexity
        analysis['estimated_complexity'] = self._estimate_complexity(analysis)

        return analysis

    def _detect_languages(self) -> List[str]:
        """Detect programming languages used in the repository"""
        languages = set()

        # File extension to language mapping
        ext_map = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript',
            '.tsx': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.r': 'R',
            '.R': 'R',
            '.jl': 'Julia',
            '.m': 'MATLAB',
            '.sh': 'Shell',
        }

        for ext, lang in ext_map.items():
            if list(self.repo_path.glob(f'**/*{ext}')):
                languages.add(lang)

        return sorted(list(languages))

    def _analyze_python_deps(self) -> Dict:
        """Analyze Python dependencies"""
        deps = {
            'requirements_files': [],
            'packages': [],
            'setup_py': False,
            'pyproject_toml': False,
            'conda_env': False,
        }

        # Check requirements.txt
        req_files = list(self.repo_path.glob('**/requirements*.txt'))
        for req_file in req_files:
            deps['requirements_files'].append(str(req_file.relative_to(self.repo_path)))
            try:
                with open(req_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            # Extract package name (before ==, >=, etc.)
                            pkg = re.split(r'[=<>!]', line)[0].strip()
                            if pkg:
                                deps['packages'].append(pkg)
            except Exception as e:
                logger.warning(f"Failed to read {req_file}: {e}")

        # Check setup.py
        setup_py = self.repo_path / 'setup.py'
        if setup_py.exists():
            deps['setup_py'] = True
            try:
                with open(setup_py) as f:
                    content = f.read()
                    # Try to extract install_requires
                    install_match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                    if install_match:
                        for line in install_match.group(1).split(','):
                            pkg = line.strip().strip('"\'')
                            if pkg:
                                pkg = re.split(r'[=<>!]', pkg)[0].strip()
                                deps['packages'].append(pkg)
            except Exception as e:
                logger.warning(f"Failed to parse setup.py: {e}")

        # Check pyproject.toml
        pyproject = self.repo_path / 'pyproject.toml'
        if pyproject.exists():
            deps['pyproject_toml'] = True

        # Check for conda environment files
        conda_files = list(self.repo_path.glob('**/environment*.yml')) + \
                     list(self.repo_path.glob('**/environment*.yaml'))
        if conda_files:
            deps['conda_env'] = True
            deps['conda_files'] = [str(f.relative_to(self.repo_path)) for f in conda_files]

            for conda_file in conda_files:
                try:
                    with open(conda_file) as f:
                        env_data = yaml.safe_load(f)
                        if env_data and 'dependencies' in env_data:
                            for dep in env_data['dependencies']:
                                if isinstance(dep, str):
                                    pkg = re.split(r'[=<>!]', dep)[0].strip()
                                    deps['packages'].append(pkg)
                                elif isinstance(dep, dict) and 'pip' in dep:
                                    for pip_pkg in dep['pip']:
                                        pkg = re.split(r'[=<>!]', pip_pkg)[0].strip()
                                        deps['packages'].append(pkg)
                except Exception as e:
                    logger.warning(f"Failed to parse {conda_file}: {e}")

        # Deduplicate packages
        deps['packages'] = sorted(list(set(deps['packages'])))

        return deps

    def _analyze_node_deps(self) -> Dict:
        """Analyze Node.js dependencies"""
        deps = {
            'package_json': False,
            'packages': [],
        }

        package_json = self.repo_path / 'package.json'
        if package_json.exists():
            deps['package_json'] = True
            try:
                with open(package_json) as f:
                    data = json.load(f)
                    packages = list(data.get('dependencies', {}).keys())
                    packages.extend(data.get('devDependencies', {}).keys())
                    deps['packages'] = sorted(list(set(packages)))
            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")

        return deps

    def _analyze_r_deps(self) -> Dict:
        """Analyze R dependencies"""
        deps = {
            'packages': [],
        }

        # Look for library() or require() calls
        r_files = list(self.repo_path.glob('**/*.R')) + list(self.repo_path.glob('**/*.r'))
        for r_file in r_files:
            try:
                with open(r_file) as f:
                    content = f.read()
                    # Find library() and require() calls
                    matches = re.findall(r'(?:library|require)\(["\']?(\w+)["\']?\)', content)
                    deps['packages'].extend(matches)
            except Exception as e:
                logger.debug(f"Failed to read {r_file}: {e}")

        deps['packages'] = sorted(list(set(deps['packages'])))
        return deps

    def _find_python_entry_points(self) -> List[Dict]:
        """Find potential Python entry points"""
        entry_points = []

        # Look for common entry point patterns
        candidates = [
            'main.py',
            'run.py',
            'train.py',
            'test.py',
            'demo.py',
            'example.py',
            'inference.py',
            'predict.py',
        ]

        for candidate in candidates:
            files = list(self.repo_path.glob(f'**/{candidate}'))
            for file in files:
                entry_points.append({
                    'file': str(file.relative_to(self.repo_path)),
                    'type': 'python_script',
                    'command': f'python {file.relative_to(self.repo_path)}',
                })

        # Check if there's a __main__.py (package entry point)
        main_files = list(self.repo_path.glob('**/__main__.py'))
        for main_file in main_files:
            package_dir = main_file.parent
            entry_points.append({
                'file': str(main_file.relative_to(self.repo_path)),
                'type': 'python_package',
                'command': f'python -m {package_dir.name}',
            })

        return entry_points

    def _find_node_entry_points(self) -> List[Dict]:
        """Find Node.js entry points"""
        entry_points = []

        package_json = self.repo_path / 'package.json'
        if package_json.exists():
            try:
                with open(package_json) as f:
                    data = json.load(f)

                    # Check main field
                    if 'main' in data:
                        entry_points.append({
                            'file': data['main'],
                            'type': 'node_main',
                            'command': f'node {data["main"]}',
                        })

                    # Check scripts
                    if 'scripts' in data:
                        for script_name, script_cmd in data['scripts'].items():
                            entry_points.append({
                                'script': script_name,
                                'type': 'npm_script',
                                'command': f'npm run {script_name}',
                                'raw_command': script_cmd,
                            })

            except Exception as e:
                logger.warning(f"Failed to parse package.json: {e}")

        return entry_points

    def _find_readme(self) -> Optional[str]:
        """Find and return path to README file"""
        readme_patterns = ['README.md', 'README', 'README.txt', 'readme.md', 'Readme.md']

        for pattern in readme_patterns:
            readme = self.repo_path / pattern
            if readme.exists():
                return str(readme.relative_to(self.repo_path))

        return None

    def _find_config_files(self) -> List[str]:
        """Find configuration files"""
        config_patterns = [
            'config.yaml', 'config.yml', 'config.json',
            'settings.yaml', 'settings.yml', 'settings.json',
            '.env.example', '.env.sample',
        ]

        config_files = []
        for pattern in config_patterns:
            files = list(self.repo_path.glob(f'**/{pattern}'))
            config_files.extend([str(f.relative_to(self.repo_path)) for f in files])

        return config_files

    def _extract_commands_from_readme(self, readme_path: str) -> Dict[str, List[str]]:
        """Extract commands from README file"""
        commands = {
            'test': [],
            'build': [],
            'run': [],
        }

        readme_full_path = self.repo_path / readme_path

        try:
            with open(readme_full_path) as f:
                content = f.read()

                # Find code blocks (```...```)
                code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', content, re.DOTALL)

                for block in code_blocks:
                    lines = block.strip().split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line or line.startswith('#'):
                            continue

                        # Categorize commands
                        if any(keyword in line.lower() for keyword in ['pytest', 'test', 'unittest']):
                            commands['test'].append(line)
                        elif any(keyword in line.lower() for keyword in ['build', 'compile', 'make']):
                            commands['build'].append(line)
                        elif any(keyword in line.lower() for keyword in ['python', 'node', 'npm start', 'run', 'execute']):
                            commands['run'].append(line)

        except Exception as e:
            logger.warning(f"Failed to extract commands from README: {e}")

        return commands

    def _detect_data_requirements(self) -> List[str]:
        """Detect if the code requires external data"""
        data_indicators = []

        # Look for data directories
        data_dirs = ['data', 'dataset', 'datasets', 'inputs', 'examples']
        for dir_name in data_dirs:
            if (self.repo_path / dir_name).exists():
                data_indicators.append(f"Found '{dir_name}' directory")

        # Look for download scripts
        download_scripts = list(self.repo_path.glob('**/download*.py')) + \
                          list(self.repo_path.glob('**/download*.sh'))
        if download_scripts:
            data_indicators.append(f"Found data download scripts: {[s.name for s in download_scripts]}")

        return data_indicators

    def _check_gpu_requirement(self) -> bool:
        """Check if code likely requires GPU"""
        gpu_keywords = ['torch.cuda', 'tensorflow.gpu', 'cupy', 'jax.gpu', 'device="cuda"', 'device=cuda']

        python_files = list(self.repo_path.glob('**/*.py'))
        for py_file in python_files[:20]:  # Check first 20 files for performance
            try:
                with open(py_file) as f:
                    content = f.read()
                    if any(keyword in content for keyword in gpu_keywords):
                        return True
            except Exception:
                pass

        return False

    def _estimate_complexity(self, analysis: Dict) -> str:
        """Estimate reproduction complexity"""
        score = 0

        # More dependencies = more complex
        total_deps = sum(len(deps.get('packages', [])) for deps in analysis['dependencies'].values())
        if total_deps > 50:
            score += 3
        elif total_deps > 20:
            score += 2
        elif total_deps > 10:
            score += 1

        # GPU requirement
        if analysis['gpu_required']:
            score += 2

        # Data requirements
        if analysis['data_requirements']:
            score += 2

        # Docker support reduces complexity
        if analysis['docker_support']:
            score -= 2

        # Clear entry points reduce complexity
        if analysis['entry_points']:
            score -= 1

        # Categorize
        if score <= 0:
            return 'low'
        elif score <= 3:
            return 'medium'
        else:
            return 'high'

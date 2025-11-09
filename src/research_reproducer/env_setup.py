"""
Environment Setup Module
Sets up isolated environments for running research code
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EnvironmentSetup:
    """Setup isolated environments for code execution"""

    def __init__(self, work_dir: Path):
        """
        Args:
            work_dir: Working directory for environments
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def setup_python_venv(
        self,
        repo_path: Path,
        analysis: Dict,
        env_name: str = 'venv',
        python_version: Optional[str] = None
    ) -> Dict:
        """
        Setup Python virtual environment

        Args:
            repo_path: Path to repository
            analysis: Repository analysis dict
            env_name: Name for the environment
            python_version: Python version (e.g., '3.9')

        Returns:
            Environment info dict
        """
        env_path = self.work_dir / env_name

        result = {
            'type': 'python_venv',
            'path': str(env_path),
            'success': False,
            'errors': [],
            'activation_command': None,
        }

        try:
            # Create virtual environment
            python_cmd = 'python3' if not python_version else f'python{python_version}'
            cmd = [python_cmd, '-m', 'venv', str(env_path)]

            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Created virtual environment at {env_path}")

            # Determine activation command
            if os.name == 'nt':  # Windows
                result['activation_command'] = str(env_path / 'Scripts' / 'activate.bat')
                pip_cmd = str(env_path / 'Scripts' / 'pip')
            else:  # Unix
                result['activation_command'] = f'source {env_path}/bin/activate'
                pip_cmd = str(env_path / 'bin' / 'pip')

            # Upgrade pip
            subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True, capture_output=True)

            # Install dependencies
            python_deps = analysis['dependencies'].get('python', {})

            # Install from requirements.txt
            if python_deps.get('requirements_files'):
                for req_file in python_deps['requirements_files']:
                    req_path = repo_path / req_file
                    if req_path.exists():
                        logger.info(f"Installing from {req_file}")
                        try:
                            subprocess.run(
                                [pip_cmd, 'install', '-r', str(req_path)],
                                check=True,
                                capture_output=True,
                                timeout=600
                            )
                        except subprocess.CalledProcessError as e:
                            error_msg = f"Failed to install from {req_file}: {e.stderr.decode()}"
                            result['errors'].append(error_msg)
                            logger.warning(error_msg)

            # Install from setup.py
            if python_deps.get('setup_py'):
                setup_path = repo_path / 'setup.py'
                if setup_path.exists():
                    logger.info("Installing from setup.py")
                    try:
                        subprocess.run(
                            [pip_cmd, 'install', '-e', str(repo_path)],
                            check=True,
                            capture_output=True,
                            timeout=600
                        )
                    except subprocess.CalledProcessError as e:
                        error_msg = f"Failed to install from setup.py: {e.stderr.decode()}"
                        result['errors'].append(error_msg)
                        logger.warning(error_msg)

            result['success'] = True
            result['python_path'] = str(env_path / 'bin' / 'python') if os.name != 'nt' else str(env_path / 'Scripts' / 'python.exe')

        except Exception as e:
            error_msg = f"Failed to setup Python environment: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)

        return result

    def setup_conda_env(
        self,
        repo_path: Path,
        analysis: Dict,
        env_name: str = 'research_env'
    ) -> Dict:
        """
        Setup Conda environment

        Args:
            repo_path: Path to repository
            analysis: Repository analysis dict
            env_name: Name for the environment

        Returns:
            Environment info dict
        """
        result = {
            'type': 'conda',
            'name': env_name,
            'success': False,
            'errors': [],
            'activation_command': f'conda activate {env_name}',
        }

        try:
            # Check if conda is available
            subprocess.run(['conda', '--version'], check=True, capture_output=True)

            python_deps = analysis['dependencies'].get('python', {})

            # If conda environment file exists, use it
            if python_deps.get('conda_env') and python_deps.get('conda_files'):
                env_file = repo_path / python_deps['conda_files'][0]
                logger.info(f"Creating conda environment from {env_file}")

                cmd = ['conda', 'env', 'create', '-f', str(env_file), '-n', env_name]
                subprocess.run(cmd, check=True, capture_output=True, timeout=600)

            else:
                # Create basic conda environment
                logger.info(f"Creating conda environment '{env_name}'")
                cmd = ['conda', 'create', '-n', env_name, 'python=3.9', '-y']
                subprocess.run(cmd, check=True, capture_output=True, timeout=300)

                # Install pip dependencies
                if python_deps.get('requirements_files'):
                    for req_file in python_deps['requirements_files']:
                        req_path = repo_path / req_file
                        if req_path.exists():
                            logger.info(f"Installing from {req_file}")
                            cmd = ['conda', 'run', '-n', env_name, 'pip', 'install', '-r', str(req_path)]
                            subprocess.run(cmd, check=True, capture_output=True, timeout=600)

            result['success'] = True

        except FileNotFoundError:
            error_msg = "Conda is not installed or not in PATH"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to setup Conda environment: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)

        return result

    def setup_docker_env(
        self,
        repo_path: Path,
        analysis: Dict,
        image_name: str = 'research_reproducer'
    ) -> Dict:
        """
        Build Docker environment

        Args:
            repo_path: Path to repository
            analysis: Repository analysis dict
            image_name: Name for Docker image

        Returns:
            Environment info dict
        """
        result = {
            'type': 'docker',
            'image': image_name,
            'success': False,
            'errors': [],
            'run_command': None,
        }

        try:
            # Check if Docker is available
            subprocess.run(['docker', '--version'], check=True, capture_output=True)

            # Find Dockerfile
            dockerfiles = [f for f in analysis['container_files'] if 'Dockerfile' in f]

            if not dockerfiles:
                error_msg = "No Dockerfile found"
                result['errors'].append(error_msg)
                logger.error(error_msg)
                return result

            dockerfile = dockerfiles[0]
            dockerfile_path = repo_path / dockerfile

            logger.info(f"Building Docker image from {dockerfile}")

            # Build Docker image
            cmd = [
                'docker', 'build',
                '-f', str(dockerfile_path),
                '-t', image_name,
                str(repo_path)
            ]

            process = subprocess.run(
                cmd,
                capture_output=True,
                timeout=1800  # 30 minutes
            )

            if process.returncode == 0:
                result['success'] = True
                result['run_command'] = f'docker run -it {image_name}'
                logger.info(f"Successfully built Docker image: {image_name}")
            else:
                error_msg = f"Docker build failed: {process.stderr.decode()}"
                result['errors'].append(error_msg)
                logger.error(error_msg)

        except FileNotFoundError:
            error_msg = "Docker is not installed or not in PATH"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except subprocess.TimeoutExpired:
            error_msg = "Docker build timed out (30 minutes)"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to setup Docker environment: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)

        return result

    def setup_node_env(
        self,
        repo_path: Path,
        analysis: Dict
    ) -> Dict:
        """
        Setup Node.js environment

        Args:
            repo_path: Path to repository
            analysis: Repository analysis dict

        Returns:
            Environment info dict
        """
        result = {
            'type': 'node',
            'success': False,
            'errors': [],
        }

        try:
            # Check if npm is available
            subprocess.run(['npm', '--version'], check=True, capture_output=True)

            node_deps = analysis['dependencies'].get('node', {})

            if node_deps.get('package_json'):
                logger.info("Installing Node.js dependencies")

                # Install dependencies
                subprocess.run(
                    ['npm', 'install'],
                    cwd=str(repo_path),
                    check=True,
                    capture_output=True,
                    timeout=600
                )

                result['success'] = True
                logger.info("Node.js dependencies installed successfully")

        except FileNotFoundError:
            error_msg = "npm is not installed or not in PATH"
            result['errors'].append(error_msg)
            logger.error(error_msg)
        except Exception as e:
            error_msg = f"Failed to setup Node.js environment: {e}"
            result['errors'].append(error_msg)
            logger.error(error_msg)

        return result

    def auto_setup(
        self,
        repo_path: Path,
        analysis: Dict,
        prefer_docker: bool = True,
        prefer_conda: bool = False
    ) -> Dict:
        """
        Automatically choose and setup the best environment

        Args:
            repo_path: Path to repository
            analysis: Repository analysis dict
            prefer_docker: Prefer Docker if available
            prefer_conda: Prefer Conda over venv for Python

        Returns:
            Environment info dict
        """
        # Try Docker first if preferred and available
        if prefer_docker and analysis.get('docker_support'):
            logger.info("Attempting Docker setup")
            result = self.setup_docker_env(repo_path, analysis)
            if result['success']:
                return result
            logger.warning("Docker setup failed, falling back to other methods")

        # Setup environments based on detected languages
        results = []

        if 'Python' in analysis['languages']:
            if prefer_conda:
                logger.info("Attempting Conda setup")
                result = self.setup_conda_env(repo_path, analysis)
                if result['success']:
                    results.append(result)
                else:
                    # Fallback to venv
                    logger.info("Conda setup failed, trying venv")
                    result = self.setup_python_venv(repo_path, analysis)
                    results.append(result)
            else:
                logger.info("Attempting Python venv setup")
                result = self.setup_python_venv(repo_path, analysis)
                results.append(result)

        if any(lang in analysis['languages'] for lang in ['JavaScript', 'TypeScript']):
            logger.info("Attempting Node.js setup")
            result = self.setup_node_env(repo_path, analysis)
            results.append(result)

        # Return combined results
        if results:
            # If any succeeded, consider it a success
            success = any(r['success'] for r in results)
            return {
                'type': 'multi',
                'environments': results,
                'success': success,
            }
        else:
            return {
                'type': 'none',
                'success': False,
                'errors': ['No compatible environment setup found'],
            }

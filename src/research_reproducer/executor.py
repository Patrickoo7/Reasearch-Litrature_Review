"""
Code Execution Engine
Executes research code with monitoring and error handling
"""

import logging
import os
import signal
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

logger = logging.getLogger(__name__)
console = Console()


class ExecutionResult:
    """Container for execution results"""

    def __init__(self):
        self.success = False
        self.exit_code = None
        self.stdout = ""
        self.stderr = ""
        self.execution_time = 0.0
        self.errors = []
        self.warnings = []
        self.start_time = None
        self.end_time = None
        self.timed_out = False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'exit_code': self.exit_code,
            'stdout': self.stdout,
            'stderr': self.stderr,
            'execution_time': self.execution_time,
            'errors': self.errors,
            'warnings': self.warnings,
            'start_time': str(self.start_time) if self.start_time else None,
            'end_time': str(self.end_time) if self.end_time else None,
            'timed_out': self.timed_out,
        }


class CodeExecutor:
    """Execute research code with monitoring"""

    def __init__(self, repo_path: Path, env_info: Dict, work_dir: Path):
        """
        Args:
            repo_path: Path to repository
            env_info: Environment setup info
            work_dir: Working directory for outputs
        """
        self.repo_path = Path(repo_path)
        self.env_info = env_info
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        command: str,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        stream_output: bool = True,
    ) -> ExecutionResult:
        """
        Execute a command

        Args:
            command: Command to execute
            timeout: Timeout in seconds (None for no timeout)
            capture_output: Capture stdout/stderr
            stream_output: Stream output to console in real-time

        Returns:
            ExecutionResult object
        """
        result = ExecutionResult()
        result.start_time = datetime.now()

        console.print(f"\n[bold cyan]Executing:[/bold cyan] {command}")
        console.print(f"[dim]Working directory: {self.repo_path}[/dim]")

        if timeout:
            console.print(f"[dim]Timeout: {timeout} seconds[/dim]")

        # Prepare environment variables
        env = os.environ.copy()

        # Add Python environment to PATH if applicable
        if self.env_info.get('type') == 'python_venv':
            venv_bin = Path(self.env_info['path']) / 'bin'
            env['PATH'] = f"{venv_bin}:{env.get('PATH', '')}"
            env['VIRTUAL_ENV'] = self.env_info['path']

        # Log output to file
        log_file = self.work_dir / f"execution_{int(time.time())}.log"

        try:
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE if capture_output else None,
                stderr=subprocess.PIPE if capture_output else None,
                cwd=str(self.repo_path),
                env=env,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Stream output if requested
            if capture_output and stream_output:
                stdout_lines = []
                stderr_lines = []

                def read_stdout():
                    for line in process.stdout:
                        stdout_lines.append(line)
                        console.print(line, end='')

                def read_stderr():
                    for line in process.stderr:
                        stderr_lines.append(line)
                        console.print(f"[red]{line}[/red]", end='')

                stdout_thread = threading.Thread(target=read_stdout)
                stderr_thread = threading.Thread(target=read_stderr)

                stdout_thread.start()
                stderr_thread.start()

                # Wait with timeout
                try:
                    process.wait(timeout=timeout)
                    stdout_thread.join(timeout=1)
                    stderr_thread.join(timeout=1)

                    result.stdout = ''.join(stdout_lines)
                    result.stderr = ''.join(stderr_lines)

                except subprocess.TimeoutExpired:
                    console.print(f"\n[yellow]⚠ Execution timed out after {timeout} seconds[/yellow]")
                    process.kill()
                    result.timed_out = True
                    result.errors.append(f"Execution timed out after {timeout} seconds")

            else:
                # Simple wait
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    if capture_output:
                        result.stdout = stdout or ""
                        result.stderr = stderr or ""
                except subprocess.TimeoutExpired:
                    console.print(f"\n[yellow]⚠ Execution timed out after {timeout} seconds[/yellow]")
                    process.kill()
                    stdout, stderr = process.communicate()
                    result.timed_out = True
                    result.errors.append(f"Execution timed out after {timeout} seconds")
                    if capture_output:
                        result.stdout = stdout or ""
                        result.stderr = stderr or ""

            result.exit_code = process.returncode
            result.success = (result.exit_code == 0) and not result.timed_out

        except Exception as e:
            error_msg = f"Execution failed: {str(e)}"
            result.errors.append(error_msg)
            logger.error(error_msg)

        result.end_time = datetime.now()
        result.execution_time = (result.end_time - result.start_time).total_seconds()

        # Write log file
        try:
            with open(log_file, 'w') as f:
                f.write(f"Command: {command}\n")
                f.write(f"Exit code: {result.exit_code}\n")
                f.write(f"Execution time: {result.execution_time:.2f}s\n")
                f.write(f"\n--- STDOUT ---\n{result.stdout}\n")
                f.write(f"\n--- STDERR ---\n{result.stderr}\n")

            console.print(f"\n[dim]Log saved to: {log_file}[/dim]")

        except Exception as e:
            logger.warning(f"Failed to write log file: {e}")

        # Display result
        if result.success:
            console.print(f"\n[green]✓ Execution completed successfully[/green]")
            console.print(f"[dim]Time: {result.execution_time:.2f}s[/dim]")
        else:
            console.print(f"\n[red]✗ Execution failed[/red]")
            if result.exit_code is not None:
                console.print(f"[red]Exit code: {result.exit_code}[/red]")

        # Analyze errors
        result.errors.extend(self._analyze_errors(result.stderr))
        result.warnings.extend(self._analyze_warnings(result.stderr))

        if result.errors:
            console.print(f"\n[red]Errors detected:[/red]")
            for error in result.errors[:5]:  # Show first 5
                console.print(f"  • {error}")

        return result

    def execute_docker(
        self,
        command: str,
        image_name: str,
        timeout: Optional[int] = None,
        mount_data: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute command in Docker container

        Args:
            command: Command to execute
            image_name: Docker image name
            timeout: Timeout in seconds
            mount_data: Path to mount as /data in container

        Returns:
            ExecutionResult object
        """
        docker_cmd = f"docker run --rm"

        # Add volume mounts
        docker_cmd += f" -v {self.repo_path}:/workspace"

        if mount_data:
            docker_cmd += f" -v {mount_data}:/data"

        # Add GPU support if nvidia-docker is available
        try:
            subprocess.run(['nvidia-smi'], check=True, capture_output=True)
            docker_cmd += " --gpus all"
            console.print("[green]✓[/green] GPU support enabled")
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass

        docker_cmd += f" -w /workspace {image_name} {command}"

        return self.execute(docker_cmd, timeout=timeout)

    def run_tests(self, test_commands: List[str], timeout: Optional[int] = None) -> List[ExecutionResult]:
        """
        Run test commands

        Args:
            test_commands: List of test commands
            timeout: Timeout per command in seconds

        Returns:
            List of ExecutionResult objects
        """
        console.print("\n[bold cyan]Running Tests[/bold cyan]")

        results = []
        for cmd in test_commands:
            result = self.execute(cmd, timeout=timeout)
            results.append(result)

            if not result.success:
                console.print(f"[yellow]⚠ Test failed: {cmd}[/yellow]")

        # Summary
        passed = sum(1 for r in results if r.success)
        console.print(f"\n[bold]Test Summary:[/bold] {passed}/{len(results)} passed")

        return results

    def _analyze_errors(self, stderr: str) -> List[str]:
        """Extract and analyze errors from stderr"""
        errors = []

        if not stderr:
            return errors

        lines = stderr.split('\n')

        # Common error patterns
        error_patterns = [
            'Error:',
            'ERROR:',
            'Exception:',
            'Traceback',
            'FAILED',
            'ImportError',
            'ModuleNotFoundError',
            'FileNotFoundError',
            'RuntimeError',
            'ValueError',
            'TypeError',
        ]

        for line in lines:
            for pattern in error_patterns:
                if pattern in line:
                    errors.append(line.strip())
                    break

        return list(set(errors))  # Deduplicate

    def _analyze_warnings(self, stderr: str) -> List[str]:
        """Extract warnings from stderr"""
        warnings = []

        if not stderr:
            return warnings

        lines = stderr.split('\n')

        for line in lines:
            if 'Warning:' in line or 'WARNING:' in line:
                warnings.append(line.strip())

        return list(set(warnings))

    def cleanup(self):
        """Cleanup resources"""
        # Future: implement cleanup logic
        pass

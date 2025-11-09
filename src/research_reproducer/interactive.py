"""
Interactive Question System
Asks intelligent questions to gather missing information
"""

import logging
from typing import Any, Dict, List, Optional

try:
    import inquirer
    from inquirer import errors
except ImportError:
    inquirer = None

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


class InteractiveSession:
    """Handle interactive questions and user input"""

    def __init__(self, analysis: Dict, paper_metadata: Dict):
        """
        Args:
            analysis: Repository analysis results
            paper_metadata: Paper metadata from ingestion
        """
        self.analysis = analysis
        self.paper_metadata = paper_metadata
        self.user_config = {}

    def gather_requirements(self) -> Dict:
        """
        Ask questions to gather all necessary information

        Returns:
            Configuration dictionary with user answers
        """
        console.print("\n[bold cyan]Configuration Assistant[/bold cyan]")
        console.print("I'll help you configure the environment to run this research code.\n")

        # Ask about repository selection if multiple found
        self._ask_repository_selection()

        # Ask about environment preferences
        self._ask_environment_preferences()

        # Ask about data requirements
        self._ask_data_requirements()

        # Ask about hardware resources
        self._ask_hardware_resources()

        # Ask about execution preferences
        self._ask_execution_preferences()

        # Ask about missing configurations
        self._ask_missing_configs()

        return self.user_config

    def _ask_repository_selection(self):
        """Ask user to select repository if multiple found"""
        repos = self.user_config.get('available_repos', [])

        if not repos:
            return

        if len(repos) == 1:
            console.print(f"[green]✓[/green] Using repository: {repos[0]['url']}")
            self.user_config['selected_repo'] = repos[0]
            return

        # Multiple repos found
        console.print(f"\n[yellow]![/yellow] Found {len(repos)} repositories:")

        table = Table(show_header=True)
        table.add_column("#", style="cyan")
        table.add_column("Repository", style="green")
        table.add_column("Stars", justify="right")
        table.add_column("Source", style="yellow")

        for idx, repo in enumerate(repos, 1):
            table.add_row(
                str(idx),
                repo['url'],
                str(repo.get('stars', 'N/A')),
                repo.get('source', 'unknown')
            )

        console.print(table)

        while True:
            choice = Prompt.ask(
                "\nSelect repository number",
                default="1"
            )
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(repos):
                    self.user_config['selected_repo'] = repos[idx]
                    console.print(f"[green]✓[/green] Selected: {repos[idx]['url']}")
                    break
                else:
                    console.print("[red]Invalid choice. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a number.[/red]")

    def _ask_environment_preferences(self):
        """Ask about environment setup preferences"""
        console.print("\n[bold]Environment Setup[/bold]")

        # Check what's available
        has_docker = self.analysis.get('docker_support', False)
        has_conda = self.analysis['dependencies'].get('python', {}).get('conda_env', False)

        options = []
        if has_docker:
            options.append("Docker (recommended)")
        if has_conda:
            options.append("Conda")
        options.append("Python venv")
        options.append("Let me choose manually")

        if has_docker:
            default = "Docker (recommended)"
        elif has_conda:
            default = "Conda"
        else:
            default = "Python venv"

        console.print(f"Available options: {', '.join(options)}")

        env_choice = Prompt.ask(
            "Preferred environment",
            choices=options,
            default=default
        )

        if "Docker" in env_choice:
            self.user_config['env_type'] = 'docker'
        elif "Conda" in env_choice:
            self.user_config['env_type'] = 'conda'
        elif "venv" in env_choice:
            self.user_config['env_type'] = 'venv'
        else:
            self.user_config['env_type'] = 'manual'

        console.print(f"[green]✓[/green] Will use: {self.user_config['env_type']}")

    def _ask_data_requirements(self):
        """Ask about data availability"""
        data_reqs = self.analysis.get('data_requirements', [])

        if not data_reqs:
            console.print("\n[green]✓[/green] No obvious data requirements detected")
            return

        console.print("\n[bold]Data Requirements[/bold]")
        console.print("This code appears to require external data:")

        for req in data_reqs:
            console.print(f"  • {req}")

        has_data = Confirm.ask("\nDo you have the required data available?", default=False)
        self.user_config['has_data'] = has_data

        if has_data:
            data_path = Prompt.ask(
                "Enter the path to your data directory",
                default="./data"
            )
            self.user_config['data_path'] = data_path
        else:
            console.print("[yellow]⚠[/yellow] You may need to download data before running")
            auto_download = Confirm.ask("Should I attempt to run data download scripts?", default=True)
            self.user_config['auto_download_data'] = auto_download

    def _ask_hardware_resources(self):
        """Ask about hardware availability"""
        console.print("\n[bold]Hardware Resources[/bold]")

        if self.analysis.get('gpu_required', False):
            console.print("[yellow]⚠[/yellow] This code appears to require GPU")

            has_gpu = Confirm.ask("Do you have GPU access?", default=False)
            self.user_config['has_gpu'] = has_gpu

            if not has_gpu:
                console.print("[yellow]⚠[/yellow] Warning: Code may fail or run very slowly without GPU")
                continue_anyway = Confirm.ask("Continue anyway?", default=True)
                self.user_config['continue_without_gpu'] = continue_anyway
        else:
            console.print("[green]✓[/green] No GPU required")

        # Ask about compute resources
        complexity = self.analysis.get('estimated_complexity', 'unknown')
        if complexity in ['medium', 'high']:
            console.print(f"\n[yellow]⚠[/yellow] Estimated complexity: {complexity}")
            console.print("This may take significant time and resources")

            timeout = Prompt.ask(
                "Maximum execution time (minutes)",
                default="30"
            )
            try:
                self.user_config['timeout_minutes'] = int(timeout)
            except ValueError:
                self.user_config['timeout_minutes'] = 30

    def _ask_execution_preferences(self):
        """Ask about how to execute the code"""
        console.print("\n[bold]Execution Preferences[/bold]")

        entry_points = self.analysis.get('entry_points', [])

        if not entry_points:
            console.print("[yellow]⚠[/yellow] No clear entry points found")
            custom_command = Prompt.ask(
                "Enter the command to run the code",
                default="python main.py"
            )
            self.user_config['run_command'] = custom_command
            return

        if len(entry_points) == 1:
            console.print(f"[green]✓[/green] Entry point: {entry_points[0].get('command', 'N/A')}")
            self.user_config['entry_point'] = entry_points[0]
            return

        # Multiple entry points
        console.print("Found multiple possible entry points:")

        for idx, ep in enumerate(entry_points, 1):
            console.print(f"  {idx}. {ep.get('command', ep.get('file', 'unknown'))}")

        choice = Prompt.ask(
            "Select entry point",
            default="1"
        )

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(entry_points):
                self.user_config['entry_point'] = entry_points[idx]
        except ValueError:
            self.user_config['entry_point'] = entry_points[0]

    def _ask_missing_configs(self):
        """Ask about missing configuration files"""
        config_files = self.analysis.get('config_files', [])

        if not config_files:
            return

        console.print("\n[bold]Configuration Files[/bold]")
        console.print("Found configuration files:")

        for config in config_files:
            console.print(f"  • {config}")

        needs_config = Confirm.ask(
            "\nDo any of these need to be customized before running?",
            default=False
        )

        if needs_config:
            console.print("[yellow]⚠[/yellow] Please review and update configuration files before execution")
            self.user_config['review_configs'] = True

    def display_summary(self):
        """Display summary of collected configuration"""
        console.print("\n[bold cyan]Configuration Summary[/bold cyan]")

        table = Table(show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        for key, value in self.user_config.items():
            if isinstance(value, dict):
                value = value.get('url', str(value))
            table.add_row(key.replace('_', ' ').title(), str(value))

        console.print(table)

        proceed = Confirm.ask("\nProceed with these settings?", default=True)
        return proceed

    def ask_simple_question(
        self,
        question: str,
        choices: Optional[List[str]] = None,
        default: Optional[str] = None
    ) -> str:
        """
        Ask a simple question

        Args:
            question: Question to ask
            choices: List of valid choices (optional)
            default: Default answer (optional)

        Returns:
            User's answer
        """
        if choices:
            return Prompt.ask(question, choices=choices, default=default)
        else:
            return Prompt.ask(question, default=default)

    def ask_yes_no(self, question: str, default: bool = True) -> bool:
        """
        Ask a yes/no question

        Args:
            question: Question to ask
            default: Default answer

        Returns:
            True for yes, False for no
        """
        return Confirm.ask(question, default=default)

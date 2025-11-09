"""
Main Orchestrator
Coordinates all modules to reproduce research papers
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import git
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .paper_ingestion import PaperIngestion
from .repo_finder import RepositoryFinder
from .repo_analyzer import RepositoryAnalyzer
from .env_setup import EnvironmentSetup
from .interactive import InteractiveSession
from .executor import CodeExecutor
from .cache import ReproducerCache
from .ai_agent import get_default_agent, AIAgent

logger = logging.getLogger(__name__)
console = Console()


class ReproductionOrchestrator:
    """Main orchestrator for reproducing research papers"""

    def __init__(self, work_dir: str = './reproductions', github_token: Optional[str] = None, use_cache: bool = True, use_ai: bool = False):
        """
        Args:
            work_dir: Working directory for reproductions
            github_token: GitHub API token
            use_cache: Enable caching for faster repeated operations
            use_ai: Enable AI agent assistance
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.paper_ingestion = PaperIngestion()
        self.repo_finder = RepositoryFinder(github_token=github_token)

        # Initialize cache
        self.use_cache = use_cache
        self.cache = ReproducerCache() if use_cache else None

        # Initialize AI agent
        self.use_ai = use_ai
        self.ai_agent = None
        if use_ai:
            self.ai_agent = get_default_agent()
            if self.ai_agent:
                console.print("[dim]AI assistant enabled[/dim]")
            else:
                console.print("[yellow]⚠ AI assistant not available. Install Ollama or set API keys.[/yellow]")

        self.session_dir = None
        self.checkpoint_file = None
        self.report = {
            'paper': {},
            'repositories': [],
            'analysis': {},
            'environment': {},
            'execution': {},
            'ai_insights': {},
            'success': False,
            'timestamp': None,
        }

    def reproduce_from_pdf(self, pdf_path: str, **kwargs) -> Dict:
        """
        Reproduce research from PDF file

        Args:
            pdf_path: Path to PDF file
            **kwargs: Additional options

        Returns:
            Reproduction report
        """
        console.print(f"\n[bold cyan]Research Reproducer[/bold cyan]")
        console.print(f"Starting reproduction from PDF: {pdf_path}\n")

        # Extract paper metadata
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting paper metadata...", total=None)
            paper_metadata = self.paper_ingestion.extract_from_pdf(pdf_path)
            progress.update(task, completed=True)

        self.report['paper'] = paper_metadata

        console.print(f"[green]✓[/green] Paper: {paper_metadata.get('title', 'Unknown')}")

        return self._reproduce_from_metadata(paper_metadata, **kwargs)

    def reproduce_from_arxiv(self, arxiv_id: str, **kwargs) -> Dict:
        """
        Reproduce research from arXiv ID

        Args:
            arxiv_id: arXiv paper ID
            **kwargs: Additional options

        Returns:
            Reproduction report
        """
        console.print(f"\n[bold cyan]Research Reproducer[/bold cyan]")
        console.print(f"Starting reproduction from arXiv: {arxiv_id}\n")

        # Try to get from cache first
        paper_metadata = None
        if self.use_cache:
            paper_metadata = self.cache.get_paper_metadata(arxiv_id)
            if paper_metadata:
                console.print(f"[dim]Using cached paper metadata[/dim]")

        # Fetch paper metadata if not cached
        if not paper_metadata:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Fetching paper from arXiv...", total=None)
                paper_metadata = self.paper_ingestion.extract_from_arxiv(arxiv_id)
                progress.update(task, completed=True)

            # Cache the result
            if self.use_cache:
                self.cache.set_paper_metadata(arxiv_id, paper_metadata)

        self.report['paper'] = paper_metadata

        console.print(f"[green]✓[/green] Paper: {paper_metadata.get('title', 'Unknown')}")

        return self._reproduce_from_metadata(paper_metadata, **kwargs)

    def reproduce_from_url(self, url: str, **kwargs) -> Dict:
        """
        Reproduce research from URL

        Args:
            url: Paper URL
            **kwargs: Additional options

        Returns:
            Reproduction report
        """
        console.print(f"\n[bold cyan]Research Reproducer[/bold cyan]")
        console.print(f"Starting reproduction from URL: {url}\n")

        # Extract paper metadata
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting paper metadata...", total=None)
            paper_metadata = self.paper_ingestion.extract_from_url(url)
            progress.update(task, completed=True)

        self.report['paper'] = paper_metadata

        return self._reproduce_from_metadata(paper_metadata, **kwargs)

    def _reproduce_from_metadata(
        self,
        paper_metadata: Dict,
        interactive: bool = True,
        auto_install: bool = True,
        timeout: Optional[int] = None,
        **kwargs
    ) -> Dict:
        """
        Core reproduction logic

        Args:
            paper_metadata: Paper metadata
            interactive: Use interactive mode
            auto_install: Automatically install dependencies
            timeout: Execution timeout in seconds
            **kwargs: Additional options

        Returns:
            Reproduction report
        """
        self.report['timestamp'] = datetime.now().isoformat()

        # Create session directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        paper_name = paper_metadata.get('title', 'unknown')[:50].replace(' ', '_')
        self.session_dir = self.work_dir / f"{paper_name}_{timestamp}"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.session_dir / '.checkpoint.json'

        console.print(f"[dim]Session directory: {self.session_dir}[/dim]\n")

        # Initialize checkpoint
        self._save_checkpoint('initialized', {'paper': paper_metadata})

        # AI paper explanation
        if self.use_ai and self.ai_agent:
            console.print("[bold]AI Paper Explanation[/bold]")
            try:
                explanation = self.ai_agent.explain_paper(paper_metadata)
                console.print(f"[dim]{explanation}[/dim]\n")
                self.report['ai_insights']['paper_explanation'] = explanation
            except Exception as e:
                logger.warning(f"AI explanation failed: {e}")

        # Step 1: Find repositories
        console.print("[bold]Step 1: Finding repositories[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Searching for code repositories...", total=None)
            repos = self.repo_finder.find_repositories(paper_metadata)
            progress.update(task, completed=True)

        if not repos:
            console.print("[red]✗ No repositories found[/red]")
            console.print("This paper may not have publicly available code.")
            self.report['success'] = False
            return self.report

        console.print(f"[green]✓[/green] Found {len(repos)} repository(ies)")
        self.report['repositories'] = repos

        # Step 2: Interactive session (if enabled)
        selected_repo = None
        user_config = {}

        if interactive:
            session = InteractiveSession(analysis={}, paper_metadata=paper_metadata)
            session.user_config['available_repos'] = repos

            # Basic questions before cloning
            session._ask_repository_selection()
            selected_repo = session.user_config.get('selected_repo')

            if not selected_repo:
                console.print("[red]✗ No repository selected[/red]")
                return self.report
        else:
            # Auto-select official or most starred repo
            selected_repo = self.repo_finder.get_official_repository(repos)

        console.print(f"\n[bold]Step 2: Cloning repository[/bold]")
        repo_path = self.session_dir / 'repo'

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Cloning {selected_repo['url']}...", total=None)
            try:
                git.Repo.clone_from(selected_repo['url'], repo_path)
                progress.update(task, completed=True)
                console.print(f"[green]✓[/green] Repository cloned")
            except Exception as e:
                console.print(f"[red]✗ Failed to clone repository: {e}[/red]")
                self.report['success'] = False
                return self.report

        # Step 3: Analyze repository
        console.print(f"\n[bold]Step 3: Analyzing repository[/bold]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing code structure...", total=None)
            analyzer = RepositoryAnalyzer(repo_path)
            analysis = analyzer.analyze()
            progress.update(task, completed=True)

        self.report['analysis'] = analysis

        console.print(f"[green]✓[/green] Analysis complete")
        console.print(f"  Languages: {', '.join(analysis['languages'])}")
        console.print(f"  Dependencies: {sum(len(d.get('packages', [])) for d in analysis['dependencies'].values())} packages")
        console.print(f"  Complexity: {analysis['estimated_complexity']}")

        # AI parameter suggestions
        if self.use_ai and self.ai_agent:
            console.print("\n[bold]AI Parameter Suggestions[/bold]")
            try:
                suggestions = self.ai_agent.suggest_parameters(analysis)
                console.print(f"[dim]{suggestions}[/dim]\n")
                self.report['ai_insights']['parameter_suggestions'] = suggestions
            except Exception as e:
                logger.warning(f"AI suggestions failed: {e}")

        # Continue interactive session with analysis
        if interactive:
            session.analysis = analysis
            remaining_config = session.gather_requirements()
            user_config.update(remaining_config)

            if not session.display_summary():
                console.print("[yellow]⚠ Reproduction cancelled by user[/yellow]")
                return self.report

        # Step 4: Setup environment
        console.print(f"\n[bold]Step 4: Setting up environment[/bold]")

        env_setup = EnvironmentSetup(self.session_dir / 'envs')

        if interactive and user_config.get('env_type'):
            env_type = user_config['env_type']
        else:
            env_type = 'auto'

        if env_type == 'docker' or (env_type == 'auto' and analysis.get('docker_support')):
            env_info = env_setup.setup_docker_env(repo_path, analysis)
        elif env_type == 'conda':
            env_info = env_setup.setup_conda_env(repo_path, analysis)
        elif env_type == 'venv':
            env_info = env_setup.setup_python_venv(repo_path, analysis)
        else:
            env_info = env_setup.auto_setup(repo_path, analysis)

        self.report['environment'] = env_info

        if not env_info.get('success'):
            console.print("[red]✗ Environment setup failed[/red]")
            if env_info.get('errors'):
                for error in env_info['errors']:
                    console.print(f"  • {error}")
            return self.report

        console.print(f"[green]✓[/green] Environment ready ({env_info.get('type', 'unknown')})")

        # Step 5: Execute code
        console.print(f"\n[bold]Step 5: Executing code[/bold]")

        executor = CodeExecutor(repo_path, env_info, self.session_dir / 'logs')

        # Determine what to run
        if interactive and user_config.get('run_command'):
            command = user_config['run_command']
        elif interactive and user_config.get('entry_point'):
            command = user_config['entry_point'].get('command')
        elif analysis.get('entry_points'):
            command = analysis['entry_points'][0].get('command')
        else:
            console.print("[yellow]⚠ No clear entry point found[/yellow]")
            console.print("Please run the code manually from:")
            console.print(f"  {repo_path}")
            return self.report

        # Execute
        exec_timeout = timeout or (user_config.get('timeout_minutes', 30) * 60)
        result = executor.execute(command, timeout=exec_timeout)

        self.report['execution'] = result.to_dict()
        self.report['success'] = result.success

        # AI error debugging
        if self.use_ai and self.ai_agent and not result.success:
            console.print("\n[bold]AI Error Analysis[/bold]")
            try:
                error_msg = result.stderr if result.stderr else result.stdout
                debug_help = self.ai_agent.debug_error(error_msg, code_context=f"Repository: {selected_repo['url']}")
                console.print(f"[dim]{debug_help}[/dim]\n")
                self.report['ai_insights']['error_analysis'] = debug_help
            except Exception as e:
                logger.warning(f"AI error debugging failed: {e}")

        # Step 6: Generate report
        self._save_report()

        # Final summary
        console.print("\n" + "="*60)
        if self.report['success']:
            console.print("[bold green]✓ Reproduction completed successfully![/bold green]")
        else:
            console.print("[bold red]✗ Reproduction failed[/bold red]")

        console.print(f"\nReport saved to: {self.session_dir / 'report.json'}")
        console.print("="*60)

        return self.report

    def _save_report(self):
        """Save reproduction report to file"""
        report_path = self.session_dir / 'report.json'

        try:
            with open(report_path, 'w') as f:
                json.dump(self.report, f, indent=2, default=str)

            logger.info(f"Report saved to {report_path}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")

    def _save_checkpoint(self, stage: str, data: Dict):
        """Save checkpoint for resuming interrupted reproductions"""
        if not self.checkpoint_file:
            return

        checkpoint = {}
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file) as f:
                    checkpoint = json.load(f)
            except:
                pass

        checkpoint[stage] = {
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        checkpoint['last_stage'] = stage

        try:
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")

    def _load_checkpoint(self) -> Dict:
        """Load checkpoint if exists"""
        if not self.checkpoint_file or not self.checkpoint_file.exists():
            return {}

        try:
            with open(self.checkpoint_file) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return {}

    def cleanup(self):
        """Cleanup resources"""
        # Future: implement cleanup logic
        pass

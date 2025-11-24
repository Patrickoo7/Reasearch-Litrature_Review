"""
Command Line Interface
Main entry point for the research reproducer tool
"""

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from .orchestrator import ReproductionOrchestrator

console = Console()


def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group()
@click.version_option(version='0.1.0')
def main():
    """
    Research Reproducer - Automatically reproduce research papers

    A tool for early and solo researchers to automatically find and run
    code from research papers.
    """
    pass


@main.command()
@click.argument('source', type=str)
@click.option('--type', 'source_type', type=click.Choice(['pdf', 'arxiv', 'url', 'auto']),
              default='auto', help='Type of source (auto-detect by default)')
@click.option('--work-dir', type=click.Path(), default='./reproductions',
              help='Working directory for reproductions')
@click.option('--github-token', envvar='GITHUB_TOKEN',
              help='GitHub API token (or set GITHUB_TOKEN env var)')
@click.option('--interactive/--no-interactive', default=True,
              help='Use interactive mode')
@click.option('--timeout', type=int, default=1800,
              help='Execution timeout in seconds (default: 1800)')
@click.option('--with-ai', is_flag=True, help='Enable AI assistant (requires Ollama or API keys)')
@click.option('--verbose', is_flag=True, help='Verbose output')
def reproduce(source, source_type, work_dir, github_token, interactive, timeout, with_ai, verbose):
    """
    Reproduce a research paper from various sources

    SOURCE can be:
    - Path to a PDF file
    - arXiv ID (e.g., 2301.12345)
    - URL to a paper

    Examples:

      # From PDF file
      research-reproduce reproduce paper.pdf

      # From arXiv
      research-reproduce reproduce 2301.12345
      research-reproduce reproduce arxiv:2301.12345

      # From URL
      research-reproduce reproduce https://arxiv.org/abs/2301.12345
    """
    setup_logging(verbose)

    # Auto-detect source type
    if source_type == 'auto':
        if source.startswith('http://') or source.startswith('https://'):
            source_type = 'url'
        elif source.endswith('.pdf') or Path(source).is_file():
            source_type = 'pdf'
        elif 'arxiv' in source.lower() or source.replace('.', '').replace('v', '').isdigit():
            source_type = 'arxiv'
            # Clean arxiv ID
            source = source.replace('arxiv:', '').replace('arXiv:', '')
        else:
            console.print("[red]Error: Could not auto-detect source type[/red]")
            console.print("Please specify --type explicitly")
            sys.exit(1)

    console.print(f"[dim]Source type: {source_type}[/dim]")

    # Create orchestrator
    orchestrator = ReproductionOrchestrator(
        work_dir=work_dir,
        github_token=github_token,
        use_ai=with_ai
    )

    try:
        # Run reproduction based on source type
        if source_type == 'pdf':
            if not Path(source).exists():
                console.print(f"[red]Error: PDF file not found: {source}[/red]")
                sys.exit(1)

            report = orchestrator.reproduce_from_pdf(
                source,
                interactive=interactive,
                timeout=timeout
            )

        elif source_type == 'arxiv':
            report = orchestrator.reproduce_from_arxiv(
                source,
                interactive=interactive,
                timeout=timeout
            )

        elif source_type == 'url':
            report = orchestrator.reproduce_from_url(
                source,
                interactive=interactive,
                timeout=timeout
            )

        # Exit with appropriate code
        if report.get('success'):
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠ Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.argument('paper_source', type=str)
@click.option('--output', '-o', type=click.Path(), help='Output file for report')
@click.option('--github-token', envvar='GITHUB_TOKEN',
              help='GitHub API token')
@click.option('--verbose', is_flag=True, help='Verbose output')
def analyze(paper_source, output, github_token, verbose):
    """
    Analyze a paper and find repositories without running code

    This is useful for quickly checking if a paper has associated code
    without going through the full reproduction process.
    """
    setup_logging(verbose)

    from .paper_ingestion import PaperIngestion
    from .repo_finder import RepositoryFinder

    console.print(f"\n[bold cyan]Research Paper Analysis[/bold cyan]\n")

    # Extract paper metadata
    ingestion = PaperIngestion()

    if paper_source.endswith('.pdf'):
        paper_metadata = ingestion.extract_from_pdf(paper_source)
    elif 'arxiv' in paper_source.lower() or paper_source.replace('.', '').isdigit():
        paper_metadata = ingestion.extract_from_arxiv(paper_source)
    else:
        paper_metadata = ingestion.extract_from_url(paper_source)

    console.print(f"[bold]Title:[/bold] {paper_metadata.get('title', 'Unknown')}")
    console.print(f"[bold]Authors:[/bold] {', '.join(paper_metadata.get('authors', [])[:3])}")

    if paper_metadata.get('arxiv_id'):
        console.print(f"[bold]arXiv ID:[/bold] {paper_metadata['arxiv_id']}")

    # Find repositories
    console.print(f"\n[bold]Finding repositories...[/bold]")
    finder = RepositoryFinder(github_token=github_token)
    repos = finder.find_repositories(paper_metadata)

    if not repos:
        console.print("[yellow]No repositories found[/yellow]")
        sys.exit(0)

    console.print(f"\n[green]Found {len(repos)} repository(ies):[/green]\n")

    from rich.table import Table
    table = Table(show_header=True)
    table.add_column("#", style="cyan")
    table.add_column("Repository", style="green")
    table.add_column("Stars", justify="right")
    table.add_column("Language", style="yellow")
    table.add_column("Source", style="dim")

    for idx, repo in enumerate(repos, 1):
        table.add_row(
            str(idx),
            repo['url'],
            str(repo.get('stars', 'N/A')),
            repo.get('language', 'N/A'),
            repo.get('source', 'unknown')
        )

    console.print(table)

    # Save report if requested
    if output:
        import json
        report = {
            'paper': paper_metadata,
            'repositories': repos,
        }
        with open(output, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        console.print(f"\n[dim]Report saved to: {output}[/dim]")


@main.command()
@click.argument('repo_path', type=click.Path(exists=True))
@click.option('--verbose', is_flag=True, help='Verbose output')
def inspect(repo_path, verbose):
    """
    Inspect a cloned repository and analyze its structure

    Useful for understanding what dependencies and entry points exist
    in a repository you've already cloned.
    """
    setup_logging(verbose)

    from .repo_analyzer import RepositoryAnalyzer
    from rich.table import Table
    from rich.panel import Panel

    console.print(f"\n[bold cyan]Repository Inspection[/bold cyan]\n")

    analyzer = RepositoryAnalyzer(Path(repo_path))
    analysis = analyzer.analyze()

    # Display results
    console.print(Panel(f"[bold]{repo_path}[/bold]", title="Repository Path"))

    console.print("\n[bold]Languages:[/bold]")
    console.print(f"  {', '.join(analysis['languages'])}")

    console.print(f"\n[bold]Complexity:[/bold] {analysis['estimated_complexity']}")

    if analysis.get('docker_support'):
        console.print(f"\n[green]✓ Docker support detected[/green]")

    if analysis.get('gpu_required'):
        console.print(f"\n[yellow]⚠ GPU likely required[/yellow]")

    # Dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    for lang, deps in analysis['dependencies'].items():
        packages = deps.get('packages', [])
        if packages:
            console.print(f"  {lang}: {len(packages)} packages")
            if verbose:
                for pkg in packages[:10]:
                    console.print(f"    - {pkg}")
                if len(packages) > 10:
                    console.print(f"    ... and {len(packages) - 10} more")

    # Entry points
    if analysis.get('entry_points'):
        console.print("\n[bold]Entry Points:[/bold]")
        for ep in analysis['entry_points']:
            console.print(f"  • {ep.get('command', ep.get('file', 'unknown'))}")

    # Data requirements
    if analysis.get('data_requirements'):
        console.print("\n[bold]Data Requirements:[/bold]")
        for req in analysis['data_requirements']:
            console.print(f"  • {req}")


@main.command()
@click.option('--share', is_flag=True, help='Create public share link')
@click.option('--port', type=int, default=7860, help='Port to run on')
@click.option('--github-token', envvar='GITHUB_TOKEN', help='GitHub API token')
@click.option('--with-ai', is_flag=True, help='Enable AI assistant')
def web(share, port, github_token, with_ai):
    """
    Launch web interface for browser-based access

    The web interface provides an easy-to-use GUI for reproducing papers.
    """
    try:
        from .web_interface import launch_web_interface

        console.print("\n[bold cyan]Launching Research Reproducer Web Interface[/bold cyan]")
        console.print(f"[dim]Server will run on http://localhost:{port}[/dim]\n")
        if with_ai:
            console.print("[dim]AI assistant enabled[/dim]\n")

        launch_web_interface(
            github_token=github_token,
            share=share,
            port=port,
            use_ai=with_ai
        )

    except ImportError:
        console.print("[red]Error: gradio not installed[/red]")
        console.print("Install with: pip install gradio")
        sys.exit(1)


@main.group()
def cache():
    """Manage cache for faster operations"""
    pass


@cache.command('stats')
def cache_stats():
    """Show cache statistics"""
    from .cache import ReproducerCache

    c = ReproducerCache()
    stats = c.get_cache_stats()

    console.print("\n[bold cyan]Cache Statistics[/bold cyan]\n")
    console.print(f"Papers cached: {stats['papers']}")
    console.print(f"Repositories cached: {stats['repositories']}")
    console.print(f"Analyses cached: {stats['analysis']}")
    console.print(f"Cache directory: {stats['cache_dir']}\n")


@cache.command('clear')
@click.option('--type', 'cache_type', type=click.Choice(['papers', 'repositories', 'analysis', 'all']),
              default='all', help='Type of cache to clear')
@click.confirmation_option(prompt='Are you sure you want to clear the cache?')
def cache_clear(cache_type):
    """Clear cache"""
    from .cache import ReproducerCache

    c = ReproducerCache()

    if cache_type == 'all':
        c.clear_cache(None)
        console.print("[green]✓[/green] All cache cleared")
    else:
        c.clear_cache(cache_type)
        console.print(f"[green]✓[/green] {cache_type} cache cleared")


@main.command()
def gpu():
    """Check GPU availability and status"""
    from .gpu_utils import get_gpu_requirements_summary
    from rich.table import Table

    console.print("\n[bold cyan]GPU Status[/bold cyan]\n")

    gpu_info = get_gpu_requirements_summary()

    if gpu_info['gpu_available']:
        console.print(f"[green]✓[/green] {gpu_info['gpu_count']} GPU(s) available")
        console.print(f"Total Memory: {gpu_info['total_memory_gb']} GB")
        console.print(f"Free Memory: {gpu_info['free_memory_gb']} GB")

        if gpu_info['cuda_version']:
            console.print(f"CUDA Version: {gpu_info['cuda_version']}")

        if gpu_info['gpus']:
            console.print("\n[bold]GPUs:[/bold]")
            table = Table(show_header=True)
            table.add_column("Index", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Memory", style="yellow")

            for gpu in gpu_info['gpus']:
                table.add_row(
                    str(gpu['index']),
                    gpu['name'],
                    f"{gpu['memory_total_mb']} MB"
                )

            console.print(table)
    else:
        console.print("[yellow]⚠[/yellow] No GPU detected")
        console.print("\nML papers may fail or run slowly without GPU.")
        console.print("Consider using cloud GPU services (Colab, Paperspace, etc.)")


if __name__ == '__main__':
    main()

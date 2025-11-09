"""
Example usage of the Research Reproducer tool

This demonstrates how to use the tool programmatically
"""

from research_reproducer.orchestrator import ReproductionOrchestrator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Example 1: Reproduce from arXiv ID
def example_arxiv():
    print("\n" + "="*60)
    print("Example 1: Reproduce from arXiv ID")
    print("="*60)

    orchestrator = ReproductionOrchestrator(
        work_dir='./example_reproductions',
        github_token=None  # Set your token or use environment variable
    )

    # Use a paper with known code repository
    # This is just an example - replace with actual arXiv ID
    report = orchestrator.reproduce_from_arxiv(
        '2301.12345',
        interactive=False,  # Non-interactive for automated runs
        timeout=600  # 10 minutes
    )

    print(f"\nReproduction success: {report['success']}")

    if report.get('repositories'):
        print(f"Found {len(report['repositories'])} repositories")

    if report.get('analysis'):
        analysis = report['analysis']
        print(f"Languages: {analysis.get('languages', [])}")
        print(f"Complexity: {analysis.get('estimated_complexity', 'unknown')}")


# Example 2: Analyze only (no execution)
def example_analyze():
    print("\n" + "="*60)
    print("Example 2: Analyze paper without execution")
    print("="*60)

    from research_reproducer.paper_ingestion import PaperIngestion
    from research_reproducer.repo_finder import RepositoryFinder

    # Ingest paper
    ingestion = PaperIngestion()
    paper_metadata = ingestion.extract_from_arxiv('2301.12345')

    print(f"Paper: {paper_metadata.get('title', 'Unknown')}")
    print(f"Authors: {', '.join(paper_metadata.get('authors', [])[:3])}")

    # Find repositories
    finder = RepositoryFinder()
    repos = finder.find_repositories(paper_metadata)

    print(f"\nFound {len(repos)} repositories:")
    for repo in repos:
        print(f"  - {repo['url']} ({repo.get('stars', 0)} stars)")


# Example 3: Custom workflow
def example_custom_workflow():
    print("\n" + "="*60)
    print("Example 3: Custom workflow with specific options")
    print("="*60)

    orchestrator = ReproductionOrchestrator(work_dir='./custom_reproductions')

    # Custom configuration
    report = orchestrator.reproduce_from_arxiv(
        '2301.12345',
        interactive=False,
        auto_install=True,
        timeout=1800,  # 30 minutes
    )

    # Access detailed results
    if report['success']:
        execution = report.get('execution', {})
        print(f"\nExecution completed in {execution.get('execution_time', 0):.2f}s")

        if execution.get('stdout'):
            print("\nOutput preview:")
            print(execution['stdout'][:500])


if __name__ == '__main__':
    # Uncomment the examples you want to run

    # Example 1: Full reproduction
    # example_arxiv()

    # Example 2: Analysis only
    example_analyze()

    # Example 3: Custom workflow
    # example_custom_workflow()

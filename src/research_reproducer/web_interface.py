"""
Gradio Web Interface for Research Reproducer
Provides browser-based UI for easy paper reproduction
"""

import gradio as gr
import logging
from pathlib import Path
from typing import Optional, Tuple
import json

from .orchestrator import ReproductionOrchestrator
from .paper_ingestion import PaperIngestion
from .repo_finder import RepositoryFinder
from .gpu_utils import get_gpu_requirements_summary

logger = logging.getLogger(__name__)


class WebInterface:
    """Web interface for Research Reproducer"""

    def __init__(self, work_dir: str = './reproductions', github_token: Optional[str] = None, use_ai: bool = False):
        """
        Args:
            work_dir: Working directory for reproductions
            github_token: GitHub API token
            use_ai: Enable AI assistant
        """
        self.work_dir = work_dir
        self.github_token = github_token
        self.use_ai = use_ai

    def reproduce_paper(
        self,
        paper_source: str,
        source_type: str,
        use_cache: bool,
        use_ai: bool,
        timeout_minutes: int,
        progress=gr.Progress()
    ) -> Tuple[str, str, str]:
        """
        Reproduce a paper

        Returns:
            Tuple of (status_message, report_json, log_output)
        """
        try:
            progress(0, desc="Initializing...")

            orchestrator = ReproductionOrchestrator(
                work_dir=self.work_dir,
                github_token=self.github_token,
                use_cache=use_cache,
                use_ai=use_ai
            )

            progress(0.2, desc="Processing paper...")

            # Determine source type and run appropriate method
            if source_type == "Auto-detect":
                if paper_source.startswith('http'):
                    report = orchestrator.reproduce_from_url(
                        paper_source,
                        interactive=False,
                        timeout=timeout_minutes * 60
                    )
                elif paper_source.endswith('.pdf'):
                    report = orchestrator.reproduce_from_pdf(
                        paper_source,
                        interactive=False,
                        timeout=timeout_minutes * 60
                    )
                else:
                    report = orchestrator.reproduce_from_arxiv(
                        paper_source,
                        interactive=False,
                        timeout=timeout_minutes * 60
                    )
            elif source_type == "arXiv ID":
                progress(0.3, desc="Fetching from arXiv...")
                report = orchestrator.reproduce_from_arxiv(
                    paper_source,
                    interactive=False,
                    timeout=timeout_minutes * 60
                )
            elif source_type == "PDF":
                progress(0.3, desc="Processing PDF...")
                report = orchestrator.reproduce_from_pdf(
                    paper_source,
                    interactive=False,
                    timeout=timeout_minutes * 60
                )
            else:  # URL
                progress(0.3, desc="Fetching from URL...")
                report = orchestrator.reproduce_from_url(
                    paper_source,
                    interactive=False,
                    timeout=timeout_minutes * 60
                )

            progress(1.0, desc="Complete!")

            # Format results
            if report['success']:
                status = f"✅ SUCCESS - Paper reproduced successfully!\n\nPaper: {report['paper'].get('title', 'Unknown')}\nExecution time: {report.get('execution', {}).get('execution_time', 0):.2f}s"
            else:
                status = f"❌ FAILED - Reproduction failed\n\nPaper: {report['paper'].get('title', 'Unknown')}"

            report_json = json.dumps(report, indent=2, default=str)

            # Get logs
            logs = ""
            if orchestrator.session_dir:
                log_files = list(Path(orchestrator.session_dir / 'logs').glob('*.log'))
                if log_files:
                    with open(log_files[0]) as f:
                        logs = f.read()

            return status, report_json, logs

        except Exception as e:
            logger.error(f"Reproduction failed: {e}")
            return f"❌ ERROR: {str(e)}", "", ""

    def analyze_paper(
        self,
        paper_source: str,
        source_type: str,
        progress=gr.Progress()
    ) -> Tuple[str, str]:
        """
        Analyze a paper without running code

        Returns:
            Tuple of (summary, details_json)
        """
        try:
            progress(0, desc="Analyzing paper...")

            ingestion = PaperIngestion()
            finder = RepositoryFinder(github_token=self.github_token)

            # Get paper metadata
            if source_type == "arXiv ID" or (source_type == "Auto-detect" and not paper_source.startswith('http')):
                progress(0.3, desc="Fetching from arXiv...")
                metadata = ingestion.extract_from_arxiv(paper_source)
            elif source_type == "PDF":
                progress(0.3, desc="Processing PDF...")
                metadata = ingestion.extract_from_pdf(paper_source)
            else:
                progress(0.3, desc="Fetching from URL...")
                metadata = ingestion.extract_from_url(paper_source)

            # Find repositories
            progress(0.6, desc="Finding repositories...")
            repos = finder.find_repositories(metadata)

            progress(1.0, desc="Complete!")

            # Format summary
            summary = f"""
📄 **Title:** {metadata.get('title', 'Unknown')}

👥 **Authors:** {', '.join(metadata.get('authors', [])[:5])}

🔗 **Repositories Found:** {len(repos)}
"""

            if repos:
                summary += "\n**Top Repositories:**\n"
                for i, repo in enumerate(repos[:3], 1):
                    summary += f"{i}. {repo['url']} (⭐ {repo.get('stars', 0)})\n"

            details = {
                'paper': metadata,
                'repositories': repos
            }

            return summary, json.dumps(details, indent=2, default=str)

        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return f"❌ ERROR: {str(e)}", ""

    def check_gpu_status(self) -> str:
        """Check GPU availability"""
        gpu_info = get_gpu_requirements_summary()

        if gpu_info['gpu_available']:
            status = f"""
✅ **GPU Available**

- **GPUs:** {gpu_info['gpu_count']}
- **Total Memory:** {gpu_info['total_memory_gb']} GB
- **Free Memory:** {gpu_info['free_memory_gb']} GB
- **CUDA Version:** {gpu_info.get('cuda_version', 'Unknown')}
"""
            for i, gpu in enumerate(gpu_info['gpus'], 1):
                status += f"\nGPU {i}: {gpu['name']} ({gpu['memory_total_mb']} MB)"
        else:
            status = """
⚠️ **No GPU Detected**

No NVIDIA GPUs found. Papers requiring GPU may fail or run slowly.
"""

        return status

    def create_interface(self):
        """Create and return Gradio interface"""

        with gr.Blocks(title="Research Reproducer", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🔬 Research Reproducer

            Automatically find and run code from research papers
            """)

            with gr.Tabs():
                # Reproduce Tab
                with gr.Tab("🚀 Reproduce"):
                    gr.Markdown("Fully reproduce a research paper by finding and running its code")

                    with gr.Row():
                        with gr.Column():
                            paper_input = gr.Textbox(
                                label="Paper Source",
                                placeholder="arXiv ID, PDF path, or URL",
                                info="Enter arXiv ID (e.g., 2301.12345), path to PDF, or paper URL"
                            )

                            source_type = gr.Radio(
                                choices=["Auto-detect", "arXiv ID", "PDF", "URL"],
                                value="Auto-detect",
                                label="Source Type"
                            )

                            with gr.Row():
                                use_cache = gr.Checkbox(
                                    value=True,
                                    label="Use cache (faster for repeated runs)"
                                )

                                use_ai_checkbox = gr.Checkbox(
                                    value=self.use_ai,
                                    label="AI Assistant (requires Ollama or API keys)"
                                )

                            timeout = gr.Slider(
                                minimum=5,
                                maximum=120,
                                value=30,
                                step=5,
                                label="Timeout (minutes)"
                            )

                            reproduce_btn = gr.Button("🚀 Reproduce Paper", variant="primary")

                        with gr.Column():
                            status_output = gr.Textbox(
                                label="Status",
                                lines=6,
                                max_lines=10
                            )

                    with gr.Accordion("📊 Full Report", open=False):
                        report_output = gr.JSON(label="Reproduction Report")

                    with gr.Accordion("📝 Execution Logs", open=False):
                        logs_output = gr.Code(label="Logs", language="text")

                    reproduce_btn.click(
                        fn=self.reproduce_paper,
                        inputs=[paper_input, source_type, use_cache, use_ai_checkbox, timeout],
                        outputs=[status_output, report_output, logs_output]
                    )

                # Analyze Tab
                with gr.Tab("🔍 Analyze"):
                    gr.Markdown("Quick analysis to check if a paper has code (no execution)")

                    with gr.Row():
                        with gr.Column():
                            analyze_input = gr.Textbox(
                                label="Paper Source",
                                placeholder="arXiv ID, PDF path, or URL"
                            )

                            analyze_source_type = gr.Radio(
                                choices=["Auto-detect", "arXiv ID", "PDF", "URL"],
                                value="Auto-detect",
                                label="Source Type"
                            )

                            analyze_btn = gr.Button("🔍 Analyze", variant="primary")

                        with gr.Column():
                            analyze_summary = gr.Markdown(label="Summary")

                    with gr.Accordion("📋 Detailed Analysis", open=False):
                        analyze_details = gr.JSON(label="Details")

                    analyze_btn.click(
                        fn=self.analyze_paper,
                        inputs=[analyze_input, analyze_source_type],
                        outputs=[analyze_summary, analyze_details]
                    )

                # GPU Check Tab
                with gr.Tab("🎮 GPU Status"):
                    gr.Markdown("Check GPU availability for running ML papers")

                    gpu_check_btn = gr.Button("Check GPU Status")
                    gpu_status_output = gr.Markdown()

                    gpu_check_btn.click(
                        fn=self.check_gpu_status,
                        outputs=gpu_status_output
                    )

                # Help Tab
                with gr.Tab("ℹ️ Help"):
                    gr.Markdown("""
                    ## How to Use

                    ### Reproduce Tab
                    1. Enter a paper source (arXiv ID like `2301.12345`, PDF path, or URL)
                    2. Click "Reproduce Paper"
                    3. Wait for the process to complete
                    4. Review the results and logs

                    ### Analyze Tab
                    Quick analysis without running code - useful to check if a paper has code available

                    ### GPU Status
                    Check if you have GPU available for running ML models

                    ## Tips
                    - Enable cache for faster repeated runs
                    - Start with "Analyze" to check if code exists before full reproduction
                    - Increase timeout for complex papers
                    - Papers requiring GPU may fail without available GPU
                    - Enable AI Assistant for paper explanations, debugging help, and parameter suggestions
                    - AI Assistant requires Ollama (local, free) or API keys (OpenAI, Anthropic, HuggingFace)

                    ## Examples
                    - arXiv: `1706.03762` (Attention Is All You Need)
                    - URL: `https://arxiv.org/abs/2301.12345`
                    - PDF: `/path/to/paper.pdf`
                    """)

            return interface

    def launch(self, **kwargs):
        """Launch the web interface"""
        interface = self.create_interface()
        interface.launch(**kwargs)


def create_web_interface(work_dir: str = './reproductions', github_token: Optional[str] = None, use_ai: bool = False):
    """
    Create and return web interface

    Args:
        work_dir: Working directory for reproductions
        github_token: GitHub API token
        use_ai: Enable AI assistant

    Returns:
        Gradio interface
    """
    web = WebInterface(work_dir=work_dir, github_token=github_token, use_ai=use_ai)
    return web.create_interface()


def launch_web_interface(
    work_dir: str = './reproductions',
    github_token: Optional[str] = None,
    share: bool = False,
    port: int = 7860,
    use_ai: bool = False
):
    """
    Launch web interface

    Args:
        work_dir: Working directory
        github_token: GitHub API token
        share: Create public share link
        port: Port to run on
        use_ai: Enable AI assistant
    """
    web = WebInterface(work_dir=work_dir, github_token=github_token, use_ai=use_ai)
    web.launch(share=share, server_port=port)


if __name__ == "__main__":
    import os
    launch_web_interface(
        github_token=os.getenv('GITHUB_TOKEN'),
        share=False
    )

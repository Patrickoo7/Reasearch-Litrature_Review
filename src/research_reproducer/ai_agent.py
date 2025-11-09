"""
AI Agent Integration
Support for open-source AI agents to assist with reproduction
"""

import logging
import os
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class AIProvider(Enum):
    """Supported AI providers"""
    OLLAMA = "ollama"  # Local, free
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"
    CUSTOM = "custom"


class AIAgent:
    """AI agent to assist with reproduction tasks"""

    def __init__(
        self,
        provider: AIProvider = AIProvider.OLLAMA,
        model: str = None,
        api_key: str = None,
        base_url: str = None
    ):
        """
        Initialize AI agent

        Args:
            provider: AI provider to use
            model: Model name (e.g., 'llama2', 'gpt-4', 'claude-3-opus')
            api_key: API key for cloud providers
            base_url: Custom base URL for API
        """
        self.provider = provider
        self.model = model or self._get_default_model(provider)
        self.api_key = api_key or os.getenv(f"{provider.value.upper()}_API_KEY")
        self.base_url = base_url or self._get_default_url(provider)

        self.client = self._initialize_client()

    def _get_default_model(self, provider: AIProvider) -> str:
        """Get default model for provider"""
        defaults = {
            AIProvider.OLLAMA: "llama2",
            AIProvider.OPENAI: "gpt-4",
            AIProvider.ANTHROPIC: "claude-3-sonnet-20240229",
            AIProvider.HUGGINGFACE: "meta-llama/Llama-2-7b-chat-hf",
        }
        return defaults.get(provider, "llama2")

    def _get_default_url(self, provider: AIProvider) -> str:
        """Get default API URL"""
        urls = {
            AIProvider.OLLAMA: "http://localhost:11434",
            AIProvider.OPENAI: "https://api.openai.com/v1",
            AIProvider.ANTHROPIC: "https://api.anthropic.com",
            AIProvider.HUGGINGFACE: "https://api-inference.huggingface.co",
        }
        return urls.get(provider)

    def _initialize_client(self):
        """Initialize API client"""
        if self.provider == AIProvider.OLLAMA:
            return self._init_ollama()
        elif self.provider == AIProvider.OPENAI:
            return self._init_openai()
        elif self.provider == AIProvider.ANTHROPIC:
            return self._init_anthropic()
        elif self.provider == AIProvider.HUGGINGFACE:
            return self._init_huggingface()
        return None

    def _init_ollama(self):
        """Initialize Ollama client"""
        try:
            import requests
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama connection successful")
                return "ollama"
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
        return None

    def _init_openai(self):
        """Initialize OpenAI client"""
        try:
            import openai
            openai.api_key = self.api_key
            return openai
        except ImportError:
            logger.warning("OpenAI library not installed: pip install openai")
        return None

    def _init_anthropic(self):
        """Initialize Anthropic client"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            logger.warning("Anthropic library not installed: pip install anthropic")
        return None

    def _init_huggingface(self):
        """Initialize HuggingFace client"""
        try:
            from huggingface_hub import InferenceClient
            return InferenceClient(token=self.api_key)
        except ImportError:
            logger.warning("HuggingFace hub not installed: pip install huggingface-hub")
        return None

    def chat(self, message: str, context: str = None) -> str:
        """
        Send message to AI agent

        Args:
            message: User message
            context: Additional context

        Returns:
            AI response
        """
        if not self.client:
            return "AI agent not available. Install dependencies or configure API key."

        full_prompt = f"{context}\n\n{message}" if context else message

        try:
            if self.provider == AIProvider.OLLAMA:
                return self._chat_ollama(full_prompt)
            elif self.provider == AIProvider.OPENAI:
                return self._chat_openai(full_prompt)
            elif self.provider == AIProvider.ANTHROPIC:
                return self._chat_anthropic(full_prompt)
            elif self.provider == AIProvider.HUGGINGFACE:
                return self._chat_huggingface(full_prompt)
        except Exception as e:
            logger.error(f"AI chat failed: {e}")
            return f"Error: {e}"

    def _chat_ollama(self, prompt: str) -> str:
        """Chat with Ollama"""
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get('response', '')

        raise Exception(f"Ollama error: {response.status_code}")

    def _chat_openai(self, prompt: str) -> str:
        """Chat with OpenAI"""
        import openai

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )

        return response.choices[0].message.content

    def _chat_anthropic(self, prompt: str) -> str:
        """Chat with Anthropic"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _chat_huggingface(self, prompt: str) -> str:
        """Chat with HuggingFace"""
        response = self.client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=1000
        )
        return response

    # ===== Specialized Functions for Research Reproduction =====

    def explain_paper(self, paper_metadata: Dict) -> str:
        """Explain paper in simple terms"""
        prompt = f"""
Explain this research paper in simple terms:

Title: {paper_metadata.get('title')}
Authors: {', '.join(paper_metadata.get('authors', [])[:5])}
Abstract: {paper_metadata.get('abstract', 'N/A')[:500]}

Provide:
1. What problem does it solve?
2. Main contribution
3. Potential impact
4. Key technical approach

Keep it brief and accessible.
"""
        return self.chat(prompt)

    def debug_error(self, error_message: str, code_context: str = None) -> str:
        """Help debug reproduction errors"""
        context = f"Code context:\n{code_context}\n\n" if code_context else ""

        prompt = f"""
{context}Error during research reproduction:

{error_message}

Provide:
1. Root cause of the error
2. Step-by-step fix
3. Alternative approaches if fix is difficult
4. Common pitfalls to avoid

Be specific and actionable.
"""
        return self.chat(prompt)

    def suggest_parameters(self, analysis: Dict) -> str:
        """Suggest parameters for reproduction"""
        prompt = f"""
Analyzing code for reproduction:

Languages: {analysis.get('languages', [])}
Dependencies: {list(analysis.get('dependencies', {}).keys())}
Complexity: {analysis.get('estimated_complexity')}
GPU Required: {analysis.get('gpu_required')}

Suggest:
1. Optimal batch size
2. Memory requirements
3. Estimated run time
4. Recommended hardware
5. Potential issues to watch for
"""
        return self.chat(prompt)

    def compare_results(self, paper_claims: str, actual_results: str) -> str:
        """Compare reproduction results to paper claims"""
        prompt = f"""
Compare reproduction results to paper claims:

Paper Claims:
{paper_claims}

Actual Results:
{actual_results}

Analyze:
1. Are results consistent?
2. Significant differences?
3. Possible reasons for discrepancies
4. Is reproduction successful?
"""
        return self.chat(prompt)

    def generate_reproduction_guide(self, analysis: Dict, repo_url: str) -> str:
        """Generate step-by-step reproduction guide"""
        prompt = f"""
Generate a reproduction guide for:

Repository: {repo_url}
Languages: {analysis.get('languages', [])}
Dependencies: {analysis.get('dependencies', {})}
Entry points: {analysis.get('entry_points', [])}

Create a step-by-step guide including:
1. Prerequisites
2. Installation steps
3. Data preparation
4. Execution commands
5. Expected outputs
6. Troubleshooting tips

Format as markdown.
"""
        return self.chat(prompt)

    def recommend_papers(self, current_paper: Dict, user_interests: List[str]) -> str:
        """Recommend related papers"""
        prompt = f"""
Current paper: {current_paper.get('title')}
User interests: {', '.join(user_interests)}

Recommend 5 related papers that:
1. Build on this work
2. Are highly reproducible
3. Match user interests
4. Have available code

Format as list with title, reason, and link.
"""
        return self.chat(prompt)

    def extract_metrics(self, paper_text: str) -> str:
        """Extract key metrics from paper"""
        prompt = f"""
Extract key metrics and results from this paper excerpt:

{paper_text[:2000]}

Extract:
1. Reported metrics (accuracy, F1, BLEU, etc.)
2. Datasets used
3. Baseline comparisons
4. Main results

Format as structured data.
"""
        return self.chat(prompt)


# Convenience functions

def get_default_agent() -> AIAgent:
    """Get default AI agent (tries Ollama first, then others)"""

    # Try Ollama (free, local)
    agent = AIAgent(provider=AIProvider.OLLAMA)
    if agent.client:
        return agent

    # Try OpenAI
    if os.getenv('OPENAI_API_KEY'):
        agent = AIAgent(provider=AIProvider.OPENAI)
        if agent.client:
            return agent

    # Try Anthropic
    if os.getenv('ANTHROPIC_API_KEY'):
        agent = AIAgent(provider=AIProvider.ANTHROPIC)
        if agent.client:
            return agent

    logger.warning("No AI agent available. Install Ollama or set API keys.")
    return None


def quick_explain(paper_metadata: Dict) -> str:
    """Quick paper explanation"""
    agent = get_default_agent()
    if agent:
        return agent.explain_paper(paper_metadata)
    return "AI agent not available"


def quick_debug(error_message: str) -> str:
    """Quick error debugging"""
    agent = get_default_agent()
    if agent:
        return agent.debug_error(error_message)
    return "AI agent not available"

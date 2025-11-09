"""
Semantic Scholar Integration
Uses Semantic Scholar API for enhanced paper discovery and metadata
"""

import logging
from typing import Dict, List, Optional
import requests
from ..retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class SemanticScholarSource:
    """Interface to Semantic Scholar API"""

    BASE_URL = 'https://api.semanticscholar.org/graph/v1'

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Semantic Scholar API key (optional, for higher rate limits)
        """
        self.api_key = api_key
        self.headers = {}
        if api_key:
            self.headers['x-api-key'] = api_key

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def get_paper_by_arxiv(self, arxiv_id: str) -> Optional[Dict]:
        """
        Get paper metadata from Semantic Scholar by arXiv ID

        Args:
            arxiv_id: arXiv ID (e.g., '2301.12345')

        Returns:
            Paper metadata dict or None
        """
        try:
            url = f'{self.BASE_URL}/paper/arXiv:{arxiv_id}'
            params = {
                'fields': 'title,authors,abstract,year,citationCount,referenceCount,publicationDate,externalIds,openAccessPdf,repository'
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return self._format_paper_data(data)
            elif response.status_code == 404:
                logger.warning(f"Paper not found in Semantic Scholar: arXiv:{arxiv_id}")
            else:
                logger.warning(f"Semantic Scholar API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to fetch from Semantic Scholar: {e}")

        return None

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def search_paper(self, query: str, limit: int = 5) -> List[Dict]:
        """
        Search for papers on Semantic Scholar

        Args:
            query: Search query (paper title, keywords)
            limit: Maximum number of results

        Returns:
            List of paper metadata dicts
        """
        try:
            url = f'{self.BASE_URL}/paper/search'
            params = {
                'query': query,
                'limit': limit,
                'fields': 'title,authors,abstract,year,citationCount,externalIds,openAccessPdf,repository'
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                papers = data.get('data', [])
                return [self._format_paper_data(p) for p in papers]

        except Exception as e:
            logger.error(f"Semantic Scholar search failed: {e}")

        return []

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def get_citations(self, arxiv_id: str, limit: int = 10) -> List[Dict]:
        """
        Get papers that cite this paper

        Args:
            arxiv_id: arXiv ID
            limit: Maximum number of citations

        Returns:
            List of citing papers
        """
        try:
            url = f'{self.BASE_URL}/paper/arXiv:{arxiv_id}/citations'
            params = {
                'limit': limit,
                'fields': 'title,authors,year,externalIds,repository'
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                citations = data.get('data', [])
                return [self._format_paper_data(c.get('citingPaper', {})) for c in citations]

        except Exception as e:
            logger.error(f"Failed to get citations: {e}")

        return []

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def get_references(self, arxiv_id: str, limit: int = 10) -> List[Dict]:
        """
        Get papers referenced by this paper

        Args:
            arxiv_id: arXiv ID
            limit: Maximum number of references

        Returns:
            List of referenced papers
        """
        try:
            url = f'{self.BASE_URL}/paper/arXiv:{arxiv_id}/references'
            params = {
                'limit': limit,
                'fields': 'title,authors,year,externalIds,repository'
            }

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                references = data.get('data', [])
                return [self._format_paper_data(r.get('citedPaper', {})) for r in references]

        except Exception as e:
            logger.error(f"Failed to get references: {e}")

        return []

    def _format_paper_data(self, data: Dict) -> Dict:
        """Format Semantic Scholar data to standard format"""
        if not data:
            return {}

        formatted = {
            'title': data.get('title'),
            'authors': [a.get('name') for a in data.get('authors', [])],
            'abstract': data.get('abstract'),
            'year': data.get('year'),
            'citation_count': data.get('citationCount', 0),
            'reference_count': data.get('referenceCount', 0),
            'publication_date': data.get('publicationDate'),
        }

        # Extract external IDs
        external_ids = data.get('externalIds', {})
        if external_ids:
            formatted['arxiv_id'] = external_ids.get('ArXiv')
            formatted['doi'] = external_ids.get('DOI')
            formatted['pmid'] = external_ids.get('PubMed')

        # Extract repository URL if available
        repository = data.get('repository')
        if repository and isinstance(repository, dict):
            repo_url = repository.get('url')
            if repo_url and 'github.com' in repo_url:
                formatted['github_url'] = repo_url

        # Extract PDF URL
        pdf_info = data.get('openAccessPdf')
        if pdf_info and isinstance(pdf_info, dict):
            formatted['pdf_url'] = pdf_info.get('url')

        return formatted

    def find_code_repositories(self, paper_title: str) -> List[str]:
        """
        Find potential code repositories for a paper

        Args:
            paper_title: Paper title

        Returns:
            List of GitHub URLs
        """
        repos = []

        # Search for the paper
        papers = self.search_paper(paper_title, limit=3)

        for paper in papers:
            # Check if paper has repository link
            if paper.get('github_url'):
                repos.append(paper['github_url'])

            # Check citations for implementations
            arxiv_id = paper.get('arxiv_id')
            if arxiv_id:
                citations = self.get_citations(arxiv_id, limit=5)
                for citation in citations:
                    if citation.get('github_url'):
                        repos.append(citation['github_url'])

        return list(set(repos))  # Deduplicate

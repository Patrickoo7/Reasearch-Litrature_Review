"""
Repository Finder Module
Finds GitHub repositories associated with research papers using multiple sources
"""

import logging
from typing import Dict, List, Optional
import requests
from bs4 import BeautifulSoup
from .retry_utils import retry_with_backoff, RetryableError

logger = logging.getLogger(__name__)


class RepositoryFinder:
    """Find GitHub repositories associated with research papers"""

    def __init__(self, github_token: Optional[str] = None):
        """
        Args:
            github_token: GitHub API token for higher rate limits
        """
        self.github_token = github_token
        self.headers = {}
        if github_token:
            self.headers['Authorization'] = f'token {github_token}'

    def find_repositories(
        self,
        paper_metadata: Dict,
        use_papers_with_code: bool = True,
        use_google_search: bool = False
    ) -> List[Dict]:
        """
        Find repositories for a paper using multiple strategies

        Args:
            paper_metadata: Metadata from PaperIngestion
            use_papers_with_code: Search Papers with Code
            use_google_search: Use Google search (requires additional setup)

        Returns:
            List of repository info dicts with keys: url, stars, description, source
        """
        repos = []

        # Strategy 1: Use URLs already extracted from paper
        if paper_metadata.get('github_urls'):
            for url in paper_metadata['github_urls']:
                repo_info = self._get_repo_info(url)
                if repo_info:
                    repo_info['source'] = 'paper_text'
                    repos.append(repo_info)

        # Strategy 2: Search Papers with Code
        if use_papers_with_code and paper_metadata.get('title'):
            pwc_repos = self._search_papers_with_code(paper_metadata['title'])
            repos.extend(pwc_repos)

        # Strategy 3: Search by arXiv ID
        if paper_metadata.get('arxiv_id'):
            arxiv_repos = self._search_by_arxiv_id(paper_metadata['arxiv_id'])
            repos.extend(arxiv_repos)

        # Deduplicate by URL
        unique_repos = {}
        for repo in repos:
            url = repo['url'].rstrip('/').lower()
            if url not in unique_repos:
                unique_repos[url] = repo

        # Sort by stars (descending)
        sorted_repos = sorted(
            unique_repos.values(),
            key=lambda x: x.get('stars', 0),
            reverse=True
        )

        return sorted_repos

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def _get_repo_info(self, github_url: str) -> Optional[Dict]:
        """Get repository information from GitHub API"""
        try:
            # Extract owner and repo name from URL
            parts = github_url.rstrip('/').split('/')
            if len(parts) < 2:
                return None

            owner, repo = parts[-2], parts[-1]
            repo = repo.replace('.git', '')

            api_url = f'https://api.github.com/repos/{owner}/{repo}'
            response = requests.get(api_url, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                return {
                    'url': github_url,
                    'full_name': data['full_name'],
                    'description': data.get('description', ''),
                    'stars': data.get('stargazers_count', 0),
                    'language': data.get('language', ''),
                    'topics': data.get('topics', []),
                    'default_branch': data.get('default_branch', 'main'),
                    'archived': data.get('archived', False),
                    'last_updated': data.get('updated_at', ''),
                }
            elif response.status_code == 404:
                logger.warning(f"Repository not found: {github_url}")
            else:
                logger.warning(f"GitHub API error: {response.status_code}")

        except Exception as e:
            logger.error(f"Failed to get repo info for {github_url}: {e}")

        return None

    @retry_with_backoff(max_attempts=3, base_delay=2.0, exceptions=(requests.exceptions.RequestException,))
    def _search_papers_with_code(self, title: str) -> List[Dict]:
        """Search Papers with Code for repositories"""
        repos = []

        try:
            # Papers with Code search API
            search_url = 'https://paperswithcode.com/api/v1/papers/'
            params = {'q': title}

            response = requests.get(search_url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()
                results = data.get('results', [])

                for paper in results[:3]:  # Check top 3 results
                    paper_id = paper.get('id')
                    if paper_id:
                        # Get repositories for this paper
                        repo_url = f'https://paperswithcode.com/api/v1/papers/{paper_id}/repositories/'
                        repo_response = requests.get(repo_url, timeout=10)

                        if repo_response.status_code == 200:
                            repo_data = repo_response.json()
                            for repo in repo_data.get('results', []):
                                github_url = repo.get('url')
                                if github_url and 'github.com' in github_url:
                                    repo_info = self._get_repo_info(github_url)
                                    if repo_info:
                                        repo_info['source'] = 'papers_with_code'
                                        repo_info['is_official'] = repo.get('is_official', False)
                                        repo_info['framework'] = repo.get('framework', '')
                                        repos.append(repo_info)

        except Exception as e:
            logger.error(f"Papers with Code search failed: {e}")

        return repos

    def _search_by_arxiv_id(self, arxiv_id: str) -> List[Dict]:
        """Search for repositories using arXiv ID"""
        repos = []

        try:
            # Try CatalyzeX (links papers to code)
            catalyzex_url = f'https://www.catalyzex.com/_next/data/*/paper/arxiv/{arxiv_id}.json'

            # This is a simplified example - actual implementation may vary
            # based on CatalyzeX's current API/structure

        except Exception as e:
            logger.debug(f"CatalyzeX search failed: {e}")

        return repos

    def verify_repository_active(self, repo_url: str) -> bool:
        """Check if a repository is active and not archived"""
        repo_info = self._get_repo_info(repo_url)
        if not repo_info:
            return False

        return not repo_info.get('archived', False)

    def get_official_repository(self, repos: List[Dict]) -> Optional[Dict]:
        """Try to identify the official/primary repository"""
        if not repos:
            return None

        # Prioritize:
        # 1. Repos marked as official (from Papers with Code)
        # 2. Repos from paper text
        # 3. Most starred repo

        official = [r for r in repos if r.get('is_official')]
        if official:
            return official[0]

        from_paper = [r for r in repos if r.get('source') == 'paper_text']
        if from_paper:
            return from_paper[0]

        return repos[0]  # Most starred (already sorted)

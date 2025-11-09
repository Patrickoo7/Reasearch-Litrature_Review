"""
Caching module for paper metadata and repository analysis
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ReproducerCache:
    """Cache for paper metadata and analysis results"""

    def __init__(self, cache_dir: str = './.cache/research_reproducer'):
        """
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.paper_cache_dir = self.cache_dir / 'papers'
        self.repo_cache_dir = self.cache_dir / 'repositories'
        self.analysis_cache_dir = self.cache_dir / 'analysis'

        self.paper_cache_dir.mkdir(exist_ok=True)
        self.repo_cache_dir.mkdir(exist_ok=True)
        self.analysis_cache_dir.mkdir(exist_ok=True)

    def _get_cache_key(self, identifier: str) -> str:
        """Generate cache key from identifier"""
        return hashlib.md5(identifier.encode()).hexdigest()

    def _is_cache_valid(self, cache_file: Path, max_age_days: int = 7) -> bool:
        """Check if cache file exists and is not too old"""
        if not cache_file.exists():
            return False

        try:
            with open(cache_file) as f:
                data = json.load(f)

            cached_time = datetime.fromisoformat(data.get('cached_at', ''))
            age = datetime.now() - cached_time

            return age < timedelta(days=max_age_days)

        except Exception as e:
            logger.warning(f"Invalid cache file {cache_file}: {e}")
            return False

    def get_paper_metadata(self, paper_id: str) -> Optional[Dict]:
        """
        Get cached paper metadata

        Args:
            paper_id: Paper identifier (arxiv ID, URL, etc.)

        Returns:
            Cached metadata or None
        """
        cache_key = self._get_cache_key(paper_id)
        cache_file = self.paper_cache_dir / f"{cache_key}.json"

        if self._is_cache_valid(cache_file, max_age_days=30):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.info(f"Using cached paper metadata for {paper_id}")
                return data['metadata']
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def set_paper_metadata(self, paper_id: str, metadata: Dict):
        """
        Cache paper metadata

        Args:
            paper_id: Paper identifier
            metadata: Metadata to cache
        """
        cache_key = self._get_cache_key(paper_id)
        cache_file = self.paper_cache_dir / f"{cache_key}.json"

        try:
            data = {
                'paper_id': paper_id,
                'metadata': metadata,
                'cached_at': datetime.now().isoformat(),
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Cached paper metadata for {paper_id}")

        except Exception as e:
            logger.warning(f"Failed to cache metadata: {e}")

    def get_repository_info(self, repo_url: str) -> Optional[Dict]:
        """Get cached repository information"""
        cache_key = self._get_cache_key(repo_url)
        cache_file = self.repo_cache_dir / f"{cache_key}.json"

        if self._is_cache_valid(cache_file, max_age_days=7):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.info(f"Using cached repo info for {repo_url}")
                return data['repo_info']
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def set_repository_info(self, repo_url: str, repo_info: Dict):
        """Cache repository information"""
        cache_key = self._get_cache_key(repo_url)
        cache_file = self.repo_cache_dir / f"{cache_key}.json"

        try:
            data = {
                'repo_url': repo_url,
                'repo_info': repo_info,
                'cached_at': datetime.now().isoformat(),
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Cached repo info for {repo_url}")

        except Exception as e:
            logger.warning(f"Failed to cache repo info: {e}")

    def get_analysis(self, repo_url: str) -> Optional[Dict]:
        """Get cached repository analysis"""
        cache_key = self._get_cache_key(repo_url)
        cache_file = self.analysis_cache_dir / f"{cache_key}.json"

        if self._is_cache_valid(cache_file, max_age_days=3):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                logger.info(f"Using cached analysis for {repo_url}")
                return data['analysis']
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        return None

    def set_analysis(self, repo_url: str, analysis: Dict):
        """Cache repository analysis"""
        cache_key = self._get_cache_key(repo_url)
        cache_file = self.analysis_cache_dir / f"{cache_key}.json"

        try:
            data = {
                'repo_url': repo_url,
                'analysis': analysis,
                'cached_at': datetime.now().isoformat(),
            }

            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Cached analysis for {repo_url}")

        except Exception as e:
            logger.warning(f"Failed to cache analysis: {e}")

    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache

        Args:
            cache_type: Type of cache to clear ('papers', 'repositories', 'analysis', or None for all)
        """
        if cache_type == 'papers' or cache_type is None:
            for file in self.paper_cache_dir.glob('*.json'):
                file.unlink()
            logger.info("Cleared paper cache")

        if cache_type == 'repositories' or cache_type is None:
            for file in self.repo_cache_dir.glob('*.json'):
                file.unlink()
            logger.info("Cleared repository cache")

        if cache_type == 'analysis' or cache_type is None:
            for file in self.analysis_cache_dir.glob('*.json'):
                file.unlink()
            logger.info("Cleared analysis cache")

    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            'papers': len(list(self.paper_cache_dir.glob('*.json'))),
            'repositories': len(list(self.repo_cache_dir.glob('*.json'))),
            'analysis': len(list(self.analysis_cache_dir.glob('*.json'))),
            'cache_dir': str(self.cache_dir),
        }

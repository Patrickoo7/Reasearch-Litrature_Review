"""
OpenReview Integration
Access papers from OpenReview (NeurIPS, ICLR, etc.)
"""

import logging
from typing import Dict, List, Optional
import requests
from ..retry_utils import retry_with_backoff

logger = logging.getLogger(__name__)


class OpenReviewSource:
    """Interface to OpenReview API"""

    BASE_URL = 'https://api2.openreview.net'

    def __init__(self):
        self.headers = {'Content-Type': 'application/json'}

    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """
        Get paper from OpenReview by ID

        Args:
            paper_id: OpenReview paper ID

        Returns:
            Paper metadata dict
        """
        try:
            url = f'{self.BASE_URL}/notes'
            params = {'id': paper_id}

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                notes = data.get('notes', [])

                if notes:
                    return self._format_paper(notes[0])

        except Exception as e:
            logger.error(f"Failed to fetch from OpenReview: {e}")

        return None

    @retry_with_backoff(max_attempts=3, base_delay=2.0)
    def search_papers(self, query: str, venue: str = None, limit: int = 10) -> List[Dict]:
        """
        Search papers on OpenReview

        Args:
            query: Search query
            venue: Conference venue (e.g., 'NeurIPS.cc/2023', 'ICLR.cc/2024')
            limit: Maximum results

        Returns:
            List of paper dicts
        """
        try:
            url = f'{self.BASE_URL}/notes/search'
            params = {
                'term': query,
                'limit': limit,
                'type': 'all'
            }

            if venue:
                params['invitation'] = f'{venue}/Conference/-/Blind_Submission'

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                notes = data.get('notes', [])
                return [self._format_paper(note) for note in notes]

        except Exception as e:
            logger.error(f"OpenReview search failed: {e}")

        return []

    def _format_paper(self, note: Dict) -> Dict:
        """Format OpenReview note to standard format"""
        content = note.get('content', {})

        formatted = {
            'id': note.get('id'),
            'title': content.get('title', {}).get('value'),
            'abstract': content.get('abstract', {}).get('value'),
            'authors': content.get('authors', {}).get('value', []),
            'venue': note.get('invitation', '').split('/')[0] if '/' in note.get('invitation', '') else None,
            'pdf_url': f"https://openreview.net/pdf?id={note.get('id')}",
            'forum_url': f"https://openreview.net/forum?id={note.get('id')}",
        }

        # Extract GitHub URLs from content
        github_urls = []
        for field in ['code', 'github', 'repository']:
            if field in content and content[field].get('value'):
                url = content[field]['value']
                if 'github.com' in url:
                    github_urls.append(url)

        formatted['github_urls'] = github_urls

        return formatted

    def get_reviews(self, paper_id: str) -> List[Dict]:
        """Get reviews for a paper"""
        try:
            url = f'{self.BASE_URL}/notes'
            params = {'forum': paper_id}

            response = requests.get(url, params=params, headers=self.headers, timeout=10)

            if response.status_code == 200:
                data = response.json()
                notes = data.get('notes', [])

                reviews = []
                for note in notes:
                    if 'Official_Review' in note.get('invitation', ''):
                        reviews.append({
                            'rating': note.get('content', {}).get('rating', {}).get('value'),
                            'confidence': note.get('content', {}).get('confidence', {}).get('value'),
                            'review': note.get('content', {}).get('review', {}).get('value'),
                        })

                return reviews

        except Exception as e:
            logger.error(f"Failed to get reviews: {e}")

        return []

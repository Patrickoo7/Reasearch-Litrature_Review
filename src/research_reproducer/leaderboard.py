"""
Reproducibility Leaderboard
Track which papers are successfully reproduced
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ReproducibilityLeaderboard:
    """Track and rank papers by reproducibility"""

    def __init__(self, db_path: str = './.cache/leaderboard.json'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        """Load leaderboard data"""
        if self.db_path.exists():
            try:
                with open(self.db_path) as f:
                    return json.load(f)
            except:
                pass
        return {'papers': {}, 'stats': {}}

    def _save(self):
        """Save leaderboard data"""
        try:
            with open(self.db_path, 'w') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save leaderboard: {e}")

    def add_result(self, paper_id: str, result: Dict):
        """
        Add a reproduction result

        Args:
            paper_id: Paper identifier (arXiv ID, etc.)
            result: Reproduction result dict
        """
        if paper_id not in self.data['papers']:
            self.data['papers'][paper_id] = {
                'title': result.get('paper', {}).get('title'),
                'authors': result.get('paper', {}).get('authors', []),
                'attempts': [],
                'success_rate': 0,
                'avg_time': 0,
                'reproducibility_score': 0,
            }

        paper = self.data['papers'][paper_id]

        # Add attempt
        attempt = {
            'timestamp': datetime.now().isoformat(),
            'success': result.get('success', False),
            'execution_time': result.get('execution', {}).get('execution_time', 0),
            'complexity': result.get('analysis', {}).get('estimated_complexity'),
            'environment': result.get('environment', {}).get('type'),
        }
        paper['attempts'].append(attempt)

        # Update stats
        successful = sum(1 for a in paper['attempts'] if a['success'])
        paper['success_rate'] = (successful / len(paper['attempts'])) * 100

        times = [a['execution_time'] for a in paper['attempts'] if a['execution_time']]
        paper['avg_time'] = sum(times) / len(times) if times else 0

        # Calculate reproducibility score (0-100)
        paper['reproducibility_score'] = self._calculate_score(paper)

        self._save()
        self._update_global_stats()

    def _calculate_score(self, paper: Dict) -> float:
        """
        Calculate reproducibility score (0-100)

        Factors:
        - Success rate (50%)
        - Number of attempts (20%)
        - Consistency (20%)
        - Setup complexity (10%)
        """
        score = 0

        # Success rate (0-50 points)
        score += paper['success_rate'] * 0.5

        # Number of attempts (0-20 points)
        attempts = len(paper['attempts'])
        score += min(attempts * 4, 20)  # Max at 5 attempts

        # Consistency (0-20 points)
        recent_attempts = paper['attempts'][-5:]  # Last 5
        if len(recent_attempts) >= 2:
            recent_success = sum(1 for a in recent_attempts if a['success'])
            consistency = (recent_success / len(recent_attempts)) * 20
            score += consistency

        # Setup complexity (0-10 points)
        if paper.get('attempts'):
            complexities = [a.get('complexity') for a in paper['attempts'] if a.get('complexity')]
            if complexities:
                # Lower complexity = higher score
                complexity_map = {'low': 10, 'medium': 6, 'high': 2}
                avg_complexity = complexity_map.get(complexities[-1], 5)
                score += avg_complexity

        return min(score, 100)

    def _update_global_stats(self):
        """Update global statistics"""
        papers = self.data['papers']

        self.data['stats'] = {
            'total_papers': len(papers),
            'total_attempts': sum(len(p['attempts']) for p in papers.values()),
            'successful_reproductions': sum(
                sum(1 for a in p['attempts'] if a['success'])
                for p in papers.values()
            ),
            'avg_success_rate': sum(p['success_rate'] for p in papers.values()) / len(papers) if papers else 0,
            'highly_reproducible': sum(1 for p in papers.values() if p['reproducibility_score'] >= 80),
            'moderately_reproducible': sum(1 for p in papers.values() if 50 <= p['reproducibility_score'] < 80),
            'low_reproducible': sum(1 for p in papers.values() if p['reproducibility_score'] < 50),
        }

        self._save()

    def get_top_papers(self, limit: int = 10) -> List[Dict]:
        """Get top reproducible papers"""
        papers = [
            {
                'paper_id': pid,
                **paper
            }
            for pid, paper in self.data['papers'].items()
        ]

        # Sort by reproducibility score
        papers.sort(key=lambda x: x['reproducibility_score'], reverse=True)

        return papers[:limit]

    def get_paper_stats(self, paper_id: str) -> Dict:
        """Get statistics for a specific paper"""
        return self.data['papers'].get(paper_id, {})

    def get_global_stats(self) -> Dict:
        """Get global statistics"""
        return self.data['stats']

    def generate_badge(self, paper_id: str) -> str:
        """
        Generate a reproducibility badge (SVG URL)

        Returns:
            shields.io badge URL
        """
        paper = self.data['papers'].get(paper_id)
        if not paper:
            return None

        score = paper['reproducibility_score']

        # Determine color
        if score >= 80:
            color = 'brightgreen'
            label = 'Highly Reproducible'
        elif score >= 50:
            color = 'yellow'
            label = 'Moderately Reproducible'
        else:
            color = 'red'
            label = 'Low Reproducibility'

        badge_url = f"https://img.shields.io/badge/Reproducibility-{int(score)}%25-{color}?style=flat-square&logo=github"

        return badge_url

    def export_markdown_table(self, limit: int = 20) -> str:
        """Export top papers as markdown table"""
        papers = self.get_top_papers(limit)

        md = "# Reproducibility Leaderboard\n\n"
        md += "| Rank | Paper | Score | Success Rate | Attempts |\n"
        md += "|------|-------|-------|--------------|----------|\n"

        for i, paper in enumerate(papers, 1):
            title = paper.get('title', 'Unknown')[:50]
            score = int(paper['reproducibility_score'])
            success_rate = int(paper['success_rate'])
            attempts = len(paper['attempts'])

            md += f"| {i} | {title} | {score}/100 | {success_rate}% | {attempts} |\n"

        return md

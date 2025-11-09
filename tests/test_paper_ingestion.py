"""
Test suite for paper ingestion module
"""

import pytest
from pathlib import Path
from research_reproducer.paper_ingestion import PaperIngestion


class TestPaperIngestion:

    def setup_method(self):
        self.ingestion = PaperIngestion()

    def test_extract_github_urls(self):
        """Test GitHub URL extraction"""
        text = """
        Our code is available at https://github.com/user/repo
        Also see http://github.com/other/project
        """
        urls = self.ingestion._extract_github_urls(text)

        assert len(urls) == 2
        assert 'https://github.com/user/repo' in urls
        assert 'http://github.com/other/project' in urls

    def test_extract_arxiv_id(self):
        """Test arXiv ID extraction"""
        test_cases = [
            ("arXiv:2301.12345", "2301.12345"),
            ("arxiv.org/abs/2301.12345v2", "2301.12345v2"),
            ("See paper 2301.12345 for details", "2301.12345"),
        ]

        for text, expected in test_cases:
            result = self.ingestion._extract_arxiv_id(text)
            assert result == expected

    def test_github_url_cleaning(self):
        """Test that GitHub URLs are cleaned properly"""
        text = "https://github.com/user/repo.git/"
        urls = self.ingestion._extract_github_urls(text)

        assert urls[0] == "https://github.com/user/repo"

    @pytest.mark.integration
    def test_arxiv_fetch(self):
        """Test fetching from arXiv (integration test)"""
        # Use a known stable paper
        metadata = self.ingestion.extract_from_arxiv("1706.03762")

        assert metadata['title'] is not None
        assert len(metadata['authors']) > 0
        assert metadata['arxiv_id'] == "1706.03762"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

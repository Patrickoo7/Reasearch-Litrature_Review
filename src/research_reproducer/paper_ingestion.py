"""
Paper Ingestion Module
Handles parsing research papers (PDFs, arXiv links) and extracting metadata
"""

import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Union
import logging

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    PyPDF2 = None
    pdfplumber = None

try:
    import arxiv
except ImportError:
    arxiv = None

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class PaperIngestion:
    """Extract metadata and links from research papers"""

    def __init__(self):
        self.github_pattern = re.compile(
            r'https?://(?:www\.)?github\.com/[\w\-\.]+/[\w\-\.]+'
        )
        self.arxiv_pattern = re.compile(
            r'(?:arxiv\.org/(?:abs|pdf)/|arXiv:)\s*(\d+\.\d+(?:v\d+)?)'
        )

    def extract_from_pdf(self, pdf_path: Union[str, Path]) -> Dict:
        """Extract text and metadata from a PDF file"""
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        metadata = {
            'source': str(pdf_path),
            'title': None,
            'authors': [],
            'abstract': None,
            'github_urls': [],
            'arxiv_id': None,
            'full_text': '',
        }

        # Try PyPDF2 first for metadata
        if PyPDF2:
            try:
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.metadata:
                        metadata['title'] = reader.metadata.get('/Title', None)
                        metadata['authors'] = reader.metadata.get('/Author', '').split(', ')

                    # Extract text
                    for page in reader.pages:
                        metadata['full_text'] += page.extract_text() or ''
            except Exception as e:
                logger.warning(f"PyPDF2 extraction failed: {e}")

        # Try pdfplumber for better text extraction
        if pdfplumber:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                    if text_parts:
                        metadata['full_text'] = '\n'.join(text_parts)
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")

        # Extract GitHub URLs
        metadata['github_urls'] = self._extract_github_urls(metadata['full_text'])

        # Extract arXiv ID
        metadata['arxiv_id'] = self._extract_arxiv_id(metadata['full_text'])

        # Try to extract abstract (usually in first 2 pages)
        metadata['abstract'] = self._extract_abstract(metadata['full_text'])

        return metadata

    def extract_from_arxiv(self, arxiv_id: str) -> Dict:
        """Fetch paper metadata from arXiv"""
        if not arxiv:
            raise ImportError("arxiv package not installed. Install with: pip install arxiv")

        # Clean arxiv ID
        match = self.arxiv_pattern.search(arxiv_id)
        if match:
            arxiv_id = match.group(1)

        metadata = {
            'source': f'arXiv:{arxiv_id}',
            'title': None,
            'authors': [],
            'abstract': None,
            'github_urls': [],
            'arxiv_id': arxiv_id,
            'full_text': '',
            'pdf_url': None,
        }

        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())

            metadata['title'] = paper.title
            metadata['authors'] = [author.name for author in paper.authors]
            metadata['abstract'] = paper.summary
            metadata['pdf_url'] = paper.pdf_url

            # Extract GitHub URLs from abstract and comments
            full_text = f"{paper.title}\n{paper.summary}\n{paper.comment or ''}"
            metadata['github_urls'] = self._extract_github_urls(full_text)
            metadata['full_text'] = full_text

        except Exception as e:
            logger.error(f"Failed to fetch arXiv paper {arxiv_id}: {e}")
            raise

        return metadata

    def extract_from_url(self, url: str) -> Dict:
        """Extract paper information from a URL (arXiv, OpenReview, etc.)"""
        if 'arxiv.org' in url:
            # Extract arXiv ID and use arxiv API
            match = self.arxiv_pattern.search(url)
            if match:
                return self.extract_from_arxiv(match.group(1))

        # For other URLs, try to scrape the page
        metadata = {
            'source': url,
            'title': None,
            'authors': [],
            'abstract': None,
            'github_urls': [],
            'arxiv_id': None,
            'full_text': '',
        }

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Try to find title
            title_tag = soup.find('h1') or soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()

            # Extract GitHub URLs
            for link in soup.find_all('a', href=True):
                href = link['href']
                if 'github.com' in href:
                    metadata['github_urls'].extend(self._extract_github_urls(href))

            # Get page text
            metadata['full_text'] = soup.get_text()

            # Try to find abstract
            abstract_tag = soup.find('div', class_='abstract') or soup.find('abstract')
            if abstract_tag:
                metadata['abstract'] = abstract_tag.get_text().strip()

        except Exception as e:
            logger.error(f"Failed to fetch URL {url}: {e}")
            raise

        return metadata

    def _extract_github_urls(self, text: str) -> List[str]:
        """Extract GitHub repository URLs from text"""
        matches = self.github_pattern.findall(text)
        # Deduplicate and clean URLs
        urls = list(set(matches))
        # Remove .git suffix and trailing slashes
        urls = [url.rstrip('/').replace('.git', '') for url in urls]
        return urls

    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID from text"""
        match = self.arxiv_pattern.search(text)
        if match:
            return match.group(1)
        return None

    def _extract_abstract(self, text: str) -> Optional[str]:
        """Try to extract abstract from paper text"""
        # Look for common abstract markers
        abstract_patterns = [
            r'Abstract[\s\n]+(.+?)(?:\n\n|Introduction|1\.|Keywords)',
            r'ABSTRACT[\s\n]+(.+?)(?:\n\n|INTRODUCTION|1\.|KEYWORDS)',
        ]

        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                abstract = match.group(1).strip()
                # Clean up
                abstract = re.sub(r'\s+', ' ', abstract)
                if len(abstract) > 100:  # Sanity check
                    return abstract

        return None

    def download_pdf(self, url: str, output_path: Union[str, Path]) -> Path:
        """Download a PDF from a URL"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()

            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded PDF to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            raise

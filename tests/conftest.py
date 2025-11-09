"""
Test configuration
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (may be slow)"
    )
    config.addinivalue_line(
        "markers", "requires_docker: marks tests that require Docker"
    )
    config.addinivalue_line(
        "markers", "requires_conda: marks tests that require Conda"
    )

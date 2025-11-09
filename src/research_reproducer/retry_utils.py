"""
Retry utilities for handling transient failures
"""

import time
import logging
from typing import Callable, Any, Optional, Type, Tuple
from functools import wraps

logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Decorator for retrying functions with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        exceptions: Tuple of exceptions to catch and retry

    Example:
        @retry_with_backoff(max_attempts=3, base_delay=2.0)
        def fetch_data():
            return requests.get('https://api.example.com/data')
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 1
            delay = base_delay

            while attempt <= max_attempts:
                try:
                    return func(*args, **kwargs)

                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts")
                        raise

                    logger.warning(
                        f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )

                    time.sleep(delay)

                    # Exponential backoff with jitter
                    delay = min(delay * exponential_base, max_delay)
                    attempt += 1

            return None

        return wrapper
    return decorator


class RetryableError(Exception):
    """Exception for errors that should be retried"""
    pass


class NonRetryableError(Exception):
    """Exception for errors that should not be retried"""
    pass


def is_retryable_error(error: Exception) -> bool:
    """
    Determine if an error should trigger a retry

    Args:
        error: The exception to check

    Returns:
        True if error is retryable
    """
    import requests

    # Network errors are typically retryable
    retryable_types = (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        RetryableError,
    )

    if isinstance(error, retryable_types):
        return True

    # HTTP errors - retry on 5xx, not on 4xx
    if isinstance(error, requests.exceptions.HTTPError):
        if hasattr(error, 'response') and error.response is not None:
            status_code = error.response.status_code
            # Retry on server errors, not client errors
            return 500 <= status_code < 600

    return False


class SmartRetry:
    """Smart retry manager with configurable strategies"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with retry logic

        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function

        Returns:
            Function result

        Raises:
            Last exception if all retries fail
        """
        attempt = 1
        delay = self.base_delay
        last_exception = None

        while attempt <= self.max_attempts:
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info(f"Succeeded on attempt {attempt}")
                return result

            except Exception as e:
                last_exception = e

                if isinstance(e, NonRetryableError):
                    logger.error(f"Non-retryable error: {e}")
                    raise

                if not is_retryable_error(e) or attempt == self.max_attempts:
                    logger.error(f"Failed after {attempt} attempts: {e}")
                    raise

                logger.warning(
                    f"Attempt {attempt}/{self.max_attempts} failed: {e}. "
                    f"Retrying in {delay:.1f}s..."
                )

                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)
                attempt += 1

        raise last_exception


# Convenience function
def smart_retry(func: Callable, *args, **kwargs) -> Any:
    """Convenience function for one-off smart retries"""
    retry_manager = SmartRetry()
    return retry_manager.execute(func, *args, **kwargs)

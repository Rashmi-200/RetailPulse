import functools
import time
from typing import Callable, Tuple, Type
from src.utils.logger import get_logger

logger = get_logger(__name__)


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
) -> Callable:
    """Retry transient failures using exponential backoff and track retry count."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            delay = base_delay
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as exc:
                    attempt += 1
                    if attempt > max_retries:
                        logger.error(f"Function '{func.__name__}' failed after {max_retries} retries: {exc}")
                        raise
                    logger.warning(
                        f"Transient error in '{func.__name__}' (attempt {attempt}/{max_retries}): {exc}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    delay *= backoff_factor

        return wrapper

    return decorator

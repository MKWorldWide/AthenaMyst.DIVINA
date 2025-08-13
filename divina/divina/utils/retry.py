""
Retry utilities with exponential backoff for async operations.
"""
import asyncio
import random
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar, Union

from loguru import logger

# Type variable for the wrapped function's return type
T = TypeVar('T')

class MaxRetriesExceededError(Exception):
    """Raised when the maximum number of retries is exceeded."""
    pass

async def async_retry(
    func: Callable[..., Awaitable[T]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff: float = 2.0,
    jitter: float = 0.1,
    exceptions: Union[Type[Exception], tuple[Type[Exception], ...]] = Exception,
    log_retries: bool = True,
    **kwargs: Any
) -> T:
    """
    Retry an async function with exponential backoff.
    
    Args:
        func: The async function to retry
        max_retries: Maximum number of retry attempts (not including the initial try)
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff: Backoff multiplier (e.g., 2.0 for exponential backoff)
        jitter: Random jitter factor (0.0 to 1.0)
        exceptions: Exception type(s) to catch and retry on
        log_retries: Whether to log retry attempts
        **kwargs: Arguments to pass to the function
        
    Returns:
        The result of the function call if successful
        
    Raises:
        MaxRetriesExceededError: If max_retries is exceeded
        Exception: Any exception raised by the function that's not in the exceptions list
    """
    delay = initial_delay
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return await func(**kwargs)
        except exceptions as e:
            last_exception = e
            
            if attempt == max_retries:
                if log_retries:
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for {func.__name__}: {e}"
                    )
                raise MaxRetriesExceededError(
                    f"Max retries ({max_retries}) exceeded for {func.__name__}"
                ) from e
            
            # Calculate delay with jitter
            jitter_amount = random.uniform(1 - jitter, 1 + jitter)
            current_delay = min(delay * jitter_amount, max_delay)
            
            if log_retries:
                logger.warning(
                    f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__} "
                    f"with {type(e).__name__}: {e}. Retrying in {current_delay:.2f}s..."
                )
            
            await asyncio.sleep(current_delay)
            delay *= backoff
    
    # This should never be reached due to the raise in the except block
    raise MaxRetriesExceededError("Unexpected error in async_retry")

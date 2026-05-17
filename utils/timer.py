import logging
import time
from functools import wraps

logger = logging.getLogger(__name__)


def watch_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func_name = func.__name__
        logger.info(f"[START] {func_name}")
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"[END] {func_name} - Duration: {duration:.2f}s")
    return wrapper
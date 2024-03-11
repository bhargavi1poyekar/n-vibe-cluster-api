import datetime
import logging
import time
from typing import Any
from typing import Callable

logger = logging.getLogger("main")
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch.setFormatter(formatter)
logger.addHandler(ch)

logger.propagate = False


def time_and_log(fn: Callable) -> Any:
    """Timing a function exectution and logging it.

    Args:
        fn: The callable function to time

    Returns:
        The output of the function called.
    """

    def wrapper(*args, **kwargs):
        logger.info(f"Start: executing {fn.__name__}.")
        toc = time.time()

        result = fn(*args, **kwargs)
        tic = time.time()

        elapsed_time = tic - toc
        logger.info(f"End: function {fn.__name__} took {elapsed_time:.5f} seconds.")

        return result

    return wrapper

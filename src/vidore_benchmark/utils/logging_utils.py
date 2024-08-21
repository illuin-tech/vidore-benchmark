import logging
import sys

from loguru import logger


def setup_logging(log_level: str = "WARNING") -> None:
    """
    Setup the logging with loguru.
    """

    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logger.remove()  # remove the default logger
    logger.add(
        sink=sys.stderr,
        level=log_level.upper(),
        format="{time:MM/DD/YYYY HH:mm:ss} - {level} - {name}:{line} - {message}",
    )
    logger.info(f"Logging level set to {log_level.upper()}.")

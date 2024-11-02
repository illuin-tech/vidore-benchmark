import logging

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = "warning") -> None:
    """
    Setup logging configuration.
    """
    numeric_level = getattr(logging, log_level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        handlers=[logging.StreamHandler()],
    )
    logging.captureWarnings(True)
    logger.info("Logging level set to %s", log_level)

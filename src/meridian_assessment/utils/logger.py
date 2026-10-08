"""Logging setup and utilities for Meridian Vendor Assessment."""

import logging
import sys
from typing import Optional


def setup_logger(
    name: str = "meridian_assessment",
    level: Optional[str] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a standard logger for the application.

    Args:
        name: Logger name.
        level: Log level string (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
        log_format: Custom log format string.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        log_level_str = (level or "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logger.setLevel(log_level)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)

        formatter = logging.Formatter(
            log_format or "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# Default application logger instance
logger = setup_logger()

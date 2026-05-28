"""Simple logging helper for Spite Analysis."""

import logging
import os
import sys


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return (or create) a named logger with a console handler."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)-7s %(name)s: %(message)s',
            datefmt='%H:%M:%S',
        ))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


# Application-level root logger
app_log = get_logger('spite_analysis')

"""
Thin wrapper around Python's logging module.
Provides a consistent log format across all project modules.
"""

import logging


def get_logger(name: str, level=logging.INFO) -> logging.Logger:
    """Return a logger with a consistent format: [LEVEL] name: message.
    
    Prevents duplicate handlers when called multiple times with the same name.
    """
    logger = logging.getLogger(name)
    
    # Avoid adding duplicate handlers on repeated calls
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    return logger

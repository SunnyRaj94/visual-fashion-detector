import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, Optional


class CustomFormatter(logging.Formatter):
    """Custom formatter to provide clear, structured logging output with ANSI colors."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    log_format = (
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s"
    )

    FORMATS = {
        logging.DEBUG: grey + log_format + reset,
        logging.INFO: log_format,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMATS.get(record.levelno, self.log_format)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(
    name: str = "fashion_detector",
    log_level: int = logging.INFO,
    log_file: Optional[str] = "logs/fashion_detector.log",
) -> logging.Logger:
    """Sets up a global logger with console and file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if setup is called multiple times
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger()


def configure_logger(
    level_name: str, log_file: Optional[str] = "logs/fashion_detector.log"
) -> None:
    """Reconfigures the default logger with a new log level and file path."""
    global logger
    level = getattr(logging, level_name.upper(), logging.INFO)

    # Remove existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    logger = setup_logger(name="fashion_detector", log_level=level, log_file=log_file)


@contextmanager
def log_duration(
    activity_name: str, extra_info: Optional[Dict[str, Any]] = None
) -> Generator[None, None, None]:
    """Context manager to measure and log the execution time of a block of code."""
    start_time = time.perf_counter()
    info_str = f" ({extra_info})" if extra_info else ""
    logger.info(f"Starting: {activity_name}{info_str}")
    try:
        yield
    finally:
        duration = time.perf_counter() - start_time
        logger.info(f"Completed: {activity_name} in {duration:.4f} seconds")


def time_it(
    activity_name: Optional[str] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to measure and log the execution time of a function."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = activity_name or func.__name__

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with log_duration(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

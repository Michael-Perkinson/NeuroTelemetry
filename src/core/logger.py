import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
_LOGGER = logging.getLogger("neurotelemetry")


def _get_logger() -> logging.Logger:
    """Configure and return the application logger on first use."""
    if not _LOGGER.handlers:
        LOG_DIR.mkdir(exist_ok=True)
        log_file = LOG_DIR / f"analysis_{datetime.now():%Y%m%d_%H%M%S}.log"
        handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")
        )
        _LOGGER.addHandler(handler)
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.propagate = False
    return _LOGGER


def log_info(message: str) -> None:
    _get_logger().info(message)


def log_error(message: str) -> None:
    _get_logger().error(message)


def log_warning(message: str) -> None:
    _get_logger().warning(message)


def log_exception(error: Exception) -> None:
    _get_logger().exception(str(error))

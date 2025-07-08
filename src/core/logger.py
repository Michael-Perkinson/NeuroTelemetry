import logging
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    filename=log_file,
    filemode='w',  # one file per run
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)


def log_info(message):
    logging.info(message)


def log_error(message):
    logging.error(message)


def log_exception(e):
    logging.exception(str(e))  # includes traceback

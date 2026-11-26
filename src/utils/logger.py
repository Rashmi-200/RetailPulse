import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def get_logger(name: str) -> logging.Logger:
    """Create a structured logger outputting to both console and logs/retailpulse.log."""
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_dir = Path(os.getenv("LOG_DIR", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(log_level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    try:
        file_handler = logging.FileHandler(log_dir / "retailpulse.log", encoding="utf-8")
    except (PermissionError, OSError):
        file_handler = logging.FileHandler("/tmp/retailpulse.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

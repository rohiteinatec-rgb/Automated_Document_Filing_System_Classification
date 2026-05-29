import logging
import os
from logging.handlers import RotatingFileHandler
import json

from config import Config


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "service": "adfs",
            "stage": getattr(record, "stage", "system"),
            "message": record.getMessage(),
            "latency_ms": getattr(record, "latency_ms", None),
            "file": record.filename,
            "funcName": record.funcName
        })

def setup_logger():
    logger = logging.getLogger("adfs")

    # Prevent duplicate handlers if imported multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.DEBUG)

    # File handler: 100MB per file, keep 10 backups
    log_file = os.path.join(Config.BASE_DIR, "adfs.log")
    fh = RotatingFileHandler(log_file, maxBytes=100 * 1024 * 1024, backupCount=10)
    fh.setLevel(logging.DEBUG)

    # Attach the structured JSON formatter
    formatter = StructuredFormatter()
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

# Create a singleton instance to be imported across the app
adfs_logger = setup_logger()
import logging
import sys
import os
import json
from Autonomous_Reasoning_System.infrastructure import config

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        # Basic structured format
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno
        }
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record)

def setup_logging(default_level=logging.INFO):
    """
    Configures the root logger with a consistent format and handlers.
    """
    # Check if we want JSON logging via env var, default to standard for readability unless specified
    json_logging = os.getenv("JSON_LOGGING", "false").lower() == "true"

    # Define the log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Create a formatter
    if json_logging:
        formatter = StructuredFormatter(datefmt=date_format)
    else:
        formatter = logging.Formatter(log_format, date_format)

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(default_level)

    # Remove existing handlers to avoid duplicate logs if re-configured
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler in same directory as DB/memory files
    log_dir = os.path.dirname(config.MEMORY_DB_PATH) or "."
    os.makedirs(log_dir, exist_ok=True)
    file_path = os.path.join(log_dir, "tyrone.log")
    file_handler = logging.FileHandler(file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    logging.info("Logging system initialized.")

def get_logger(name):
    """
    Returns a logger with the specified name.
    """
    return logging.getLogger(name)

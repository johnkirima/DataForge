"""DataForge Logging Configuration"""
import logging
import os
from logging.handlers import RotatingFileHandler

from config import LOG_MAX_BYTES, LOG_BACKUP_COUNT

# Ensure logs directory exists
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"

# Configure root logger once
_root_configured = False


def _setup_root_logger() -> None:
    """Configure the root logger with file and console handlers."""
    global _root_configured
    if _root_configured:
        return
    
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Rotating file handler with UTF-8 encoding
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    _root_configured = True


def get_logger(agent_name: str) -> logging.Logger:
    """Get a logger instance for an agent.
    
    Args:
        agent_name: Name of the agent requesting the logger
        
    Returns:
        Configured logger instance
    """
    _setup_root_logger()
    return logging.getLogger(agent_name)

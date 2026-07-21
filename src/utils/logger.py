"""
Runtime Logging Configuration Subsystem.

This module establishes the global logging framework for the entire application suite.
Upon initial module import, it programmatically configures the root logger with a unified 
format strategy and provisions dual output pipelines targeting both the active interactive 
terminal session and a persistent disk log matrix.

Key Architecture Features:
    * **Global Initialization Override**: Invokes ``logging.basicConfig`` with ``force=True`` 
    to explicitly flush and reset any pre-existing root logger configurations established by third-party modules during startup.
    * **Dual Stream Aggregation**: Multiplexes runtime events simultaneously to the console 
    (via standard output stream) and an operational tracking file.
    * **Hierarchical Namespace Routing**: Provides factory accessors to inherit parent 
    stream properties while isolating subsystem identities.

Module Global Parameters:
    LOG_FORMAT (str): Structural layout template enforcing the tracking syntax:
        ``[Timestamp: SeverityLevel: ModuleName: Message]``
    LOG_DIR (str): Filesystem path referencing the directory container for log archives.
    LOG_FILEPATH (str): Concrete destination path for writing the persistent log stream.

Requirements:
    Sphinx extension `sphinx.ext.napoleon` must be enabled in `conf.py` to parse
    the Google-style docstrings used throughout this module.
"""

import os
import sys
import logging

# Logging format specification: Time - Level - Module Name - Message
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

LOG_DIR = "./logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")

# Ensure the log file directory structure exists at runtime module initialization
os.makedirs(LOG_DIR, exist_ok=True)

# Globally configure the root logger handler stack upon module import
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),  # Append operational records to local disk file
        logging.StreamHandler(sys.stdout)   # Stream active log entries to terminal standard output
    ],
    force=True  # Reset any existing root log configurations across external imports
)


def get_logger(name: str) -> logging.Logger:
    """Retrieves or instantiates a named Logger instance bound to a specific runtime namespace.

    This factory method provides modules with access to the pre-configured logging 
    pipeline. By using the standard dot-separated module path (typically via ``__name__``), 
    loggers inherit the global root properties while retaining granular diagnostic 
    isolation for filtering and trace attribution.

    Args:
        name (str): The unique structural namespace identifier for the logger instance, 
            typically initialized using the calling module's ``__name__`` context variable.

    Returns:
        logging.Logger: A configured and active logger instance mapped to the requested 
        subsystem pipeline channel.

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Subsystem successfully initialized.")
    """
    return logging.getLogger(name)
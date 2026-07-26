"""
Description:
    Centralized logging utility designed to manage runtime diagnostics, application tracing,
    and concurrent file/stream logging output orchestration across an entire software ecosystem.

Main components :

    * get_logger: Factory function to retrieve or instantiate named module-level loggers.

Main features :

    * Automatic runtime generation of local log directories.
    * Dual-routing pipeline streaming records concurrently to disk files and standard output.
    * Forced root logger synchronization to override competing external package configurations.

General architecture :
    Statically configures the root logger at module load time utilizing Python's native 
    built-in logging framework, establishing a parent channel that child namespaces inherit.

General data flow :

    .. code-block:: text

        [Application Event] ---> Subsystem Logger (Child Namespace)
                                           |
                                           v
                                  [Root Logger Stack]
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
        [FileHandler: running_logs.log]               [StreamHandler: sys.stdout]

Optimisations :

    * Idempotent directory tree generation execution (`os.makedirs` with `exist_ok=True`) 
      to eliminate OS-level IO validation overhead.
    * Atomic runtime synchronization with `force=True` parameter to bypass distributed 
      package configuration deadlocks during sub-module import cycles.

Example:

    .. code-block:: python

        from src.utils.logger import get_logger

        logger = get_logger(__name__)
        logger.info("Operational pipeline successfully online.")

Note:
    The file handler appends operational records continuously. For long-running server 
    environments or intensive training pipelines, consider migrating to a rotating 
    file handler structure to balance disk constraints.

References:

    * PEP 282 -- A Logging System: https://peps.python.org/pep-0282/
    * Python Logging Documentation: https://docs.python.org/3/library/logging.html

Author:
    Goudjou Borel

Version:
    1.0.0
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

    This factory method provides modules with instant access to the pre-configured logging
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
    
        .. code-block:: python

            from src.utils.logger import get_logger
            logger = get_logger(__name__)
            logger.info("Subsystem successfully initialized.")
    """
    return logging.getLogger(name)
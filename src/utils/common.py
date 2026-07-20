import os
from pathlib import Path
from typing import List, Union
import yaml
from box import ConfigBox  # pip install python-box
from box.exceptions import BoxValueError

from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML configuration file and wraps its contents in a ConfigBox.

    Wrapping the dictionary inside a ConfigBox allows attribute-style access 
    to keys using dot notation (e.g., `config.alpha` instead of `config['alpha']`).

    Args:
        path_to_yaml (Path): File system path pointing to the target YAML file.

    Returns:
        ConfigBox: A box container wrapping the parsed YAML key-value contents.

    Raises:
        ValueError: If the target YAML file is empty or parsing yields no data.
        Exception: Re-raises any underlying system or filesystem access errors.
    """
    try:
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("The provided YAML file is empty.")
    except Exception as e:
        raise e


def create_directories(path_to_directories: List[Union[str, Path]], verbose: bool = True) -> None:
    """Creates a collection of directories recursively if they do not already exist.

    Args:
        path_to_directories (List[Union[str, Path]]): A list containing folder paths 
            represented as strings or Path objects.
        verbose (bool, optional): If True, logs a confirmation message to the tracking 
            logger for every folder created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory verified or created at destination: {path}")
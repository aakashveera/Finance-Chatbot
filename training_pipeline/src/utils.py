import json
import yaml
import logging
from pathlib import Path
from typing import List, Union


def create_logger(log_file_path:str)->logging.Logger:
    """
    Create and configure a logger at INFO level with a log file.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler for the log file
    file_handler = logging.FileHandler(log_file_path)
    
    # Create a stream handler for console output
    console_handler = logging.StreamHandler()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def load_json(path: Path) -> dict:
    """
    Load JSON data from a file.

    Args:
        path (Path): The path to the JSON file.

    Returns:
        dict: The JSON data as a dictionary.
    """

    with path.open("r") as f:
        data = json.load(f)

    return data


def write_json(data: Union[dict, List[dict]], path: Path) -> None:
    """
    Write a dictionary or a list of dictionaries to a JSON file.

    Args:
        data (Union[dict, List[dict]]): The data to be written to the file.
        path (Path): The path to the file.

    Returns:
        None
    """

    with path.open("w") as f:
        json.dump(data, f, indent=4)


def load_yaml(path: Path) -> dict:
    """
    Load a YAML file from the given path and return its contents as a dictionary.

    Args:
        path (Path): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with path.open("r") as f:
        config = yaml.safe_load(f)

    return config
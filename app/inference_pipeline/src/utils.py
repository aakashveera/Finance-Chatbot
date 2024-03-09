import os
import yaml
import logging
from pathlib import Path
from typing import Optional

import qdrant_client

from .constants import QDRANT_URL

def build_qdrant_client(
    url: Optional[str] = QDRANT_URL,
    api_key: Optional[str] = None,
):
    """
    Builds a Qdrant client object using the provided URL and API key.

    Args:
        url (Optional[str]): The URL of the Qdrant server. If not provided, the function will attempt
            to read it from the QDRANT_URL environment variable.
        api_key (Optional[str]): The API key to use for authentication. If not provided, the function will attempt
            to read it from the QDRANT_API_KEY environment variable.

    Raises:
        KeyError: If the URL or API key is not provided and cannot be read from the environment variables.

    Returns:
        qdrant_client.QdrantClient: A Qdrant client object.
    """

    if api_key is None:
        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )

    client = qdrant_client.QdrantClient(url, api_key=api_key)

    return client

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
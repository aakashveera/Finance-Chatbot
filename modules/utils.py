import os
import yaml
import torch
import random
import logging

import numpy as np

from transformers import set_seed, PreTrainedTokenizerFast
from typing import List, Dict, Union


def seed_everything(seed: int=42) -> None:
    """ Seeds all the random variables and random number generation for results reproducability.
    
    Args:
        seed (int, optional): A random integer for seeding all random number generators. Defaults to 42.
        
    Returns: None
    
    """    
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    set_seed(seed)


def load_config(config_file_path: str) -> Dict:
    """
    Load a configuration file in YAML format and return it as a dictionary.

    Args:
        config_file (str): The path to the YAML configuration file.

    Returns:
        Dict: A dictionary containing the configuration.

    """
    
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
        
    return config


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
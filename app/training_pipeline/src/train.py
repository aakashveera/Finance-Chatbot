from pathlib import Path
from typing import Dict, Union

from .utils import create_logger, load_yaml
from src.model import TrainerAPI

logger = create_logger("logs/outputs.log")

def train(
    config: Dict[str, Union[str,int]]
):
    """
    Trains a machine learning model using the specified configuration file and dataset.

    Args:
        config_file (Dict[str, Union[str,int]]): A dictionary containing all the necassary params and hyperparams.
    """

    trainer = TrainerAPI(config)
    trainer.train()


if __name__ == "__main__":
    config = load_yaml(Path('src/config.yml'))     
    train(config)
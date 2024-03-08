import argparse
from pathlib import Path
from typing import Dict, Union, Optional

from .utils import create_logger, load_yaml
from src.model import InferenceAPI

logger = create_logger("logs/outputs.log")
config = load_yaml(Path('src/config.yml'))

def main(
    peft_model_id: str,
    config: Dict[str, Union[str,int]],
    output_path: Optional[str] = None
):
    """
    Trains a machine learning model using the specified configuration file and dataset.

    Args:
        peft_model_id (str): Finetuned lora adapter name or path.
        config_file (Dict[str, Union[str,int]]): A dictionary containing all the necassary params and hyperparams.
        output_path (str, optional): File path to save the output responses.
    """

    output_path = Path(output_path) if output_path else Path(config['general']['output_filepath'])

    model = InferenceAPI(peft_model_id=peft_model_id,config=config)    
    model.infer_all(output_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Inference Script for the finetuned model")

    # Add arguments
    parser.add_argument("--model_name", type=str, required=True, help="Name/Path of the finetuned LORA peft adapter from model registry")
    parser.add_argument("--output_path", type=str, help="Output Path to store the infered response. Optional argument")
    args = parser.parse_args()
    
    main(args.model_name, config, args.output_path)
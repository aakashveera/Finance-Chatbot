from pathlib import Path
from typing import Tuple, Dict, Union, Optional

import os
import torch
from comet_ml import API
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel

from src.utils import create_logger
logger = create_logger("logs/outputs.log")

def get_model_and_tokenizer(
    config: Dict[str,Union[str,int]],
    peft_pretrained_model_name_or_path: Optional[str]=None,
    gradient_checkpointing: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name.

    Args:
        config (Dict): A dictionary containing all parameters and hyperparameters as key/value pairs.
        gradient_checkpointing (bool): Whether or not to enable gradient checkpoint while training.
        peft_pretrained_model_name_or_path (str, optional): The name or path of the pretrained lora adapter to use. Defaults to None.

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the built model, tokenizer,
            and PeftConfig.
    """
    
    bnb_config =  BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['model_name'],
        quantization_config=bnb_config if config['model']['use_qlora'] else None,
        device_map=config['model']['device']
        )

    tokenizer = AutoTokenizer.from_pretrained(config['model']['model_name'])
    
    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    if peft_pretrained_model_name_or_path:
        is_model_name = not os.path.isdir(peft_pretrained_model_name_or_path)
        if is_model_name:
            logger.info(
                f"Downloading {peft_pretrained_model_name_or_path} from Comet ML's model registry."
            )
            peft_pretrained_model_name_or_path = download_from_model_registry(
                model_id=peft_pretrained_model_name_or_path,
                cache_dir=config['general']['cache_dir'],
            )

        logger.info(f"Loading Lora Confing from: {peft_pretrained_model_name_or_path}")
        lora_config = LoraConfig.from_pretrained(peft_pretrained_model_name_or_path)
        assert (
            lora_config.base_model_name_or_path == config['model']['model_name']
        ), f"Lora Model trained on different base model than the one requested: \
        {lora_config.base_model_name_or_path} != {config['model']['model_name']}"

        logger.info(f"Loading Peft Model from: {peft_pretrained_model_name_or_path}")
        model = PeftModel.from_pretrained(model, peft_pretrained_model_name_or_path)
    else:
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(**config['peft'])
        model = get_peft_model(model, peft_config)
        
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = (
            False  # Gradient checkpointing is not compatible with caching.
        )
    else:
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        
    return model, tokenizer


def download_from_model_registry(model_id: str, cache_dir: str):
    """
    Downloads a model from the Comet ML Learning model registry.

    Args:
        model_id (str): The ID of the model to download, in the format "workspace/model_name:version".
        cache_dir (str): The directory to cache the downloaded model in. Defaults to the value of
            `constants.CACHE_DIR`.

    Returns:
        Path: The path to the downloaded model directory.
    """

    output_folder = Path(cache_dir) / model_id

    already_downloaded = output_folder.exists()
    if not already_downloaded:
        workspace, model_id = model_id.split("/")
        model_name, version = model_id.split(":")

        api = API()
        model = api.get_model(workspace=workspace, model_name=model_name)
        model.download(version=version, output_folder=output_folder, expand=True)
    else:
        logger.info(f"Model {model_id=} already downloaded to: {output_folder}")

    subdirs = [d for d in output_folder.iterdir() if d.is_dir()]
    if len(subdirs) == 1:
        model_dir = subdirs[0]
    else:
        raise RuntimeError(
            f"There should be only one directory inside the model folder. \
                Check the downloaded model at: {output_folder}"
        )

    logger.info(f"Model {model_id=} downloaded from the registry to: {model_dir}")

    return model_dir
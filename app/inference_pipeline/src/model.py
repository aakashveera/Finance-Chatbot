from pathlib import Path
from typing import Tuple, Dict, Union, Optional, List

import os
import torch
from comet_ml import API
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
    pipeline)

from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from langchain.llms import HuggingFacePipeline

from .constants import *

from src.utils import create_logger
logger = create_logger("logs/outputs.log")

def get_model_and_tokenizer(
    config: Dict[str,Union[str,int]],
    peft_pretrained_model_name_or_path: Optional[str]=None,
    gradient_checkpointing: bool = True,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Function that builds a QLoRA LLM model based on the given HuggingFace name:
        1.   Create and prepare the bitsandbytes configuration for QLoRa's quantization
        2.   Download, load, and quantize on-the-fly Mistral-Instruct-7b
        3.   Create and prepare the LoRa configuration
        4.   Load and configuration Mistral-7B's tokenizer

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


class StopOnTokens(StoppingCriteria):
    """
    A stopping criteria that stops generation when a specific token is generated.

    Args:
        stop_ids (List[int]): A list of token ids that will trigger the stopping criteria.
    """

    def __init__(self, stop_ids: List[int]):
        super().__init__()

        self._stop_ids = stop_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Check if the last generated token is in the stop_ids list.

        Args:
            input_ids (torch.LongTensor): The input token ids.
            scores (torch.FloatTensor): The scores of the generated tokens.

        Returns:
            bool: True if the last generated token is in the stop_ids list, False otherwise.
        """

        for stop_id in self._stop_ids:
            if input_ids[0][-1] == stop_id:
                return True

        return False
    
    
def build_pipeline(
    config: Dict[str,Union[str,int]],
    llm_lora_model_id: str,
    gradient_checkpointing: bool = False,
    use_streamer: bool = False
) -> Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]:
    """
    Builds a HuggingFace pipeline for text generation using a custom LLM + Finetuned checkpoint.

    Args:
        config (Dict): A dictionary containing all parameters and hyperparameters as key/value pairs.
        llm_lora_model_id (str): The ID or path of the LLM LoRA model.
        gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to False.
        use_streamer (bool, optional): Whether to use a text iterator streamer. Defaults to False.

    Returns:
        Tuple[HuggingFacePipeline, Optional[TextIteratorStreamer]]: A tuple containing the HuggingFace pipeline
            and the text iterator streamer (if used).
    """

    model, tokenizer= get_model_and_tokenizer(
        config=config,
        peft_pretrained_model_name_or_path=llm_lora_model_id,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.eval()

    if use_streamer:
        streamer = TextIteratorStreamer(
            tokenizer, timeout=10.0, skip_prompt=True, skip_special_tokens=True
        )
        stop_on_tokens = StopOnTokens(stop_ids=[tokenizer.eos_token_id])
        stopping_criteria = StoppingCriteriaList([stop_on_tokens])
    else:
        streamer = None
        stopping_criteria = StoppingCriteriaList([])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        streamer=streamer,
        stopping_criteria=stopping_criteria,
    )
    hf = HuggingFacePipeline(pipeline=pipe)

    return hf, streamer, tokenizer.eos_token
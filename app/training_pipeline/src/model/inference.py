import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Union

import comet_llm
from datasets import Dataset
from peft import PeftConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model import get_model_and_tokenizer
from src.data import FinanceDataset
from src.utils import create_logger, write_json

logger = create_logger("logs/outputs.log")


class InferenceAPI:
    """
    A class for performing inference using a trained LLM model.

    Args:
        peft_model_id (str): Id/name of the finetuned LORA adapter to use.
        config (Dict): A dictionary containing all parameters and hyperparameters as key/value pairs.
    """

    def __init__(
        self,
        peft_model_id: str,
        config: Dict[str,Union[str,int,Dict]],
        dataset_filepath: Optional[str] = None
    ):
        self._config = config
        self._peft_model_id = peft_model_id
        self._model_id = config['model']['model_name']
        self._root_dataset_dir = config['data']['dataset_path']
        self._test_dataset_file = config['data']['test_filename']
        self._model_cache_dir = config['general']['cache_dir']
        self._log_prompt = config['general']['log_prompt']
        self._device = config['model']['device']

        self._model, self._tokenizer = self.load_model()
        self._dataset = self.load_data(dataset_filepath)
        
        try:
            self._comet_project_name = os.environ["COMET_PROJECT_NAME"]
        except KeyError:
            raise RuntimeError("Please set the COMET_PROJECT_NAME environment variable.")


    def load_data(self,
                  dataset_filepath: Optional[str] = None) -> Dataset:
        """
        Loads the QA dataset.
        
        Args:
            dataset_filepath (str, optional): Filepath of the dataset to be infered. By default uses the test data for inference.

        Returns:
            Dataset: The loaded QA dataset.

        """

        logger.info(f"Loading QA dataset from {self._root_dataset_dir=}")
        
        if not dataset_filepath:
            dataset_filepath = Path(self._root_dataset_dir + self._test_dataset_file)

        dataset = FinanceDataset(
            data_path=dataset_filepath,
            mode='inference',
        ).to_dataset_object()

        logger.info(f"Loaded {len(dataset)} samples for inference")

        return dataset
    

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]:
        """
        Loads the LLM model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer, PeftConfig]: A tuple containing the loaded LLM model, tokenizer,
                and PEFT config.

        """

        logger.info(f"Loading model using {self._model_id=} and {self._peft_model_id=}")

        model, tokenizer = get_model_and_tokenizer(
            config=self._config,
            peft_pretrained_model_name_or_path=self._peft_model_id,
            gradient_checkpointing=False
        )
        model.eval()

        return model, tokenizer
    

    def infer(self, infer_prompt: str, infer_payload: dict) -> str:
        """
        Performs inference using the loaded LLM model.

        Args:
            infer_prompt (str): The prompt to use for inference.
            infer_payload (dict): The payload to use for inference.

        Returns:
            str: The generated answer.

        """

        start_time = time.time()
        answer = self.generate_response(
            query_text=infer_prompt,
            return_only_answer=True,
        )
        end_time = time.time()
        duration_milliseconds = (end_time - start_time) * 1000

        if self._log_prompt:
            comet_llm.log_prompt(
                project=f"{self._comet_project_name}-monitor-prompts",
                prompt=infer_prompt,
                output=answer,
                prompt_template_variables=infer_payload,
                # TODO: Count tokens instead of using len().
                metadata={
                    "usage.prompt_tokens": len(infer_prompt),
                    "usage.total_tokens": len(infer_prompt) + len(answer),
                    "usage.actual_new_tokens": len(answer),
                    "model": self._model_id,
                    "peft_model": self._peft_model_id,
                },
                duration=duration_milliseconds,
            )

        return answer
    

    def infer_all(self, output_file: Optional[Path] = None) -> None:
        """
        Performs inference on all samples in the loaded dataset.

        Args:
            output_file (Optional[Path], optional): The file to save the output to. Defaults to None.

        """

        assert (
            self._dataset is not None
        ), "Dataset not loaded. Provide a dataset directory to the constructor: 'root_dataset_dir'."

        prompt_and_answers = []
        should_save_output = output_file is not None
        for sample in tqdm(self._dataset):
            answer = self.infer(
                infer_prompt=sample["prompt"], infer_payload=sample["payload"]
            )

            if should_save_output:
                prompt_and_answers.append(
                    {
                        "prompt": sample["prompt"],
                        "answer": answer,
                    }
                )

        if should_save_output:
            write_json(prompt_and_answers, output_file)
        
            
    def generate_response(self,
        query_text: str,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        return_only_answer: bool = False,
    ):
        """
        Generates text based on the input text using the provided model and tokenizer.

        Args:
            query_text (str): The input text to generate text from.
            max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 40.
            temperature (float, optional): The temperature to use for text generation. Defaults to 1.0.
            return_only_answer (bool, optional): Whether to return only the generated text or the entire generated sequence.
                Defaults to False.

        Returns:
            str: The generated text.
        """

        inputs = self._tokenizer(query_text, return_tensors="pt", return_token_type_ids=False).to(
            self._device
        )

        outputs = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, temperature=temperature
        )

        output = outputs[
            0
        ]  # The input to the model is a batch of size 1, so the output is also a batch of size 1.
        if return_only_answer:
            input_ids = inputs.input_ids
            input_length = input_ids.shape[-1]
            output = output[input_length:]

        output = self._tokenizer.decode(output, skip_special_tokens=True)

        return output
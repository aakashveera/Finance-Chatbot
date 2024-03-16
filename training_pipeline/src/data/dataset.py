from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset
from unstructured.cleaners.core import clean_extra_whitespace, group_broken_paragraphs

from src.utils import load_json


@dataclass(frozen=True)
class DataSample:
    """
    A data sample for a question answering model.

    Attributes:
        user_context (str): The user's context for the question.
        news_context (str): The news context for the question.
        chat_history (str): The chat history for the question.
        question (str): The question to be answered.
        answer (str): The answer to the question.
    """

    user_context: str = ""
    news_context: str = ""
    chat_history: str = ""
    question: str = ""
    answer: str = ""


class FinanceDataset:
    def __init__(
        self,
        data_path: Path,
        mode: Optional[str] = "training"
    ):
        """
        A class representing a finance dataset.

        Args:
            data_path (Path): The path to the data file.
            mode (str, optional): The mode to use for the dataset. Defaults to "training".
        """

        self._mode = mode
        self._raw_data = self.load(data_path)

    def load(self, data_path: Path) -> List[DataSample]:
        """
        Loads the data from the specified path.

        Args:
            data_path (Path): The path to the data file.

        Returns:
            List[DataSample]: The loaded data.
        """

        data = load_json(data_path)

        return self.deserialize(data)

    def deserialize(self, data: List[dict]) -> List[DataSample]:
        """
        Deserializes the data.

        Args:
            data (List[dict]): The data to deserialize.

        Returns:
            List[DataSample]: The deserialized data.
        """

        if self._mode == 'training':
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                    answer=sample["response"],
                )
                for sample in data
            ]
        else:
            return [
                DataSample(
                    user_context=sample["about_me"],
                    news_context=sample["context"],
                    chat_history=sample.get("chat_history", ""),
                    question=sample["question"],
                )
                for sample in data
            ]
    
    def _get_training_prompt(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Converts the provided training sample into a prompt."""

        prompt = f"""[INST] You are helpful assistant with expertise on finance and investment. You will be provided a short context about the user, a question and a latest news related to the question.
Using the provided information, try to answer the question with a right justification.

### ABOUT ME: {sample['user_context']}

### LATEST NEWS: {sample['news_context']}

### QUESTION: {sample['question']} [/INST]

### RESPONSE: {sample['answer']} </s>"""

        return {"prompt": prompt, "payload": sample}
    
    def _get_inference_prompt(self, sample: Dict[str, str]) -> Dict[str, Union[str, Dict]]:
        """Converts the provided inference sample into a prompt."""

        prompt = f"""[INST] You are helpful assistant with expertise on finance and investment. You will be provided a short context about the user, a question and a latest news related to the question.
Using the provided information, try to answer the question with a right justification.

### ABOUT ME: {sample['user_context']}

### LATEST NEWS: {sample['news_context']}

### QUESTION: {sample['question']} [/INST]"""

        return {"prompt": prompt, "payload": sample}

    def to_dataset_object(self) -> Dataset:
        """
        Preprocesses the data & returns a HuggingFace dataset.

        Returns:
            Dataset: The HuggingFace dataset.
        """
        
        if self._mode == 'training':
            template_mapping_func = self._get_training_prompt
        else:
            template_mapping_func = self._get_inference_prompt
            
        data_as_dict = [asdict(sample) for sample in self._raw_data]
        dataset = Dataset.from_list(data_as_dict)
        
        dataset = dataset.map(self.clean)
        dataset = dataset.map(template_mapping_func, remove_columns=dataset.column_names)

        return dataset

    def clean(self, samples: Dict[str, str]) -> Dict[str, str]:
        """
        Cleans the samples.

        Args:
            samples (Dict[str, str]): The samples to clean.

        Returns:
            Dict[str, str]: The cleaned samples.
        """

        for key, sample in samples.items():
            cleaned_sample = clean_extra_whitespace(sample)
            cleaned_sample = group_broken_paragraphs(cleaned_sample)

            samples[key] = cleaned_sample

        return samples
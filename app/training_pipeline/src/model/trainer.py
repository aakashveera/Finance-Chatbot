import comet_ml
from pathlib import Path
from typing import Tuple, Dict, Union
import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EvalPrediction,
    TrainerCallback,
    TrainerState,
    TrainingArguments,
    TrainerControl
)
from trl import SFTTrainer

from src.model import get_model_and_tokenizer
from src.data import FinanceDataset
from src.utils import create_logger

logger = create_logger("logs/outputs.log")

class BestModelToModelRegistryCallback(TrainerCallback):
    """
    Callback that logs the best model checkpoint to the Comet.ml model registry.

    Args:
        model_id (str): The ID of the model to log to the model registry.
    """

    def __init__(self, model_id: str):
        self._model_id = model_id

    @property
    def model_name(self) -> str:
        """
        Returns the name of the model to log to the model registry.
        """

        return f"financial_assistant/{self._model_id}"

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        """
        Event called at the end of training.

        Logs the best model checkpoint to the Comet.ml model registry.
        """

        best_model_checkpoint = state.best_model_checkpoint
        has_best_model_checkpoint = best_model_checkpoint is not None
        if has_best_model_checkpoint:
            best_model_checkpoint = Path(best_model_checkpoint)
            logger.info(
                f"Logging best model from {best_model_checkpoint} to the model registry..."
            )

            self.to_model_registry(best_model_checkpoint)
        else:
            logger.warning(
                "No best model checkpoint found. Skipping logging it to the model registry..."
            )

    def to_model_registry(self, checkpoint_dir: Path):
        """
        Logs the given model checkpoint to the Comet.ml model registry.

        Args:
            checkpoint_dir (Path): The path to the directory containing the model checkpoint.
        """

        checkpoint_dir = checkpoint_dir.resolve()

        assert (
            checkpoint_dir.exists()
        ), f"Checkpoint directory {checkpoint_dir} does not exist"

        # Get the stale experiment from the global context to grab the API key and experiment ID.
        stale_experiment = comet_ml.get_global_experiment()
        # Resume the expriment using its API key and experiment ID.
        experiment = comet_ml.ExistingExperiment(
            api_key=stale_experiment.api_key, experiment_key=stale_experiment.id
        )
        logger.info(f"Starting logging model checkpoint @ {self.model_name}")
        experiment.log_model(self.model_name, str(checkpoint_dir))
        experiment.end()
        logger.info(f"Finished logging model checkpoint @ {self.model_name}")


class TrainerAPI:
    """
    A class for training a Qlora model.

    Args:
        config (Dict): A dictionary containing all parameters and hyperparameters as key/value pairs.
    """

    def __init__(
        self,
        config: Dict[str,Union[str,int,Dict]],
    ):
                
        self._config = config
        self._root_dataset_dir = config['data']['dataset_path']
        self._training_filename = config['data']['training_filename']
        self._testing_filename = config['data']['test_filename']
        self._model_id = config['model']['model_name']
        self._max_seq_length = config['model']['max_seq_len']
        self._model_cache_dir = config['general']['cache_dir']
        
        self._training_dataset, self._validation_dataset = self.load_data()
        self._model, self._tokenizer = self.load_model()
        self._training_arguments = TrainingArguments(**config['training_arguments'])


    def load_data(self) -> Tuple[Dataset, Dataset]:
        """
        Loads the training and validation datasets.

        Returns:
            Tuple[Dataset, Dataset]: A tuple containing the training and validation datasets.
        """

        logger.info(f"Loading QA datasets from {self._root_dataset_dir=}")

        training_dataset = FinanceDataset(
            data_path=Path(self._root_dataset_dir + self._training_filename),
            mode='training',
        ).to_dataset_object()
        
        validation_dataset = FinanceDataset(
            data_path=Path(self._root_dataset_dir + self._testing_filename),
            mode='inference',
        ).to_dataset_object()

        logger.info(f"Training dataset size: {len(training_dataset)}")
        logger.info(f"Validation dataset size: {len(validation_dataset)}")

        return training_dataset, validation_dataset
    

    def load_model(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """
        Loads the model.

        Returns:
            Tuple[AutoModelForCausalLM, AutoTokenizer]: A tuple containing the model, tokenizer.
        """

        logger.info(f"Loading model using {self._model_id=}")
        
        model, tokenizer = get_model_and_tokenizer(
            config=self._config,
            gradient_checkpointing=True
        )

        return model, tokenizer
    

    def train(self) -> SFTTrainer:
        """
        Trains the model.

        Returns:
            SFTTrainer: The trained model.
        """

        logger.info("Training model...")

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=self._training_dataset,
            eval_dataset=self._validation_dataset,
            dataset_text_field="prompt",
            max_seq_length=self._max_seq_length,
            tokenizer=self._tokenizer,
            args=self._training_arguments,
            packing=True,
            compute_metrics=self.compute_metrics,
            callbacks=[BestModelToModelRegistryCallback(model_id=self._model_id)]
        )
        trainer.train()

        return trainer
    

    def compute_metrics(self, eval_pred: EvalPrediction):
        """
        Computes the perplexity metric.

        Args:
            eval_pred (EvalPrediction): The evaluation prediction.

        Returns:
            dict: A dictionary containing the perplexity metric.
            
        """
        
        perplexity = np.exp(eval_pred.predictions.mean()).item()

        return {"perplexity": perplexity}
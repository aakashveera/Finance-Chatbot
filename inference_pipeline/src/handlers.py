import comet_llm
from langchain.callbacks.base import BaseCallbackHandler
from typing import Dict, Any

from .constants import *



class CometLLMMonitoringHandler(BaseCallbackHandler):
    """
    A callback handler for monitoring LLM models using Comet.ml.

    Args:
        project_name (str): The name of the Comet.ml project to log to.
        llm_model_id (str): The ID of the LLM model to use for inference.
        llm_qlora_model_id (str): The ID of the PEFT model to use for inference.
    """
    
    def __init__(
        self,
        project_name: str = None,
        llm_model_id: str = LLM_MODEL_ID,
        llm_qlora_model_id: str = LLM_QLORA_CHECKPOINT
    ):
        self._project_name = project_name
        self._llm_model_id = llm_model_id
        self._llm_qlora_model_id = llm_qlora_model_id

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """
        A callback function that logs the prompt and output to Comet.ml.

        Args:
            outputs (Dict[str, Any]): The output of the LLM model.
            **kwargs (Any): Additional arguments passed to the function.
        """

        should_log_prompt = "metadata" in kwargs
        if should_log_prompt:
            metadata = kwargs["metadata"]

            comet_llm.log_prompt(
                project=self._project_name,
                prompt=metadata["prompt"],
                output=outputs["answer"],
                prompt_template_variables=metadata["prompt_template_variables"],
                metadata={
                    "usage.prompt_tokens": metadata["usage.prompt_tokens"],
                    "usage.total_tokens": metadata["usage.total_tokens"],
                    "usage.actual_new_tokens": metadata["usage.actual_new_tokens"],
                    "model": self._llm_model_id,
                    "peft_model": self._llm_qlora_model_id,
                },
                duration=metadata["duration_milliseconds"],
            )
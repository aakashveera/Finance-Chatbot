import numpy as np
from transformers import AutoModel, AutoTokenizer

import hashlib
import traceback
from typing import *
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel

from unstructured.cleaners.core import clean, clean_non_ascii_chars, replace_unicode_quotes
from unstructured.partition.html import partition_html
from unstructured.staging.huggingface import chunk_by_attention_window

from .constants import *
from .utils import create_logger

# Creating an logger
logger = create_logger("logs/outputs.log")


class EmbeddingModel:
    """
    A Class that provides a pre-trained transformer model for generating embeddings of input text.

    Args:
        model_id (str): The identifier of the pre-trained transformer model to use.
        max_input_length (int): The maximum length of input text to tokenize.
        device (str): The device to use for running the model (e.g. "cpu", "cuda").
        cache_dir (Optional[Path]): The directory to cache the pre-trained model files.
            If None, the default cache directory is used.

    Attributes:
        max_input_length (int): The maximum length of input text to tokenize.
        tokenizer (AutoTokenizer): The tokenizer used to tokenize input text.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_input_length: int = MAX_LENGTH,
        device: str = DEVICE,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initializes the EmbeddingModelSingleton instance.

        Args:
            model_id (str): The identifier of the pre-trained transformer model to use.
            max_input_length (int): The maximum length of input text to tokenize.
            device (str): The device to use for running the model (e.g. "cpu", "cuda").
            cache_dir (Optional[Path]): The directory to cache the pre-trained model files.
                If None, the default cache directory is used.
        """

        self._device = device
        self._max_input_length = max_input_length

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name,cache_dir=cache_dir).to(self._device)
        self._model.eval()

    @property
    def max_input_length(self) -> int:
        """
        Returns the maximum length of input text to tokenize.

        Returns:
            int: The maximum length of input text to tokenize.
        """

        return self._max_input_length

    @property
    def tokenizer(self) -> AutoTokenizer:
        """
        Returns the tokenizer used to tokenize input text.

        Returns:
            AutoTokenizer: The tokenizer used to tokenize input text.
        """

        return self._tokenizer

    def __call__(
        self, input_text: str, to_list: bool = True
    ) -> Union[np.ndarray, list]:
        """
        Generates embeddings for the input text using the pre-trained transformer model.

        Args:
            input_text (str): The input text to generate embeddings for.
            to_list (bool): Whether to return the embeddings as a list or numpy array. Defaults to True.

        Returns:
            Union[np.ndarray, list]: The embeddings generated for the input text.
        """

        try:
            tokenized_text = self._tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self._max_input_length,
            ).to(self._device)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(f"Error tokenizing the following input text: {input_text}")

            return [] if to_list else np.array([])

        try:
            result = self._model(**tokenized_text)
        except Exception:
            logger.error(traceback.format_exc())
            logger.error(
                f"Error generating embeddings for the following model_id: {self._model_id} and input text: {input_text}"
            )

            return [] if to_list else np.array([])

        embeddings = result.last_hidden_state[:, 0, :].cpu().detach().numpy()
        if to_list:
            embeddings = embeddings.flatten().tolist()

        return embeddings
    
class NewsArticle(BaseModel):
    """
    Represents a news article.

    Attributes:
        id (int): News article ID
        headline (str): Headline or title of the article
        summary (str): Summary text for the article (may be first sentence of content)
        author (str): Original author of news article
        created_at (datetime): Date article was created (RFC 3339)
        updated_at (datetime): Date article was updated (RFC 3339)
        url (Optional[str]): URL of article (if applicable)
        content (str): Content of the news article (might contain HTML)
        symbols (List[str]): List of related or mentioned symbols
        source (str): Source where the news originated from (e.g. Benzinga)
    """
    
    id: int
    headline: str
    summary: str
    author: str
    created_at: datetime
    updated_at: datetime
    url: Optional[str]
    content: str
    symbols: List[str]
    source: str
    
    def to_document(self) -> "Document":
        """
        Converts the news article to a Document object.

        Returns:
            Document: A Document object representing the news article.
        """
        
        document_id = hashlib.md5(self.content.encode()).hexdigest()
        document = Document(id=document_id)

        article_elements = partition_html(text=self.content)
        cleaned_content = clean_non_ascii_chars(
            replace_unicode_quotes(clean(" ".join([str(x) for x in article_elements])))
        )
        cleaned_headline = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.headline))
        )
        cleaned_summary = clean_non_ascii_chars(
            replace_unicode_quotes(clean(self.summary))
        )

        document.text = [cleaned_headline, cleaned_summary, cleaned_content]
        document.metadata["headline"] = cleaned_headline
        document.metadata["summary"] = cleaned_summary
        document.metadata["url"] = self.url
        document.metadata["symbols"] = self.symbols
        document.metadata["author"] = self.author
        document.metadata["created_at"] = self.created_at

        return document
    
    
class Document(BaseModel):
    """
    A Pydantic model representing a document.

    Attributes:
        id (str): The ID of the document.
        group_key (Optional[str]): The group key of the document.
        metadata (dict): The metadata of the document.
        text (list): The text of the document.
        chunks (list): The chunks of the document.
        embeddings (list): The embeddings of the document.

    Methods:
        to_payloads: Returns the payloads of the document.
        compute_chunks: Computes the chunks of the document.
        compute_embeddings: Computes the embeddings of the document.
    """

    id: str
    group_key: Optional[str] = None
    metadata: dict = {}
    text: list = []
    chunks: list = []
    embeddings: list = []
    
    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        """
        Returns the payloads of the document.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing the IDs and payloads of the document.
        """

        payloads = []
        ids = []
        for chunk in self.chunks:
            payload = self.metadata
            payload.update({"text": chunk})
            # Create the chunk ID using the hash of the chunk to avoid storing duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads

    def compute_chunks(self, model: EmbeddingModel) -> "Document":
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModel): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """

        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)

        return self

    def compute_embeddings(self, model: EmbeddingModel) -> "Document":
        """
        Computes the embeddings for each chunk in the document using the specified embedding model.

        Args:
            model (EmbeddingModel): The embedding model to use for computing the embeddings.

        Returns:
            Document: The document object with the computed embeddings.
        """

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self
import os

from bytewax.outputs import DynamicOutput, StatelessSink
from qdrant_client import QdrantClient
from qdrant_client.http.api_client import UnexpectedResponse
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct

from .constants import *
from .models import Document


class QdrantVectorOutput(DynamicOutput):
    """A class representing a Qdrant vector output.

    This class is used to create a Qdrant vector output, which is a type of dynamic output that supports
    at-least-once processing. Messages from the resume epoch will be duplicated right after resume.

    Args:
        vector_size (int): The size of the vector.
        collection_name (str, optional): The name of the collection.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
        client (Optional[QdrantClient], optional): The Qdrant client. Defaults to None.
    """

    def __init__(
        self,
        vector_size: int,
        collection_name: str = COLLECTION_NAME
    ):
        self._collection_name = collection_name
        self._vector_size = vector_size
        
        api_key = self._get_credentials()
        self.client = QdrantClient(QDRANT_URL, api_key=api_key)

        try:
            self.client.get_collection(collection_name=self._collection_name)
        except (UnexpectedResponse, ValueError):
            self.client.recreate_collection(
                collection_name=self._collection_name,
                vectors_config=VectorParams(
                    size=self._vector_size, distance=Distance.COSINE
                ),
            )
            
    def _get_credentials(self):

        try:
            api_key = os.environ["QDRANT_API_KEY"]
        except KeyError:
            raise KeyError(
                "QDRANT_API_KEY must be set as environment variable or manually passed as an argument."
            )
            
        return api_key
        

    def build(self, worker_index, worker_count):
        """Builds a QdrantVectorSink object.

        Args:
            worker_index (int): The index of the worker.
            worker_count (int): The total number of workers.

        Returns:
            QdrantVectorSink: A QdrantVectorSink object.
        """

        return QdrantVectorSink(self.client, self._collection_name)


class QdrantVectorSink(StatelessSink):
    """
    A sink that writes document embeddings to a Qdrant collection.

    Args:
        client (QdrantClient): The Qdrant client to use for writing.
        collection_name (str, optional): The name of the collection to write to.
            Defaults to constants.VECTOR_DB_OUTPUT_COLLECTION_NAME.
    """

    def __init__(
        self,
        client: QdrantClient,
        collection_name: str = COLLECTION_NAME,
    ):
        self._client = client
        self._collection_name = collection_name

    def write(self, document: Document):
        ids, payloads = document.to_payloads()
        points = [
            PointStruct(id=idx, vector=vector, payload=_payload)
            for idx, vector, _payload in zip(ids, document.embeddings, payloads)
        ]

        self._client.upsert(collection_name=self._collection_name, points=points)
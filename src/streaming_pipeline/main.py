import src.constants as config
from .qdrant import QdrantVectorOutput
from .models import EmbeddingModel, NewsArticle
from .alpaca_stream import AlpacaNewsStreamInput
from .alpaca_batch import AlpacaNewsBatchInput

from typing import List
from datetime import datetime

from bytewax.dataflow import Dataflow
from pydantic import parse_obj_as

def get_inputStreamer(mode):
    if mode=='real-time':
        return AlpacaNewsStreamInput()
    elif mode=='batch':
        date_format = '%d/%m/%Y'
        startDate = datetime.strptime(config.START_DATE, date_format)
        endDate = datetime.strptime(config.END_DATE, date_format)
        return AlpacaNewsBatchInput(startDate, endDate)
    else:
        raise ValueError(f"Mode has to either 'batch' or 'real-time'. {mode} is not accepted.")

streamSource = get_inputStreamer(config.MODE)
dbSource = QdrantVectorOutput(vector_size=config.MAX_LENGTH)
model = EmbeddingModel()

flow = Dataflow()
flow.input('input',streamSource)
flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
flow.map(lambda article: article.to_document())
flow.map(lambda document: document.compute_chunks(model))
flow.map(lambda document: document.compute_embeddings(model))
flow.output("output", dbSource)
from .constants import *
from .utils import create_logger
from .qdrant import QdrantVectorOutput
from .models import EmbeddingModel, NewsArticle
from .alpaca_stream import AlpacaNewsStreamInput
from .alpaca_batch import AlpacaNewsBatchInput

from typing import List, Union
from datetime import datetime

from bytewax.dataflow import Dataflow
from pydantic import parse_obj_as

logger = create_logger("logs/outputs.log")

def get_flow(
    streamSource: Union[AlpacaNewsBatchInput, AlpacaNewsStreamInput],
    debug:bool
    )-> Dataflow:
    """Creates and returns a Dataflow pipeline using ByteWax which includes streaming, pre-processing, embedding functions.

    Args:
        streamSource (Union[AlpacaNewsBatchInput, AlpacaNewsStreamInput]): _description_
        debug (bool): Whether to print the fetched news content before inserting it on DB.

    Returns:
        Dataflow: The dataflow pipeline for processing news articles.
    """
    
    dbSource = QdrantVectorOutput(vector_size=MAX_LENGTH)
    model = EmbeddingModel()
    
    flow = Dataflow()
    flow.input('input',streamSource)
    flow.flat_map(lambda messages: parse_obj_as(List[NewsArticle], messages))
    flow.map(lambda article: article.to_document())
    if debug:
        flow.inspect(print)
    flow.map(lambda document: document.compute_chunks(model))
    flow.map(lambda document: document.compute_embeddings(model))
    flow.output("output", dbSource)
    
    return flow
    

def real_time_mode(debug: bool = False)->Dataflow:
    """Initiate the streaming pipeline in real-time mode. 
    Continuously monitors Alpaca for latest news, embeds the news and pushes it to Vector DB.
    
    Args:
        debug (bool): Whether to print the fetched news content before inserting it on DB. Defaults to False

    Returns:
        Dataflow: The dataflow pipeline for processing news articles.
    """
    
    logger.info("Initiating a streaming pipeline in real time mode.")
    
    try:
        streamSource = AlpacaNewsStreamInput()
        return get_flow(streamSource, debug=debug)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        raise Exception(error_message) from e
        


def batch_mode(
    start_date:str,
    end_date:str,
    debug: bool = False
    )->Dataflow:
    """Initiate the streaming pipeline in batch mode. 
    Fetches the finance news on the specified time-frame, embeds the news and pushes it to Vector DB.

    Args:
        start_date (str): Starting Date to fetch news in dd/mm/yyyy format.
        end_date (str): Ending Date to fetch news in dd/mm/yyyy format.
        debug (bool): Whether to print the fetched news content before inserting it on DB. Defaults to False

    Returns:
        Dataflow: The dataflow pipeline for processing news articles
    """
    
    try:
        date_format = '%d/%m/%Y'
        startDate = datetime.strptime(start_date, date_format)
        endDate = datetime.strptime(end_date, date_format)
    except:
        raise ValueError(f"{start_date} and {end_date} should be in format dd/mm/yyyy. Please check your format.")
    
    logger.info(f"Initiating a streaming pipeline in batch mode. Fetching all news between {start_date} and {end_date}.")
    
    try:
        streamSource = AlpacaNewsBatchInput(startDate, endDate)
        return get_flow(streamSource, debug=debug)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        raise Exception(error_message) from e
        
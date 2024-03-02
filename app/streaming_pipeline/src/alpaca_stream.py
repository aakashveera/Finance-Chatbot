import json
import os
from typing import List, Union

from bytewax.inputs import DynamicInput, StatelessSource
from websocket import create_connection

from .constants import *
from .utils import create_logger

# Creating an logger
logger = create_logger("logs/outputs.log")

class AlpacaNewsStreamInput(DynamicInput):
    """Input class to receive streaming news data
    from the Alpaca real-time news API.
    """

    def __init__(self):
        self._tickers = ["*"]

    def build(self, worker_index, worker_count):
        """
        Distributes the tickers to the workers. If parallelized,
        workers will establish their own websocket connection and
        subscribe to the tickers they are allocated.

        Args:
            worker_index (int): The index of the current worker.
            worker_count (int): The total number of workers.

        Returns:
            AlpacaNewsStreamSource: An instance of the AlpacaNewsStreamSource class
            with the worker's allocated tickers.
        """

        prods_per_worker = int(len(self._tickers) / worker_count)
        worker_tickers = self._tickers[
            int(worker_index * prods_per_worker) : int(
                worker_index * prods_per_worker + prods_per_worker
            )
        ]

        return AlpacaNewsStreamSource(tickers=worker_tickers)


class AlpacaNewsStreamSource(StatelessSource):
    """
    A source for streaming news data from Alpaca API.

    Args:
        tickers (List[str]): A list of ticker symbols to subscribe to.

    Attributes:
        _alpaca_client (AlpacaStreamClient): An instance of the AlpacaStreamClient class.
    """

    def __init__(self, tickers: List[str]):
        """
        Initializes the AlpacaNewsStreamSource object.

        Args:
            tickers (List[str]): A list of ticker symbols to subscribe to.
        """
        self._alpaca_client = AlpacaNewsStreamClient(tickers=tickers)
        self._alpaca_client.start()
        self._alpaca_client.subscribe()
    
    def next(self):
        """
        Returns the next news item from the Alpaca API.

        Returns:
            dict: A dictionary containing the news item data.
        """
        return self._alpaca_client.recv()


class AlpacaNewsStreamClient:
    """
    Alpaca News Stream Client that uses a web socket to stream news data.

    References used to implement this class:
    * Alpaca Docs: https://alpaca.markets/docs/api-references/market-data-api/news-data/realtime/
    * Source of implementation inspiration: https://github.com/alpacahq/alpaca-py/blob/master/alpaca/common/websocket.py
    """

    def __init__(self, tickers: List[str]):
        """
        Initializes the AlpacaNewsStreamClient.

        Args:
            tickers (List[str]): A list of tickers to subscribe to.
        """
        self._tickers = tickers
        self._ws = None

    def start(self):
        """
        Starts the AlpacaNewsStreamClient.
        """

        self._connect()
        self._auth()

    def _connect(self):
        """
        Connects to the Alpaca News Stream.
        """

        self._ws = create_connection(STREAM_URL)

        msg = self.recv()

        if msg[0]["T"] != "success" or msg[0]["msg"] != "connected":
            raise ValueError("connected message not received")
        else:
            logger.info("[AlpacaNewsStream]: Connected to Alpaca News Stream.")

    def _auth(self):
        """
        Authenticates with the Alpaca News Stream.
        """
        
        try:
            api_key = os.environ["ALPACA_API_KEY"]
        except KeyError:
            raise KeyError(
                "ALPACA_API_KEY must be set as environment variable or manually passed as an argument."
            )

        try:
            api_secret = os.environ["ALPACA_API_SECRET"]
        except KeyError:
            raise KeyError(
                "ALPACA_API_SECRET must be set as environment variable or manually passed as an argument."
            )

        self._ws.send(
            self._build_message(
                {
                    "action": "auth",
                    "key": api_key,
                    "secret": api_secret,
                }
            )
        )

        msg = self.recv()
        if msg[0]["T"] == "error":
            raise ValueError(msg[0].get("msg", "auth failed"))
        elif msg[0]["T"] != "success" or msg[0]["msg"] != "authenticated":
            raise ValueError("failed to authenticate")
        else:
            logger.info("[AlpacaNewsStream]: Authenticated with Alpaca News Stream.")

    def subscribe(self):
        """
        Subscribes to the Alpaca News Stream.
        """

        self._ws.send(
            self._build_message({"action": "subscribe", "news": self._tickers})
        )

        msg = self.recv()
        if msg[0]["T"] != "subscription":
            raise ValueError("failed to subscribe")
        else:
            logger.info("[AlpacaNewsStream]: Subscribed to Alpaca News Stream.")

    def _build_message(self, message: dict) -> str:
        """
        Builds a message to send to the Alpaca News Stream.

        Args:
            message (dict): The message to build.

        Returns:
            str: The built message.
        """

        return json.dumps(message)

    def recv(self) -> Union[dict, List[dict]]:
        """
        Receives a message from the Alpaca News Stream.

        Returns:
            Union[dict, List[dict]]: The received message.
        """

        if self._ws:
            message = self._ws.recv()
            logger.info(f"[AlpacaNewsStream]: Received message: {message}")
            message = json.loads(message)

            return message
        else:
            raise RuntimeError("Websocket not initialized. Call start() first.")
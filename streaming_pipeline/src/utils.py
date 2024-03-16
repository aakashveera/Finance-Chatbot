import logging
import datetime
from typing import List, Tuple
    
def split_time_range_into_intervals(
    from_datetime: datetime.datetime, to_datetime: datetime.datetime, n: int
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Splits a time range [from_datetime, to_datetime] into N equal intervals.

    Args:
        from_datetime (datetime): The starting datetime object.
        to_datetime (datetime): The ending datetime object.
        n (int): The number of intervals.

    Returns:
        List of tuples: A list where each tuple contains the start and end datetime objects for each interval.
    """

    # Calculate total duration between from_datetime and to_datetime.
    total_duration = to_datetime - from_datetime

    # Calculate the length of each interval.
    interval_length = total_duration / n

    # Generate the interval.
    intervals = []
    for i in range(n):
        interval_start = from_datetime + (i * interval_length)
        interval_end = from_datetime + ((i + 1) * interval_length)
        if i + 1 != n:
            # Subtract 1 microsecond from the end of each interval to avoid overlapping.
            interval_end = interval_end - datetime.timedelta(minutes=1)

        intervals.append((interval_start, interval_end))
        
    return intervals

def create_logger(log_file_path:str)->logging.Logger:
    """
    Create and configure a logger at INFO level with a log file.

    Args:
        log_file_path (str): The path to the log file.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler for the log file
    file_handler = logging.FileHandler(log_file_path)
    
    # Create a stream handler for console output
    console_handler = logging.StreamHandler()

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
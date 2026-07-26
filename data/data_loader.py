import pandas as pd
from src.utils.logger import logger

def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file into a pandas DataFrame.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded successfully: {file_path}")
        return df
    except FileNotFoundError:
        logger.error(f"FileNotFoundError: The specified CSV file was not found: {file_path}")
        raise
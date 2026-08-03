import pandas as pd

from src.utils.logger import logger
from src.config.config import REQUIRED_COLUMNS


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate the input DataFrame before preprocessing.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        bool:
            True if validation is successful.
            False if critical validation checks fail.
    """

    # Check if the DataFrame is empty
    if df.empty:
        logger.error("The DataFrame is empty.")
        return False

    # Check if all required columns exist
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        logger.error(f"Missing required columns: {missing_columns}")
        return False

    # Check for null values in required columns
    for col in REQUIRED_COLUMNS:
        null_count = df[col].isnull().sum()

        if null_count > 0:
            logger.warning(
                f"Column '{col}' contains {null_count} missing value(s). "
                "These will be handled during preprocessing."
            )

    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:
        logger.warning(
            f"Found {duplicate_count} duplicate row(s). "
            "These will be removed during preprocessing."
        )

    logger.info("Data validation completed successfully.")

    return True
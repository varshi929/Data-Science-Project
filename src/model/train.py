import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression

from src.utils.logger import logger


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> LogisticRegression:
    """
    Train a Logistic Regression model.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (np.ndarray): Training labels.

    Returns:
        LogisticRegression: Trained Logistic Regression model.
    """

    logger.info("Training Logistic Regression model...")

    model = LogisticRegression(random_state=42)

    model.fit(X_train, y_train)

    logger.info("Model training completed successfully.")

    return model
import numpy as np
import pandas as pd

from typing import Tuple

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.utils.logger import logger
from src.config.config import (
    TARGET_COLUMN,
    COMBINED_TEXT_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
)


def encode_labels(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode the target labels into numerical values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple:
            - Encoded target labels (y)
            - Fitted LabelEncoder
    """
    logger.info("Encoding target labels.")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])

    logger.info("Target labels encoded successfully.")

    return y, label_encoder


def vectorize_text(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, TfidfVectorizer]:
    """
    Convert text into TF-IDF feature vectors.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        tuple:
            - TF-IDF feature DataFrame (X)
            - Fitted TF-IDF Vectorizer
    """
    logger.info("Vectorizing text using TF-IDF.")

    tfidf_vectorizer = TfidfVectorizer()

    tfidf_matrix = tfidf_vectorizer.fit_transform(
        df[COMBINED_TEXT_COLUMN]
    )

    X = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out(),
    )

    logger.info("Text vectorization completed successfully.")

    return X, tfidf_vectorizer


def split_dataset(
    X: pd.DataFrame,
    y: np.ndarray,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    """
    Split the dataset into training and testing sets.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (np.ndarray): Target labels.

    Returns:
        tuple:
            - X_train
            - X_test
            - y_train
            - y_test
    """
    logger.info(
        f"Splitting dataset (test_size={TEST_SIZE}, random_state={RANDOM_STATE})"
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logger.info("Dataset split successfully.")

    return X_train, X_test, y_train, y_test
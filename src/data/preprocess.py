from src.utils.logger import logger
from nltk.stem import WordNetLemmatizer
import string
import pandas as pd
from src.config.config import REQUIRED_COLUMNS, TEXT_COLUMNS, COLUMNS_TO_COMBINE, COMBINED_TEXT_COLUMN

def remove_duplicate_rows(df):
    """
    Remove duplicate rows from the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove duplicates.

    Returns:
    pd.DataFrame: A DataFrame with duplicate rows removed.
    """
    initial_row_count = len(df)
    df_cleaned = df.drop_duplicates()
    final_row_count = len(df_cleaned)
    
    if initial_row_count != final_row_count:
        logger.info(f"Removed {initial_row_count - final_row_count} duplicate rows.")
    else:
        logger.info("No duplicate rows found.")
    
    return df_cleaned

def remove_null_values(df, required_columns):
    """
    Remove rows with null values in the specified required columns.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to remove rows with null values.
    required_columns (list): List of columns to check for null values.

    Returns:
    pd.DataFrame: A DataFrame with rows containing null values in the required columns removed.
    """
    initial_row_count = len(df)
    df_cleaned = df.dropna(subset=required_columns)
    final_row_count = len(df_cleaned)
        
    if initial_row_count != final_row_count:
        logger.info(f"Removed {initial_row_count - final_row_count} rows with null values in columns: {required_columns}.")
    else:
        logger.info("No rows with null values found in the specified columns.")
    
    return df_cleaned

def clean_whitespace(df, columns):
    """
    Clean whitespace from specified string columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to clean.
    columns (list): List of column names to clean whitespace from.

    Returns:
    pd.DataFrame: A DataFrame with whitespace cleaned from the specified columns.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()
            logger.info(f"Cleaned whitespace from column: {column}.")
        else:
            logger.warning(f"Column '{column}' not found in DataFrame.")
    
    return df

def convert_to_lowercase(df, columns):
    """
    Convert specified string columns in the DataFrame to lowercase.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list): List of column names to convert to lowercase.

    Returns:
    pd.DataFrame: A DataFrame with specified columns converted to lowercase.
    """
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.lower()
            logger.info(f"Converted column '{column}' to lowercase.")
        else:
            logger.warning(f"Column '{column}' not found in DataFrame.")
    
    return df

def remove_punctuation(df, columns):
    """
    Remove punctuation from specified string columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list): List of column names to remove punctuation from.

    Returns:
    pd.DataFrame: A DataFrame with punctuation removed from the specified columns.
    """    
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).str.translate(str.maketrans('', '', string.punctuation))
            logger.info(f"Removed punctuation from column: {column}.")
        else:
            logger.warning(f"Column '{column}' not found in DataFrame.")
    
    return df

def combine_columns(df, new_column_name, columns_to_combine):
    """
    Combine specified columns into a new column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    new_column_name (str): The name of the new column to create.
    columns_to_combine (list): List of column names to combine.

    Returns:
    pd.DataFrame: A DataFrame with the new combined column added.
    """
    for column in columns_to_combine:
        if column not in df.columns:
            logger.warning(f"Column '{column}' not found in DataFrame. Cannot combine.")
            return df
    
    df[new_column_name] = df[columns_to_combine].astype(str).agg(' '.join, axis=1)
    logger.info(f"Combined columns {columns_to_combine} into new column '{new_column_name}'.")
    
    return df

def lemmatize_text(df, columns):
    """
    Lemmatize text in specified string columns of the DataFrame.

    Parameters:
    df (pd.DataFrame): The DataFrame to modify.
    columns (list): List of column names to lemmatize.

    Returns:
    pd.DataFrame: A DataFrame with lemmatized text in the specified columns.
    """
    lemmatizer = WordNetLemmatizer()
    
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype(str).apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
            logger.info(f"Lemmatized text in column: {column}.")
        else:
            logger.warning(f"Column '{column}' not found in DataFrame.")
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    
    df = remove_duplicate_rows(df)
    df = remove_null_values(df, REQUIRED_COLUMNS)
    df = clean_whitespace(df, TEXT_COLUMNS)
    df = convert_to_lowercase(df, TEXT_COLUMNS)
    df = remove_punctuation(df, TEXT_COLUMNS)
    df = combine_columns(df, COMBINED_TEXT_COLUMN, COLUMNS_TO_COMBINE)
    df = lemmatize_text(df, [COMBINED_TEXT_COLUMN])

    return df

"""
Application configuration constants.
"""

# Required columns for validation
REQUIRED_COLUMNS = [
    "Title",
    "Description",
    "Sentiment"
]

# Text columns used during preprocessing
TEXT_COLUMNS = [
    "Title",
    "Description",
    "Source",
    "Author",
    "Type"
]

# Columns to combine into a single text feature
COLUMNS_TO_COMBINE = [
    "Title",
    "Description"
]

# Target column for sentiment analysis
TARGET_COLUMN = "Sentiment"

# Name of the combined text column
COMBINED_TEXT_COLUMN = "Text"

# Dataset path
DATASET_PATH = "data/raw/news_sentiment_analysis.csv"

#Random state for reproducibility
RANDOM_STATE = 42

# Test size for train-test split
TEST_SIZE = 0.2
from data.data_loader import load_csv as load_data
from src.data.validate_data import validate_data
from src.data.preprocess import preprocess_data
from src.utils.logger import logger
from src.config.config import REQUIRED_COLUMNS, DATASET_PATH
from src.features.feature_engineering import encode_labels, vectorize_text, split_dataset

def main():

    # Load dataset
    logger.info("Loading dataset...")
    df = load_data(DATASET_PATH)

    # Validate dataset
    logger.info("Validating dataset...")

    if not validate_data(df):
        logger.error("Data validation failed.")
        return

    # Preprocess dataset
    logger.info("Preprocessing dataset...")
    df = preprocess_data(df)

    # Display processed data
    logger.info("Displaying processed dataset.")
    print(df.head())
    
    y, label_encoder = encode_labels(df)

    X, tfidf_vectorizer = vectorize_text(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

if __name__ == "__main__":
    main()
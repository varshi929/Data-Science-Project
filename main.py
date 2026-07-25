from src.utils.logger import logger
from src.utils.text_utils import clean_text, count_words

def main():
    sample_text = "   Hello, World! This is a sample text for testing.   "
    
    # Clean the text
    cleaned_text = clean_text(sample_text)
    logger.info(f"Cleaned Text: '{cleaned_text}'")
    
    # Count words in the cleaned text
    word_count = count_words(cleaned_text)
    logger.info(f"Word Count: {word_count}")

if __name__ == "__main__":
    main()
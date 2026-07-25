def clean_text(text:str) -> str:
    """
    Converts the input text to lowercase and removes any leading or trailing whitespace.
    """
    return text.lower().strip()

def count_words(text:str) -> int:
    """
    Counts the number of words in the input text.
    """
    return len(text.split())
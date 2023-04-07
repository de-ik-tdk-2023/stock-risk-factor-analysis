import re

from nltk import WordNetLemmatizer, word_tokenize


def preprocess_text(text):
    """
    Removes special characters and numbers from the given text,
    converts words to lowercase and lemmatizes them.

    Parameters:
    text (str): the raw text

    Returns:
    str: the clean text
    """
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = re.sub('\d+', '', text)
    text = text.lower()
    wnl = WordNetLemmatizer()
    lem_sentence = []
    for word in word_tokenize(text):
        lem_sentence.append(wnl.lemmatize(word))
    return " ".join(lem_sentence)

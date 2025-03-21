import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords

from typing import List, Union
import string

STOP_WORDS = list(stopwords.words('english'))
PUNCTUATION = string.punctuation

def filter_out_punctuation(tokens):
    return [token for token in tokens if token not in PUNCTUATION]

def _word_tokenize_list(text: List[str]) -> List[str]:
    words = []
    for chunk in text:
        words += filter_out_punctuation(
            nltk.tokenize.word_tokenize(chunk, language = 'english')
        )
    return words

def _word_tokenize_str(text: str) -> List[str]:
    words = filter_out_punctuation(
        nltk.tokenize.word_tokenize(text, language = 'english')
    )
    return words

def word_tokenize(text: Union[str, List[str]]) -> List[str]:
    if isinstance(text, list):
        return _word_tokenize_list(text)
    else:
        return _word_tokenize_str(text)

def uncased(tokens: List[str]) -> List[str]:
    return [token.lower() for token in tokens]

def remove_stop_words(tokens: List[str]) -> List[str]:
    return [token for token in tokens if token not in STOP_WORDS]

def stem(tokens: List[str]) -> List[str]:
    stemmer = nltk.stem.SnowballStemmer('english')
    return [stemmer.stem(token) for token in tokens]
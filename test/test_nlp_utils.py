import pytest
from utils.nlp_utils import word_tokenize

def test_nltk_word_tokenize():
    assert word_tokenize('Good morning!') == ['Good','morning']
    assert word_tokenize(["hello","good morning"]) == ['hello','good','morning']
    assert word_tokenize(["hello",""]) == ['hello',]
    assert word_tokenize(["hello, how are you?","good morning"]) == ['hello','how', 'are', 'you', 'good','morning']
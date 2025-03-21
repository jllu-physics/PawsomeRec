import pytest
from schemas.data_models import Review
from feature.rating_matrix import RatingMatrix
import numpy as np

test_reviews = [
    Review(rating = 4, title = 'OK', text = "OK", images = [], asin = 'a', 
           parent_asin = 'a', user_id = '1', timestamp = 1, helpful_vote = 10,
           verified_purchase = True),
    Review(rating = 2, title = 'Bad', text = "Not OK", images = [], asin = 'b', 
        parent_asin = 'b', user_id = '2', timestamp = 1, helpful_vote = 8,
        verified_purchase = True),
]

def test_add_review():
    rm = RatingMatrix()
    for review in test_reviews:
        rm.add_review(review)
    assert rm.get_rating('a','1') == 4
    assert rm.get_rating('b','2') == 2
    assert rm.get_rating('b','1', None) == None
    assert rm.get_rating('c','12', 3) == 3
    assert np.allclose(rm.to_scipy_csr_array_with_offset().toarray(), np.array([[1,0,],[0,-1]]))
from feature.rating_matrix import RatingMatrix
from utils.data_utils import get_review_from_json
import numpy as np
import scipy
from tqdm import tqdm
from models.collaborative.collaborative_filtering_baseline import CollaborativeFilteringBaseline

def construct_rating_matrix(path_to_review_jsonl, filename):
    rating_matrix = RatingMatrix()

    with open(path_to_review_jsonl, 'r', encoding = 'utf-8') as f:
        for review_json_str in tqdm(f):
            review = get_review_from_json(review_json_str)
            rating_matrix.add_review(review)
    
    rating_matrix.serialize(filename)
    
    # rating_csr_matrix_with_offset = rating_matrix.to_scipy_csr_array_with_offset()

    #return rating_csr_matrix_with_offset

    return rating_matrix

def sparse_svd(sparse_matrix, k, oversampling_factor = 1):
    oversampled_k = k * oversampling_factor
    u_oversample, s_oversample, vh_oversample = scipy.sparse.linalg.svds(sparse_matrix, oversampled_k)
    top_k_index = np.argsort(-s_oversample)[:k]
    u = u_oversample[:,top_k_index]
    s = s_oversample[top_k_index]
    vh = vh_oversample[top_k_index]
    return u, s, vh

if __name__ == '__main__':
    #try:
    #    rating_matrix = RatingMatrix()
    #    rating_matrix.deserialize('pet_supplies_rating.json')
    #    sparse_matrix = rating_matrix.to_scipy_csr_array_with_offset()
    #    del rating_matrix
    #    print('Loaded serialized matrix')
    #except:
    #    sparse_matrix = construct_rating_matrix('./data/Pet_Supplies.jsonl')
    #u,s,vh = sparse_svd(sparse_matrix, 16)
    #print(u.shape, s.shape, vh.shape)
    try:
        rating_matrix_train = RatingMatrix()
        rating_matrix_train.deserialize('./checkpoints/pet_supplies_rating_matrix_train.json')
    except:
        rating_matrix_train = construct_rating_matrix('./data/Pet_Supplies_train.jsonl', 'pet_supplies_rating_train.json')
    cf_baseline = CollaborativeFilteringBaseline(24)
    cf_baseline.train(rating_matrix_train)
    cf_baseline.serialize('./checkpoints/cf_model.json')

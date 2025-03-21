import numpy as np
import scipy
import json
from feature.rating_matrix import RatingMatrix

class CollaborativeFilteringBaseline:

    def __init__(
        self,
        embed_dim: int = 16
    ):
        self.embed_dim = embed_dim
    
    def serialize(
        self,
        filename: str
    ):
        json_dict = {'embed_dim': self.embed_dim}
        if hasattr(self, 'asin_to_row_index'):
            json_dict['asin_to_row_index'] = self.asin_to_row_index
        if hasattr(self, 'user_id_to_col_index'):
            json_dict['user_id_to_col_index'] = self.user_id_to_col_index
        if hasattr(self, 'prod_rating_mle'):
            json_dict['prod_rating_mle'] = self.prod_rating_mle.tolist()
        if hasattr(self, 'user_rating_mle'):
            json_dict['user_rating_mle'] = self.user_rating_mle.tolist()
        if hasattr(self, 'rating_mle'):
            json_dict['rating_mle'] = self.rating_mle.tolist()
        if hasattr(self, 'rating_offset'):
            json_dict['rating_offset'] = self.rating_offset
        if hasattr(self, 'prod_embed'):
            json_dict['prod_embed'] = self.prod_embed.tolist()
        if hasattr(self, 'user_embed'):
            json_dict['user_embed'] = self.user_embed.tolist()
        
        with open(filename, 'w', encoding = 'utf-8') as f:
            json.dump(json_dict, f, ensure_ascii = False)
    
    def deserialize(
        self,
        path_to_json
    ):
        with open(path_to_json, 'r') as f:
            json_dict = json.load(f)
        self.embed_dim = json_dict['embed_dim']
        if 'asin_to_row_index' in json_dict:
            self.asin_to_row_index = json_dict['asin_to_row_index']
        if 'user_id_to_col_index' in json_dict:
            self.user_id_to_col_index = json_dict['user_id_to_col_index']
        if 'prod_rating_mle' in json_dict:
            self.prod_rating_mle = np.array(json_dict['prod_rating_mle'], dtype = np.int8)
        if 'user_rating_mle' in json_dict:
            self.user_rating_mle = np.array(json_dict['user_rating_mle'], dtype = np.int8)
        if 'rating_mle' in json_dict:
            self.rating_mle = np.array(json_dict['rating_mle'], dtype = np.int8)
        if 'rating_offset' in json_dict:
            self.rating_offset = json_dict['rating_offset']
        if 'prod_embed' in json_dict:
            self.prod_embed = np.array(json_dict['prod_embed'], dtype = np.float16)
        if 'user_embed' in json_dict:
            self.user_embed = np.array(json_dict['user_embed'], dtype = np.float16)
        

    def train(
        self,
        rating_matrix: RatingMatrix
    ):
        self.asin_to_row_index = rating_matrix.asin_to_row_index
        self.user_id_to_col_index = rating_matrix.user_id_to_col_index
        sparse_rating_matrix = rating_matrix.to_scipy_csr_array_with_offset()
        self.prod_rating_mle, self.user_rating_mle, self.rating_mle = self.get_rating_histograms(rating_matrix)
        u, s, vh = scipy.sparse.linalg.svds(sparse_rating_matrix, self.embed_dim)
        self.rating_offset = rating_matrix.rating_offset
        self.prod_embed = u*s
        self.user_embed = vh.T
    
    def get_rating_histograms(
        self,
        rating_matrix: RatingMatrix
    ):
        n_rating = len(rating_matrix.val)
        prod_rating_hist = np.zeros((rating_matrix.n_row,5), dtype = np.int8)
        user_rating_hist = np.zeros((rating_matrix.n_col,5), dtype = np.int8)
        grand_rating_hist = np.zeros(5, dtype = np.int8)

        for i in range(n_rating):
            row = rating_matrix.row[i]
            col = rating_matrix.col[i]
            rating = int(rating_matrix.val[i] + rating_matrix.rating_offset - 1)
            prod_rating_hist[row, rating] += 1
            user_rating_hist[col, rating] += 1
            grand_rating_hist[rating] += 1
        
        prod_rating_mle = np.argmax(prod_rating_hist, axis=1) + 1
        user_rating_mle = np.argmax(user_rating_hist, axis=1) + 1
        rating_mle = np.argmax(grand_rating_hist) + 1

        return prod_rating_mle, user_rating_mle, rating_mle
    
    def predict(
        self,
        product_asin,
        user_id
    ):
        if product_asin not in self.asin_to_row_index:
            if user_id not in self.user_id_to_col_index:
                return self.rating_mle
            else:
                return self.user_rating_mle[self.user_id_to_col_index[user_id]]
        else:
            if user_id not in self.user_id_to_col_index:
                return self.prod_rating_mle[self.asin_to_row_index[product_asin]]
            else:
                prod_vector = self.prod_embed[self.asin_to_row_index[product_asin]]
                user_vector = self.user_embed[self.user_id_to_col_index[user_id]]
                score = prod_vector.dot(user_vector)
                rating = round(score + self.rating_offset)
                if rating > 5:
                    rating = 5
                elif rating < 1:
                    rating = 1
                return rating
    
    def score(
        self,
        product_asin,
        user_id
    ):
        if product_asin not in self.asin_to_row_index:
            if user_id not in self.user_id_to_col_index:
                return self.rating_mle - self.rating_offset
            else:
                return self.user_rating_mle[self.user_id_to_col_index[user_id]] - self.rating_offset
        else:
            if user_id not in self.user_id_to_col_index:
                return self.prod_rating_mle[self.asin_to_row_index[product_asin]] - self.rating_offset
            else:
                prod_vector = self.prod_embed[self.asin_to_row_index[product_asin]]
                user_vector = self.user_embed[self.user_id_to_col_index[user_id]]
                score = prod_vector.dot(user_vector)
                return score
# TODO Add binary rating and pred
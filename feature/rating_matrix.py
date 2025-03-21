from typing import Optional
import scipy
from schemas.data_models import Review
import json


class RatingMatrix:
    def __init__(self):
        self.row_index_to_asin = []
        self.col_index_to_user_id = []
        self.asin_to_row_index = {}
        self.user_id_to_col_index = {}
        self.n_row = 0
        self.n_col = 0
        self.sparse_array_rows = []
        self.row = []
        self.col = []
        self.val = []
        self.rating_offset = 3
    
    def serialize(
        self,
        filename
    ):
        json_dict = {
            'row_index_to_asin': self.row_index_to_asin,
            'col_index_to_user_id': self.col_index_to_user_id,
            'asin_to_row_index': self.asin_to_row_index,
            'user_id_to_col_index': self.user_id_to_col_index,
            'n_row': self.n_row,
            'n_col': self.n_col,
            'sparse_array_rows': self.sparse_array_rows,
            'row': self.row,
            'col': self.col,
            'val': self.val,
            'rating_offset': self.rating_offset
        }

        if self._filename_has_suffix(filename,suffix = '.json'):
            full_filename = filename
        else:
            full_filename = filename + '.json'
        
        with open(full_filename, 'w', encoding = 'utf-8') as f:
            json.dump(json_dict, f, ensure_ascii = False)
    
    def deserialize(
        self,
        path_to_json
    ):
        with open(path_to_json, 'r') as f:
            json_dict = json.load(f)
        self.row_index_to_asin = json_dict['row_index_to_asin']
        self.col_index_to_user_id = json_dict['col_index_to_user_id']
        self.asin_to_row_index = json_dict['asin_to_row_index']
        self.user_id_to_col_index = json_dict['user_id_to_col_index']
        self.n_row = json_dict['n_row']
        self.n_col = json_dict['n_col']
        self.sparse_array_rows = json_dict['sparse_array_rows']
        self.row = json_dict['row']
        self.col = json_dict['col']
        self.val = json_dict['val']
        self.rating_offset = json_dict['rating_offset']
    
    def _filename_has_suffix(self, filename, suffix = '.json'):
        suffix_len = len(suffix)
        return filename[-suffix_len:] == suffix
    
    def add_review(
        self,
        review: Review
    ):
        product_asin = review.parent_asin
        user_id = review.user_id
        rating = review.rating

        if product_asin not in self.asin_to_row_index:
            self.asin_to_row_index[product_asin] = self.n_row
            self.row_index_to_asin.append(product_asin)
            self.sparse_array_rows.append({})
            self.n_row += 1
        if user_id not in self.user_id_to_col_index:
            self.user_id_to_col_index[user_id] = self.n_col
            self.col_index_to_user_id.append(user_id)
            self.n_col += 1
        
        row_index = self.asin_to_row_index[product_asin]
        col_index = self.user_id_to_col_index[user_id]
        self.sparse_array_rows[row_index][col_index] = rating - self.rating_offset
        self.row.append(row_index)
        self.col.append(col_index)
        self.val.append(float(rating - self.rating_offset))
    
    def get_rating(
        self,
        product_asin: str,
        user_id: str,
        default_value: Optional[int] = 3
    ) -> Optional[int]:
        if product_asin in self.asin_to_row_index:
            row_index = self.asin_to_row_index[product_asin]
        else:
            return default_value
        if user_id in self.user_id_to_col_index:
            col_index = self.user_id_to_col_index[user_id]
        else:
            return default_value
        if default_value is None:
            rating = self.sparse_array_rows[row_index].get(
                col_index, 
                default_value
            )
        else:
            rating = self.sparse_array_rows[row_index].get(
                col_index, 
                default_value - self.rating_offset
            ) + self.rating_offset
        return rating

    def to_scipy_csr_array_with_offset(
        self
    ) -> scipy.sparse.csr_array:
        csr_array = scipy.sparse.csr_array((self.val, (self.row, self.col)))
        return csr_array
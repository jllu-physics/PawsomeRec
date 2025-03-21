from utils.data_utils import train_valid_test_split

if __name__ == '__main__':
    review_json_file_path = './data/Pet_Supplies.jsonl'
    meta_json_file_path = './data/meta_Pet_Supplies.jsonl'
    train_valid_test_split(
        review_json_file_path=review_json_file_path,
        meta_json_file_path=meta_json_file_path
    )
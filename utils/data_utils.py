import json
import os
import requests
from schemas.data_models import Product, Review
from tqdm import tqdm
import datetime

def get_product_from_json(product_json_str):
    prod_data = json.loads(product_json_str)
    product = Product(
        main_category = prod_data['main_category'],
        title = prod_data['title'],
        avg_rating = prod_data['average_rating'],
        rating_cnt = prod_data['rating_number'],
        features = prod_data['features'],
        description = prod_data['description'],
        price = prod_data['price'],
        images = get_large_image_links(prod_data['images']),
        store = prod_data['store'],
        categories = prod_data['categories'],
        details = prod_data['details'],
        parent_asin = prod_data['parent_asin']
    )
    return product

def get_review_from_json(review_json_str):
    review_data = json.loads(review_json_str)
    review = Review(
        rating = int(review_data['rating']),
        title = review_data['title'],
        text = review_data['text'],
        images = get_large_image_links(review_data['images'], key = 'large_image_url'),
        asin = review_data['asin'],
        parent_asin = review_data['parent_asin'],
        user_id = review_data['user_id'],
        timestamp = review_data['timestamp'],
        helpful_vote = review_data['helpful_vote'],
        verified_purchase = review_data['verified_purchase']
    )
    #print(type(review))
    return review

def get_large_image_links(image_dict_list, key = 'large'):
    large_image_links = []
    for image_dict in image_dict_list:
        if key in image_dict:
            large_image_links.append(image_dict[key])
    return large_image_links

def download_image(image_url, save_dir="images", filename=None):
    """Download an image from a URL and save it with a given filename."""
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Generate a filename if not provided
    if filename is None:
        filename = image_url.split("/")[-1]  # Extracts "51fOm+oAZbL._AC_.jpg"
    
    save_path = os.path.join(save_dir, filename)

    if file_exists(save_dir, filename):
        return True

    # Download the image
    response = requests.get(image_url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)
        print(f"✅ Saved: {save_path}")
        return True
    else:
        print(f"❌ Failed to download {image_url}")
        return False


def file_exists(target_dir, filename):
    existing_files = os.listdir(target_dir)
    return filename in existing_files

def pre_download_all_images(json_file_path, save_dir):
    with open(json_file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            image_links = get_large_image_links(data['images'])
            for image_link in image_links:
                download_image(image_link, save_dir)

def _train_valid_test_split(
        json_file_path: str,
        valid_start_time: datetime.datetime,
        test_start_time: datetime.datetime,
        ts_key = 'timestamp'
    ):
    valid_start_timestamp = valid_start_time.timestamp()
    test_start_timestamp = test_start_time.timestamp()

    json_train_path = json_file_path[:-6] + '_train.jsonl'
    json_valid_path = json_file_path[:-6] + '_valid.jsonl'
    json_test_path = json_file_path[:-6] + '_test.jsonl'
    f_train = open(json_train_path, 'w', encoding = "utf-8")
    f_valid = open(json_valid_path, 'w', encoding = "utf-8")
    f_test =  open(json_test_path, 'w', encoding = "utf-8")
    with open(json_file_path, "r", encoding = "utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            timestamp = data[ts_key] / 1000
            if timestamp < valid_start_timestamp:
                f_train.write(line)
            elif timestamp < test_start_timestamp:
                f_valid.write(line)
            else:
                f_test.write(line)
    
    f_train.close()
    f_valid.close()
    f_test.close()
            
def train_valid_test_split(
        review_json_file_path: str,
        meta_json_file_path: str,
        valid_start_time: datetime.datetime = datetime.datetime(2022,6,1,0,0,0),
        test_start_time: datetime.datetime = datetime.datetime(2023,1,1,0,0,0),
    ):
    _train_valid_test_split(
        review_json_file_path,
        valid_start_time,
        test_start_time,
        ts_key = 'timestamp'
    )
import pytest
from utils.data_utils import download_image

def test_download_image():
    # Example usage
    image_url = "https://m.media-amazon.com/images/I/51fOm+oAZbL._AC_.jpg"
    assert download_image(image_url, save_dir = './data/test_image', filename="my_pet_product.jpg")
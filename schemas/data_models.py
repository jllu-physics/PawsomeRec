from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class Product(BaseModel):
    main_category: str
    title: str
    avg_rating: float
    rating_cnt: int
    features: List[str]
    description: List[str]
    price: Optional[float]
    images: List[str]
    store: str
    categories: List[str]
    details: Dict[str, str]
    parent_asin: str

class Review(BaseModel):
    rating: int = Field(ge = 1, le = 5)
    title: str
    text: str
    images: List[str]
    asin: str
    parent_asin: str
    user_id: str
    timestamp: int
    helpful_vote: int
    verified_purchase: bool
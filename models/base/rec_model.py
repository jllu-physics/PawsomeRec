from abc import abstractmethod
import pandas as pd

class RecModel:
    def __init__(
        self, 
        user_embed_dim: int, 
        prod_embed_dim: int
    ):
        """
        Initialize recommendation model

        Parameters:
            user_embed_dim: the dimension of user embedding vector
            prod_embed_dim: the dimension of product embedding vector
        """
        pass

    @abstractmethod
    def train(
        self,
        train_df: pd.DataFrame
    ):
        """
        Train model
        """
        pass

    @abstractmethod
    def predict(
        self,
        pred_df: pd.DataFrame
    ):
        """
        Make new predictions
        """
        pass

# TODO: Add pydantic schema, ORM class and replace df with appropriate classes

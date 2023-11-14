import pandas as pd

class Normalizer:
    def __init__(self) -> None:
        self.mean = None
        self.std = None

    def normalize(self, data:pd.DataFrame) -> pd.DataFrame:
        self.mean = data.mean()
        self.std = data.std()
        normalized_df = (data - self.mean)/self.std
        return normalized_df


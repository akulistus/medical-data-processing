import pandas as pd
import numpy as np

class ForwardSelection():
    def __init__(self, estimator, n_features_to_select:int) -> None:
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select

    def fit(self, X:pd.DataFrame, y:pd.DataFrame, val_X:pd.DataFrame, val_Y:pd.DataFrame):
        
import pandas as pd
import numpy as np

def delete_corr(data_train:pd.DataFrame, data_test:pd.DataFrame, threshold:float) -> pd.DataFrame:

    tmp = data_train.corr().abs()
    upper = tmp.where(np.triu(np.ones(tmp.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data_train.drop(to_drop, axis=1, inplace=True)
    data_test.drop(to_drop, axis=1, inplace=True)

    return data_train, data_test
import numpy as np
from sklearn.datasets import load_iris

def get_irises() -> dict:
    data = load_iris()
    X = data.data
    Y = data.target
    Y_str = data.target_names

    iris_data = dict()
    for name in Y_str:
        flower_inds = Y == np.where(Y_str == name)
        flower_data = X[np.ravel(flower_inds), :]
        iris_data[name] = flower_data

    return iris_data, Y
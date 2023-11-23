import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def delete_corr(data_train:pd.DataFrame, data_test:pd.DataFrame, threshold:float) -> pd.DataFrame:

    tmp = data_train.corr().abs()
    upper = tmp.where(np.triu(np.ones(tmp.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    data_train.drop(to_drop, axis=1, inplace=True)
    data_test.drop(to_drop, axis=1, inplace=True)

    return data_train, data_test

def plot_hist(data_X:pd.DataFrame, data_Y:pd.DataFrame):
    class_1 = data_X[data_Y == 1]
    class_2 = data_X[data_Y == 0]

    sns.set()
    fig, ax = plt.subplots(4,4)
    rows = 0
    cols = 0
    for j in class_1.columns:
        if cols == 4:
            rows += 1
            cols = 0
        ax[rows,cols].hist(class_1[f"{j}"], alpha = 0.5)
        ax[rows,cols].hist(class_2[f"{j}"], alpha = 0.5)
        ax[rows,cols].set_title(f"{j}")
        cols += 1
    
    plt.show()

def split_result(data:pd.DataFrame) -> pd.DataFrame:
    data_Y = data["class"]
    data_X = data.drop(columns=["class"], axis=1)

    return data_X, data_Y

def add_ones(train_X:np.ndarray, test_X:np.ndarray):
    N = train_X.shape[0]
    train_X = np.hstack((train_X, np.ones((N,1))))
    N = test_X.shape[0]
    test_X = np.hstack((test_X, np.ones((N,1))))

    return train_X, test_X
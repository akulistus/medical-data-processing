import numpy as np
import pandas as pd
import funcs as f
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

class LogitRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.acc_loss = []
        self.val_loss = []
        self.acc_val_loss = []
    
    def sigmoid(self, X):
        Z = np.matmul(self.W, X.T)
        sigm = 1/(1+np.exp(-Z))
        return sigm
    
    def update_weights(self, X, Y):
        res = self.sigmoid(X)
        Y_T = Y.T
        eps=0
        self.loss.append((-1/self.m)*(np.sum((Y_T*np.log(res+eps)) + ((1-Y_T))*(np.log(1-res+eps)))))

        dW = (1/self.m)*np.matmul(X.T, (res - Y.T).T)

        self.W = self.W - self.learning_rate * dW.T
        
        res = np.where( res > 0.5, 1, 0)

        self.acc_loss.append(accuracy_score(Y.ravel(), res.ravel()))
        return self
    
    def update_val_loss(self, X, Y):
        k, p = X.shape
        res = self.sigmoid(X)
        Y_T = Y.T
        eps=0
        self.val_loss.append((-1/k)*(np.sum((Y_T*np.log(res+eps)) + ((1-Y_T))*(np.log(1-res+eps)))))

        res = np.where( res > 0.5, 1, 0)

        self.acc_val_loss.append(accuracy_score(Y.ravel(), res.ravel()))

    def fit(self, X, Y, val_X, val_Y):
        self.m, self.n = X.shape
        self.W = np.ones((1, self.n))

        for _ in range(self.iterations):
            self.update_val_loss(val_X, val_Y)
            self.update_weights(X, Y)
        return self

    def predict(self, X):
        Y = self.sigmoid(X)
        return Y
        # return np.where( Y > 0.5, 1, 0)

class Fisher():
    def __init__(self, learning_rate:float) -> None:
        self.learning_rate = learning_rate
        self.W = None
    
    def fit(self, data_X:pd.DataFrame, data_Y:pd.DataFrame):
        class_1 = data_X[data_Y == 1]*(-1)
        class_2 = data_X[data_Y == 0]
        new_dataframe = pd.concat([class_1, class_2])
        f.add_zeors()
        self.W = np.zeros((new_dataframe.shape[0],1))

        data_X = np.array(new_dataframe)
        projections = np.matmul(data_X, self.W)
        while min(projections) < 0:
            for ind, data_sample in enumerate(data_X):
                projections[ind] = np.matmul(data_sample, self.W)
                if self.projections[ind] < 0:
                    self.W = self.W + data_sample * self.learning_rate
            projections = np.matmul(data_X, self.W)

        return self
        
    def predict(self, data_X:pd.DataFrame,):
        data = np.array(data_X)
        return np.matmul(data, self.W)
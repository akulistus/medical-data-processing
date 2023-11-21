import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

class LogitRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def sigmoid(self, X):
        Z = np.matmul(self.W, X.T)
        sigm = np.exp(Z) / (1 + np.exp(Z))
        return sigm
    
    def update_weights(self, X, Y):
        res = self.sigmoid(X)
        Y_T = Y.T
        eps=1e-8
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
        eps=1e-8
        self.val_loss.append((-1/k)*(np.sum((Y_T*np.log(res+eps)) + ((1-Y_T))*(np.log(1-res+eps)))))

        res = np.where( res > 0.5, 1, 0)

        self.acc_val_loss.append(accuracy_score(Y.ravel(), res.ravel()))

    def fit(self, X, Y, val_X, val_Y):
        self.m, self.n = X.shape
        self.W = np.ones((1, self.n))
        self.loss = []
        self.acc_loss = []
        self.val_loss = []
        self.acc_val_loss = []

        for _ in range(self.iterations):
            self.update_val_loss(val_X, val_Y)
            self.update_weights(X, Y)
        return self

    def predict(self, X):
        Y = self.sigmoid(X)
        return Y
        # return np.where( Y > 0.5, 1, 0)

class Fisher():
    def __init__(self) -> None:
        self.W = None
    
    def fit(self, data_X:pd.DataFrame, data_Y:pd.DataFrame):
        class_1 = data_X[data_Y == 1]
        class_2 = data_X[data_Y == 0]

        class_1 = np.array(class_1)
        class_2 = np.array(class_2)
        
        diff_mean = np.mean(class_1, axis=0) - np.mean(class_2, axis=0)
        sum_covariance = np.cov(class_1, rowvar=0) + np.cov(class_2, rowvar=0)
        self.W = np.matmul(np.linalg.inv(sum_covariance), diff_mean)
        self.W = self.W/np.linalg.norm(self.W)
        return self
    
    def predict(self, data_X:pd.DataFrame, data_Y:pd.DataFrame):
        class_1 = np.array(data_X[data_Y == 1])
        class_2 = np.array(data_X[data_Y == 0])
        class_1 = np.matmul(class_1, self.W)
        class_2 = np.matmul(class_2, self.W)
        return class_1, class_2
    
    def _find_threashold(self,):
        pass
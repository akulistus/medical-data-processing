import numpy as np
from sklearn.preprocessing import normalize

class LogitRegression():

    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
    
    def sigmoid(self, X):
        Z = np.matmul(self.W, X.T)
        sigm = 1/(1+np.exp(-Z))
        return sigm
    
    def update_weights(self, X, Y):
        res = self.sigmoid(X)
        Y_T = Y.T
        self.loss.append((-1/self.m)*(np.sum((Y_T*np.log(res)) + ((1-Y_T))*(np.log(1-res)))))

        dW = (1/self.m)*np.matmul(X.T, (res - Y.T).T)

        self.W = self.W - self.learning_rate * dW.T

        return self

    def fit(self, X, Y):
        self.m, self.n = X.shape
        self.W = np.ones((1, self.n))

        for _ in range(self.iterations):
            self.update_weights(X, Y)
        return self

    def predict(self, X):
        Y = self.sigmoid(X)
        return np.where( Y > 0.5, 1, 0)


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

        self.threshold = self._find_threshold(class_1, class_2)

        return self
    
    def predict(self, data:pd.DataFrame):
        proj = np.matmul(data, self.W)

        return np.where(proj > self.threshold, 1, 0)
    
    def _find_threshold(self, class_1:np.ndarray, class_2:np.ndarray):
        proj1 = np.matmul(class_1, self.W)
        proj2 = np.matmul(class_2, self.W)

        proj1_mean = np.mean(proj1)
        proj2_mean = np.mean(proj2)

        proj1_std = np.std(proj1)
        proj2_std = np.std(proj2)

        threshold = self._solve_pdfs(proj1_mean, proj2_mean, proj1_std, proj2_std)
        return threshold

    def _solve_pdfs(self, pdf_mean_1, pdf_mean_2, pdf_std_1, pdf_std_2):
        coeff1 = 1/(2*pdf_std_1**2) - 1/(2*pdf_std_2**2)
        coeff2 = pdf_mean_2/(pdf_std_2**2) - pdf_mean_1/(pdf_std_1**2)
        coeff3 = pdf_mean_1**2 /(2*pdf_std_1**2) - pdf_mean_2**2 / (2*pdf_std_2**2) - np.log(pdf_std_2/pdf_std_1)

        coeffs = [coeff1, coeff2, coeff3]

        roots_of_eq = np.roots(coeffs)
        threshold = roots_of_eq[1]
        return threshold
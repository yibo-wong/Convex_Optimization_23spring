import numpy as np


class logistic:
    def __init__(self, X: np.array, y: np.array):  # w 1,000,y 10,000,x:10,000*1,000
        self.X = X
        self.y = y
        self.N = X.shape[0]
        self.dim = X.shape[1]

    def f(self, w: np.array):
        return np.average(np.log(1 + np.exp(-self.y * (self.X @ w)))

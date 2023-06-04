import numpy as np
import matplotlib.pyplot as plt
import time
import random

np.random.seed(1919810)
random.seed(1919810)

max_time = 5


class GD:
    def __init__(self, m, n, X, y, w0, alpha_ls, beta_ls):
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.X = X
        self.y = y
        self.w = w0.copy()
        self.alpha_ls = alpha_ls
        self.beta_ls = beta_ls
        self.f_his = []
        self.t_his = []
        self.steps = 0

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def line_search(self, x, direction):
        t = 5.0
        cur = self.f(x)
        df = self.df(x)
        while self.f(x + t * direction) > cur + self.alpha_ls * t * np.dot(df, direction):
            t *= self.beta_ls
        return t

    def step(self):
        d = -self.df(self.w)
        t = self.line_search(self.w, d)
        self.w += t*d
        cur = self.f(self.w)
        self.f_his.append(cur)
        self.steps += 1
        print("steps:", self.steps)
        print("f:", cur)

    def start(self):
        start = time.time()
        while True:
            self.step()
            end = time.time()
            if end-start > max_time:
                return
            self.t_his.append(end-start)


class AGD:
    def __init__(self, m, n, X, y, w0):
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.X = X
        self.y = y
        self.w = w0.copy()
        self.beta = np.linalg.norm(X, 2)**2 / (4*n)
        self.lam = 1
        self.gamma = 0
        self.steps = 1
        self.f_his = []
        self.t_his = []
        self.v = np.zeros_like(self.w)

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def step(self):
        lam_old = self.lam
        v_old = self.v
        self.lam = (1+np.sqrt(1+4*self.lam**2))/2
        self.gamma = (1-lam_old)/self.lam
        df = self.df(self.w)
        self.v = self.w-df/self.beta
        self.w = (1-self.gamma)*self.v+self.gamma*v_old
        cur = self.f(self.w)
        self.f_his.append(cur)
        self.steps += 1
        print("steps:", self.steps)
        print("f:", cur)

    def start(self):
        start = time.time()
        while True:
            self.step()
            end = time.time()
            if end-start > max_time:
                return
            self.t_his.append(end-start)


if __name__ == "__main__":
    X = np.random.randn(1000, 10000)
    y = np.random.randint(0, 2, size=10000) * 2 - 1
    w = np.array([0.0]*1000)

    gd = GD(1000, 10000, X.copy(), y.copy(), w.copy(), 0.4, 0.8)
    gd.start()

    agd = AGD(1000, 10000, X.copy(), y.copy(), w.copy())
    agd.start()

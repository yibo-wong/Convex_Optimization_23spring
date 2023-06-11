import numpy as np
import matplotlib.pyplot as plt
import time
import random

np.random.seed(1919810)

max_time = 30.0
max_iter = 1000


class GD:
    def __init__(self, m, n, X, y, w0, alpha, beta):
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.X = X
        self.y = y
        self.w = w0.copy()
        self.alpha = alpha
        self.beta = beta
        self.f_his = []
        self.t_his = []
        self.steps = 0
        self.eta = 0

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def line_search(self, x, direction):
        t = 5.0
        cur = self.f(x)
        df = self.df(x)
        while self.f(x + t * direction) > cur + self.alpha * t * np.dot(df, direction):
            t *= self.beta
        return t

    def step(self):
        d = -self.df(self.w)
        t = self.line_search(self.w, d)
        self.w += t*d
        cur = self.f(self.w)
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
            self.f_his.append(self.f(self.w))

    def export_data(self, name: str):
        return [name, self.f_his, self.t_his]


class pLADMPSAP:
    def __init__(self, m, n, X, y, w0, beta, eta):
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.X = X
        self.y = y
        self.w0 = w0.copy()
        self.W = None
        self.t0 = None
        self.T = None
        self.tau0 = None
        self.TAU = None
        self.eta0 = eta
        self.ETA = None
        self.f_his = []
        self.t_his = []
        self.steps = 0
        self.beta = beta
        self.LAM = None

    def setParameter(self):
        self.T = np.linalg.norm(self.X, ord=2, axis=0) / (self.n * 4) + 1.0
        self.t0 = 1.0
        self.eta0 = (self.n)**2.5 + 1.0
        self.ETA = (self.n+5.0)*np.ones(self.n)
        self.tau0 = self.t0 + self.beta * self.eta0
        self.TAU = self.T + self.beta * self.ETA
        self.W = np.zeros((self.m, self.n))
        for i in range(self.n):
            self.W[:, i] = self.w0.copy()
        self.LAM = np.zeros((self.m, self.n))

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def df_i(self, w, i):
        y = self.y[i]
        return (1 - 1 / (np.exp(-(y*self.X[:, i]) @ w) + 1)) * (-y * self.X[:, i])

    def step(self):
        dLAM = self.LAM.copy()
        w0 = self.w0.copy()
        dw0 = (1 / self.tau0) * np.sum(self.LAM, axis=1)
        self.w0 += dw0
        for i in range(self.n):
            self.W[:, i] -= (1 / self.TAU[i]) * (self.LAM[:, i] + self.df_i(self.W[:, i], i))
            dLAM[:, i] = self.W[:, i] - w0
        self.LAM += self.beta * dLAM
        cur = self.f(self.w0)
        self.steps += 1
        print("steps:", self.steps)
        print("f:", cur)

    def start(self):
        plad.setParameter()
        start = time.time()
        while True:
            self.step()
            end = time.time()
            if end-start >= max_time:
                return
            self.t_his.append(end-start)
            self.f_his.append(self.f(self.w0))

    def export_data(self, name: str):
        return [name, self.f_his, self.t_his]

# f: 0.5921396703288766


class Plot:
    def __init__(self) -> None:
        self.name = []
        self.f = []
        self.t = []

    def recv_data(self, obj):
        self.name.append(obj[0])
        self.f.append(obj[1])
        self.t.append(obj[2])

    def plot(self):
        plt.figure()
        for i in range(len(self.name)):
            print(i)
            plt.plot(self.t[i], self.f[i], label=self.name[i])
        plt.legend()
        plt.savefig("pLADMPSAP.png")
        plt.show()


if __name__ == "__main__":
    X = np.random.randn(100, 500)
    y = np.random.randint(0, 2, size=500) * 2 - 1
    w = np.array([0.0]*100)

    gd = GD(100, 500, X.copy(), y.copy(), w.copy(), 0.4, 0.8)
    gd.start()

    plad = pLADMPSAP(100, 500, X.copy(), y.copy(), w.copy(), 0.001, 0.0001)
    plad.start()

    plot = Plot()
    plot.recv_data(gd.export_data("gd"))
    plot.recv_data(plad.export_data("pLADMPSAP"))
    plot.plot()

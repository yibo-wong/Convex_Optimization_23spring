import numpy as np
import matplotlib.pyplot as plt
import time


class ConjGrad:
    def __init__(self, x0: np.array, alpha: float, beta: float, eta: float, a: float):
        self.x0 = x0
        self.x = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.f_his = []
        self.time_his = []
        self.total_time = 0
        self.start_time = time.time()
        self.g = None
        self.d = None
        self.a = a
        self.first_time = 1

    def f(self, x: np.array):
        xx = x.copy().reshape(-1, 2)
        x1 = xx[:, 0]
        x2 = xx[:, 1]
        return np.sum(self.a * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2)

    def df(self, x: np.array):
        xx = x.copy().reshape(-1, 2)
        x1 = xx[:, 0]
        x2 = xx[:, 1]
        xc1 = 2 * self.a * (x2 - x1 ** 2)
        xc0 = 4 * self.a * (x1 ** 2 - x2) * x1 + 2 * (x1 - 1)
        return np.hstack([xc0.reshape(-1, 1), xc1.reshape(-1, 1)]).reshape(-1)

    def cg_step(self, method):
        print(self.x, file=fl)
        t = 1.0
        g = self.df(self.x)
        dfdf = np.dot(g, g)
        if dfdf < self.eta * self.eta:
            return False
        f = self.f(self.x)
        d = None
        if self.first_time:
            d = -g
        else:
            beta = 1
            if method == "Hestenes-Stiefel":
                assert np.dot(self.d, g - self.g) != 0
                beta = np.dot(g, g - self.g) / np.dot(self.d, g - self.g)
            elif method == "Polak-Ribiere":
                beta = np.dot(g, g - self.g) / np.dot(self.g, self.g)
            elif method == "Fletcher-Reeves":
                beta = np.dot(g, g) / np.dot(self.g, self.g)
            d = beta * self.d - g
        while self.f(self.x + t * d) > f + self.alpha * t * np.dot(g, d):
            t = self.beta * t

        self.x += t * d
        self.d = d
        self.g = g

        self.f_his.append(f)
        self.time_his.append(time.time() - self.start_time)
        self.first_time = 0
        return True

    def plot(self, figName, time=0):
        if time == 1:
            plt.figure()
            plt.plot(self.time_his, [np.log(i) for i in self.f_his], "o-", color="r")
            ax = plt.gca()
            self.total_time = self.time_his[len(self.time_his-1)]
            ax.set_xlabel(f"total time:{round(self.total_time*1000,3)} s")
            ax.set_ylabel(r"$log(f(x)-p^*)$")
            plt.savefig(figName + ".png")
            plt.show()
        else:
            plt.figure()
            plt.plot(range(len(self.f_his)), [np.log(i) for i in self.f_his], "o-", color="g")
            ax = plt.gca()
            ax.set_xlabel(f"steps:{len(self.f_his)-1}")
            ax.set_ylabel(r"$log(f(x)-p^*)$")
            plt.savefig(figName + ".png")
            plt.show()

    def start(self, method):
        while self.cg_step(method):
            pass


if __name__ == "__main__":
    n = 100
    x0 = np.array([-1.0] * n)
# Hestenes-Stiefel
    fl = open("hs.log", mode="w")

    HSconj = ConjGrad(x0.copy(), 0.8, 0.5, 1e-5, 1.0)
    HSconj.start("Hestenes-Stiefel")
    HSconj.plot("Hestenes-Stiefel")

    fl.close()
# Polak-Ribiere
    fl = open("pr.log", mode="w")

    PRconj = ConjGrad(x0.copy(), 0.8, 0.5, 1e-5, 1.0)
    PRconj.start("Polak-Ribiere")
    PRconj.plot("Polak-Ribiere")

    fl.close()
# Fletcher-Reeves
    fl = open("fr.log", mode="w")

    FRconj = ConjGrad(x0.copy(), 0.8, 0.5, 1e-5, 1.0)
    FRconj.start("Fletcher-Reeves")
    FRconj.plot("Fletcher-Reeves")

    fl.close()

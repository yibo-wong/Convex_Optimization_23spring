import numpy as np
import matplotlib.pyplot as plt
import time


class DFP:
    def __init__(self, x0: np.array, H: np.array, alpha: float, beta: float, eta: float):
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
        self.H = H

    def f(self, x: np.array):
        x1, x2 = x[0], x[1]
        return x1 ** 4 / 4 + x2 ** 2 / 2 - x1 * x2 + x1 - x2

    def df(self, x: np.array):
        x1, x2 = x[0], x[1]
        return np.array([x1 ** 3 - x2 + 1, x2 - x1 - 1])

    def dfp_step(self):
        pass
        print(self.x, file=fl)
        alpha = 1.0
        g = self.df(self.x)
        dfdf = np.dot(g, g)
        if dfdf < self.eta * self.eta:
            return False
        f = self.f(self.x)
        d = -self.H @ g

        while self.f(self.x + alpha * d) > f + self.alpha * alpha * np.dot(g, d):
            alpha = self.beta * alpha
        dx = alpha * d
        self.x += dx

        dg = self.df(self.x) - g

        assert np.dot(dx, dg) != 0
        assert np.dot(dg, self.H@dg) != 0

        self.H += np.outer(dx, dx)/np.dot(dx, dg)-np.outer(self.H@dg, self.H@dg)/np.dot(dg, self.H@dg)

        self.d = d
        self.g = g

        self.f_his.append(f)
        self.time_his.append(time.time() - self.start_time)

        return True

    def plot(self, figName, time=0):
        if time == 1:
            plt.figure()
            plt.plot(self.time_his, self.f_his, "o-", color="r")
            ax = plt.gca()
            self.total_time = self.time_his[len(self.time_his-1)]
            ax.set_xlabel(f"total time:{round(self.total_time*1000,3)} s")
            ax.set_ylabel(r"$f(x)$")
            plt.savefig(figName + ".png")
            plt.show()
        else:
            plt.figure()
            plt.plot(range(len(self.f_his)), self.f_his, "o-", color="g")
            ax = plt.gca()
            ax.set_xlabel(f"steps:{len(self.f_his)-1}")
            ax.set_ylabel(r"$f(x)$")
            plt.savefig(figName + ".png")
            plt.show()

    def start(self):
        while self.dfp_step():
            pass


if __name__ == "__main__":
    H0 = np.identity(2)
    x0_1 = np.array([0.0, 0.0])
    x0_2 = np.array([1.5, 1.0])

    fl = open("dfp_1.log", mode="w")
    dfp = DFP(x0_1, H0.copy(), 0.5, 0.5, 1e-5)
    dfp.start()
    dfp.plot("dfp_1")
    fl.close()

    fl = open("dfp_2.log", mode="w")
    dfp = DFP(x0_2, H0.copy(), 0.5, 0.5, 1e-5)
    dfp.start()
    dfp.plot("dfp_2")
    fl.close()

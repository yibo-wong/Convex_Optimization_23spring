import numpy as np
import matplotlib.pyplot as plt
import time


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


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
        self.pos = []

    def f(self, x: np.array):
        x1, x2, x3 = x[0], x[1], x[2]
        return (3 - x1) ** 2 + 7 * (x2 - x1 ** 2) ** 2 + 9 * (x3 - x1 - x2 ** 2) ** 2

    def df(self, x: np.array):
        x1, x2, x3 = x[0], x[1], x[2]
        return np.array([2 * (x1 - 3) + 28 * (x1 ** 2 - x2) * x1 + 18 * (-x3 + x1 + x2 ** 2),
                         36 * x2 ** 3 + 36 * x1 * x2 - 36 * x2 * x3 + 14 * x2 - 14 * x1 ** 2,
                         -18 * x2 ** 2 - 18 * x1 + 18 * x3])

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

        # check whether positive definite
        if is_pos_def(self.H):
            self.pos.append(1)
        else:
            self.pos.append(0)

        while self.f(self.x + alpha * d) > f + self.alpha * alpha * np.dot(g, d):
            alpha = self.beta * alpha
        dx = alpha * d
        self.x += dx

        dg = self.df(self.x) - g

        Hdg = self.H @ dg
        dgdx = np.dot(dg, dx)
        dgHdg = np.dot(dg, Hdg)
        assert dgdx != 0 and dgHdg != 0

        self.H += np.outer(dx, dx) / dgdx - np.outer(Hdg, Hdg) / dgHdg

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

    def plot_pos(self):
        plt.figure()
        plt.plot(range(len(self.pos)), self.pos, "o-", color="purple")
        ax = plt.gca()
        ax.set_xlabel("1 means it's positive definite,0 means it's not")
        plt.savefig("dfp_pos_def.png")
        plt.show()

    def start(self):
        while self.dfp_step():
            pass


class BFGS:
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
        self.pos = []

    def f(self, x: np.array):
        x1, x2, x3 = x[0], x[1], x[2]
        return (3 - x1) ** 2 + 7 * (x2 - x1 ** 2) ** 2 + 9 * (x3 - x1 - x2 ** 2) ** 2

    def df(self, x: np.array):
        x1, x2, x3 = x[0], x[1], x[2]
        return np.array([2 * (x1 - 3) + 28 * (x1 ** 2 - x2) * x1 + 18 * (-x3 + x1 + x2 ** 2),
                         36 * x2 ** 3 + 36 * x1 * x2 - 36 * x2 * x3 + 14 * x2 - 14 * x1 ** 2,
                         -18 * x2 ** 2 - 18 * x1 + 18 * x3])

    def bfgs_step(self):
        pass
        print(self.x, file=fl)
        alpha = 1.0
        g = self.df(self.x)
        dfdf = np.dot(g, g)
        if dfdf < self.eta * self.eta:
            return False
        f = self.f(self.x)

        # check whether positive definite
        if is_pos_def(self.H):
            self.pos.append(1)
        else:
            self.pos.append(0)

        d = -self.H @ g

        while self.f(self.x + alpha * d) > f + self.alpha * alpha * np.dot(g, d):
            alpha = self.beta * alpha
        dx = alpha * d
        self.x += dx

        dg = self.df(self.x) - g
        Hdg = self.H @ dg
        dgdx = np.dot(dg, dx)
        Hdgdx = np.outer(Hdg, dx)
        assert dgdx != 0

        self.H += (1 + np.dot(dg, Hdg) / dgdx) / dgdx * np.outer(dx, dx) - (Hdgdx + Hdgdx.transpose()) / dgdx

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

    def plot_pos(self):
        plt.figure()
        plt.plot(range(len(self.pos)), self.pos, "o-", color="purple")
        ax = plt.gca()
        ax.set_xlabel("1 means it's positive definite,0 means it's not")
        plt.savefig("bfgs_pos_def.png")
        plt.show()

    def start(self):
        while self.bfgs_step():
            pass


if __name__ == "__main__":
    H0 = np.identity(3)
    x0 = np.array([1.0, 1.0, 1.0])

    fl = open("dfp.log", mode="w")
    dfp = DFP(x0.copy(), H0.copy(), 0.8, 0.5, 1e-5)
    dfp.start()
    dfp.plot("dfp")
    dfp.plot_pos()
    fl.close()

    fl = open("bfgs.log", mode="w")
    bfgs = BFGS(x0.copy(), H0.copy(), 0.8, 0.5, 1e-5)
    bfgs.start()
    bfgs.plot("bfgs")
    bfgs.plot_pos()
    fl.close()

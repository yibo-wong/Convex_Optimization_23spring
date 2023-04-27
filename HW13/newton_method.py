import numpy as np
import matplotlib.pyplot as plt
import time


class Newton:
    def __init__(self, x0: np.array, alpha: float, beta: float, eta: float):
        self.x0 = x0
        self.x = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.f_his = []
        self.time_his = []
        self.total_time = 0

    def f(self, x: np.array):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(self, x: np.array):
        return np.array([2 * (x[0] - 1) + 400 * (x[0] ** 2 - x[1]) * x[0], 200 * (x[1] - x[0] ** 2)])

    def ddf(self, x: np.array):
        return np.array([[800 * x[0] ** 2 + 2 + 400 * (x[0] ** 2 - x[1]), -400 * x[0]], [-400 * x[0], 200],])

    def rx(self, x: np.array):
        return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])

    def jx(self, x: np.array):
        return np.array([[-20 * x[0], 10], [-1, 0]])

    def step_damped(self):
        start = time.perf_counter()

        t = 1.0
        df = self.df(self.x)
        dx = -np.linalg.inv(self.ddf(self.x)) @ df
        dfdf = np.dot(df, df)

        cur = self.f(self.x)
        self.f_his.append(cur)

        end = time.perf_counter()
        self.total_time += end - start
        self.time_his.append(self.total_time)

        if dfdf < self.eta * self.eta:
            return False
        while self.f(self.x + t * dx) > cur + self.alpha * t * np.dot(df, dx):
            t = self.beta * t
        self.x += t * dx
        return True

    def step_gauss(self):
        start = time.perf_counter()

        j = self.jx(self.x)
        jj = np.linalg.inv(j.T @ j) @ j.T
        dx = -jj @ self.rx(self.x)
        df = self.df(self.x)
        cur = self.f(self.x)

        self.f_his.append(cur)

        end = time.perf_counter()
        self.total_time += end - start
        self.time_his.append(self.total_time)

        dfdf = np.dot(df, df)
        if dfdf < self.eta * self.eta:
            return False
        self.x += dx
        return True

    def start(self, method: str):
        if method == "damped":
            while self.step_damped():
                pass
        elif method == "gauss":
            while self.step_gauss():
                pass

    def plot(self, figName, time=0):
        if time == 1:
            plt.figure()
            plt.plot(self.time_his, [np.log(i) for i in self.f_his], "o-", color="r")
            ax = plt.gca()
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


if __name__ == "__main__":
    x0 = np.array([-2.0, 2.0])
    damped_newton = Newton(x0, 0.5, 0.8, 1e-9)
    damped_newton.start("damped")
    damped_newton.plot("damped_newton")
    damped_newton.plot("damped_newton_time", time=1)
    print("steps,", len(damped_newton.f_his)-1)
    x1 = np.array([-2.0, 2.0])
    gauss_newton = Newton(x1, 0.5, 0.5, 1e-9)
    gauss_newton.start("gauss")
    gauss_newton.plot("gauss_newton")
    gauss_newton.plot("gauss_newton_time", time=1)
    print("steps,", len(gauss_newton.f_his)-1)

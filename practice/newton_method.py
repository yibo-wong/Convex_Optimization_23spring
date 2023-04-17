import numpy as np


class Newton:
    def __init__(self, x0: np.array, alpha: float, beta: float, eta: float):
        self.x0 = x0
        self.x = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.iteTime = 0

    def f(self, x: np.array):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def df(self, x: np.array):
        return np.array(
            [2 * (x[0] - 1) + 400 * (x[0] ** 2 - x[1]) * x[0], 200 * (x[1] - x[0] ** 2)]
        ).T

    def ddf(self, x: np.array):
        return np.array(
            [
                [800 * x[0] ** 2 + 2 + 400 * (x[0] ** 2 - x[1]), -400 * x[0]],
                [-400 * x[0], 200],
            ]
        )

    def rx(self, x: np.array):
        return np.array([10 * (x[1] - x[0] ** 2), (1 - x[0])])

    def jx(self, x: np.array):
        return np.array([[-20 * x[0], 10], [-1, 0]])

    def step_damped(self):
        t = 1.0
        df = self.df(self.x)
        dx = -np.linalg.inv(self.ddf(self.x)) @ df
        dfdf = np.dot(df, df)
        if dfdf < self.eta * self.eta:
            return False
        cur = self.f(self.x)
        print(f"ite{self.iteTime}:{cur}")
        while self.f(self.x + t * dx) > cur + self.alpha * t * np.dot(df, dx):
            t = self.beta * t
        self.x += t * dx
        # self.x_his.append(self.x)
        # cur = self.f(self.x)
        # self.f_his.append(cur)
        self.iteTime += 1
        return True

    def step_gauss(self):
        j = self.jx(self.x)
        jj = np.linalg.inv(j.T @ j) @ j.T
        dx = -jj @ self.rx(self.x)
        df = self.df(self.x)
        cur = self.f(self.x)
        print(f"ite{self.iteTime}:{cur}")
        dfdf = np.dot(df, df)
        if dfdf < self.eta * self.eta:
            return False
        self.x += dx
        self.iteTime += 1
        return True

    def start(self, method: str):
        if method == "damped":
            while self.step_damped():
                pass
        elif method == "GN":
            while self.step_gauss():
                pass


if __name__ == "__main__":
    x0 = np.array([-2.0, 2.0])
    damped_newton = Newton(x0, 0.5, 0.5, 1e-10)
    damped_newton.start("damped")
    x1 = np.array([-2.0, 2.0])
    gauss_newton = Newton(x1, 0.5, 0.5, 1e-10)
    gauss_newton.start("GN")

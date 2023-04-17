import numpy as np
import matplotlib.pyplot as plt


class Gradient:
    def __init__(
        self, A: np.array, x0: np.array, alpha: float, beta: float, eta: float
    ):
        self.A = A
        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.x = x0
        self.x_his = []
        self.f_his = []

    def f(self, x: np.array):
        return np.sum(np.exp(self.A.T @ x) + np.exp(-self.A.T @ x))

    def df(self, x: np.array):
        return self.A @ (np.exp(self.A.T @ x) - np.exp(-self.A.T @ x))

    def step(self):
        t = 1.0
        dx = self.df(self.x)
        dxdx = np.dot(dx, dx)
        if dxdx < self.eta * self.eta:
            return False
        cur = self.f(self.x)
        while self.f(self.x - t * dx) > cur - self.alpha * t * dxdx:
            t = self.beta * t
        self.x -= t * dx
        self.x_his.append(self.x)
        cur = self.f(self.x)
        self.f_his.append(cur)
        return True

    def descend(self):
        while self.step():
            # print(self.x)
            pass

    def plot_f(self):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his)
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$f(x)=\sum e^{a_i^Tx}+e^{-a_i^Tx}$")
        plt.show()


def start(alpha, beta):
    np.random.seed(2023)
    A = np.random.randn(3, 3)
    print(A)
    x0 = np.array([1.0, 1.0, 1.0]).T
    gd = Gradient(A, x0, alpha, beta, eta=1e-6)
    gd.descend()
    gd.plot_f()


if __name__ == "__main__":
    # np.random.seed(2023)
    # A = np.random.randn(3, 3)
    # x0 = np.array([1.0, 1.0, 1.0]).T
    # print(A @ x0)

    start(0.5, 0.8)

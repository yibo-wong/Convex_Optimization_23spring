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
        self.step_length_his = []
        self.f_opt = 2 * A.shape[1]
        self.total_steps = 0

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
        self.f_his.append(cur - self.f_opt)
        self.step_length_his.append(np.linalg.norm(t * dx, ord=2))
        self.total_steps += 1
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
        ax.set_ylabel(r"$f(x)-p^*=\sum e^{a_i^Tx}+e^{-a_i^Tx}-p^*$")
        ax.set_yscale("log")
        plt.show()

    def plot_step_length(self):
        plt.figure()
        plt.plot(range(len(self.step_length_his)), self.step_length_his)
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$||x_{k+1}-x_{k}||_2$")
        ax.set_yscale("log")
        plt.show()


def start(alpha, beta):
    np.random.seed(1919810)
    m = 10  # col
    n = 5  # row
    A = np.random.randn(n, m)
    # print(A)
    x0 = np.array([1.0] * n)
    # print(x0)
    gd = Gradient(A, x0, alpha, beta, eta=1e-6)
    gd.descend()
    gd.plot_f()
    gd.plot_step_length()


if __name__ == "__main__":
    # np.random.seed(2023)
    # A = np.random.randn(3, 3)
    # x0 = np.array([1.0, 1.0, 1.0]).T
    # print(A @ x0)

    start(0.6, 0.8)

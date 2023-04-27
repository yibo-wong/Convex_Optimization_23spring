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
        df = self.df(self.x)
        dfdf = np.dot(df, df)
        if dfdf < self.eta * self.eta:
            return False
        cur = self.f(self.x)
        dx = np.array([-float(np.sign(i)) for i in df])

        while self.f(self.x + t * dx) > cur + self.alpha * t * np.dot(dx, df):
            t = self.beta * t
        self.x += t * dx
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
        plt.plot(range(len(self.f_his)), [np.log(i) for i in self.f_his])
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$log(f(x)-p^*)=log(\sum e^{a_i^Tx}+e^{-a_i^Tx}-p^*)$")
        # ax.set_yscale("log")
        plt.savefig("gd_function_value_figure.png")
        plt.show()

    def plot_step_length(self):
        plt.figure()
        plt.plot(range(len(self.step_length_his)),
                 self.step_length_his, color="g")
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)},log-scale y-axis")
        ax.set_ylabel(r"$||x_{k+1}-x_{k}||_2$")
        ax.set_yscale("log")
        plt.savefig("gd_step_length_figure.png")
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
    start(0.3, 0.8)

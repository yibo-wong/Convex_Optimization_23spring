import numpy as np
import matplotlib.pyplot as plt


class Penalty:
    def __init__(
        self, A: np.array, b: np.array, x0: np.array, alpha: float, beta: float, gamma: float, eta: float
    ):
        self.A = A
        self.b = b
        self.x0 = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.x = x0
        self.f_his = []
        self.q_his = []
        self.gamma = gamma
        self.x_opt = self.A.T @ np.linalg.inv(self.A @ self.A.T) @ self.b
        self.steps = 0

    def f(self, x: np.array):
        return np.dot(x, x) * 0.5

    def f_1(self, x: np.array):
        return np.linalg.norm(x, 2) * 0.5

    def q_abs(self, x: np.array):
        return np.dot(x, x) * 0.5 + self.gamma * np.linalg.norm(self.A @ x - self.b, 1)

    def q_cb(self, x: np.array):
        return np.dot(x, x) * 0.5 + self.gamma * np.sum((self.A @ x - self.b) ** 2)

    def dq_abs(self, x: np.array):
        return x + self.gamma * self.A.T @ (self.A @ x - self.b)

    def dq_cb(self, x: np.array):
        return x + self.gamma * 2 * self.A.T @ (self.A @ x - self.b)

    def step(self, func: str, method: str):
        if func == "dot":
            dq = 0
            d = 0
            cur = 0
            t = 1.0
            if method == "abs":
                dq = self.dq_abs(self.x)
                d = dq/np.linalg.norm(dq, 2)
                cur = self.q_abs(self.x)
                dd = np.dot(d, d)
                while self.q_abs(self.x - t * d) > cur - self.alpha * t * dd:
                    t = self.beta * t
            elif method == "cb":
                dq = self.dq_cb(self.x)
                d = dq/np.linalg.norm(dq, 2)
                cur = self.q_cb(self.x)
                dd = np.dot(d, d)
                while self.q_cb(self.x - t * d) > cur - self.alpha * t * dd:
                    t = self.beta * t
            self.x -= t * d
            old_value = cur
            if method == "abs":
                cur = self.q_abs(self.x)
            elif method == "cb":
                cur = self.q_cb(self.x)
            self.f_his.append(self.f(self.x))
            self.q_his.append(cur)
            self.steps += 1

    def plot(self, func: str, name: str):
        if func == "dot":
            f_opt = np.dot(self.x_opt, self.x_opt)
            plt.figure()
            plt.plot(range(len(self.f_his)), [np.log(np.abs(i - f_opt)) for i in self.f_his])
            # plt.plot(range(len(self.f_his)), self.f_his)
            ax = plt.gca()
            ax.set_xlabel(f"steps:{len(self.f_his)}")
            ax.set_ylabel(r"$log(|f(x)-f^*|)$")
            plt.savefig(name + ".png")
            plt.show()

    def start(self, func: str, method: str, name: str):
        while True:
            print("steps: ", self.steps)
            print("x: ", self.f(self.x))
            print("p: ", np.linalg.norm(self.A@self.x-self.b, 1))
            print()
            self.step(func, method)
            self.gamma += 5
            if np.linalg.norm(self.x-self.x_opt, 2) < 1e-5:
                break
        self.plot(func, name)


if __name__ == "__main__":
    np.random.seed(1919810)
    A = np.random.randn(200, 300)
    b = np.random.randn(200)
    x0 = np.array([0.0]*300)
    penalty_abs = Penalty(A.copy(), b.copy(), x0.copy(), 0.5, 0.8, 10, 1e-10)
    penalty_abs.start("dot", "abs", "pen_abs")
    penalty_cb = Penalty(A.copy(), b.copy(), x0.copy(), 0.5, 0.8, 10, 1e-10)
    penalty_cb.start("dot", "cb", "pen_cb")

import numpy as np
import matplotlib.pyplot as plt


class L_BFGS:
    def __init__(self, n: int, x0: np.array, alpha: float, beta: float, eta: float, a: float, m_size: int):
        self.x = x0
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.f_his = []
        self.m_size = m_size
        self.memory = [[None, None] for i in range(m_size)]
        self.a = a
        self.k = 0
        self.dx_temp = None
        self.dg_temp = None

    def update(self, x, g):
        self.memory.pop(0)
        self.memory.append([x, g])

    def get_x(self, i: int):
        return self.memory[self.m_size - i - 1][0]

    def get_g(self, i: int):
        return self.memory[self.m_size - i - 1][1]

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

    def l_bfgs_step(self):
        print(self.x)
        g = self.df(self.x)
        if self.k != 0:
            self.update(self.dx_temp, g - self.dg_temp)
        dfdf = np.dot(g, g)
        if dfdf < self.eta * self.eta:
            return False
        f = self.f(self.x)
        self.f_his.append(f)
        # calc H_k^0. there is no need to write it in matrix form.
        gamma = 1.0
        if self.k != 0:
            gamma = np.dot(self.get_g(0), self.get_x(0)) / np.dot(self.get_g(0), self.get_g(0))
        # loop 1
        alpha_his = []
        q = g.copy()
        for i in range(min(self.m_size, self.k)):
            alpha = np.dot(self.get_x(i), q) / np.dot(self.get_x(i), self.get_g(i))
            q = q - alpha * self.get_g(i)
            alpha_his.append(alpha)
        # p <== q
        p = gamma * q
        # loop 2
        for i in range(min(self.m_size, self.k)-1, -1, -1):
            beta = np.dot(self.get_g(i), p) / np.dot(self.get_x(i), self.get_g(i))
            p = p + (alpha_his[i] - beta)*self.get_x(i)
        d = -p
        # line search
        t = 1.0
        while self.f(self.x + t * d) > f + self.alpha * t * np.dot(g, d):
            t = self.beta * t
        dx = t * d
        # update memory
        self.x += dx
        self.dx_temp = dx
        self.dg_temp = g
        self.k += 1
        return True

    def plot(self, figName, time=0):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his, "o-", color="g")
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)-1}")
        ax.set_ylabel(r"$f(x)$")
        plt.savefig(figName + "_m_5.png")
        plt.show()

    def start(self):
        while self.l_bfgs_step():
            pass


if __name__ == "__main__":
    n = 2000
    x0 = np.array([-1.0] * n)
    bfgs = L_BFGS(n, x0.copy(), 0.5, 0.5, 1e-7, 0.5, 5)
    bfgs.start()
    bfgs.plot("bfgs")

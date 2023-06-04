import numpy as np
import matplotlib.pyplot as plt


class FW:
    def __init__(
        self, y: np.array, D: np.array, x0: np.array, eta: float
    ):
        self.x0 = x0
        self.y = y
        self.D = D
        self.eta = eta
        self.x = x0
        self.x_his = []
        self.f_his = []
        self.k = 0

    def f(self, x: np.array):
        ydx = self.y-self.D@x
        return np.dot(ydx, ydx)

    def df(self, x: np.array):
        return 2*self.D.T@(-self.y+self.D@x)

    def step(self, method: str):
        if method == "inf":
            self.x_his.append(self.x)
            cur = self.f(self.x)
            self.f_his.append(cur)

            g = self.df(self.x)
            s = -np.sign(g)
            gamma = 2/(self.k+2)
            dx = gamma*(s-self.x)

            if np.dot(dx, dx) < self.eta or self.k > 5000:
                return False

            self.x += dx
            self.k += 1
            return True
        elif method == "1":
            self.x_his.append(self.x)
            cur = self.f(self.x)
            self.f_his.append(cur)

            g = self.df(self.x)
            s = np.zeros_like(g)
            max_index = np.argmax(np.abs(g))
            s[max_index] = -np.sign(g[max_index])
            gamma = 2/(self.k+2)
            dx = gamma*(s-self.x)

            if np.dot(dx, dx) < self.eta or self.k > 5000:
                return False

            self.x += dx
            self.k += 1
            return True

    def descend(self, method: str):
        while self.step(method):
            print("step: ", self.k)
            print(self.f(self.x))
            pass

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his)
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        plt.savefig(name+".png")
        plt.show()


if __name__ == "__main__":
    np.random.seed(114514)
    D = np.random.randn(200, 300)
    y = np.random.randn(200)
    # x0 = np.array([0.0]*300)
    x0 = np.random.randn(300)

    fw_inf = FW(y.copy(), D.copy(), x0.copy(), 1e-5)
    fw_inf.descend("inf")
    fw_inf.plot("frank-wolfe_inf_norm")

    fw_one = FW(y.copy(), D.copy(), x0.copy(), 1e-5)
    fw_one.descend("1")
    fw_one.plot("frank-wolfe_1_norm")

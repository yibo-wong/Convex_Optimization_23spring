import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


class DA:
    def __init__(
        self, a: np.array, b: np.array, x0: np.array, y0: np.array, lam_0: np.array, alpha: float, beta: float, eta: float
    ):
        self.x = x0
        self.y = y0
        self.lam = lam_0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.x_his = []
        self.f_his = []
        self.total_steps = 0
        self.a = a
        self.b = b

    def f0(self, x: np.array, y: np.array):
        return np.linalg.norm(x-self.a, 2) + np.linalg.norm(y-self.b, 1)

    def f(self, x: np.array, y: np.array, lam):
        return np.linalg.norm(x-self.a, 2) + np.linalg.norm(y-self.b, 1)+np.dot(lam, (x-y))

    def f_x(self, x: np.array, lam):
        return np.linalg.norm(x-self.a, 2)+np.dot(lam, x)

    def f_y(self, y: np.array, lam):
        return np.linalg.norm(y-self.b, 1)-np.dot(lam, y)

    def df_x(self, x: np.array, lam):
        return (x-self.a)/np.linalg.norm(x-self.a, 2)+lam

    def df_y(self, y: np.array, lam):
        return np.sign(y-self.b)-lam

    def step_f(self):
        t = 1.0
        dx = self.df_x(self.x, self.lam)
        dy = self.df_y(self.y, self.lam)
        dxdx = np.dot(dx, dx)
        dydy = np.dot(dy, dy)
        cur = self.f(self.x, self.y, self.lam)
        while self.f(self.x - t * dx, self.y - t * dy, self.lam) > cur - self.alpha * t * (dxdx+dydy):
            t = self.beta * t
        self.x -= t * dx
        self.y -= t * dy
        if t*t*(dxdx+dydy) < self.eta:
            return False
        self.f_his.append(self.f0(self.x, self.y))
        return True

    def step_lam(self):
        self.lam += (self.x-self.y)

    def descend(self):
        for i in range(100000):
            while self.step_f():
                # print(self.f(self.x, self.y, self.lam))
                pass
            self.step_lam()
            self.total_steps += 1
            print("step: ", self.total_steps)
            print(self.f0(self.x, self.y), "\n")

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his)
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$f(x)$")
        plt.savefig(name+".png")
        plt.show()


np.random.seed(1919810)
a = np.random.randn(100)
b = np.random.randn(100)
x0 = np.random.randn(100)
y0 = np.random.randn(100)
lam = np.array([0.0]*100)


def func(x: np.array):
    return np.linalg.norm(x-a, 2) + np.linalg.norm(x-b, 1)


p = optimize.minimize(func, x0)
print(p)

# da = DA(a.copy(), b.copy(), x0.copy(), y0.copy(), lam, 0.5, 0.5, 1e-3)
# da.descend()

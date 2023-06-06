import numpy as np
import matplotlib.pyplot as plt
import time


class ADMM:
    def __init__(self, a: np.array, b: np.array, x0: np.array, y0: np.array, lam_0: np.array, beta: float, tau: float):
        self.x = x0.copy()
        self.y = y0.copy()
        self.lam = lam_0.copy()
        self.x_his = []
        self.f_his = []
        self.total_steps = 0
        self.a = a
        self.b = b
        self.beta = beta
        self.tau = tau

    def f(self, x: np.array, y: np.array):
        return np.linalg.norm(x-self.a, 2) + np.linalg.norm(y-self.b, 1)

    def L(self, x: np.array, y: np.array, lam):
        return np.linalg.norm(x-self.a, 2) + np.linalg.norm(y-self.b, 1) + np.dot(lam, (x-y)) + 0.5*self.beta*np.dot(x-y, x-y)

    def L_y(self, y: np.array):
        return np.linalg.norm(self.x-self.a, 2) + np.linalg.norm(y-self.b, 1) + np.dot(self.lam, (self.x-y)) + 0.5*self.beta*np.dot(self.x-y, self.x-y)

    def step_x(self):
        d = self.y - (self.lam)/self.beta - self.a
        dist = np.linalg.norm(d, 2)
        self.x = self.a + (dist-self.beta)*d/dist

    def step_y(self):
        b = self.x+self.lam/self.beta
        y_hat = np.zeros_like(self.y)
        a = self.b
        coe = b-a
        for i in range(100):
            if coe[i] < -1/self.beta:
                y_hat[i] = coe[i]+1/self.beta
            elif coe[i] > 1/self.beta:
                y_hat[i] = coe[i]-1/self.beta
            else:
                y_hat[i] = 0
        self.y = y_hat+a

    def step_lam(self):
        self.lam += self.tau*self.beta*(self.x-self.y)

    def start(self):
        while True:
            self.step_x()
            self.step_y()
            self.step_lam()
            cur = self.f(self.x, self.y)
            print("f:", cur)
            self.f_his.append(cur)
            constr = np.linalg.norm(self.x-self.y, 2)
            print("c:", constr, "\n")
            if abs(cur-f_opt) < eps1 and constr < eps2:
                return

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his)
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$f(x)$")
        plt.savefig(name+".png")
        plt.show()


# def func(x: np.array):
#     return np.linalg.norm(x-a, 2) + np.linalg.norm(x-b, 1)


f_opt = 16.336948
eps1 = 1e-5
eps2 = 1e-8


if __name__ == "__main__":
    np.random.seed(1919810)
    a = np.random.randn(100)
    b = np.random.randn(100)
    x0 = np.random.randn(100)
    y0 = np.random.randn(100)
    lam = np.array([0.0]*100)

    admm = ADMM(a.copy(), b.copy(), x0.copy(), y0.copy(), lam.copy(), 0.01, 1.0)
    admm.start()
    admm.plot("ADMM")

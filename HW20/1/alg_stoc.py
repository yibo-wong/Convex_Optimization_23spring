import numpy as np
import matplotlib.pyplot as plt
import time
import random

np.random.seed(1919810)
random.seed(1919810)

max_time = 10.0


class Stochastic:
    def __init__(self, m, n, X, y, w0, eta):
        self.m = m  # m=1000,n=10000,X(m,n)
        self.n = n
        self.X = X
        self.y = y
        self.w = w0.copy()
        self.eta = eta
        self.w_his = []
        self.f_his = []
        self.t_his = []
        self.steps = 0

    def f(self, w):
        return np.sum(np.log(1 + np.exp(-(w @ self.X) * self.y))) / self.n

    def df(self, w):
        return self.X @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def df_i(self, w, i):
        x_i = self.X[:, i]
        return (-self.y[i] * (1 - 1 / (np.exp(-(w.T @ x_i) * self.y[i]) + 1))) * x_i / self.n

    def df_xi(self, w, i):
        return self.X[i] @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def SGD_step(self, i):
        self.w -= self.df_i(self.w, i) * self.eta
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def SGD_start(self):
        start = time.time()
        while True:
            print("SGD", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.SGD_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def Momentum_step(self, i):
        self.b = self.b*self.gamma + self.eta * self.df_i(self.w, i)
        self.w -= self.b
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def Momentum_start(self, gamma):
        start = time.time()
        self.gamma = gamma
        self.b = np.zeros(1000)
        while True:
            print("Momentum", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.Momentum_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def NAG_start(self, gamma):
        start = time.time()
        self.gamma = gamma
        self.b = np.zeros(1000)
        while True:
            print("NAG", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.NAG_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def NAG_step(self, i):
        d = self.df_i(self.w - self.eta * self.b, i)
        self.b = self.b*self.gamma + self.eta * d
        self.w -= self.b
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def Adagrad_step(self, i):
        d = self.df_i(self.w, i)
        self.G[i] += np.dot(d, d)
        self.w -= self.eta_adagrad / np.sqrt(1e-8 + self.G[i]) * d
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def Adagrad_start(self, eta):
        start = time.time()
        self.G = np.zeros(10000)
        self.eta_adagrad = eta
        while True:
            print("Adagrad", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.Adagrad_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def Adadelta_start(self, gamma):
        start = time.time()
        self.gamma = gamma
        self.Eg = np.zeros_like(self.w)
        self.dw = np.zeros_like(self.w)
        self.dwsq = np.zeros_like(self.w)
        while True:
            print("Adadelta", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.Adadelta_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def Adadelta_step(self, i):
        d = self.df_i(self.w, i)
        self.Eg = self.gamma*self.Eg+(1-self.gamma)*d**2
        dw = np.sqrt(self.dwsq+1e-8) / np.sqrt(self.Eg+1e-8)*d
        self.dwsq = self.gamma*self.dwsq + (1-self.gamma)*dw ** 2
        self.w -= dw
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def Adam_start(self, gamma):
        start = time.time()
        self.eta = 0.00001
        self.beta1 = 0.9
        self.eps = 1e-8
        self.beta2 = 0.009
        self.m = np.zeros_like(self.w)
        self.v = np.zeros_like(self.w)
        while True:
            print("Adam", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.Adam_step(i)
            end = time.time()
            self.t_his.append(end-start)
            self.steps += 1
            if end-start > max_time:
                return

    def Adam_step(self, i):
        d = self.df_i(self.w, i)
        self.m = self.beta1 * self.m + (1 - self.beta1) * d
        self.v = self.beta2 * self.v + (1 - self.beta2) * d ** 2
        m_hat = self.m / (1 - self.beta1 ** (1+self.steps))
        v_hat = self.v / (1 - self.beta2 ** (1+self.steps))
        self.w -= self.eta / (np.sqrt(v_hat) + self.eps) * m_hat
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def Adan_start(self, gamma):
        start = time.time()
        self.eta = 0.0001
        self.eps = 1e-8
        self.beta1 = 0.02
        self.beta2 = 0.08
        self.beta3 = 0.01
        self.lam = 0.01
        self.mk = np.zeros_like(self.w)
        self.vk = np.zeros_like(self.w)
        self.uk = np.zeros_like(self.w)
        self.nk = np.zeros_like(self.w)
        self.gk = np.zeros_like(self.w)
        while True:
            print("Adan", self.f(self.w))
            i = random.randint(0, self.n-1)
            self.Adan_step(i)
            end = time.time()
            self.t_his.append(end-start)
            self.steps += 1
            if end-start > max_time:
                return

    def Adan_step(self, i):
        g = self.df_i(self.w, i)
        self.vk = (1-self.beta2)*self.vk+self.beta2*(g-self.gk)
        self.mk = (1-self.beta1)*self.mk+self.beta1*g
        self.uk = self.mk+(1-self.beta2)*self.vk
        self.nk = (1-self.beta3)*self.nk+self.beta3*(g+(1-self.beta2)*(g-self.gk))**2
        self.gk = g
        self.w = (1/(1+self.lam*self.eta))*(self.w-self.eta*self.uk/np.sqrt(self.nk+self.eps))
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def df_i_rcd(self, w, j):  # scalar!
        return self.X[j] @ (-self.y * (1 - 1 / (np.exp(-(w.T @ self.X) * self.y) + 1))) / self.n

    def RCD_start(self, gamma):
        self.rcd_gamma = gamma
        self.rcd_beta = np.linalg.norm(X, axis=1)**2/(4*self.n)
        self.rcd_p = self.rcd_beta**self.rcd_gamma/np.sum(self.rcd_beta**self.rcd_gamma)
        start = time.time()
        while True:
            print("RCD", self.f(self.w))
            i = np.random.choice(self.m, p=self.rcd_p)
            self.RCD_step(i)
            end = time.time()
            self.t_his.append(end-start)
            if end-start > max_time:
                return

    def RCD_step(self, i):
        self.w[i] -= self.df_i_rcd(self.w, i) / self.rcd_beta[i]
        self.w_his.append(self.w)
        self.f_his.append(self.f(self.w))

    def export_data(self, name: str):
        return [name, self.f_his, self.t_his]


class Plot:
    def __init__(self) -> None:
        self.name = []
        self.f = []
        self.t = []

    def recv_data(self, obj):
        self.name.append(obj[0])
        self.f.append(obj[1])
        self.t.append(obj[2])

    def plot(self):
        plt.figure()
        # plt.ylim((0.685, 0.7))
        # plt.yticks(np.arange(0.685, 0.7, 0.001))
        for i in range(len(self.name)):
            print(i)
            plt.plot(self.t[i], self.f[i], label=self.name[i])
        plt.legend()
        plt.savefig("stochastic_algorithm(with RCD).png")
        plt.show()


if __name__ == "__main__":
    X = np.random.randn(1000, 10000)
    y = np.random.randint(0, 2, size=10000) * 2 - 1
    w = np.array([0.0]*1000)
    plot = Plot()

    stoc_sgd = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_sgd.SGD_start()
    plot.recv_data(stoc_sgd.export_data("SGD"))

    stoc_momentum = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_momentum.Momentum_start(0.8)
    plot.recv_data(stoc_momentum.export_data("momentum"))

    stoc_nag = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_nag.NAG_start(0.8)
    plot.recv_data(stoc_nag.export_data("NAG"))

    stoc_adagrad = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_adagrad.Adagrad_start(0.001)
    plot.recv_data(stoc_adagrad.export_data("adagrad"))

    stoc_adadelta = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_adadelta.Adadelta_start(0.8)
    plot.recv_data(stoc_adadelta.export_data("adadelta"))

    stoc_adam = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_adam.Adam_start(0.9)
    plot.recv_data(stoc_adam.export_data("adam"))

    stoc_adan = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_adan.Adan_start(0.9)
    plot.recv_data(stoc_adan.export_data("adan"))

    stoc_rcd = Stochastic(1000, 10000, X.copy(), y.copy(), w.copy(), 0.001)
    stoc_rcd.RCD_start(0.9)
    plot.recv_data(stoc_rcd.export_data("rcd"))

    plot.plot()

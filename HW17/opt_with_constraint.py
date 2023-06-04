import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


f_opt = 6.2163884296


class Opt:
    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, beta: float, eta: float, n: int, p: int):
        self.A = A
        self.b = b
        self.x0 = x0
        self.x = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.iteTime = 0
        self.oldValue = -100
        self.nowValue = 0
        self.f_his = []
        self.n = n
        self.p = p

    def init(self):
        self.iteTime = 0
        self.x = self.x0

    def f(self, x: np.array):
        return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))

    def df(self, x: np.array):
        x1 = x - np.max(x)
        return np.exp(x1) / sum(np.exp(x1))

    def ddf(self, x: np.array):
        x1 = x - np.max(x)
        df = np.exp(x1) / sum(np.exp(x1))
        ddf = -np.outer(df, df)+np.diag(df)
        return ddf

    def projection(self, x: np.array, A: np.array):
        return x - A.T @ np.linalg.inv(A @ A.T) @ A @ x

    def line_search(self, x, direction):
        t = 5.0
        cur = self.f(x)
        df = self.df(x)
        while self.f(x + t * direction) > cur + self.alpha * t * np.dot(df, direction):
            t *= self.beta
        return t

    def direct_step(self):
        df = self.df(self.x)
        dfdf = np.dot(df, df)
        d = -self.projection(df / dfdf, self.A)
        t = self.line_search(self.x, d)
        self.x += t * d
        self.oldValue = self.nowValue
        self.nowValue = self.f(self.x)
        self.f_his.append(self.nowValue)
        if abs(self.nowValue-f_opt) < self.eta and self.iteTime > 50:
            return False
        self.iteTime += 1
        if self.iteTime > 1000:
            return False
        return True

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), [np.log(i - f_opt) for i in self.f_his])
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$log(f(x)-f^*)$")
        plt.savefig(name + ".png")
        plt.show()

    def newton_step(self):
        df = self.df(self.x)
        ddf = self.ddf(self.x)
        N_upper = np.concatenate([ddf, self.A.T], axis=1)
        N_lower = np.concatenate([self.A, np.zeros((self.p, self.p))], axis=1)
        N = np.concatenate([N_upper, N_lower], axis=0)
        r = np.concatenate([-df, np.zeros(self.p)])
        dxw = np.linalg.solve(N, r)
        dx = dxw[:500]
        t = self.line_search(self.x, dx)
        self.x += t * dx
        self.f_his.append(self.f(self.x))
        lam = np.dot(dx, ddf@dx)
        if lam/2 < self.eta:
            return False
        return True

    def start(self, method: str):
        if method == "direct":
            while self.direct_step():
                print("f(x)", self.f(self.x))
                print("Ax-b", np.linalg.norm(self.A@(self.x) - self.b, 2))
                print()
                pass
            self.plot("direct")
        elif method == "newton":
            while self.newton_step():
                print("f(x)", self.f(self.x))
                print("Ax-b", np.linalg.norm(self.A@(self.x) - self.b, 2))
                print()
                pass
            self.plot("newton")
        self.init()


class Elim:
    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, beta: float, eta: float, n: int, p: int):
        self.A = A
        self.b = b
        self.x0 = x0
        self.x = x0
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.iteTime = 0
        self.f_his = []
        self.n = n
        self.p = p
        self.z = A.T @ np.linalg.inv(A @ A.T) @ b
        self.F = np.identity(n) - A.T @ np.linalg.inv(A @ A.T) @ A

    def init(self):
        self.iteTime = 0
        self.x = self.x0

    def f0(self, x: np.array):
        return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))

    def df0(self, x: np.array):
        x1 = x - np.max(x)
        return np.exp(x1) / sum(np.exp(x1))

    def f(self, x: np.array):
        return self.f0(self.F @ x+self.z)

    def df(self, x: np.array):
        return self.F.T @ self.df0(self.F @ x + self.z)

    def line_search(self, x, d):
        t = 5.0
        cur = self.f(x)
        while self.f(x + t * d) > cur - self.alpha * t * np.dot(d, d) and t > 0.00001:
            t *= self.beta
            # print(t)
        return t

    def step(self):
        df = self.df(self.x)
        d = -df
        t = self.line_search(self.x, d)
        self.x += t * d
        cur = self.f(self.x)
        self.f_his.append(cur)
        if abs(cur-f_opt) < self.eta:
            return False
        self.iteTime += 1
        return True

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), [np.log(i - f_opt) for i in self.f_his])
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$log(f(x)-f^*)$")
        plt.savefig(name + ".png")
        plt.show()

    def start(self):
        while self.step():
            print("f(x)", self.f(self.x))
            print("Ax-b", np.linalg.norm(self.A@(self.F @ self.x + self.z) - self.b, 2))
            print()
            pass
        self.plot("eliminate")
        self.init


class Dual:
    def __init__(self, A: np.array, b: np.array, x0: np.array, alpha: float, beta: float, eta: float, gamma: float, n: int, p: int):
        self.A = A
        self.b = b
        self.x0 = x0
        self.x = x0
        self.lam = np.zeros_like(b)
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.iteTime = 0
        self.f_his = []
        self.n = n
        self.p = p
        self.steps = 0

    def init(self):
        self.iteTime = 0
        self.x = self.x0

    def f(self, x: np.array):
        return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))

    def df(self, x: np.array):
        x1 = x - np.max(x)
        return np.exp(x1) / sum(np.exp(x1))

    def L(self, x: np.array):
        Axb = self.A@x-self.b
        return self.f(self.x) + np.dot(self.lam, Axb) + 0.5*self.gamma*np.dot(Axb, Axb)

    def dL(self, x: np.array):
        return self.df(x) + self.A.T@self.lam + self.gamma * self.A.T@(self.A@x-b)

    def line_search(self, x, d):
        t = 5.0
        cur = self.L(x)
        while self.L(x + t * d) > cur - self.alpha * t * np.dot(d, d):
            t *= self.beta
        return t

    def step(self, flag=True):
        if flag:
            self.x = minimize(self.L, self.x, jac=self.dL).x
            self.lam += self.gamma*(self.A@self.x - self.b)
            cur = self.f(self.x)
            self.f_his.append(cur)
            self.steps += 1
            if abs(cur-f_opt) < self.eta or self.steps > 300:
                return False
            return True
        else:
            for i in range(50):
                d = -self.dL(self.x)
                t = self.line_search(self.x, d)
                self.x += t*d
            self.lam += self.gamma*(self.A@self.x - self.b)
            cur = self.f(self.x)
            self.f_his.append(cur)
            self.steps += 1
            if abs(cur-f_opt) < self.eta or self.steps > 300:
                return False
            return True

    def plot(self, name: str):
        plt.figure()
        plt.plot(range(len(self.f_his)), [np.log(i - f_opt) for i in self.f_his])
        ax = plt.gca()
        ax.set_xlabel(f"steps:{len(self.f_his)}")
        ax.set_ylabel(r"$log(f(x)-f^*)$")
        plt.savefig(name + ".png")
        plt.show()

    def start(self):
        while self.step():
            print("f(x)", self.f(self.x))
            print("Ax-b", np.linalg.norm(self.A@(self.x) - self.b, 2))
            print()
        self.plot("dual")
        self.init


if __name__ == "__main__":
    np.random.seed(1919810)

    n = 500
    p = 100
    A = np.random.normal(5, 2, (p, n))
    b = np.random.normal(5, 2, p)
    eta = 1e-4
    x0 = 10 * np.random.rand(n)
    x0 = x0 - A.T @ np.linalg.inv(A @ A.T) @ A @ x0 + A.T @ np.linalg.inv(A @ A.T) @ b
    opt_direct = Opt(A, b, x0.copy(), 0.4, 0.8, eta, n, p)
    opt_direct.start("direct")

    opt_newton = Opt(A, b, x0.copy(), 0.4, 0.8, eta, n, p)
    opt_newton.start("direct")

    x1 = np.array([0.0]*n)
    elim = Elim(A, b, x1.copy(), 0.4, 0.5, eta, n, p)
    elim.start()

    x2 = np.array([0.0]*n)
    dual = Dual(A, b, x2.copy(), 0.4, 0.5, eta, 1, n, p)
    dual.start()

import numpy as np


class Opt:
    def __init__(
        self,
        A: np.array,
        b: np.array,
        x0: np.array,
        alpha: float,
        beta: float,
        eta: float,
    ):
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

    def init(self):
        self.iteTime = 0
        self.x = self.x0

    def f(self, x: np.array):
        return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))

    def df(self, x: np.array):
        x = x - np.max(x)
        return np.exp(x) / sum(np.exp(x))

    def projection(self, x: np.array, A: np.array):
        return x - A.T @ np.linalg.inv(A @ A.T) @ A @ x

    def line_search(self, x, direction):
        t = 5.0
        cur = self.f(x)
        df = self.df(x)
        while self.f(x + t * direction) > cur + self.alpha * np.dot(df, direction):
            t *= self.beta
        return t

    def direct_step(self):
        print(f"step{self.iteTime}:{self.f(self.x)}")
        df = self.df(self.x)
        # print(df)
        dfdf = np.dot(df, df)
        d = -self.projection(df / dfdf, self.A)
        t = self.line_search(self.x, d)
        self.x += t * d
        self.oldValue = self.nowValue
        self.nowValue = self.f(self.x)
        if abs(self.nowValue - self.oldValue) < self.eta:
            return False
        self.iteTime += 1
        if self.iteTime > 100:
            return False
        return True

    def newton_step(self):
        pass

    def start(self, method: str):
        if method == "direct":
            while self.direct_step():
                pass
        self.init()


if __name__ == "__main__":
    np.random.seed(19890817)
    n = 500
    p = 100
    A = np.random.normal(10, 5, (p, n))
    b = np.random.normal(10, 5, p)
    # x0 = np.random.normal(10, 5, n)
    x0 = 10 * np.random.rand(n)
    x0 = x0 - A.T @ np.linalg.inv(A @ A.T) @ A @ x0 + A.T @ np.linalg.inv(A @ A.T) @ b
    opt = Opt(A, b, x0, 0.4, 0.8, 1e-10)
    opt.start("direct")

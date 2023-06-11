import numpy as np
import matplotlib.pyplot as plt


class BCD:
    def __init__(self, Y, lam: float):
        # Y 200x500    D 200x400    X 400x500
        self.steps = 0
        self.Y = Y.copy()
        self.lam = lam
        self.D = np.random.randn(200, 400)
        for j in range(400):
            self.D[:, j] = self.D[:, j] / np.linalg.norm(self.D[:, j])
        self.X = np.random.randn(400, 500)
        self.f_his = []

    def f(self, X, D):
        return 0.5*np.linalg.norm(self.Y-D@X, "fro") + self.lam * np.linalg.norm(X, 1)

    def step(self, i):
        # use K-SVD and low-rank approximation algorithm.
        # see article "K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation"
        d = self.D[:, i]
        x = self.X[i, :]
        E = self.Y-self.D@self.X+np.outer(d, x)
        U, S, V = np.linalg.svd(E)
        d_opt = U[:, 0]
        x_opt = V[0, :]*S[0]
        for j in range(x_opt.shape[0]):
            if x_opt[j] >= self.lam:
                x_opt[j] -= self.lam
            elif x_opt[j] <= self.lam:
                x_opt[j] += self.lam
            else:
                x_opt[j] = 0
        self.X[i, :] = x_opt
        self.D[:, i] = d_opt
        cur = self.f(self.X, self.D)
        self.f_his.append(cur)

    def start(self):
        while True:
            print("step:", self.steps)
            print("f:", self.f(self.X, self.D), "\n")
            for i in range(400):
                if (i % 100 == 0):
                    print(i, " / 400")
                self.step(i)
            self.steps += 1
            if self.steps >= 5:
                return

    def plot_f(self):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his, color="blue", linewidth=1)
        plt.savefig("bcd_dict.png")
        plt.show()


if __name__ == "__main__":
    np.random.seed(1919810)
    Y = np.random.randn(200, 500)
    lam = 1.0
    bcd = BCD(Y.copy(), lam)
    bcd.start()
    bcd.plot_f()

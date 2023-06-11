import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1919810)


class PPA:
    def __init__(self, Y0: np.array, D: np.array) -> None:
        self.Y0 = Y0.copy()
        self.D = D
        self.Y = Y0.copy()
        self.x = np.random.randn(2)
        self.steps = 0
        self.f_his = []

    def f(self):
        sum = 0
        for i in range(5):
            sum += np.linalg.norm(self.Y[:, i]-self.x, 2)**2 / 2
        return sum

    def step_x(self):
        self.x = 0.2*np.sum(self.Y, axis=1)

    def step_y(self, i):
        y = self.Y[:, i]
        y0 = self.Y0[:, i]
        if np.linalg.norm(y-self.x, 2) < self.D[i]:
            y = self.x
        else:
            y = y0 + (self.x-y0) * self.D[i] / np.linalg.norm((self.x-y0), 2)
        self.Y[:, i] = y

    def start(self):
        while True:
            self.steps += 1
            self.step_x()
            for i in range(5):
                self.step_y(i)
            self.f_his.append(self.f())
            print("step:", self.steps)
            print("f:", self.f())
            if self.steps >= 30:
                return

    def plot_f(self):
        plt.figure()
        plt.plot(range(len(self.f_his)), self.f_his, color="blue", linewidth=1)
        plt.savefig("ppa.png")
        plt.show()


if __name__ == "__main__":
    Y0 = 10*np.random.randn(2, 5)
    D0 = 5*np.random.randn(5)
    ppa = PPA(Y0, D0)
    ppa.start()
    ppa.plot_f()

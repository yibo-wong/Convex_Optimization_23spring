import numpy as np
import matplotlib.pyplot as plt


class BCD:
    def __init__(self, D, omega_list: list):
        self.steps = 0
        self.D = D.copy()
        self.omega = omega_list
        self.U = np.random.randn(200, 5)
        self.V = np.random.randn(300, 5)
        self.A = np.random.randn(200, 300)
        self.f_his = []
        self.diff_his = []

    def f(self, U, V, A):
        return np.linalg.norm(U@V.T-A, "fro")

    def set_mat(self):
        for pos in self.omega:
            self.A[pos[0]][pos[1]] = self.D[pos[0]][pos[1]]

    def step(self):
        self.U = self.A@np.linalg.pinv(self.V.T)
        self.V = (np.linalg.pinv(self.U)@self.A).T
        self.A = self.U@self.V.T
        self.set_mat()
        diff = np.linalg.norm(self.A-self.D, "fro")
        self.diff_his.append(diff)
        print("step:", self.steps)
        print("diff:", diff)
        cur = self.f(self.U, self.V, self.A)
        self.f_his.append(cur)
        print("loss:", cur)

    def start(self):
        while True:
            self.steps += 1
            self.step()
            if self.steps > 2000:
                return


if __name__ == "__main__":
    np.random.seed(1919810)
    U0 = np.random.randn(200, 5)
    V0 = np.random.randn(300, 5)
    D = U0@V0.T
    omega = []
    for i in range(int(0.1*200*300)):
        x = np.random.choice(200)
        y = np.random.choice(300)
        omega.append([x, y])
    bcd = BCD(D.copy(), omega.copy())
    bcd.start()

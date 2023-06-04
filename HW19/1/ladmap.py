import numpy as np
from matplotlib import pyplot as plt


class LADMAP:
    def __init__(self, n, p, lam_coe: float, D: np.array, Z0: np.array, E0: np.array, LAM: np.array, beta: float, beta_m: float, rho: float, eps1: float, eps2: float):
        self.n = n
        self.p = p
        self.lam_coe = lam_coe
        self.D = D.copy()
        self.Z = Z0.copy()
        self.E = E0.copy()
        T1n = np.ones((1, n))
        T0p = np.zeros((1, p))
        Ep = np.identity(p)
        self.A1 = np.concatenate((D, T1n), axis=0)
        self.A2 = np.concatenate((Ep, T0p), axis=0)
        self.B = self.A1.copy()
        self.eta1 = np.linalg.norm(self.A1, 2)**2 + 1
        self.eta2 = np.linalg.norm(self.A2, 2)**2 + 1
        self.beta = beta
        self.beta_m = beta_m
        self.rho = rho
        self.eps1 = eps1
        self.eps2 = eps2
        self.f_his = []
        self.LAM = LAM
        self.dZ = np.zeros_like(self.Z)
        self.dE = np.zeros_like(self.E)
        self.criteria_1 = False
        self.criteria_2 = False
        self.steps = 0

    def f1(self, Z: np.array):
        return np.linalg.norm(Z, "nuc")

    def f2(self, E: np.array):
        return self.lam_coe * np.linalg.norm(np.linalg.norm(E, 2, axis=0), 1)

    def prox_1(self, Z, beta):
        U, Sigma, VT = np.linalg.svd(Z)
        Sigma = np.maximum(Sigma - 1/beta, np.zeros_like(Sigma))
        return U @ np.diag(Sigma) @ VT

    def prox_2(self, E, beta):
        Xv = np.linalg.norm(E, ord=2, axis=0)
        P = np.zeros_like(E)
        for i in range(P.shape[1]):
            P[:, i] = max((1 - self.lam_coe / (beta * Xv[i])), 0) * E[:, i]
        return P

    def step_Z(self):
        W = self.Z - (1 / (self.beta * self.eta1)) * self.A1.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.B))
        Z = self.prox_1(W, self.beta * self.eta1)
        self.dZ = Z-self.Z
        self.Z = Z.copy()

    def step_E(self):
        W = self.E - (1 / (self.beta * self.eta2)) * self.A2.T @ (self.LAM + self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.B))
        E = self.prox_2(W, self.beta * self.eta2)
        self.dE = E-self.E
        self.E = E.copy()

    def step_LAM(self):
        self.LAM += self.beta * (self.A1 @ self.Z + self.A2 @ self.E - self.B)

    def step_beta(self):
        rho = 1
        temp = self.beta * max(np.sqrt(self.eta1) * np.linalg.norm(self.dZ), np.sqrt(self.eta2) * np.linalg.norm(self.dE)) / np.linalg.norm(self.B)
        if temp < self.eps2:
            rho = self.rho
        self.beta = min(self.beta_m, self.beta*rho)

    def set_criterion(self):
        self.criteria_1 = False
        self.criteria_2 = False
        if np.linalg.norm(self.A1 @ self.Z + self.A2 @ self.E - self.B)/np.linalg.norm(self.B) < self.eps1:
            self.criteria_1 = True
        if self.beta * max(np.sqrt(self.eta1) * np.linalg.norm(self.dZ), np.sqrt(self.eta2) * np.linalg.norm(self.dE)) / np.linalg.norm(self.B) < self.eps2:
            self.criteria_2 = True

    # def step(self):
    #     cur = self.f1(self.Z) + self.f2(self.E)
    #     self.f_his.append(cur)
    #     self.step_Z()
    #     self.step_E()
    #     self.step_LAM()
    #     self.step_beta()
    #     self.set_criterion()

    def start(self):
        while True:
            cur = self.f1(self.Z) + self.f2(self.E)
            self.f_his.append(cur)
            print("step", self.steps)
            print("fx", cur)
            self.step_Z()
            self.step_E()
            self.step_LAM()
            self.step_beta()
            self.set_criterion()
            self.steps += 1
            if self.criteria_1 and self.criteria_2:
                return


if __name__ == "__main__":
    np.random.seed(1919810)
    n = 300
    p = 200
    lam_coe = 1.0
    D0 = np.random.randn(200, 300)
    E0 = np.zeros((200, 300))
    Z0 = np.zeros((300, 300))
    LAM0 = np.zeros((201, 300))
    beta = 0.0001
    beta_m = 100
    rho = 1.8
    eps1 = 1e-4
    eps2 = 1e-4
    ladmap = LADMAP(n, p, lam_coe, D0, Z0, E0, LAM0, beta, beta_m, rho, eps1, eps2)
    ladmap.start()

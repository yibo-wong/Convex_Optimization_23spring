import numpy as np

p = 0.5


def dev(x: np.array):
    norm = np.linalg.norm(x, p)
    return (norm ** (1 - p)) * (x ** (p - 1))


def hessian(x: np.array):
    norm = np.linalg.norm(x, p)
    return (norm ** (1 - p)) * np.diag((p - 1) * x ** (p - 2)) + (1 - p) * norm ** (
        1 - 2 * p
    ) * np.outer(x ** (p - 1), x ** (p - 1))


# if __name__ == "__main__":
#     x = np.array([1, 2, 3, 4])
#     v = np.array([4, 3, 2, 1])
#     t = 0.00001
#     print((dev(x + t * v) - dev(x)) / t)
#     print(hessian(x) @ v)


if __name__ == "__main__":
    A = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    A1 = np.linalg.norm(A, 2)
    print(A1)
    cond = np.linalg.norm(A, 2) * np.linalg.norm(np.linalg.inv(A), 2)
    print(cond)
    cond = np.linalg.cond(A)
    print(cond)

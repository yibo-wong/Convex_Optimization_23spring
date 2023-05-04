import numpy as np

np.random.seed(114514)
# x = np.random.normal(0, 1, (5))
g = np.random.normal(0, 1, (5))
B = np.random.normal(0, 1, (5, 5))
H = np.linalg.inv(B)

x = -0.5 * H @ g

B_1 = B+np.outer(g, g)/np.dot(g, x)-np.outer(B@x, B@x)/np.dot(x, B@x)

rho = 1/np.dot(g, x)
V = np.identity(5)-rho*np.outer(g, x)

H_1 = V.T@H@V+rho*np.outer(x, x)

print(B_1)
print(H_1)
print(B_1@H_1)

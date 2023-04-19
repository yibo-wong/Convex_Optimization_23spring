from math import *

history = []
x = 2
eps = 1e-6
while abs(x - log(x) - 2) > eps:
    x = 2 + log(x)
    history.append(x)
print(history)

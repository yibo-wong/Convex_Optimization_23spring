import numpy as np
import math


class Add:
    def forward(a, b):
        return a + b

    def diff_a(a, b):
        return 1

    def diff_b(a, b):
        return 1


class Sub:
    def forward(a, b):
        return a - b

    def diff_a(a, b):
        return 1

    def diff_b(a, b):
        return -1


class Mul:
    def forward(a, b):
        return a * b

    def diff_a(a, b):
        return b.forward()

    def diff_b(a, b):
        return a.forward()


class Div:
    def forward(a, b):
        assert b != 0
        return a / b

    def diff_a(a, b):
        return 1 / b.forward()

    def diff_b(a, b):
        return -a.forward() / (b.forward() * b.forward())


class Pow:
    def forward(a, b):
        assert a >= 0
        return a ** b

    def diff_a(a, b):
        return b.forward() * (a.forward() ** (b.forward() - 1))

    def diff_b(a, b):
        return (a.forward() ** b.forward()) * math.log(a.forward())


class Log:
    def forward(a):
        return math.log(a)

    def diff(a):
        return 1 / a.forward()


def log(a):
    return Node(a, 0, Log, False)


class Exp:
    def forward(a):
        return math.exp(a)

    def diff(a):
        return math.exp(a.forward())


def exp(a):
    return Node(a, 0, Exp, False)


class Sin:
    def forward(a):
        return math.sin(a)

    def diff(a):
        return math.cos(a.forward())


def sin(a):
    return Node(a, 0, Sin, False)


class Cos:
    def forward(a):
        return math.cos(a)

    def diff(a):
        return -math.sin(a.forward())


def cos(a):
    return Node(a, 0, Cos, False)


class Tan:
    def forward(a):
        return math.tan(a)

    def diff(a):
        return 1 / (math.cos(a.forward())) ** 2


def tan(a):
    return Node(a, 0, Tan, False)


class Node:
    def __init__(self, a, b=0, op=None, binary=True):
        self.a = a
        self.b = b
        self.op = op
        self.result = None
        self.binary = binary

    def __add__(self, x):
        return Node(self, x, Add)

    def __radd__(self, x):
        return Node(x, self, Add)

    def __sub__(self, x):
        return Node(self, x, Sub)

    def __rsub__(self, x):
        return Node(x, self, Sub)

    def __mul__(self, x):
        return Node(self, x, Mul)

    def __rmul__(self, x):
        return Node(x, self, Mul)

    def __truediv__(self, x):
        return Node(self, x, Div)

    def __pow__(self, x):
        return Node(self, x, Pow)

    def forward(self):
        if self.result is not None:
            return self.result
        if self.binary:
            ans = self.op.forward(self.a.forward(), self.b.forward())
        else:
            ans = self.op.forward(self.a.forward())
        self.result = ans
        return ans

    def backward(self, chain=1):
        if self.binary:
            self.a.backward(chain * self.op.diff_a(self.a, self.b))
            self.b.backward(chain * self.op.diff_b(self.a, self.b))
        else:
            self.a.backward(chain * self.op.diff(self.a))


class Variable(Node):
    def __init__(self, value):
        self.value = value
        self.diff = 0

    def forward(self):
        return self.value

    def diff(self):
        return self.diff

    def backward(self, chain):
        self.diff += chain


if __name__ == "__main__":
    x1 = Variable(1.0)
    x2 = Variable(2.0)
    x3 = Variable(3.0)
    one = Variable(1.0)
    two = Variable(2.0)

    f = (sin(x1 + one) + cos(two * x2)) * tan(log(x3)) + (
        sin(x2 + one) + cos(two * x1)
    ) * exp(one + sin(x3))

    print(f"f = {f.forward()}")
    f.backward()
    print(f"df/dx1 = {x1.diff}")
    print(f"df/dx2 = {x2.diff}")
    print(f"df/dx3 = {x3.diff}")
    print("----------test:numerically df/dx_1------------")
    t = 1e-7
    x1_test = Variable(1.0 + t)
    x2_test = Variable(2.0)
    x3_test = Variable(3.0)
    f_test = (sin(x1_test + one) + cos(two * x2_test)) * tan(log(x3_test)) + (
        sin(x2_test + one) + cos(two * x1_test)
    ) * exp(one + sin(x3_test))
    ans = (f_test.forward() - f.forward()) / t
    print(f"df/dx1 = {ans}")

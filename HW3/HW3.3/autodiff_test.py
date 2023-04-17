# Yibo Wang,2100011025,coe@pku,convex opt 23 spring.
import numpy as np
import math
from autodiff import *

# define the expressions and their partial derivations(just for verification)
def expression(x1, x2, x3):
    return (sin(x1 + 1.0) + cos(2.0 * x2)) * tan(log(x3)) + (
        sin(x2 + 1.0) + cos(2.0 * x1)
    ) * exp(1.0 + sin(x3))


def expression_x1(x1, x2, x3):
    return cos(x1 + 1.0) * tan(log(x3)) - 2.0 * sin(2.0 * x1) * exp(1.0 + sin(x3))


def expression_x2(x1, x2, x3):
    return -2.0 * sin(2.0 * x2) * tan(log(x3)) + cos(x2 + 1.0) * exp(1.0 + sin(x3))


def expression_x3(x1, x2, x3):
    return (sin(x1 + 1.0) + cos(2.0 * x2)) / (x3 * cos(log(x3)) ** 2) + (
        sin(x2 + 1.0) + cos(2.0 * x1)
    ) * exp(1.0 + sin(x3)) * cos(x3)


if __name__ == "__main__":
    v1 = 5
    v2 = 2.6
    v3 = 3.0

    x1 = Variable(v1)
    x2 = Variable(v2)
    x3 = Variable(v3)

    f = expression(x1, x2, x3)
    f_x1 = expression_x1(x1, x2, x3)
    f_x2 = expression_x2(x1, x2, x3)
    f_x3 = expression_x3(x1, x2, x3)
    f_value = f.forward()
    print(f"f = {f_value}")
    f.backward()
    print("-------automatic differential test------------")
    print(f"df/dx1 = {x1.diff}")
    print(f"df/dx2 = {x2.diff}")
    print(f"df/dx3 = {x3.diff}")
    print("-------mathematically derivation test---------")
    print(f"f_x1 = {f_x1.forward()}")
    print(f"f_x2 = {f_x2.forward()}")
    print(f"f_x3 = {f_x3.forward()}")

    t = 1e-7

    print("----------test:numerically derivation----------")
    x1_test = Variable(v1 + t)
    x2_test = Variable(v2)
    x3_test = Variable(v3)
    f_test_1 = expression(x1_test, x2_test, x3_test)
    ans = (f_test_1.forward() - f_value) / t
    print(f"df/dx1 = {ans}")

    x1_test = Variable(v1)
    x2_test = Variable(v2 + t)
    x3_test = Variable(v3)
    f_test_2 = expression(x1_test, x2_test, x3_test)
    ans = (f_test_2.forward() - f_value) / t
    print(f"df/dx2 = {ans}")

    x1_test = Variable(v1)
    x2_test = Variable(v2)
    x3_test = Variable(v3 + t)
    f_test_3 = expression(x1_test, x2_test, x3_test)
    ans = (f_test_3.forward() - f_value) / t
    print(f"df/dx2 = {ans}")

    print(
        "The correctness can be verified,",
        "if the values calculated from the three ways above are all close.",
    )

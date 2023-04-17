# Yibo Wang,2100011025,coe@pku,convex opt 23 spring.
import numpy as np
import math
from autodiff_for_resnet import *

if __name__ == "__main__":
    np.random.seed(1919810)

    # size setting

    width1 = height1 = 10
    width2 = height2 = 3

    # initial value setting.I just randomize them.

    inputInit = np.random.normal(0, 1, (width1, height1))
    convInit = np.random.normal(0, 1, (width2, height2))
    sampleInit = np.random.normal(0, 1, (width1, height1))

    # BP to calculate the value and the derivation

    resnet = resNet(width1, height1, width2, height2, inputInit, convInit)
    resnet.start(sampleInit)
    resnetLoss = resnet.loss.forward()

    print("MSE loss =", resnetLoss)

    resnet.loss.backward()
    resnetGradient = [
        [resnet.gradient(i, j) for i in range(width2)] for j in range(height2)
    ]
    gradientMatrix = np.array(resnetGradient)

    print("gradient matrix:")
    print(gradientMatrix)

    # using [f(X+a*t)-f(X)]/t = <f'(X),a> tp test the program

    gradient_test = np.random.normal(5, 1, (width2, height2))
    t = 0.000001

    resnetTest = resNet(
        width1, height1, width2, height2, inputInit, convInit + t * gradient_test
    )
    resnetTest.start(sampleInit)
    resnetLossTest = resnetTest.loss.forward()
    numericalTest = (resnetLossTest - resnetLoss) / t
    print("---------------------------------------------")
    print("numerically derivation:     ", numericalTest)

    approx = np.sum(gradient_test * gradientMatrix)

    print("first-order approx:         ", approx)
    print("---------------------------------------------")
    print(
        "The correctness can be verified,",
        "if the two numbers above are close enough.^_^",
    )

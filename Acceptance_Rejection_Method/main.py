

from math import exp, gamma, log
import numpy as np
from numpy.random import rand
from matplotlib import pyplot as plt

alpha = 1.3
lam = 5.6
f = lambda x: lam**alpha*x**(alpha-1)*exp(-lam*x)/gamma(alpha)
g = lambda x: 4*exp(-4*x)
C = 1.2

def accept_reject():
    found = False
    while not found:
        # use inverse-transformation method to draw sample from the exponential distribution
        x = -log(rand())/4
        if C*g(x)*rand()<=f(x):
            found = True
    return x

if __name__ == "__main__":
    samples = []
    N = 10000
    for i in range(N):
        x = accept_reject()
        samples.append(x)
    x_true = np.linspace(0, 2, 1000)
    y_true = np.array([f(x) for x in x_true])

    fig = plt.figure()
    plt.plot(x_true, y_true, label="gamma density curve")
    plt.hist(samples, density=True, label="simulated histogram")
    plt.legend()
    plt.show()
    plt.close()
    fig.savefig("simulated_histogram.png")




        

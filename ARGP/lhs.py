import numpy as np


def latin_hypercube_1d(low, high, samples, seed=0):

    # for reproducibility
    np.random.seed(seed)

    # Generate intervals of the sample space
    cut = np.linspace(low, high, samples + 1)

    # Generate uniform random samples
    u = np.random.rand(samples) 
    a = cut[:-1]
    b = cut[1:]

    # Transform uniform random samples, one into each interval
    return u * (b - a) + a


if __name__ == '__main__':

    np.random.seed(0)
    N2 = latin_hypercube_1d(0.8, 2.34, 100)
    print(N2)

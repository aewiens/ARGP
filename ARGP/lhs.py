import numpy as np


def latin_hypercube_samples(low, high, samples, seed=0, dtype='float'):

    # for reproducibility
    np.random.seed(seed)

    # Generate intervals of the sample space
    cut = np.linspace(low, high, samples + 1)

    # Generate uniform random samples
    u = np.random.rand(samples) 
    a = cut[:-1]
    b = cut[1:]

    # Transform uniform random samples, one into each interval
    samples = u * (b - a) + a

    if dtype == 'float':
        return samples

    elif dtype == 'int':
        return np.array([int(i) for i in samples])


if __name__ == '__main__':
    test = latin_hypercube_samples(0, 100, 20, dtype='int')
    print(test)

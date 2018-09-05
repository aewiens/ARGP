import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary

E = np.loadtxt("5z.dat", delimiter=',', skiprows=1)
E[:, -1] += 76.297669654097


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Test set
Xtest = E[:, :-1]
dim = len(Xtest.T)

# Training set sizes
Nt = [375]

f = open("rmse.dat", "a")

for n in Nt:

    seeds = np.arange(10)
    avg_rmse = 0

    for s in seeds:
        np.random.seed(s)

        idx = np.random.randint(0, len(E), size=n)
        T = E[idx]
        X1, Y1 = np.split(T, [dim], axis=1)

        #  Ordinary GP
        kernel = GPy.kern.RBF(dim, ARD=True)
        model = ordinary.optimize(X1, Y1, kernel, normalize=True, restarts=12)
        mu, v = ordinary.predict(model, Xtest)

        # Calculate error
        Exact = E[:, -1].reshape(-1, 1)
        error = 219474 * RMSE(mu, Exact)

        avg_rmse += error 

    avg_rmse /= len(seeds)
    f.write("{:<5d}{:>10.5f}\n".format(n, avg_rmse))

f.close()

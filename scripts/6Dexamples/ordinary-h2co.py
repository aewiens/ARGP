import GPy
import numpy as np
from ARGP import ordinary
from ARGP import lhs

E = np.loadtxt("surfaces/5z-grid1.dat", delimiter=',', skiprows=1)
E[:, -1] += 114.438967635300


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Test set
Xtest = E[:, :-1]
dim = len(Xtest.T)

# Training set sizes
Nt = 80
errors = []
seeds = np.arange(10)


# Average over 10 Gaussian processes
for i in range(10):

    #idx = lhs.latin_hypercube_samples(0, len(E), Nt, seed=i, dtype='int')
    idx = np.random.randint(0, len(E), size=Nt)
    T = E[idx]
    X1, Y1 = np.split(T, [dim], axis=1)

    #  Ordinary GP
    kernel = GPy.kern.RBF(dim, ARD=True)
    model = ordinary.optimize(X1, Y1, kernel, normalize=True, restarts=10)
    mu, v = ordinary.predict(model, Xtest)

    # Calculate error
    Exact = E[:, -1].reshape(-1, 1)
    error = 1000 * RMSE(mu, Exact)
    errors.append(error)


print("----------------------------------------------")
print("  10 GP Regressions with {:2d} Data Points".format(Nt))
print("----------------------------------------------")
print("|    First RMSE: {:>10.5f}                   |".format(errors[0]))
print("|     Best RMSE: {:>10.5f}                   |".format(min(errors)))
print("|    Worst RMSE: {:>10.5f}                   |".format(max(errors)))
print("|  Average RMSE: {:>10.5f}                   |".format(np.mean(errors)))
print("----------------------------------------------")

import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary

E1 = np.loadtxt("surfaces/ccsd-t-tz.dat", delimiter=',', skiprows=1)
E2 = np.loadtxt("surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)

# shift relative to global minimum
E1[:, -1] += 76.136681747014
E2[:, -1] += 76.172634420318


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Test set
Xtest = E2[:, 0:-1]
Nts = len(Xtest)

# Dimensions of system
dim = 2
active_dimensions=np.arange(0, dim)

np.random.seed(5)

avg_rmse = 0
seeds = [0, 1, 2, 3, 4, 5, 8, 10, 11, 12]

f = open("rmse.dat", "a")
for s in seeds:

    np.random.seed(s)

    # Level 1 training set
    N1 = 400
    idx = np.random.randint(0, len(E1), size=N1)
    T1 = E1[idx]
    X1, Y1 = np.split(T1, [dim], axis=1)

    #  Train level 1
    k1 = GPy.kern.RBF(dim, ARD=True)
    m1 = ordinary.optimize(X1, Y1, k1, normalize=True, restarts=12)
    mu1, v1 = ordinary.predict(m1, Xtest, full_cov=True)

    # Level 2 training set
    N2 = 180
    idx2 = np.random.choice(idx, size=N2)
    T2 = E2[idx2]
    X2, Y2 = np.split(T2, [dim], axis=1)

    # Predict level 1 at X2
    mu1_, v1_ = ordinary.predict(m1, X2, full_cov=True)

    # Train level 2
    XX = np.hstack((X2, mu1_))
    k2 = GPy.kern.RBF(1, active_dims = [dim]) * GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True) \
        + GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True)
    m2 = GPy.models.GPRegression(X=XX, Y=Y2, kernel=k2, normalizer=True)
    m2.optimize(max_iters=500)
    m2.optimize_restarts(12, optimizer='bfgs', max_iters=1000)

    # Predict level 2
    nsamples = 100
    Z = np.random.multivariate_normal(mu1.flatten(), v1, nsamples)

    tmp_m = np.zeros((nsamples, Nts))
    tmp_v = np.zeros((nsamples, Nts))

    for i in range(nsamples):
        XX = np.hstack((Xtest, Z[i, :][:, np.newaxis]))
        mu, v = m2.predict(XX)
        tmp_m[i, :] = mu.flatten()
        tmp_v[i, :] = v.flatten()

    mu2 = np.mean(tmp_m, axis=0).reshape(-1, 1)
    C2 = np.abs(np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0))
    C2 = C2.reshape(-1, 1)

    # Calculate error
    exact = E2[:, 2].reshape(-1, 1)
    error = 1000 * RMSE(mu2, exact)
    f.write("{:>3d}{:>10.5f}\n".format(s, error))
    avg_rmse += error 

avg_rmse /= len(seeds)

f.write("mean: {:>10.5f}".format(avg_rmse))
f.close()

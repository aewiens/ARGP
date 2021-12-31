import GPy
import numpy as np
from matplotlib import pyplot as plt
from ARGP import ordinary
from ARGP import data_sampler

E1 = np.loadtxt("surfaces/ccsd-t-dz.dat", delimiter=',', skiprows=1)
E2 = np.loadtxt("surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)

# shift relative to global minimum
E1[:, -1] += 76.240075372915
E2[:, -1] += 76.174584084555

# Test set
Xtest = E2[:, :-1]
Ntest = len(Xtest)

# Training set sizes
Nt1 = [140]
Nt2 = [25]

# Dimensions of system
dim = 3
active_dimensions=np.arange(0, dim)

f = open("rmse.dat", "a")

for N1, N2 in zip(Nt1, Nt2):

    # Level 1 training set
    train, test = data_sampler.smart_random(E1, N1, n_test=None)
    X1, Y1 = np.split(E1[train], [dim], axis=1)

    #  Train level 1
    k1 = GPy.kern.RBF(dim, ARD=True)
    m1 = ordinary.optimize(X1, Y1, k1, normalize=True, restarts=12)
    mu1, v1 = ordinary.predict(m1, Xtest, full_cov=True)

    # Level 2 training set
    train2, test2 = data_sampler.smart_random2(E2, N2, train, test)
    X2, Y2 = np.split(E2[train2], [dim], axis=1)

    # Predict level 1 at X2
    mu1_, v1_ = ordinary.predict(m1, X2, full_cov=True)

    # Train level 2
    XX = np.hstack((X2, mu1_))
    k2 = GPy.kern.RBF(1, active_dims = [dim]) * GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True) \
        + GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True)
    m2 = GPy.models.GPRegression(X=XX, Y=Y2, kernel=k2, normalizer=True)
    m2.optimize(max_iters=1000)
    m2.optimize_restarts(12, optimizer='bfgs', max_iters=1000)

    # Predict level 2
    nsamples = 500
    Z = np.random.multivariate_normal(mu1.flatten(), v1, nsamples)

    tmp_m = np.zeros((nsamples, Ntest))
    tmp_v = np.zeros((nsamples, Ntest))

    for j in range(nsamples):
        XX = np.hstack((Xtest, Z[j, :][:, np.newaxis]))
        mu, v = m2.predict(XX)
        tmp_m[j, :] = mu.flatten()
        tmp_v[j, :] = v.flatten()

    mu2 = np.mean(tmp_m, axis=0)
    C2 = np.abs(np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0))

    # Calculate error
    exact = E2[:, -1]
    error = 1000 * np.sqrt(np.mean((mu2[test2] - exact[test2])**2))
    etrain = 1000 * np.sqrt(np.mean((mu2[train2] - np.ravel(Y2))**2))
    print("Prediction error: {:>5.3} mEh".format(error))
    print("Training set error: {:>5.3} mEh".format(etrain))
    f.write("{:>3d}{:>7.3f}{:>7.3f}".format(N2, error, etrain))

f.close()

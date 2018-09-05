import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary

E1 = np.loadtxt("surfaces/ccsd-t-dz.dat", delimiter=',', skiprows=1)
E2 = np.loadtxt("surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)

# shift relative to global minimum
E1[:, -1] += 76.240075372915
E2[:, -1] += 76.174584084555


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))



# Test set
Xtest = E2[:, :-1]
Ntest = len(Xtest)

# Training set sizes
N1 = 300
N2 = 50

# Dimensions of system
dim = 3
active_dimensions=np.arange(0, dim)

# Average over 10 GPs
errors=[]
for i in range(10):

    # Level 1 training set
    idx = np.random.randint(0, len(E1), size=N1)
    T1 = E1[idx]
    X1, Y1 = np.split(T1, [dim], axis=1)

    #  Train level 1
    k1 = GPy.kern.RBF(dim, ARD=True)
    m1 = ordinary.optimize(X1, Y1, k1, normalize=True, restarts=10)
    mu1, v1 = ordinary.predict(m1, Xtest, full_cov=True)

    # Level 2 training set
    idx2 = np.random.choice(idx, size=N2)
    T2 = E2[idx]
    X2, Y2 = np.split(T2, [dim], axis=1)

    # Predict level 1 at X2
    mu1_, v1_ = ordinary.predict(m1, X2, full_cov=True)

    # Train level 2
    XX = np.hstack((X2, mu1_))
    k2 = GPy.kern.RBF(1, active_dims = [dim]) * GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True) \
        + GPy.kern.RBF(dim, active_dims = active_dimensions, ARD = True)
    m2 = GPy.models.GPRegression(X=XX, Y=Y2, kernel=k2, normalizer=True)
    m2.optimize(max_iters=500)
    m2.optimize_restarts(10, optimizer='bfgs', max_iters=100)

    # Predict level 2
    nsamples = 200
    Z = np.random.multivariate_normal(mu1.flatten(), v1, nsamples)

    tmp_m = np.zeros((nsamples, Ntest))
    tmp_v = np.zeros((nsamples, Ntest))

    for i in range(nsamples):
        XX = np.hstack((Xtest, Z[i, :][:, np.newaxis]))
        mu, v = m2.predict(XX)
        tmp_m[i, :] = mu.flatten()
        tmp_v[i, :] = v.flatten()

    mu2 = np.mean(tmp_m, axis=0)
    C2 = np.abs(np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0))

    # Calculate error
    exact = E2[:, -1]
    error = 219474 * RMSE(mu2, exact)
    errors.append(error)


print("----------------------------------------------")
print("10 ARGP Regressions with {:3d} and {:3d} Data Points".format(N1, N2))
print("----------------------------------------------")
print("|    First RMSE: {:>10.5f}                   |".format(errors[0]))
print("|     Best RMSE: {:>10.5f}                   |".format(min(errors)))
print("|    Worst RMSE: {:>10.5f}                   |".format(max(errors)))
print("|  Average RMSE: {:>10.5f}                   |".format(np.mean(errors)))
print("----------------------------------------------")

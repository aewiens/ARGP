import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary

E1 = np.loadtxt("surfaces/atz-grid1.dat", delimiter=',', skiprows=1)
E2 = np.loadtxt("surfaces/a5z-grid1.dat", delimiter=',', skiprows=1)

# shift relative to global minimum
E1[:, -1] += 114.40044229471 
E2[:, -1] += 114.438967635300


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Test set
Xtest = E2[:, :-1]
Ntest = len(Xtest)

# Training set sizes
N1 = 150
N2 = 70

# Dimensions of system
dim = len(Xtest.T)
active_dimensions=np.arange(0, dim)

# Average over 10 GPs
errors=[]
#for i in range(10):
for i in [0, 1, 3]:

    # Level 1 training set
    idx = np.random.randint(0, len(E1), size=N1)
    T1 = E1[idx]
    X1, Y1 = np.split(T1, [dim], axis=1)

    #  Train level 1
    k1 = GPy.kern.RBF(dim, ARD=True)
    m1 = ordinary.optimize(X1, Y1, k1, normalize=True, restarts=15)
    mu1, v1 = ordinary.predict(m1, Xtest, full_cov=True)

    # Level 2 training set
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
    m2.optimize_restarts(15, optimizer='bfgs', max_iters=500)

    # Predict level 2
    nsamples = 300
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
    error = 1000 * RMSE(mu2, exact)
    errors.append(error)


print("----------------------------------------------")
print("10 ARGP Regressions with {:2d} and {:2d} Data Points".format(N1, N2))
print("----------------------------------------------")
print("|    First RMSE: {:>10.5f}                   |".format(errors[0]))
print("|     Best RMSE: {:>10.5f}                   |".format(min(errors)))
print("|    Worst RMSE: {:>10.5f}                   |".format(max(errors)))
print("|  Average RMSE: {:>10.5f}                   |".format(np.mean(errors)))
print("----------------------------------------------")

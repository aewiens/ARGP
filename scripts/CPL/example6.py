#!/usr/bin/env python3
import os
import numpy as np
import ARGP as GP


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


np.random.seed(4)

# Load ab initio surface
E0 = np.loadtxt("surfaces/mrci-pcvtz.tab")
E = np.loadtxt("surfaces/mrci-pcv5z.tab")

# Set zeros at dissociation limits
E0[:, 1] += 109.12399268
E[:, 1] += 109.15851906

Ex, Ey = np.split(E, [-1], axis=1)

# Level 0 training set
train0 = np.load("indices/train0_10.npy")
X0, Y0 = np.split(E0[train0], 2, axis=1)

# Level 1 training set
train = np.load("indices/train1_10.npy")
test = np.array([i for i in range(len(E)) if i not in train])
X, Y = Ex[train], Ey[train]

# Train ordinary model
m0 = GP.ordinary.optimize(X0, Y0, normalize=True, restarts=20)
mu00, C00 = GP.ordinary.predict(m0, X, full_cov=True)
mu0, C0 = GP.ordinary.predict(m0, Ex, full_cov=True)
S0 = np.sqrt(np.diag(C0))

# row vectors
mu0, S0 = np.ravel(mu0), np.ravel(S0)

# Train NARGP model
m1 = GP.nargp.optimize(mu00, X, Y, normalize=True)
mu, C = GP.nargp.predict(m1, mu0, C0, Ex)
mu, S = np.ravel(mu), np.ravel(np.sqrt(C))

# Calculate errors
rmse_test = 1000 * RMSE(mu[test], Ey[test])
rmse_train = 1000 * RMSE(mu[train], Y)
rmse_test_string = "RMSE$_{\mathrm{test}}$: " + "{:>3.3f} ".format(rmse_test) + "m$E_{\mathrm{h}}$"
rmse_train_string = "RMSE$_{\mathrm{train}}$: " + "{:>3.3f} ".format(rmse_train) + "m$E_{\mathrm{h}}$"

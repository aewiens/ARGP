#!/usr/bin/env python3
import os
import numpy as np
import ARGP as GP


def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Load ab initio surface
E0 = np.loadtxt("surfaces/cas-631g.tab")
E = np.loadtxt("surfaces/mrci-pcv5z.tab")

# Set zeros at dissociation limits
E0[:, 1] += 108.759112
E[:, 1] += 109.15851906

# Level 0 training set
train0 = np.load("indices/train0_5.npy")
T0 = np.array([E0[i] for i in train0])
X0, Y0 = np.split(T0, 2, axis=1)

# Level 2 training set
train = np.load("indices/train1_5.npy")
test = np.array([i for i in range(len(E)) if i not in train])
Ex, Ey = np.split(E, [-1], axis=1)
X, Y = Ex[train], Ey[train]

# Train ordinary model
m0 = GP.ordinary.optimize(np.exp(-X0), Y0, normalize=True, restarts=20)

# Predict ordinary model at training points
mu00, C00 = GP.ordinary.predict(m0, np.exp(-X))

# Predict ordinary model at test points
mu0, C0 = GP.ordinary.predict(m0, np.exp(-Ex), full_cov=True)
S0 = np.sqrt(np.diag(C0))
mu0, S0 = np.ravel(mu0), np.ravel(S0)

# Train ARGP model and predict at test points
m1 = GP.nargp.optimize(mu00, np.exp(-X), Y, normalize=True, restarts=20)
mu, C = GP.nargp.predict(m1, mu0, C0, np.exp(-Ex), nsamples=200)
mu, S = np.ravel(mu), np.ravel(np.sqrt(C))

# Calculate errors
rmse_test = 1000 * RMSE(mu[test], Ey[test])
rmse_train = 1000 * RMSE(mu[train], Y)
rmse_test_string = "RMSE$_{\mathrm{test}}$: " + "{:>3.3f} ".format(rmse_test) + "m$E_{\mathrm{h}}$"
rmse_train_string = "RMSE$_{\mathrm{train}}$: " + "{:>3.3f} ".format(rmse_train) + "m$E_{\mathrm{h}}$"

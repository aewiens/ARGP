import numpy as np
import ARGP as GP

def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


# Load ab initio surface
E = np.loadtxt('surfaces/mrci-pcv5z.tab')
E[:, 1] += 109.15851906  # dissociation limit

Ex, Ey = np.split(E, [-1], axis=1)

# Training set
train = np.load("indices/train1_5.npy")
test = np.array([i for i in range(len(E)) if i not in train])
T = E[train]
X, Y = Ex[train], Ey[train]


# Train ordinary model
model = GP.ordinary.optimize(np.exp(-X), Y, normalize=True)
mu, C = GP.ordinary.predict(model, np.exp(-Ex), full_cov=True)
S = np.sqrt(np.diag(C))
mu, S = np.ravel(mu), np.ravel(S)
rmse_test = 1000 * RMSE(mu[test], Ey[test])
rmse_train = 1000 * RMSE(mu[train], Y)
rmse_test_string = "RMSE$_{\mathrm{test}}$: " + "{:>3.3f} ".format(rmse_test) + "m$E_{\mathrm{h}}$"
rmse_train_string = "RMSE$_{\mathrm{train}}$: " + "{:>3.3f} ".format(rmse_train) + "m$E_{\mathrm{h}}$"

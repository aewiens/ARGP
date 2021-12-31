import GPy
import numpy as np
from ARGP import ordinary
from ARGP import data_sampler

E = np.loadtxt("surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)
E[:, -1] += 76.203896662997

# Test set
Xtest = E[:, :-1]
dim = len(Xtest.T)

# Training set size
Nt = 120

# Sample training set
train, test = data_sampler.smart_random(E, Nt, n_test=None)
T = E[train]
X1, Y1 = np.split(T, [dim], axis=1)

#  Ordinary GP regression
kernel = GPy.kern.RBF(dim, ARD=True)
model = ordinary.optimize(X1, Y1, kernel, normalize=True, restarts=10)
mu, v = ordinary.predict(model, Xtest)

# Calculate error
exact = E[:, -1].reshape(-1, 1)
error = 1000 * np.sqrt(np.mean((mu - exact)**2))
print("Test error: {:>5.3} mEh".format(error))

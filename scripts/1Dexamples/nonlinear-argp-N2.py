#!/usr/bin/env python3
import os
import GPy
import numpy as np
import matplotlib.pyplot as plt
from ARGP import nargp
from ARGP import matrix
from ARGP import ordinary

np.random.seed(10)

# Size of confidence interval
ns = 3

# Load ab initio surfaces
E_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'surfaces')
E0 = np.loadtxt(os.path.join(E_path, 'cas-sto3g.tab'))
E1 = np.loadtxt(os.path.join(E_path, 'mrci-pcv5z.tab'))

# Set zeros at dissociation limits
E0[:, 1] += 107.43802032
E1[:, 1] += 109.15851906

# Level 0 training set
Nt0 = 36
i0 = np.random.randint(0, len(E0), size=Nt0)
T0 = np.array([E0[i] for i in i0])
X0, Y0 = np.split(T0, 2, axis=1)

# Level 1 training set
Nt1 = 8
i1 = np.random.choice(i0, size=Nt1)
T1 = np.array([E1[i] for i in i1])
X1, Y1 = np.split(T1, 2, axis=1)

# Test set
Ntest = 100
Xtest = matrix.Col(np.linspace(0.8, 2.34, Ntest))

# Optimize level 1
m0 = ordinary.optimize(X0, Y0)

# Optimize level 2
mu0, C0 = ordinary.predict(m0, X1, full_cov=True)
m1 = nargp.optimize(mu0, X1, Y1)

# Predict level 1
mu0, C0 = ordinary.predict(m0, Xtest, full_cov=True)
S0 = np.sqrt(np.diag(C0))

# Predict Level 2
mu1, C1 = nargp.predict(m1, mu0, C0, Xtest)
S1 = np.sqrt(C1)

# Row vectors for plotting
mu0, S0 = np.ravel(mu0), np.ravel(S0)
mu1, S1 = np.ravel(mu1), np.ravel(S1)

# Prediction error
rmse = 1000 * matrix.RMSE(mu1, E1[:, 1])
print(rmse)

if __name__ == '__main__':
    # Plotting
    plt.xlim(0.8, 2.35)
    plt.ylim(-0.4, 0.6)

    grid = np.ravel(Xtest)

    # OGP
    plt.plot(grid, mu0 + 0.2, 'k--', lw=1)

    # LARGP 
    plt.scatter(X1, Y1, c='r', s=45, zorder=10, edgecolors=(0, 0, 0))
    plt.fill_between(grid, mu1 + ns * S1, mu1 - ns * S1, alpha=0.2, color='k')
    plt.plot(E1[:, 0], E1[:, 1], c='r', lw=2)
    plt.plot(grid, mu1, 'k--', lw=2)
    plt.tight_layout()
    plt.savefig("1-nargp-N2.pdf", transparent=True)

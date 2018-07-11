#!/usr/bin/env python3
import os
import GPy
import numpy as np
import matplotlib.pyplot as plt
from ARGP import largp
from ARGP import matrix
from ARGP import ordinary

# Size of confidence interval
ns = 3

# Load ab initio surface
E_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'surfaces')
E0 = np.loadtxt(os.path.join(E_path, 'cas-sto3g.tab'))
#E0 = np.loadtxt(os.path.join(E_path, 'mrci-pcvtz.tab'))
E = np.loadtxt(os.path.join(E_path, 'mrci-pcv5z.tab'))

# Set zeros at dissociation limits
E0[:, 1] += 107.43802032
#E0[:, 1] += 109.15038716
E[:, 1] += 109.15851906

# Level 0 training set
Nt0 = 40
i0 = np.random.randint(0, len(E), size=Nt0)
T0 = np.array([E0[i] for i in i0])
X0, Y0 = np.split(T0, 2, axis=1)

# Level 2 training set
Nt1 = 10
i1 = np.random.randint(0, len(E), size=Nt1)
T = np.array([E[i] for i in i1])
X, Y = np.split(T, 2, axis=1)

# Test set
Ntest = 100
Xtest = matrix.Col(np.linspace(0.8, 2.34, Ntest))

# Train ordinary model
m0 = ordinary.optimize(X0, Y0)
mu0, C0 = m0.predict(Xtest, full_cov=True)
S0 = np.sqrt(np.diag(C0))

# row vectors
mu0, S0 = np.ravel(mu0), np.ravel(S0)

# Train LARGP model
m1 = largp.optimize(m0, X, Y)
mu, C = largp.predict(m1, mu0, C0, Xtest)
mu, S = np.ravel(mu), np.ravel(np.sqrt(C))
rmse = 1000 * matrix.RMSE(mu, E[:, 1])

if __name__ == '__main__':
    # Plotting
    plt.xlim(0.8, 2.35)
    plt.ylim(-0.4, 0.6)

    # OGP
    plt.plot(Xtest, mu0 + 0.2, 'k--', lw=1)

    # LARGP 
    plt.scatter(X, Y, c='r', s=45, zorder=10, edgecolors=(0, 0, 0))
    plt.fill_between(Xtest[:, 0], mu + ns * S, mu - ns * S, alpha=0.2, color='k')
    plt.plot(E[:, 0], E[:, 1], c='r', lw=2)
    plt.plot(Xtest, mu, 'k--', lw=2)
    plt.tight_layout()
    plt.savefig("largp.pdf", transparent=True)

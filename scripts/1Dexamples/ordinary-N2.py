#!/usr/bin/env python3
import os
import numpy as np
from ARGP import ordinary
from ARGP import matrix
import matplotlib.pyplot as plt


# Load ab initio surface
E_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'surfaces')
E = np.loadtxt(os.path.join(E_path, 'mrci-pcv5z.tab'))
E[:, 1] += 109.15851906  # dissociation limit

# Training set
Nt = 20
index = np.random.randint(0, len(E), size=Nt)
T = np.array([E[i] for i in index])
X, Y = np.split(T, 2, axis=1)

# Test set
Ntest = 100
Xtest = matrix.Col(np.linspace(0.8, 2.34, Ntest))

# Train ordinary model
m = ordinary.optimize(X, Y, normalize=True)
mu, C = m.predict(Xtest, full_cov=True)
S = np.sqrt(np.diag(C))
mu, S = np.ravel(mu), np.ravel(S)
rmse = 1000*matrix.RMSE(mu, E[:, 1])

print("Prediction Error: {:>9.4f} cm-1".format(rmse))

if __name__ == '__main__':

    # Size of confidence interval
    ns = 3

    # Plotting
    plt.xlim(0.8, 2.35)
    plt.ylim(-0.4, 0.6)

    grid = np.ravel(Xtest)

    # OGP
    plt.scatter(X, Y, c='r', s=45, zorder=10, edgecolors=(0, 0, 0), label='Training point')
    plt.fill_between(grid, mu + ns * S, mu - ns * S, alpha=0.2, color='k', label='Confidence interval')
    plt.plot(E[:, 0], E[:, 1], c='r', lw=2, label='Target')
    plt.plot(grid, mu, 'k--', lw=2, label='GP prediction')
    plt.tight_layout()
    plt.legend()
    plt.savefig("ordinary-N2.pdf", transparent=True)

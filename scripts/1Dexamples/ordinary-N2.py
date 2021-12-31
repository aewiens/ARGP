import numpy as np
from ARGP import ordinary
import matplotlib.pyplot as plt

np.random.seed(0)

# Load ab initio surface
E = np.loadtxt("surfaces/mrci-pcv5z.tab")
E[:, 1] += 109.15851906

# Training set
Nt = 10
index = np.random.randint(0, len(E), size=Nt)
X, Y = np.split(E[index], 2, axis=1)

# Test set
Xtest = E[:, 0].reshape(-1, 1)

# Train ordinary model
m = ordinary.optimize(X, Y, normalize=True)
mu, C = m.predict(Xtest, full_cov=True)
S = np.sqrt(np.diag(C))
mu, S = np.ravel(mu), np.ravel(S)
rmse = 1000 * np.sqrt(((mu - E[:, 1])**2).mean())

print("Prediction Error: {:>9.4f} mEh".format(rmse))


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

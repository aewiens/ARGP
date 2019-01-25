#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import example1 as ex1
import example2 as ex2
import example3 as ex3
import example4 as ex4
import example5 as ex5
import example6 as ex6

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['axes.xmargin'] = 0

# Test set
Ntest = 100

# True ab initio surface (for comparison)
E = np.loadtxt('surfaces/mrci-pcv5z.tab')
E[:, 1] += 109.15851906  # dissociation limit
Xtest = E[:, 0]

# Size of confidence interval
ns = 3


def RMSE(prediction, target):
    return np.sqrt(((prediction - target)**2).mean())


# Plotting
nrows, ncols = 3, 2
dx, dy = 2, 1
figsize = (9.5, 2.5 * nrows)
fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

for i, ax in enumerate(axes.flat):
    ax.set_ylim(-0.38, 0.4)
    if i % 2 == 0:
        ax.set_ylabel('Energy/ $E_{\mathrm{h}}$', fontsize=12)

    if i >= 4:
        ax.set_xlabel('Bond length/ \AA', fontsize=12)
    ax.plot(E[:, 0], E[:, 1], c='r', lw=2, label='Target')

ax1, ax2, ax3, ax4, ax5, ax6, = axes.flat
Xtest = np.ravel(Xtest)

# Example 1: LARGP with 6 training points and MRCI/cc-pCVTZ base model
ax1.scatter(ex1.X, ex1.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0), label='Training point')
ax1.fill_between(Xtest, ex1.mu + ns * ex1.S, ex1.mu - ns * ex1.S, alpha=0.2, color='k', label='3$\sigma$ confidence interval')
ax1.plot(Xtest, ex1.mu, 'k--', lw=2, label=r'Prediction ($\tilde{\mu}$)')
ax1.text(2.2, 0.3, ex1.rmse_test_string, ha='right', fontsize=11)
ax1.text(2.2, 0.22, ex1.rmse_train_string, ha='right', fontsize=11)
ax1.text(2.2, -0.3, "(a)", ha='center', fontsize=12)
ax1.legend(loc=2, fontsize='10')

# Example 2: OGP with 12 training points and MRCI/cc-pCVTZ base model
ax2.scatter(ex2.X, ex2.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
ax2.fill_between(Xtest, ex2.mu + ns * ex2.S, ex2.mu - ns * ex2.S, alpha=0.2, color='k')
ax2.plot(Xtest, ex2.mu, 'k--', lw=2)
ax2.text(2.2, 0.3, ex2.rmse_test_string, ha='right', fontsize=11)
ax2.text(2.2, 0.22, ex2.rmse_train_string, ha='right', fontsize=11)
ax2.text(2.2, -0.3, "(b)", ha='center', fontsize=12)


# Example 3: LARGP with 6 training points and MRCI/cc-pCVTZ base model
ax3.plot(Xtest, ex3.mu0 + 0.1, 'k--', lw=1, label='Base model prediction ($\hat{y_0}$)')
ax3.scatter(ex3.X, ex3.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
ax3.fill_between(Xtest, ex3.mu + ns * ex3.S, ex3.mu - ns * ex3.S, alpha=0.2, color='k')
ax3.plot(Xtest, ex3.mu, 'k--', lw=2, label='Prediction ($\hat{y_1}$)')
ax3.text(2.2, 0.3, ex3.rmse_test_string, ha='right', fontsize=11)
ax3.text(2.2, 0.22, ex3.rmse_train_string, ha='right', fontsize=11)
ax3.text(2.2, -0.3, "(c)", ha='center', fontsize=11)
ax3.text(2.25, 0.05, r"$\tilde{\mu}$")

# Example 4: LARGP with 12 training points and MRCI/cc-pCVTZ base model
ax4.plot(Xtest, ex4.mu0 + 0.1, 'k--', lw=1)
ax4.scatter(ex4.X, ex4.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
ax4.fill_between(Xtest, ex4.mu + ns * ex4.S, ex4.mu - ns * ex4.S, alpha=0.2, color='k')
ax4.plot(Xtest, ex4.mu, 'k--', lw=2)
ax4.text(2.2, 0.3, ex4.rmse_test_string, ha='right', fontsize=11)
ax4.text(2.2, 0.22, ex4.rmse_train_string, ha='right', fontsize=11)
ax4.text(2.2, -0.3, "(d)", ha='center', fontsize=12)

# Example 5: LARGP with 6 training points and MRCI/cc-pCVTZ base model
ax5.plot(Xtest, ex5.mu0 + 0.1, 'k--', lw=1, label='Base model prediction ($\hat{y_0}$)')
ax5.scatter(ex5.X, ex5.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
ax5.fill_between(Xtest, ex5.mu + ns * ex5.S, ex5.mu - ns * ex5.S, alpha=0.2, color='k')
ax5.plot(Xtest, ex5.mu, 'k--', lw=2, label='Prediction ($\hat{y_1}$)')
ax5.text(2.2, 0.3, ex5.rmse_test_string, ha='right', fontsize=11)
ax5.text(2.2, 0.22, ex5.rmse_train_string, ha='right', fontsize=11)
ax5.text(2.2, -0.3, "(e)", ha='center', fontsize=12)

# Example 6: LARGP with 12 training points and MRCI/cc-pCVTZ base model
ax6.plot(Xtest, ex6.mu0 + 0.1, 'k--', lw=1)
ax6.scatter(ex6.X, ex6.Y, c='r', s=30, zorder=10, edgecolors=(0, 0, 0))
ax6.fill_between(Xtest, ex6.mu + ns * ex6.S, ex6.mu - ns * ex6.S, alpha=0.2, color='k')
ax6.plot(Xtest, ex6.mu, 'k--', lw=2, label='Prediction ($\hat{y_1}$)')
ax6.text(2.2, 0.3, ex6.rmse_test_string, ha='right', fontsize=11)
ax6.text(2.2, 0.22, ex6.rmse_train_string, ha='right', fontsize=11)
ax6.text(2.2, -0.3, "(f)", ha='center', fontsize=11)

pad = 0.05
xpad, ypad = dx * pad, dy * pad
fig.subplots_adjust(left=xpad, right=1-xpad, top=1-ypad, bottom=0.1)

# Save
plt.tight_layout()
plt.savefig("plot.pdf", transparent=True)

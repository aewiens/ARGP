import os
import GPy
import numpy as np
import matplotlib.pyplot as plt
from ARGP import largp
from ARGP import matrix
from ARGP import ordinary

np.random.seed(4)

def col(v):
    return np.ravel(v)[:, np.newaxis]


def row(v):
    return np.ravel(v)


def RMSE(prediction, target):
    return np.sqrt(((prediction - target)**2).mean())


# Load ab initio energy surfaces
E_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'surfaces')
E0= np.loadtxt(os.path.join(E_path, 'cas-sto3g.tab'))
E1= np.loadtxt(os.path.join(E_path, 'mrci-pcvtz.tab'))
E2= np.loadtxt(os.path.join(E_path, 'mrci-pcv5z.tab'))

# Set zeros at dissociation limits
E0[:, 1] += 107.43802032
E1[:, 1] += 109.12399268
E2[:, 1] += 109.15851906

# Level 0 training set
Nt0 = 100
i0 = np.random.randint(0, 100, size=Nt0)
T0 = np.array([E0[i] for i in i0])
X0, Y0 = np.split(T0, 2, axis=1)

# Level 1 training set
Nt1 = 60
i1 = np.random.choice(i0, size=Nt1)
T1 = np.array([E1[i] for i in i1])
X1, Y1 = np.split(T1, 2, axis=1)

# Level 2 training set
Nt2 = 15
i2 = np.random.randint(0, len(E2), size=Nt2)
T2 = np.array([E2[i] for i in i2])
X2, Y2 = np.split(T2, 2, axis=1)

# Test set
Ntest = 100
Xtest = np.linspace(0.8, 2.34, Ntest)[:, np.newaxis]

nsamples = 100

# Train level 0 (ordinary GP)
m0 = ordinary.optimize(X0, Y0)
mu0, C0 = ordinary.predict(m0, X1, full_cov=True)

# Train Level 1 (LARGP)
m1 = largp.optimize(mu0, X1, Y1)

# Prepare for level 2
mu0, C0 = ordinary.predict(m0, X2, full_cov=True)
mu1, C1 = largp.predict(m1, mu0, C0, X2, nsamples=nsamples)

# Train Level 2 (LARGP)
m2 = largp.optimize(mu1, X2, Y2)

mu0, C0 = ordinary.predict(m0, Xtest, full_cov=True)
Z0 = np.random.multivariate_normal(row(mu0), C0, nsamples)

# push samples through f_1 and f_2
tmp_m = np.zeros((nsamples**2, Ntest))
tmp_v = np.zeros((nsamples**2, Ntest))
cnt = 0
for i in range(nsamples):
    mu, C = m1.predict(np.hstack((Xtest, col(Z0[i,:]))), full_cov=True)
    Q = np.random.multivariate_normal(mu.flatten(), C, nsamples)
    for j in range(nsamples):
        mu, v = m2.predict(np.hstack((Xtest, col(Q[j,:]))))
        tmp_m[cnt,:] = mu.flatten()
        tmp_v[cnt,:] = v.flatten()
        cnt = cnt + 1

# get f_2 posterior mean and variance at Xtest
mu2 = np.mean(tmp_m, axis = 0)
C2 = np.abs(np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0))
S2 = np.sqrt(C2)
rmse = RMSE(mu2, E2[:,1]) * 1000
print(rmse)

# Predict model 1 at test points for plotting
mu1, C1 = largp.predict(m1, mu0, C0, Xtest, nsamples=nsamples)

# row vectors for plotting
mu0, S0 = row(mu0), row(np.sqrt(np.diag(C0)))
mu1, S1 = row(mu1), row(np.sqrt(C1))

# PLOTTING

# Size of confidence interval
ns = 3

plt.xlim(0.8, 2.35)
plt.ylim(-0.4, 0.6)

grid = np.ravel(Xtest)

# OGP
plt.plot(grid, mu0 + 0.2, 'k--', lw=1)
plt.plot(grid, mu1 + 0.1, 'k--', lw=1.5)

# LARGP 
plt.scatter(X2, Y2, c='r', s=45, zorder=10, edgecolors=(0, 0, 0))
plt.fill_between(grid, mu2 + ns * S2, mu2 - ns * S2, alpha=0.2, color='k')
plt.plot(E2[:, 0], E2[:, 1], c='r', lw=2)
plt.plot(grid, mu2, 'k--', lw=2)

plt.tight_layout()
plt.savefig("2-largp.pdf", transparent=True)

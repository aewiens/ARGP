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
E1[:, 1] += 109.15038716
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
#i2 = np.array([94, 15, 23, 11, 72, 13, 78, 36, 8, 69, 18, 88])
#Nt2 = len(i2)
Nt2 = 15
i2 = np.random.randint(0, len(E2), size=Nt2)
T2 = np.array([E2[i] for i in i2])
X2, Y2 = np.split(T2, 2, axis=1)
print(i2)

# Test set
nx = 100
x = col(np.linspace(0.8, 2.34, nx))

# Train level 0 using ordinary GP
k0 = GPy.kern.RBF(1.)
m0 = GPy.models.GPRegression(X=X0, Y=Y0, kernel=k0)
m0.optimize(max_iters=1000)
m0.optimize_restarts(30, optimizer="bfgs", max_iters=1000)
mu0, C0 = m0.predict_noiseless(col(x), full_cov=True)
Z0 = np.random.multivariate_normal(row(mu0), C0, size=1000)


# Train Level 1 (LARGP)
XX1 = np.hstack((X1, m0.predict(X1)[0]))
k1 = GPy.kern.RBF(1, active_dims=[1]) + GPy.kern.RBF(1, active_dims=[0])
m1 = GPy.models.GPRegression(X=XX1, Y=Y1, kernel=k1)
m1.optimize(max_iters=1000)
m1.optimize_restarts(30, optimizer="bfgs", max_iters=1000)

nsamples=100
Z1 = np.random.multivariate_normal(row(mu0), C0, size=nsamples)
mu1, C1 = largp.predict(m1, mu0, C0, col(x), nsamples=nsamples)

# Prepare for level 2
mu00, C00 = m0.predict(X2, full_cov=True)
Z00 = np.random.multivariate_normal(row(mu00), C00, size=nsamples)
tmp_m = np.zeros((nsamples, Nt2))
tmp_v = np.zeros((nsamples, Nt2))

for i in range(nsamples):
    temp = np.hstack((X2, col(Z00[i, :])))
    mu, v = m1.predict(temp)
    tmp_m[i,:] = row(mu)
    tmp_v[i,:] = row(v)

# mean and variance of model1 at X2:
mu02 = np.mean(tmp_m, axis=0)
v02 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
v02 = np.abs(row(v02))

# Train Level 2 (LARGP)
XX2 = np.hstack((X2, col(mu02)))
k2 = GPy.kern.RBF(1, active_dims = [1]) + GPy.kern.RBF(1, active_dims=0)
m2 = GPy.models.GPRegression(X=XX2, Y=Y2, kernel=k2)
m2.optimize(max_iters=500)
m2.optimize_restarts(30, optimizer='bfgs', max_iters=1000)

# push samples through f_2 and f_3
tmp_m = np.zeros((nsamples**2, nx))
tmp_v = np.zeros((nsamples**2, nx))
cnt = 0
for i in range(nsamples):
    mu, C = m1.predict(np.hstack((col(x), col(Z0[i,:]))), full_cov=True)
    Q = np.random.multivariate_normal(mu.flatten(), C, nsamples)
    for j in range(nsamples):
        mu, v = m2.predict(np.hstack((col(x), col(Q[j,:]))))
        tmp_m[cnt,:] = mu.flatten()
        tmp_v[cnt,:] = v.flatten()
        cnt = cnt + 1

# get f_2 posterior mean and variance at Xtest
mu2 = np.mean(tmp_m, axis = 0)
C2 = np.abs(np.mean(tmp_v, axis = 0) + np.var(tmp_m, axis = 0))
S2 = np.sqrt(C2)
rmse = RMSE(mu2, E2[:,1]) * 1000
print(rmse)

# row vectors
mu0, S0 = row(mu0), row(np.sqrt(np.diag(C0)))
mu1, S1 = row(mu1), row(np.sqrt(C1))

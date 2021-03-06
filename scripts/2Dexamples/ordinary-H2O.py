iport GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary
from ARGP import lhs 

E = np.loadtxt("surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)
E[:, 2] += 76.172634420318

np.random.seed(10)

def RMSE(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))

# Create test set
Xtest = E[:, :-1]

# Create training set
N1 = 100
dim = 2

#idx = lhs.latin_hypercube_samples(0, len(E), N1, dtype='int')
idx = np.random.randint(0, len(E), size=N1)
T = E[idx]
X1, Y1 = np.split(T, [2], axis=1)

#  Ordinary GP
kernel = GPy.kern.RBF(dim, ARD=True)
model = ordinary.optimize(X1, Y1, kernel, normalize=True, restarts=10)
mu, v = ordinary.predict(model, Xtest)

# Calculate error
rmse = 1000 * RMSE(mu.flatten(), E[:, -1])  # millihartrees
print("RMSE: {:>10.5f} mEh".format(rmse))

"""
# Exact plot
X, Y = np.meshgrid(X1[:, 0], X1[:, 1])
Exactplot = ml.griddata(Xtest[:,0], Xtest[:,1], E[:, -1], X, Y, interp='linear')
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Exactplot, color = '#377eb8', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)

# Plot ordinary GP prediction
GPplot = ml.griddata(Xtest[:,0],Xtest[:,1], mu.flatten(), X, Y, interp = 'linear')
fig = plt.figure(2)                            
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, GPplot, color = 'red', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)
plt.show()
"""

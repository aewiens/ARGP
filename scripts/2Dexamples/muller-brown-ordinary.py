import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from ARGP import ordinary

np.random.seed(10)

def my_high(x, y):
    x0 = np.array((1, 0, -0.5, -1))
    y0 = np.array((0, 0.5, 1.5, 1))
    e = np.array((-200, -100, -170, 15))
    a = np.array((-1, -1, -6.5, 0.7))
    b = np.array((0, 0, 11, 0.6))
    c = np.array((-10, -10, -6.5, 0.7))
    return np.sum(e * np.exp(a * (x - x0) ** 2 + b * (x - x0) * (y - y0) + 
                  c * (y - y0) ** 2))

def high(x):
    x1 = x[:,0]
    x2 = x[:,1]
    return (-1.275*x1**2 / np.pi**2 + 5.0*x1/np.pi + x2 - 6.0)**2 + (10.0 - 5.0/(4.0*np.pi))*np.cos(x1) + 10.0

def scale_range(x,ub,lb):
    Np = x.shape[0]
    dim = x.shape[1]
    for i in range(0,Np):
        for j in range(0,dim):
            tmp = ub[j] -lb[j]
            x[i][j] = tmp*x[i][j] + lb[j]
    return x


def rmse(pred, truth):
    pred = pred.flatten()
    truth = truth.flatten()
    return np.sqrt(np.mean((pred-truth)**2))


''' Create training set '''
N1 = 200
dim = 2
lb = np.array([-1.5, -0.5])
ub = np.array([1.0, 1.0])

tmp = np.random.rand(1000, dim)
Xtrain = scale_range(tmp, ub, lb)
idx = np.random.permutation(1000)
X1 = Xtrain[idx[0:N1], :]
Y1 = np.array([my_high(*i) for i in X1])[:, np.newaxis]

lb = np.array([-1.5, -0.5])
ub = np.array([0.9, 0.75])
x1 = np.linspace(lb[0], ub[0], 50)
x2 = np.linspace(lb[1], ub[1], 50)

tmp = np.random.rand(1000,2)
Xtest = scale_range(tmp,ub,lb)

active_dimensions = np.arange(0,dim)

#  Ordinary GP
kernel = GPy.kern.RBF(dim, ARD=True)
model = ordinary.optimize(X1, Y1, kernel, normalize=False)
mu, v = ordinary.predict(model, Xtest)

# Calculate error
Exact = np.array([my_high(*i) for i in Xtest])
Exact = Exact[:, np.newaxis]
ogp_error = np.linalg.norm(Exact - mu)/np.linalg.norm(Exact)
print("error: ", ogp_error)

# Exact plot
X, Y = np.meshgrid(x1, x2)
Exactplot = ml.griddata(Xtest[:,0], Xtest[:,1], Exact[:, 0], X, Y, interp='linear')
fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Exactplot, color = '#377eb8', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)
plt.savefig("MB-exact.pdf")

# Plot ordinary GP prediction
GPplot = ml.griddata(Xtest[:,0],Xtest[:,1], mu.flatten(), X, Y, interp = 'linear')
fig = plt.figure(2)                            
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, GPplot, color = 'red', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)
plt.savefig("MB-ordinary.pdf")

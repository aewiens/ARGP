import GPy
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as ml
from itertools import starmap

np.random.seed(10)

"""
A = np.array([-200, -100, -170, 15])
B = np.array([[1, 0], [0, 0.5], [-0.5, 1.5], [-1, 1]])
C = np.array([[[-1, 0], [0, -10]], [[-1, 0], [0, -10]],
                 [[-6.5, 5.5], [5.5, -6.5]], [[0.7, 0.3], [0.3, 0.7]]])


def my_high(x):
    eqfs = starmap(_exponential_quadratic_function, zip(A, B, C))
    return sum(eqf(x) for eqf in eqfs)


def _exponential_quadratic_function(a, b, c):

    def _f(x):
        dx = x - _cast(b, 0, np.ndim(x))
        cdx = np.tensordot(c, dx, axes=(1, 0))
        return a * np.exp(np.sum(dx * cdx, axis=0))

    return _f
"""

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

"""
def my_high(x, y):
    return (-1.275*x**2 / np.pi**2 + 5.0*x/np.pi + y - 6.0)**2 + (10.0 - 5.0/(4.0*np.pi)) * np.cos(x) + 10.0
"""

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
#Y1 = high(X1)[:, None]
Y1 = np.array([my_high(*i) for i in X1])[:, np.newaxis]

#nn = 40
lb = np.array([-1.5, -0.5])
ub = np.array([0.9, 0.75])
x1 = np.linspace(lb[0], ub[0], 50)
x2 = np.linspace(lb[1], ub[1], 50)
X, Y = np.meshgrid(x1, x2)

tmp = np.random.rand(1000,2)
Xtest = scale_range(tmp,ub,lb)

#Exact = high(Xtest)
Exact = np.array([my_high(*i) for i in Xtest])

Exactplot = ml.griddata(Xtest[:,0], Xtest[:,1], Exact, X, Y, interp = 'linear')
active_dimensions = np.arange(0,dim)

#  Ordinary GP
k4 = GPy.kern.RBF(dim, ARD=True)
m4 = GPy.models.GPRegression(X=X1, Y=Y1, kernel=k4)

m4[".*Gaussian_noise"] = m4.Y.var()*0.01
m4[".*Gaussian_noise"].fix()

m4.optimize(max_iters = 500)

m4[".*Gaussian_noise"].unfix()
m4[".*Gaussian_noise"].constrain_positive()

m4.optimize_restarts(20, optimizer = "bfgs",  max_iters = 1000)
mu4, v4 = m4.predict(Xtest)
Exact = Exact[:, np.newaxis]
ogp_error = np.linalg.norm(Exact - mu4)/np.linalg.norm(Exact)
print("error: ", ogp_error)

GPplot = ml.griddata(Xtest[:,0],Xtest[:,1], mu4.flatten(), X, Y, interp = 'linear')

fig = plt.figure(1)
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(X, Y, Exactplot, color = '#377eb8', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)
plt.savefig("plots/ordinary-1.pdf")

fig = plt.figure(2)                            
ax2 = fig.add_subplot(111, projection='3d')
ax2.plot_surface(X, Y, GPplot, color = 'red', rstride=2, cstride=2,
                 linewidth=0, antialiased=True, shade = True, alpha = 0.6)

plt.savefig("plots/ordinary-2.pdf")

"""
fig = plt.figure(3)
plt.pcolor(X, Y, GPplot, cmap='jet')
plt.plot(X1[:,0], X1[:,1], marker='o', linestyle = '')
plt.colorbar()
plt.savefig("plots/ordinary-3.pdf")
"""

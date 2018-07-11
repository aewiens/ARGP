import GPy
import numpy as np
from . import matrix

def optimize(mu0, X, Y):
    kernel = GPy.kern.RBF(1, active_dims=[1]) + GPy.kern.RBF(1, active_dims=[0])

    XX = np.hstack((X, mu0))
    model = GPy.models.GPRegression(X=XX, Y=Y, kernel=kernel)
    model.optimize(max_iters=500)
    model.optimize_restarts(30, optimizer='bfgs', max_iters=1000)

    return model

def predict(model, mu0, C0, grid, nsamples=1000):

    Nts = len(grid)

    # Predict level 1 at test points
    Z = np.random.multivariate_normal(mu0.flatten(), C0, nsamples)

    # Predict level 2 at test points 
    tmp_m = np.zeros((nsamples, Nts))
    tmp_v = np.zeros((nsamples, Nts))
    for i in range(nsamples):
        XX = np.hstack((grid, matrix.Col(Z[i,:])))
        mu, v = model.predict_noiseless(XX)
        tmp_m[i,:] = mu.flatten()
        tmp_v[i,:] = v.flatten()

    # Get posterior mean and variance
    mu2 = matrix.Col(np.mean(tmp_m, axis = 0))
    C2 = np.abs(matrix.Col(np.mean(tmp_v, axis = 0)) + matrix.Col(np.var(tmp_m, axis = 0)))

    return mu2, C2

#!/usr/bin/env python3
import numpy as np
from scipy import linalg
import GPy
from . import matrix
from . import kernel

def optimize(x, y, maxiter=500, restarts=30, optimizer='bfgs'):
    """ Optimize kernel hyperparameters on the training set (x, y)
        using maximum likelihood estimation (MLE) in GPy

    Parameters
    ----------
    x : array_like
        Set of training points
    y : array_like
        Labels for the training points

    Return
    ------
    variance : ndarray
        Numpy 1d-array of optimized variance parameters
    lengthscale : 1d-array
        Numpy 1d-array of optimized lengthscale
    """
    # how to make this part general?

    kernel = GPy.kern.RBF(1)

    print(kernel)
    model = GPy.models.GPRegression(X=x, Y=y, kernel=kernel)
    model.optimize(max_iters=maxiter)
    model.optimize_restarts(30, optimizer="bfgs", max_iters=1000)
    print(kernel)

    gaussian_variance = model.likelihood.gaussian_variance(None)[0]

    variance = np.array(kernel.variance)
    lengthscale = np.array(kernel.lengthscale)

    return model
    #return variance, lengthscale, gaussian_variance


def predict(model, grid, full_cov=False):
    """ OGP prediction """
    return model.predict(grid, full_cov=full_cov)


def my_predict(x, y, grid, k_params):

    Theta = (k_params[0], k_params[1])
    gaussian_variance = k_params[2]

    KXX = kernel.RBF(Theta, grid)
    KTX = kernel.RBF(Theta, x, grid)
    KTT = kernel.RBF(Theta, x)

    for i in range(len(KTT)):
        KTT[i, i] += gaussian_variance + 1e-8
        
    LW = matrix.jitchol(KTT)
    alpha = matrix.dpotrs(LW, y, lower=1)[0]
    mean = np.dot(KTX.T, alpha)

    tmp = linalg.lstsq(LW, KTX)[0]
    var = KXX - np.dot(tmp.T, tmp)

    return mean, var

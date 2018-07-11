import numpy as np
from . import matrix

def RBF(theta, X, X2=None):
    """ Radial basis function kernel """
    var, ls = theta[0], theta[1]
    r = matrix.euclidean_distance(X, X2=X2) / ls
    return var * np.exp(-0.5 * r ** 2)

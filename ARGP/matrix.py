# Copyright (c) 2012, GPy authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

# tdot function courtesy of Ian Murray:
# Iain Murray, April 2013. iain contactable via iainmurray.net
# http://homepages.inf.ed.ac.uk/imurray2/code/tdot/tdot.py
import numpy as np
from scipy import linalg
from scipy.linalg import lapack


def Col(v):
    """ Return a contiguous flattened array in column format

    Parameters
    ----------
    v : array_like
        Read into a 1D array, then transposed into a column vector

    Return
    ------
    array_like
        Output array of the same subtype as a, with shape (1, v.size)
    """
    return np.ravel(v)[:, np.newaxis]


def RMSE(prediction, target):
    """

    Parameters
    ----------

    Return
    ------
    """
    return np.sqrt(((prediction - target)**2).mean())
    

def jitchol(A, maxtries=5):
    A = np.ascontiguousarray(A)
    L, info = lapack.dpotrf(A, lower=1)
    if info == 0:
        return L
    else:
        diagA = np.diag(A)
        if np.any(diagA <= 0.):
            raise linalg.LinAlgError("not pd: non-positive diagonal elements")
        jitter = diagA.mean() * 1e-6
        num_tries = 1
        while num_tries <= maxtries and np.isfinite(jitter):
            try:
                L = linalg.cholesky(A + np.eye(A.shape[0]) * jitter, lower=True)
                return L
            except:
                jitter *= 10
            finally:
                num_tries += 1
        raise linalg.LinAlgError("not positive definite, even with jitter.")
    import traceback
    try: raise
    except:
        print('\n'.join(['Added jitter of {:.10e}'.format(jitter),
            '  in '+traceback.format_list(traceback.extract_stack(limit=3)[-2:-1])[0][2:]]))
    return L


def euclidean_distance(X, X2=None):
    """
    Compute the Euclidean distance between each row of X and X2.
    (between each pair of rows of X if X2 is None)
    """
    if X2 is None:
        Xsq = np.sum(np.square(X), 1)
        r2 = -2.*np.dot(X, X.T) + (Xsq[:, None] + Xsq[None, :])
        # Force diagnoal to be zero
        for i in range(len(r2)):
            r2[i,i] = 0.
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)

    else:
        X1sq = np.sum(np.square(X), 1)
        X2sq = np.sum(np.square(X2), 1)
        r2 = -2.*np.dot(X, X2.T) + (X1sq[:, None] + X2sq[None, :])
        r2 = np.clip(r2, 0, np.inf)
        return np.sqrt(r2)


def dpotrs(A, B, lower=1):
    """
    Wrapper for lapack dpotrs function
    :param A: Matrix A
    :param B: Matrix B
    :param lower: is matrix lower (true) or upper (false)
    :returns:
    """
    A = np.asfortranarray(A)
    return linalg.lapack.dpotrs(A, B, lower=lower)

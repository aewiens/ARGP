import numpy as np
from scipy import stats
from scipy import sparse
from itertools import chain


def indexable(*iterables):
    """Make arrays indexable for cross-validation.

    Checks consistent length, passes through None, and ensures that everything
    can be indexed by converting sparse matrices to csr and converting
    non-interable objects to arrays.

    Parameters
    ----------
    *iterables : lists, dataframes, arrays, sparse matrices
        List of objects to ensure sliceability.
    """
    result = []
    for X in iterables:
        if sparse.issparse(X):
            result.append(X.tocsr())
        elif hasattr(X, "__getitem__") or hasattr(X, "iloc"):
            result.append(X)
        elif X is None:
            result.append(X)
        else:
            result.append(np.array(X))
    return result


def split(X, n_train, n_test=None, seed=None):

    n_samples = len(X)
    if n_test is None:
        n_test = n_samples - n_train

    # random partition
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(n_samples)
    ind_test = permutation[:n_test]
    ind_train = permutation[n_test:(n_test + n_train)]

    return ind_train, ind_test


def train_test_split(*arrays, n_train, n_test=None, seed):
    arrays = indexable(*arrays)
    train, test = split(X=arrays[0], n_train=n_train, n_test=n_test, seed=seed)
    return list(chain.from_iterable((a[train], a[test]) for a in arrays))


def train_test_split2(*arrays, ind_train_1, ind_test_1, n_train, seed):

    arrays = indexable(*arrays)
    train_, test_ = split(ind_train_1, n_train=n_train, seed=seed)
    train = ind_train_1[train_]
    test = np.concatenate((ind_train_1[test_], ind_test_1))

    return list(chain.from_iterable((a[train], a[test]) for a in arrays))
    #return X[train2], X[test2], y[train2], y[test2]


def smart_random(dataset, n_train, n_test, save_to=None):
    """
    choose a random training set that has an energy distribution most resembling that of the full dataset.
    uses the chi-squared method to estimate the similarity of the energy distrubtions.
    """
    # dataset must be a pandas dataframe
    x, y = np.split(dataset, [-1], axis=1)
    full_dataset_dist, binedges = np.histogram(y, bins='auto', density=True)
    pvalues = []
    for seed in range(500):
    #for seed in range(1):
        y_train = train_test_split(x, y, n_train=n_train, n_test=n_test, seed=seed)[2]
        train_dist, tmpbin = np.histogram(y_train, bins=binedges, density=True)
        chisq, p = stats.chisquare(train_dist, f_exp=full_dataset_dist)
        pvalues.append(p)
    best_seed = np.argmax(pvalues)
    indices = np.arange(len(dataset))
    ind_train, ind_test  = train_test_split(indices, n_train=n_train, n_test=n_test, seed=best_seed)

    if save_to:
        np.savetxt(save_to + "_train.dat", ind_train)
        np.savetxt(save_to + "_test.dat", ind_test)

    return ind_train, ind_test


def smart_random2(dataset, n_train, i1, i1_test):
    """
    choose a random training set that has an energy distribution most resembling that of the full dataset.
    uses the chi-squared method to estimate the similarity of the energy distrubtions.
    """
    # dataset must be a pandas dataframe
    x, y = np.split(dataset, [-1], axis=1)
    full_dataset_dist, binedges = np.histogram(y, bins='auto', density=True)
    pvalues = []
    for seed in range(500): 
        y_train = train_test_split2(x, y, ind_train_1=i1, ind_test_1=i1_test, n_train=n_train, seed=seed)[2]
        train_dist, tmpbin = np.histogram(y_train, bins=binedges, density=True)
        chisq, p = stats.chisquare(train_dist, f_exp=full_dataset_dist)
        pvalues.append(p)
    best_seed = np.argmax(pvalues)
    indices = np.arange(len(dataset))
    ind_train, ind_test  = train_test_split2(indices, ind_train_1 = i1, ind_test_1=i1_test, n_train=n_train, seed=best_seed)
    return ind_train, ind_test


if __name__ == '__main__':

    E1 = np.loadtxt("../scripts/3Dexamples/surfaces/ccsd-t-dz.dat", delimiter=',', skiprows=1)
    E2 = np.loadtxt("../scripts/3Dexamples/surfaces/ccsd-t-5z.dat", delimiter=',', skiprows=1)

    E1x, E1y = np.split(E1, [-1], axis=1)
    E2x, E2y = np.split(E2, [-1], axis=1)

    train1, test1 = split(E1x, 10)
    train2, test2 = smart_random2(E2, 5, train1, test1)

__author__ = 'jrx'

import itertools as it
import numpy as np


def least_squares_regression(X, y):
    """ Calculate least squares regression """
    # Get data shape
    N, p = X.shape

    # Calculate beta coefficients
    Z = np.linalg.inv(np.dot(X.T, X))
    betahat = np.dot(np.dot(Z, X.T), y)

    # Calculate standard errors
    epsilon = y - np.dot(X, betahat)
    rss = np.dot(epsilon, epsilon)
    sigma2 = 1.0 / (N - p) * rss
    sigma = np.sqrt(sigma2)
    errors = sigma * np.sqrt(np.matrix.diagonal(Z))

    return {
        'betahat': betahat,
        'rss': rss,
        'sigma': sigma,
        'sigma2': sigma2,
        'errors': errors
    }


def powerset(iterable):
    """ powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3) """
    s = list(iterable)
    return it.chain.from_iterable(it.combinations(s, r) for r in range(len(s)+1))

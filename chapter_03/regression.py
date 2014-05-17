__author__ = 'jrx'

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
    sigma2 = 1.0 / (N - p) * np.dot(epsilon, epsilon)
    sigma = np.sqrt(sigma2)
    errors = sigma * np.sqrt(np.matrix.diagonal(Z))

    return betahat, errors

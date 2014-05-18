__author__ = 'jrx'

import itertools as it
import numpy as np


def least_squares_regression(X, y):
    """ Calculate least squares regression """
    Z = np.linalg.inv(np.dot(X.T, X))
    betahat = np.dot(np.dot(Z, X.T), y)
    return betahat


def least_squares_regression_with_std_errors(X, y):
    """ Calculate least squares regression with standard errors calculated """
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

    return betahat, errors


def best_subset_selection(X, y, size):
    """ Find the best (in terms of square error) subset of input labels of given sample """
    N, p = X.shape

    # Remember best values
    best_combination = None
    best_error = None
    best_beta = None

    for combination in it.combinations(range(p-1), size):
        # Add the intercept
        indices = [0] + [x + 1 for x in combination]
        Xsel = X[:, indices]
        beta = least_squares_regression(Xsel, y)

        residuals = y - np.dot(Xsel, beta)
        rss = np.dot(residuals, residuals)

        if (best_error is None) or (rss < best_error):
            best_error = rss
            best_combination = indices
            best_beta = beta

    return best_beta, best_combination


def test_error(beta, X, y):
    """ Calculate the error of the model using methodology from the book """
    residuals = y - np.dot(X, beta)
    residuals2 = residuals * residuals

    N, p = X.shape

    avg_error = np.average(residuals2)
    std_error = np.std(residuals2, ddof=1) / np.sqrt(N)

    return avg_error, std_error

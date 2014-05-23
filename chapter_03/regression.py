__author__ = 'jrx'

import itertools as it
import numpy as np
import pandas as pd

import common.base as base


def test_error(model, coords, values):
    """ Calculate the error of the model using methodology from the book """
    residuals = values - model.calculate(coords)
    residuals2 = residuals * residuals

    n, p = coords.shape

    avg_error = np.average(residuals2)
    std_error = np.std(residuals2, ddof=1) / np.sqrt(n)

    return avg_error, std_error


class LeastSquaresRegression(base.Regression):
    """ Linear least squares regression """

    def __init__(self, coords, values):
        super(LeastSquaresRegression, self).__init__()

        # Get data shape
        n, p = coords.shape

        # Calculate beta coefficients
        z = np.linalg.inv(np.dot(coords.T, coords))
        self.betahat = np.dot(np.dot(z, coords.T), values)

        # Calculate standard errors
        epsilon = values - np.dot(coords, self.betahat)
        self.rss = np.dot(epsilon, epsilon)

        sigma2 = 1.0 / (n - p) * self.rss
        self.sigma = np.sqrt(sigma2)
        self.std_errors = self.sigma * np.sqrt(np.matrix.diagonal(z))

    def calculate(self, samples):
        return pd.Series(np.dot(samples, self.betahat), index=samples.index)


class BestSubsetSelection(base.Regression):
    """ Best subset selection algorithm """

    def __init__(self, coords, values, size):
        super(BestSubsetSelection, self).__init__()

        n, p = coords.shape

        # Remember best values
        best_combination = None
        best_error = None
        best_beta = None

        for combination in it.combinations(range(p-1), size):
            # Add the intercept
            indices = [0] + [x + 1 for x in combination]
            x_sel = coords.iloc[:, indices]

            z = np.linalg.inv(np.dot(x_sel.T, x_sel))
            beta = np.dot(np.dot(z, x_sel.T), values)

            residuals = values - np.dot(x_sel, beta)
            rss = np.dot(residuals, residuals)

            if (best_error is None) or (rss < best_error):
                best_error = rss
                best_combination = indices
                best_beta = beta

        self.betahat = best_beta
        self.best_combination = best_combination

    def calculate(self, samples):
        return pd.Series(np.dot(samples.values[:, self.best_combination], self.betahat), index=samples.index)

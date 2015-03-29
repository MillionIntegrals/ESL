__author__ = 'jrx'

import collections

import numpy as np
import pandas as pd

import common.base as base
import common.math as cm_math


def indicator_matrix(values):
    """ Transform a series of classes into a 0-1 classification matrix """
    # TODO(jerry): maybe there is a better way to do indicator matrix in the standard library
    # Take all the classes from y
    classes = sorted(set(values))

    y = pd.DataFrame()

    # Transform classes into a classification matrix
    for cls in classes:
        y[cls] = values == cls

    return y.astype(int)


def classification_error_rate(classifier, coords, values):
    """ Calculate classification error rate """
    y_hat = classifier.classify(coords)
    return (y_hat != values).mean()


class LeastSquaresClassifier(base.Classification):
    """ Train linear least squares classifier """

    def __init__(self, coords, values):
        super(LeastSquaresClassifier, self).__init__()

        self.classes = sorted(set(values))

        value_indicator_matrix = indicator_matrix(values)
        inverse_cov = np.linalg.inv(np.dot(coords.T, coords))

        self.betahat = pd.DataFrame(
            # TODO(jerry): check if there exists a multi-dot-product
            np.dot(np.dot(inverse_cov, coords.T), value_indicator_matrix),
            columns=self.classes, index=coords.columns
        )

    def classify(self, samples):
        """ Classify x """
        y_hat = samples.dot(self.betahat)
        # TODO(jerry): check if this could be done in one go
        return y_hat.apply(lambda x: x.idxmax(), axis=1)


class LinearDiscriminantClassifier(base.Classification):
    """ Train linear discriminant classifier """

    def __init__(self, coords, values):
        super(LinearDiscriminantClassifier, self).__init__()

        # TODO(jerry): think about dropping pandas classes for calculations
        self.classes = sorted(set(values))

        k = len(self.classes)

        n, p = coords.shape

        # Per-class probabilities
        self.probabilities = pd.Series(collections.Counter(values), index=self.classes) / values.size

        # Calculate means
        x_copy = coords.copy()
        x_copy['y'] = values
        self.means = x_copy.groupby('y').mean()

        self.sigma = np.zeros((p, p))

        for idx, x in x_copy.iterrows():
            mu = self.means.loc[int(x.y)]
            xv = x.drop('y') - mu
            self.sigma += np.outer(xv, xv)

        self.sigma /= (n - k)
        self.sigma_inv = np.linalg.inv(self.sigma)

        # The second part of this expression calculates mu^T Sigma mu per each class
        self.constants = (
            np.log(self.probabilities) - 0.5 * cm_math.double_product(self.means, self.sigma_inv, self.means)
        )

        self.discrimination_matrix = np.dot(self.sigma_inv, self.means.values.T)

    def classify(self, samples):
        """ Classify X """
        df = pd.DataFrame(np.dot(samples, self.discrimination_matrix), columns=self.classes)
        # TODO(jerry): check if this can be done in one go
        return (df + self.constants).apply(lambda row: row.idxmax(), axis=1)


class QuadraticDiscriminantClassifier(base.Classification):
    """ Train quadratic discriminant classifier """

    def __init__(self, coords, values):
        super(QuadraticDiscriminantClassifier, self).__init__()

        self.classes = sorted(set(values))

        # Per-class probabilities
        self.probabilities = pd.Series(collections.Counter(values), index=self.classes) / values.size

        # Calculate means
        x_copy = coords.copy()
        x_copy['y'] = values
        self.means = x_copy.groupby('y').mean()

        self.covariance = {}
        self.covariance_inv = {}
        self.covariance_det = {}

        for cls in self.classes:
            selected = x_copy[x_copy['y'] == cls]
            mu = self.means.loc[cls]

            cov = np.cov(selected.drop('y', axis=1) - mu, rowvar=0, ddof=1)

            self.covariance[cls] = cov
            self.covariance_inv[cls] = np.linalg.inv(cov)
            self.covariance_det[cls] = np.linalg.det(cov)

        # Constant part of the discrimination function
        self.constants = np.log(self.probabilities) - 0.5 * np.log(pd.Series(self.covariance_det, index=self.classes))

    def classify(self, samples):
        """ Classify X """
        columns = {}

        for cls in self.classes:
            const = self.constants[cls]
            mu = self.means.loc[cls]
            xs = samples - mu
            sigma_inv = self.covariance_inv[cls]
            columns[cls] = const - 0.5 * cm_math.double_product(xs, sigma_inv, xs)

        discriminant_matrix = pd.concat(columns, axis=1)
        return discriminant_matrix.apply(lambda row: row.idxmax(), axis=1)

__author__ = 'jrx'

import collections

import numpy as np
import pandas as pd

import common.base as base
import common.math as cm_math


def classification_error_rate(classifier, coords, values):
    """ Calculate classification error rate """
    y_hat = classifier.classify(coords)
    return np.mean(y_hat != values)


class LeastSquaresClassifier(base.Classification):
    """ Train linear least squares classifier """

    def __init__(self, coords, values):
        super(LeastSquaresClassifier, self).__init__()

        np_coords = coords.values
        value_indicator_matrix = pd.get_dummies(values)

        self.classes = np.array(value_indicator_matrix.columns)
        np_value_indicator_matrix = value_indicator_matrix.values

        inverse_cov = np.linalg.inv(np.dot(np_coords.T, np_coords))
        self.betahat = np.dot(np.dot(inverse_cov, np_coords.T), np_value_indicator_matrix)

    def classify(self, samples):
        """ Classify x """
        y_hat = np.dot(samples.values, self.betahat)
        return pd.Series(self.classes[np.argmax(y_hat, axis=1)], index=samples.index)


class LinearDiscriminantClassifier(base.Classification):
    """ Train linear discriminant classifier """

    def __init__(self, coords, values):
        super(LinearDiscriminantClassifier, self).__init__()

        np_coords = coords.values
        np_values = values.values

        self.classes = np.array(sorted(set(np_values)))

        k = len(self.classes)
        n, p = np_coords.shape

        # Per-class probabilities
        self.probabilities = (
            pd.Series(collections.Counter(np_values), index=self.classes, dtype=float).values / np_values.size
        )

        # Calculate means
        self.means = coords.groupby(values).mean()
        self.sigma = np.zeros((p, p))

        for idx in xrange(np_coords.shape[0]):
            mu = self.means.loc[np_values[idx]].values
            xv = np_coords[idx] - mu
            self.sigma += np.outer(xv, xv)

        self.sigma /= (n - k)
        self.sigma_inv = np.linalg.inv(self.sigma)
        self.np_means = self.means.values

        # The second part of this expression calculates mu^T Sigma mu per each class
        self.constants = (
            np.log(self.probabilities) - 0.5 * cm_math.double_product(self.np_means, self.sigma_inv, self.np_means)
        )

        self.discrimination_matrix = np.dot(self.sigma_inv, self.np_means.T)

    def classify(self, samples):
        """ Classify X """
        discriminant = np.dot(samples.values, self.discrimination_matrix) + self.constants
        return self.classes[np.argmax(discriminant, axis=1)]


class QuadraticDiscriminantClassifier(base.Classification):
    """ Train quadratic discriminant classifier """

    def __init__(self, coords, values):
        super(QuadraticDiscriminantClassifier, self).__init__()

        np_coords = coords.values
        np_values = values.values

        self.classes = np.array(sorted(set(values)))
        self.class_idx = dict(zip(self.classes, xrange(len(self.classes))))

        self.n = len(self.classes)

        # Per-class probabilities
        self.probabilities = (
            pd.Series(collections.Counter(values), index=self.classes, dtype=float).values / values.size
        )

        # Calculate means
        self.means = coords.groupby(values).mean().values

        self.covariance = {}
        self.covariance_inv = {}
        self.covariance_det = np.zeros(self.n)

        for idx, cls in enumerate(self.classes):
            selected = np_coords[np_values == cls]
            mu = self.means[idx]

            cov = np.cov(selected - mu, rowvar=0, ddof=1)

            self.covariance[cls] = cov
            self.covariance_inv[cls] = np.linalg.inv(cov)
            self.covariance_det[idx] = np.linalg.det(cov)

        # Constant part of the discrimination function
        self.constants = (np.log(self.probabilities) - 0.5 * np.log(self.covariance_det))

    def classify(self, samples):
        """ Classify X """
        columns = np.zeros((len(samples), self.n))

        for idx, cls in enumerate(self.classes):
            const = self.constants[idx]
            mu = self.means[idx]
            xs = samples - mu
            sigma_inv = self.covariance_inv[cls]
            columns[:, idx] = const - 0.5 * cm_math.double_product(xs, sigma_inv, xs)

        return self.classes[np.argmax(columns, axis=1)]

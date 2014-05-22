__author__ = 'jrx'

import collections

import numpy as np
import pandas as pd


def indicator_matrix(y):
    """ Transform a series of classes into a 0-1 classification matrix """
    # Take all the classes from y
    classes = sorted(set(y))

    Y = pd.DataFrame()

    # Transform classes into a classification matrix
    for cls in classes:
        Y[cls] = y == cls

    return Y.astype(int)


def classification_error_rate(classifier, X, y):
    """ Calculate classification error rate """
    yhat = classifier.classify(X)
    return (yhat != y).mean()


class LeastSquaresClassifier(object):
    """ Train linear least squares classifier """

    def __init__(self, X, y):
        self.classes = sorted(set(y))

        Y = indicator_matrix(y)
        Z = np.linalg.inv(np.dot(X.T, X))

        self.betahat = pd.DataFrame(
            np.dot(np.dot(Z, X.T), Y),
            columns=self.classes, index=X.columns
        )

    def classify(self, x):
        """ Classify x """
        Yhat = x.dot(self.betahat)
        yhat = Yhat.apply(lambda x: x.idxmax(), axis=1)
        return yhat


class LinearDiscriminantClassifier(object):
    """ Train linear discriminant classifier """

    def __init__(self, X, y):
        self.classes = sorted(set(y))

        K = len(self.classes)

        N, p = X.shape

        # Per-class probabilities
        self.probabilities = pd.Series(collections.Counter(y)) / y.size

        # Calculate means
        X2 = X.copy()
        X2['y'] = y
        self.means = X2.groupby('y').apply(lambda x: x.mean()).drop('y', 1)

        self.sigma = np.zeros([p, p])

        for idx, x in X2.iterrows():
            mu = self.means.loc[x.y]
            xv = x.drop('y') - mu
            self.sigma += np.outer(xv, xv)

        self.sigma /= (N - K)
        self.sigma_inv = np.linalg.inv(self.sigma)

        constants = {}

        for cls in self.classes:
            mu = self.means.loc[cls]
            constants[cls] = np.log(self.probabilities[cls]) - 0.5 * np.dot(mu, np.dot(self.sigma_inv, mu))

        self.constants = pd.Series(constants, index=self.classes)

        self.discrimination_matrix = np.dot(self.sigma_inv, self.means.values.T)

    def classify(self, x):
        """ Classify X """
        df = pd.DataFrame(np.dot(x, self.discrimination_matrix), columns=self.classes)
        return (df + self.constants).apply(lambda row: row.idxmax(), axis=1)

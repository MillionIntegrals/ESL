__author__ = 'jrx'

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


def indicator_classify(X, betahat):
    """ Return classification values """
    Yhat = X.dot(betahat)
    yhat = Yhat.apply(lambda x: x.idxmax(), axis=1)
    return yhat


def linear_regression(X, y):
    """ Calculate linear regression classifier """
    Y = indicator_matrix(y)
    Z = np.linalg.inv(np.dot(X.T, X))
    betahat = np.dot(np.dot(Z, X.T), Y)
    return betahat


def classification_error_rate(X, y, betahat):
    """ Calculate classification error rate """
    yhat = indicator_classify(X, betahat)
    return (yhat != y).mean()

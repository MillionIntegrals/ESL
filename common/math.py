__author__ = 'jrx'

import numpy as np


def normalize(m, normalizant=None):
    """ Normalize the data """
    if normalizant is None:
        normalizant = m

    return (m - np.mean(normalizant)) / np.std(normalizant)


def double_product(x, m, y):
    """ Calculate the x M x^t product for multiple vectors stored in the matrix form """
    return (x * np.dot(m, y.T).T).sum(1)
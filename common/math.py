__author__ = 'jrx'

import numpy as np


def normalize(m):
    """ Normalize the data """
    return (m - m.mean()) / m.std()


def double_product(x, m, y):
    """ Calculate the x M x^t product for multiple vectors stored in the matrix form """
    return (x * np.dot(m, y.T).T).sum(1)
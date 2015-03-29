__author__ = 'jrx'


def normalize(m):
    """ Normalize the data """
    return (m - m.mean()) / m.std()
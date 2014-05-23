__author__ = 'jrx'


class Classification(object):
    """ Base class for classification algorithms """

    def __init__(self):
        pass

    def classify(self, samples):
        """ Classify given input """
        raise NotImplementedError


class Regression(object):
    """ Base class for regression algorithms """

    def __init__(self):
        pass

    def calculate(self, samples):
        """ Return function values for given input """
        raise NotImplementedError

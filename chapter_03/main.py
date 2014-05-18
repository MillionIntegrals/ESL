#!/usr/bin/env python

import pandas as pd
import numpy as np

import chapter_03.data as data
import chapter_03.regression as regression


def print_title(title):
    print
    print title
    print "=" * len(title)


def training_data_correlations():
    """ Print the correlations in training data """
    print_title("Data correlations")
    prostate_data = data.read_csv_prostate()
    training_data = prostate_data[prostate_data.train == 'T'].drop('train', 1)
    print training_data.corr().to_string(float_format=lambda x: '%.3f' % x)


def training_data_regression():
    """ Run least squares regression on the training data """
    print_title("Least squares regression")
    prostate_data = data.read_csv_prostate()

    # Select corresponding columns
    yraw = prostate_data.lpsa
    train = prostate_data.train
    Xraw = prostate_data.drop(['lpsa', 'train'], 1)

    # Normalize data
    Xraw = (Xraw - Xraw.mean()) / np.sqrt(Xraw.var())

    # Insert intercept column
    Xraw.insert(0, 'intercept', 1.0)

    # Select training set
    y = yraw[train == 'T']
    X = Xraw[train == 'T']


    # Regresion
    model = regression.least_squares_regression(X.values, y.values)

    result = pd.DataFrame({'Coefficient': model['betahat'], 'Std. Error': model['errors']}, index=X.columns)
    result['Z Score'] = result['Coefficient'] / result['Std. Error']

    print result.to_string(float_format=lambda x: '%.3f' % x)

    ytest = yraw[train == 'F']
    Xtest = Xraw[train == 'F']

    test_residual = ytest - Xtest.dot(model['betahat'])
    test_rss = test_residual.dot(test_residual)

    N,p = Xtest.shape

    test_sigma2 = 1.0 / (N - p) * test_rss
    test_sigma = np.sqrt(test_sigma2)

    print
    print 'RSS = ', test_rss
    print 'sigma2=', test_sigma2
    print 'avg err = %.3f' % np.average(np.abs(test_residual))
    print "Test error:", test_sigma


if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 3"
    data.download_data()
    training_data_correlations()
    training_data_regression()

#!/usr/bin/env python

import pandas as pd
import numpy as np

import chapter_03.data as data
import chapter_03.regression as regression


def print_title(title):
    print
    print title
    print "=" * len(title)


def print_training_data():
    """ Just print the training data """
    prostate_data = data.read_csv_prostate()
    training_data = prostate_data[prostate_data.train == 'T'].drop('train', 1)
    print training_data.to_string()


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
    y = prostate_data.lpsa
    train = prostate_data.train
    X = prostate_data.drop(['lpsa', 'train'], 1)

    # Normalize data
    X = (X - X.mean()) / np.sqrt(X.var())

    # Select training set
    y = y[train == 'T']
    X = X[train == 'T']

    # Insert intercept column
    X.insert(0, 'intercept', 1.0)

    # Regresion
    betahat, errors = regression.least_squares_regression(X.values, y)

    result = pd.DataFrame({'Coefficient': betahat, 'Std. Error': errors}, index=X.columns)
    result['Z Score'] = result['Coefficient'] / result['Std. Error']

    print result.to_string(float_format=lambda x: '%.2f' % x)

if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 3"
    data.download_data()
    training_data_correlations()
    training_data_regression()

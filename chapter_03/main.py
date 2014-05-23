#!/usr/bin/env python

import pandas as pd
import numpy as np

import chapter_03.data as data
import chapter_03.regression as regression

from common.ui import print_title


def training_data_correlations():
    """ Print the correlations in training data """
    print_title("Data correlations")
    prostate_data = data.read_prostate_data()
    training_data = prostate_data[prostate_data.train == 'T'].drop('train', 1)
    print training_data.corr().to_string(float_format=lambda x: '%.3f' % x)


def training_data_regression():
    """ Run least squares regression on the training data """
    print_title("Least squares regression")
    prostate_data = data.read_prostate_data()

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
    least_squares = regression.LeastSquaresRegression(X, y)
    # betahat, errors = regression.least_squares_regression_with_std_errors(X.values, y.values)

    result = pd.DataFrame({
        'Coefficient': least_squares.betahat,
        'Std. Error': least_squares.std_errors
    }, index=X.columns)

    result['Z Score'] = result['Coefficient'] / result['Std. Error']

    print result.to_string(float_format=lambda x: '%.2f' % x)


def shrinkage_methods():
    """ Test various shrinkage methods """
    print_title("Shrinkage methods")
    prostate_data = data.read_prostate_data()

    # Select corresponding columns
    yraw = prostate_data.lpsa
    train = prostate_data.train
    x_raw = prostate_data.drop(['lpsa', 'train'], 1)

    # Normalize data
    x_raw = (x_raw - x_raw.mean()) / np.sqrt(x_raw.var())

    # Insert intercept column
    x_raw.insert(0, 'intercept', 1.0)

    # Select training set
    y = yraw[train == 'T']
    x = x_raw[train == 'T']

    # Select test set
    y_test = yraw[train == 'F']
    x_test = x_raw[train == 'F']

    #############################################################
    # Ordinary least squares
    least_squares = regression.LeastSquaresRegression(x, y)
    test_error, std_error = regression.test_error(least_squares, x_test, y_test)

    ls_series = pd.Series(least_squares.betahat, index=x.columns)
    ls_series.set_value('Test Error', test_error)
    ls_series.set_value('Std Error', std_error)

    #############################################################
    # Best subset selection
    best_subset_parameter = 2  # These parameters where chosen by the authors of the book

    subset_model = regression.BestSubsetSelection(x, y, best_subset_parameter)
    test_error, std_error = regression.test_error(subset_model, x_test, y_test)

    best_subset_series = pd.Series(subset_model.betahat, index=x.columns[subset_model.best_combination])
    best_subset_series.set_value('Test Error', test_error)
    best_subset_series.set_value('Std Error', std_error)

    #############################################################
    # Ridge regression
    ridge_lambda_parameter = 24.176431  # selected so that df(lambda) = 5.0

    ridge_model = regression.RidgeRegression(x, y, ridge_lambda_parameter)
    test_error, std_error = regression.test_error(ridge_model, x_test, y_test)

    ridge_series = pd.Series(ridge_model.betahat, index=x.columns)
    ridge_series.set_value('Test Error', test_error)
    ridge_series.set_value('Std Error', std_error)


    #############################################################
    # Print results
    result = pd.DataFrame({
        'LS': ls_series,
        'Best Subset': best_subset_series,
        'Ridge': ridge_series
    },
        index=ls_series.index,
        columns=['LS', 'Best Subset', 'Ridge']
    )

    print result.to_string(float_format=lambda x: '---' if np.isnan(x) else '%.3f' % x)


if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 3"
    training_data_correlations()
    training_data_regression()
    shrinkage_methods()

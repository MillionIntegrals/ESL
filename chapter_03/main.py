#!/usr/bin/env python

import pandas as pd

import chapter_03.data as data
import chapter_03.regression as regression
import common.formatters as formatters
import common.math as cm_math
import common.profiling as profiling

from common.ui import print_title


def training_data_correlations(prostate_data):
    """ Print the correlations in training data """
    print_title("Data correlations")
    training_data = prostate_data[prostate_data.train == 'T'].drop('train', axis=1)
    print training_data.corr().to_string(float_format=formatters.float_precision_formatter(3))


def training_data_regression(prostate_data):
    """ Run least squares regression on the training data """
    print_title("Least squares regression")

    # Select corresponding columns
    y_raw = prostate_data.lpsa
    x_raw = prostate_data.drop(['lpsa', 'train'], axis=1)

    train_indicator = prostate_data.train

    # Normalize data
    x_raw = cm_math.normalize(x_raw)

    # Insert intercept column
    x_raw.insert(0, 'intercept', 1.0)

    # Select training set
    y = y_raw[train_indicator == 'T']
    x = x_raw[train_indicator == 'T']

    # Regression
    least_squares = regression.LeastSquaresRegression(x, y)

    result = pd.DataFrame({
        'Coefficient': least_squares.betahat,
        'Std. Error': least_squares.std_errors
    }, index=x.columns)

    result['Z Score'] = result['Coefficient'] / result['Std. Error']

    print result.to_string(float_format=formatters.float_precision_formatter(2))


def shrinkage_methods(prostate_data):
    """ Test various shrinkage methods """
    print_title("Shrinkage methods")

    # Select corresponding columns
    y_raw = prostate_data.lpsa
    x_raw = prostate_data.drop(['lpsa', 'train'], axis=1)

    train_indicator = prostate_data.train

    # Normalize data
    x_raw = cm_math.normalize(x_raw)

    # Insert intercept column
    x_raw.insert(0, 'intercept', 1.0)

    # Select training set
    y = y_raw[train_indicator == 'T']
    x = x_raw[train_indicator == 'T']

    # Select test set
    y_test = y_raw[train_indicator == 'F']
    x_test = x_raw[train_indicator == 'F']

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

    print result.to_string(float_format=formatters.float_precision_formatter(3))


def main():
    """ Main function to run """
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 3"
    prostate_data = data.read_prostate_data()

    with profiling.measure_time('Training data correlations', precision=4):
        training_data_correlations(prostate_data)

    with profiling.measure_time('Training data regression', precision=4):
        training_data_regression(prostate_data)

    with profiling.measure_time('Shrinkage methods', precision=4):
        shrinkage_methods(prostate_data)


if __name__ == '__main__':
    main()

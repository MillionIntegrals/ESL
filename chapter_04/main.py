__author__ = 'jrx'

import pandas as pd

import chapter_04.data as data
import chapter_04.classification as classification

from common.ui import print_title


def vowel_classification():
    """ Run the vowel classification algorithms """
    print_title('Vowel classification')

    train_data = data.read_vowel_train()
    test_data = data.read_vowel_test()

    X = train_data.drop('y', 1)
    y = train_data.y

    Xtest = test_data.drop('y', 1)
    ytest = test_data.y

    # Perform normalization in the same way on training and testing set

    ############################################################
    # LINEAR REGRESSION
    X_linear_train = (X - X.mean()) / X.std()
    X_linear_train.insert(0, 'intercept', 1.0)

    X_linear_test = (Xtest - X.mean()) / X.std()
    X_linear_test.insert(0, 'intercept', 1.0)

    least_squares = classification.LeastSquaresClassifier(X_linear_train, y)

    train_error_rate = classification.classification_error_rate(least_squares, X_linear_train, y)
    test_error_rate = classification.classification_error_rate(least_squares, X_linear_test, ytest)

    linear_regression_series = pd.Series({
        'Training Error Rate': train_error_rate,
        'Test Error Rate': test_error_rate
    }, index=['Training Error Rate', 'Test Error Rate'])

    #############################################################
    # LINEAR DISCRIMINANT ANALYSIS
    lda = classification.LinearDiscriminantClassifier(X, y)

    train_error_rate = classification.classification_error_rate(lda, X, y)
    test_error_rate = classification.classification_error_rate(lda, Xtest, ytest)

    lda_series = pd.Series({
        'Training Error Rate': train_error_rate,
        'Test Error Rate': test_error_rate
    }, index=['Training Error Rate', 'Test Error Rate'])

    #############################################################
    # QUADRATIC DISCRIMINANT ANALYSIS
    qda = classification.QuadraticDiscriminantClassifier(X, y)

    train_error_rate = classification.classification_error_rate(qda, X, y)
    test_error_rate = classification.classification_error_rate(qda, Xtest, ytest)

    qda_series = pd.Series({
        'Training Error Rate': train_error_rate,
        'Test Error Rate': test_error_rate
    }, index=['Training Error Rate', 'Test Error Rate'])

    result = pd.DataFrame({
        'Linear regression': linear_regression_series,
        'LDA': lda_series,
        'QDA': qda_series,
    }, columns=['Linear regression', 'LDA', 'QDA'])

    print result.T.to_string(float_format=lambda x: '%.6f' % x)


if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 4"
    vowel_classification()

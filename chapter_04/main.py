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

    X_train_raw = train_data.drop('y', 1)

    # Perform normalization in the same way on training and testing set
    X = (X_train_raw - X_train_raw.mean()) / X_train_raw.std()
    X.insert(0, 'intercept', 1.0)

    y = train_data.y

    Xtest = (test_data.drop('y', 1) - X_train_raw.mean()) / X_train_raw.std()
    Xtest.insert(0, 'intercept', 1.0)

    ytest = test_data.y

    ############################################################
    # LINEAR REGRESSION
    betahat = pd.DataFrame(
        classification.linear_regression(X, y),
        columns=sorted(set(y)), index=X.columns
    )

    train_error_rate = classification.classification_error_rate(X, y, betahat)
    test_error_rate = classification.classification_error_rate(Xtest, ytest, betahat)

    linear_regression_series = pd.Series({
        'Training Error Rate': train_error_rate,
        'Test Error Rate': test_error_rate
    }, index=['Training Error Rate', 'Test Error Rate'])

    result = pd.DataFrame({
        'Linear regression': linear_regression_series
    })

    print result.T.to_string(float_format=lambda x: '%.2f' % x)


if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 4"
    vowel_classification()

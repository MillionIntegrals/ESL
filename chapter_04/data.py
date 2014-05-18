__author__ = 'jrx'

import pandas as pd

import common.data as cdata


VOWEL_TRAIN_DATA = 'vowel.train'
VOWEL_TEST_DATA = 'vowel.test'


def read_vowel_test():
    """ Read the vowel train file accompanying the book and return it as a dataframe """
    cdata.download_data_file(VOWEL_TEST_DATA)
    filename = cdata.data_path(VOWEL_TEST_DATA)
    return pd.read_csv(filename, index_col=0)


def read_vowel_train():
    """ Read the vowel train file accompanying the book and return it as a dataframe """
    cdata.download_data_file(VOWEL_TRAIN_DATA)
    filename = cdata.data_path(VOWEL_TRAIN_DATA)
    return pd.read_csv(filename, index_col=0)

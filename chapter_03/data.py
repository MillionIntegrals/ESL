__author__ = 'jrx'

import pandas as pd

import common.data as cdata


PROSTATE_DATA_NAME = 'prostate.data'


def read_prostate_data():
    """ Read the prostate.data file accompanying the book and return it as a dataframe """
    filename = cdata.download_data_file(PROSTATE_DATA_NAME)
    return pd.read_csv(filename, delimiter='\t', index_col=0)


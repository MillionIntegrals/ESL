__author__ = 'jrx'

import contextlib
import inspect
import os
import urllib2

import pandas as pd


PROSTATE_DATA_URL = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'

DATA_DIRECTORY_NAME = '../data'
PROSTATE_DATA_NAME = '../data/prostate.data'


def this_script_directory():
    """ Return directory this script is in """
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def download_data():
    """ Download data from the book website """
    directory = this_script_directory()
    data_directory = os.path.join(directory, DATA_DIRECTORY_NAME)
    data_filename = os.path.join(directory, PROSTATE_DATA_NAME)

    if os.path.exists(data_filename):
        return  # My job is done

    # If there is no data directory, create it
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Download the file and save
    with contextlib.closing(urllib2.urlopen(PROSTATE_DATA_URL)) as url_reader:
        prostate_data = url_reader.read()

    with open(data_filename, 'wt') as prostate_writer:
        prostate_writer.write(prostate_data)


def read_prostate_data():
    """ Read the prostate.data file accompanying the book and return it as a dataframe """
    directory = this_script_directory()
    filename = os.path.join(directory, PROSTATE_DATA_NAME)
    return pd.read_csv(filename, delimiter='\t', index_col=0)


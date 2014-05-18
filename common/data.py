__author__ = 'jrx'

import contextlib
import inspect
import os
import urllib2


DATA_URL = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

DATA_DIRECTORY_NAME = '../data'


def this_script_directory():
    """ Return directory this script is in """
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def download_data_file(fname):
    """ Download data from the book website """
    directory = this_script_directory()
    data_directory = os.path.join(directory, DATA_DIRECTORY_NAME)
    data_filename = os.path.join(data_directory, fname)

    if os.path.exists(data_filename):
        return  # My job is done

    # If there is no data directory, create it
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)


    file_url = DATA_URL + fname

    # Download the file and save
    with contextlib.closing(urllib2.urlopen(file_url)) as url_reader:
        prostate_data = url_reader.read()

    with open(data_filename, 'wt') as prostate_writer:
        prostate_writer.write(prostate_data)


def data_path(fname):
    """ Return absolute path to the data file """
    return os.path.join(this_script_directory(), DATA_DIRECTORY_NAME, fname)

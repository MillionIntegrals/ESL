__author__ = 'jrx'

import contextlib
import inspect
import os
import urllib2
import urlparse


import common.constants as constants


def this_script_directory():
    """ Return directory this script is in """
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def download_data_file(filename):
    """ Download data from the book website """
    data_directory = os.path.join(this_script_directory(), constants.DATA_DIRECTORY_NAME)
    data_filename = data_path(filename)

    if os.path.exists(data_filename):
        return data_filename  # My job is done

    # If there is no data directory, create it
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    file_url = urlparse.urljoin(constants.DATA_BASE_URL, filename)

    # Download the file and save
    with contextlib.closing(urllib2.urlopen(file_url)) as url_reader:
        prostate_data = url_reader.read()

    with open(data_filename, 'wt') as file_writer:
        file_writer.write(prostate_data)

    return data_filename


def data_path(fname):
    """ Return absolute path to the data file """
    return os.path.join(this_script_directory(), constants.DATA_DIRECTORY_NAME, fname)

__author__ = 'jrx'

import contextlib
import inspect
import os
import urllib2

import pandas as pd


PROSTATE_DATA_URL = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/prostate.data'

DATA_DIRECTORY_NAME = 'data'
PROSTATE_DATA_NAME = 'data/prostate.data'
PROSTATE_CSV_NAME = 'data/prostate.csv'


def this_script_directory():
    """ Return directory this script is in """
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def download_data():
    """ Download data from the book website """
    directory = this_script_directory()
    data_directory = os.path.join(directory, DATA_DIRECTORY_NAME)
    data_filename = os.path.join(directory, PROSTATE_DATA_NAME)
    csv_filename = os.path.join(directory, PROSTATE_CSV_NAME)

    if os.path.exists(csv_filename):
        return  # My job is done

    # If there is no data directory, create it
    if not os.path.exists(data_directory):
        os.mkdir(data_directory)

    # Download the file and save
    with contextlib.closing(urllib2.urlopen(PROSTATE_DATA_URL)) as url_reader:
        prostate_data = url_reader.read()

    with open(data_filename, 'wt') as prostate_writer:
        prostate_writer.write(prostate_data)

    # Convert to a csv file
    df = read_raw_prostate()
    df.to_csv(csv_filename)


def read_raw_prostate():
    """ Read the prostate.data file accompanying the book and return it as a dataframe """
    directory = this_script_directory()
    filename = os.path.join(directory, PROSTATE_DATA_NAME)
    content = open(filename).read()

    processed = [line.split("\t") for line in content.split("\n") if line != '']

    headers = processed[0][1:]

    output = []

    for line in processed[1:]:
        sub_output = []

        for word in line[1:]:
            word = word.strip()

            if word[0].isalpha():
                # If starts with a letter, assume string
                sub_output.append(word)
            else:
                if '.' in word:
                    # Floating point
                    sub_output.append(float(word))
                else:
                    sub_output.append(int(word))

        output.append(sub_output)

    return pd.DataFrame(output, columns=headers)


def read_csv_prostate():
    """ Read the prostate data from an alternative csv file """
    directory = this_script_directory()
    filename = os.path.join(directory, PROSTATE_CSV_NAME)
    return pd.read_csv(filename, index_col=0)

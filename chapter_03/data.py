__author__ = 'jrx'

import inspect
import os

import pandas as pd


PROSTATE_DATA_NAME = 'data/prostate.data'
PROSTATE_CSV_NAME = 'data/prostate.csv'


def this_script_directory():
    """ Return directory this script is in """
    return os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


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

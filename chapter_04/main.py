__author__ = 'jrx'

import chapter_04.data as data
from common.ui import print_title


def vowel_classification():
    """ Run the vowel classification algorithms """
    print_title('Vowel classification')

    train_data = data.read_vowel_train()


if __name__ == '__main__':
    print ">>>>>>>>>>>>>>>>>>>>>>>>>>"
    print "Running code for chapter 4"
    vowel_classification()

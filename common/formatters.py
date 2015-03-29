__author__ = 'jerry'

import numpy as np


def float_precision_formatter(precision=2, nan_str='---'):
    """ Return the function formatting floats according to the precision """
    format_str = '{:.%df}' % precision

    def inner_fn(x):
        if np.isnan(x):
            return nan_str
        else:
            return format_str.format(x)

    return inner_fn

__author__ = 'jerry'

import cStringIO
import contextlib
import logging
import pstats
import time


@contextlib.contextmanager
def measure_time(label=None, logger=None, loglevel=logging.INFO, precision=2):
    """ Measure how much time has passed between beginning and end of a block """
    start = time.clock()

    try:
        yield
    finally:
        end = time.clock()

        message = ("Elapsed time: %.{}fs".format(precision)) % (end - start)

        if logger is None:
            if label is None:
                print message
            else:
                print "[%s]" % label, message
        else:
            logger.log(loglevel, '[%s] %s', label, message)


@contextlib.contextmanager
def profiling(outfile=None, print_stats=False):
    """ Run a python profiler on a given block of code and present the results """
    import cProfile

    profiler = cProfile.Profile()
    try:
        profiler.enable()
        yield
    finally:
        profiler.disable()

        s = cStringIO.StringIO()
        profiler_stats = pstats.Stats(profiler, stream=s).sort_stats('tottime')

        if print_stats:
            profiler_stats.print_stats()
            print s.getvalue()

        if outfile is not None:
            profiler_stats.dump_stats(outfile)


@contextlib.contextmanager
def line_profiling(*functions):
    import line_profiler
    profiler = line_profiler.LineProfiler(*functions)

    try:
        with profiler:
            yield
    finally:
        profiler.print_stats()

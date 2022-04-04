from functools import wraps
from time import time

"""
Utility package to measure the time of a function.
@Author: Neville Loh
"""


def measure_elapsed_time(f):
    """
    Decorator to print the execution time of a function
    Parameters
    ----------
    f : function
        Function to be decorated
    Returns
    -------
        function wrapper
    """

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r elapsed time: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap

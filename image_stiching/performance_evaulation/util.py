from functools import wraps
from time import time


def timeit(f):
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

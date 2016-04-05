"""
Tools for testing this library

the name begins with an underscore so that the functions aren't run as tests
"""

import time
import contextlib

import joblib
import numpy as np

import nose.tools
from nose.plugins.attrib import attr

eq_ = nose.tools.eq_  # for compatibility
# TODO rename to assert_equal and all other functions similarly
equal = nose.tools.eq_
raises = nose.tools.raises


@contextlib.contextmanager
def assert_time(min_time, max_time):
    """Raise an exception if the time to process the block is within the
    expected range of (min_time, max_time), measured in ms.

    Example:
    with assert_time(1, 10):
        do_something_for_5ms()
    """
    start_time = time.time()
    yield
    end_time = time.time()
    actual_time = end_time - start_time
    assert min_time <= actual_time <= max_time, dict(min=min_time,
                                                     max=max_time,
                                                     actual=actual_time)


def hash_equal(thing1, thing2, msg=None):
    equal(joblib.hash(thing1, coerce_mmap=True),
          joblib.hash(thing2, coerce_mmap=True),
          msg)

hash_eq = hash_equal  # for compatibility

numpy_equal = np.testing.assert_array_equal

numpy_almost_equal = np.testing.assert_array_almost_equal


def numpy_allclose(a, b, msg=None, **kwargs):
    """
    can be better than numpy_almost_equal because you have more
    fine-grained control over absolute and relative tolerance
    """
    assert np.allclose(a, b, **kwargs), dict(
        a=a,
        b=b,
        msg=msg,
    )


def assertion_error_complement(fn):
    """
    returns a function raises an assertion error if the inner funtion does not

    NOTE: resulting functions are not serializable
    TODO reimplement with class to make them serializable
    """
    def inner(*args, **kwargs):
        try:
            fn(*args, **kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError(dict(
                fn=fn,
                args=args,
                kwargs=kwargs,
            ))
    return inner

not_equal = assertion_error_complement(equal)
numpy_not_almost_equal = assertion_error_complement(numpy_almost_equal)
numpy_not_equal = assertion_error_complement(numpy_equal)
numpy_not_allclose = assertion_error_complement(numpy_allclose)


# decorator to mark slow tests
# ---
# this allows you to run only non-slow tests with "nosetests -a '!slow'"
slow = attr("slow")

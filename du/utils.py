"""
Core utility library handling configuration, logging, imports, and other
useful tools that the rest of the library benefits from.
"""

from __future__ import print_function
import os
import logging
import logging.config
import functools
import warnings
import time
import contextlib

import yaml
import joblib
import numpy as np


# ################################# settings #################################

LIB_PATH = os.path.abspath(os.path.dirname(__file__))
DEFAULT_SETTINGS_PATH = os.path.join(LIB_PATH, "default_settings.yml")
DEFAULT_LOG_SETTINGS_PATH = os.path.join(LIB_PATH, "default_logging.cfg")

with open(DEFAULT_SETTINGS_PATH) as f:
    s_raw = f.read()
    s = os.path.expandvars(s_raw)
    config = yaml.load(s)

# ####################### importing into this package #######################

if config["toolz"]["use_cytoolz"]:
    import cytoolz as toolz
    from cytoolz.functoolz import (compose, identity)
else:
    import toolz
    from toolz.functoolz import (compose, identity)

from functools import partial

# ################################# logging #################################

logging.config.fileConfig(DEFAULT_LOG_SETTINGS_PATH)

DEFAULT_LOGGER = logging.getLogger()
SIMPLE_LOGGER = logging.getLogger("simple")

debug = DEFAULT_LOGGER.debug
info = DEFAULT_LOGGER.info
warning = DEFAULT_LOGGER.warning
error = DEFAULT_LOGGER.error
critical = DEFAULT_LOGGER.critical
exception = DEFAULT_LOGGER.exception

simple_debug = SIMPLE_LOGGER.debug
simple_info = SIMPLE_LOGGER.info
simple_warning = SIMPLE_LOGGER.warning
simple_error = SIMPLE_LOGGER.error
simple_critical = SIMPLE_LOGGER.critical
simple_exception = SIMPLE_LOGGER.exception


def noop_fn(*args, **kwargs):
    pass

log_level_to_log_fn = {
    "debug": simple_debug,
    "info": simple_info,
    "warning": simple_warning,
    "error": simple_error,
    "critical": simple_critical,
    "none": noop_fn,
    None: noop_fn
}

# a 1-time warning that can be disabled -- different from logging.warning
warn_once = warnings.warn

# ################################ decorators ################################


def deprecated(func):
    """ warns if decorated function is used
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warn_once("Deprecated: {}".format(func))
        return func(*args, **kwargs)
    return wrapped


def untested(func):
    """ warns if decorated function is used
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        warn_once("Untested: {}".format(func))
        return func(*args, **kwargs)
    return wrapped


def todo(msg):
    """ alerts about a todo if decorated function is used
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            warn_once("TODO: {}; {}".format(func, msg))
            return func(*args, **kwargs)
        return wrapped
    return decorator


def trace(func):
    """
    Logs input, output, and time takes of a decorated function.
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        simple_debug('Calling %s', func)
        simple_debug('INPUT (args)  : %s', args)
        simple_debug('INPUT (kwargs): %s', kwargs)
        start_time = time.time()
        try:
            output = func(*args, **kwargs)
        finally:
            simple_debug('Returning %s', func)
            simple_debug('Took: %lf secs', time.time() - start_time)
        simple_debug('OUTPUT', output)
        return output
    return wrapped


def memoize(func):
    """stores inputs and outputs of a function
    """
    cache = func.cache = {}

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        key = joblib.hash((args, kwargs), coerce_mmap=True)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapped


def ignore_output(func):
    """Make function return None instead of its output
    useful for functions with side-effects (eg. memoization)
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        func(*args, **kwargs)
    return wrapped


# ################################## cache ##################################

PERSISTENT_CACHE = joblib.Memory(cachedir=config["cache"]["dir"],
                                 verbose=config["cache"]["verbose"],
                                 mmap_mode=config["cache"]["mmap_mode"])

# NOTE: this is a function decorator
persistent_cache = PERSISTENT_CACHE.cache


# ################################ attr dict ################################


class AttrDict(dict):

    """
    dict subclass that allows indexing keys using dot-notation

    example:
    d.AttrDict(foo=3, bar=2).foo
    d.AttrDict({"choo": 2}).choo

    https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute-in-python
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# ############################# rotations utils #############################


def rotations_to_radians(rotations):
    """
    converts radians to rotations
    """
    return np.pi * 2 * rotations

# ############################# exception handler ###########################


@contextlib.contextmanager
def finally_callback(fn):
    """
    A context manager that calls the given function in a finally block.
    If an exception is raised in the finally function and the context manager,
    the exception in the context manager is thrown.
    """
    e1 = None
    try:
        yield
    except Exception as e1:
        pass
    finally:
        try:
            fn()
        finally:
            if e1 is not None:
                raise e1

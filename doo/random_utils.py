import contextlib
import random
import string
import struct
import numpy as np
import joblib


@contextlib.contextmanager
def seed_random(seed=None):
    """
    seeds RNG for both random and numpy.random
    """
    if seed is None:
        yield
    elif isinstance(seed, int):
        # save state
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        # set state
        random.seed(seed)
        np.random.seed(seed)
        try:
            yield
        finally:
            # reset state
            random.setstate(random_state)
            np.random.set_state(np_random_state)
    else:
        raise TypeError("Improper random seed type")


def float_to_hex(f):
    return hex(struct.unpack('<I', struct.pack('<f', f))[0])


def str_to_random_state(string, salt=0):
    return hex_to_random_state(joblib.hash(string), salt=salt)


def hex_to_random_state(hex_str, salt=0):
    return long_to_random_state(int(hex_str, 16), salt=salt)


def int_to_random_state(i, salt=0):
    return np.random.RandomState((i + salt) % (1 << 32 - 1))


def long_to_random_state(l, salt=0):
    return int_to_random_state(l, salt=salt)


def float_to_random_state(f, salt=0):
    return hex_to_random_state(float_to_hex(f), salt=salt)


def anything_to_random_state(thing, salt=0):
    return hex_to_random_state(joblib.hash(thing, coerce_mmap=True), salt=salt)


def rand_alnum(length):
    # http://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters-and-digits-in-python/23728630#23728630
    return ''.join(random.SystemRandom().choice(
        string.uppercase + string.digits + string.lowercase
    ) for _ in xrange(length))

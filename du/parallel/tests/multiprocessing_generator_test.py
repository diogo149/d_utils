import time

import numpy as np

from du.parallel.multiprocessing_generator import mp_generator
import du._test_utils
from du._test_utils import assert_time


FOO_COUNT = 10
FOO_DELAY = 0.5


def foo(index, delay=FOO_DELAY):
    np.random.seed(index)
    for _ in range(FOO_COUNT):
        time.sleep(delay)
        z = np.random.randn(100, 100)
        yield z


@du._test_utils.slow
def test_mp_generator():
    N_JOBS = 3
    sums = set()
    with assert_time(FOO_COUNT * FOO_DELAY, FOO_COUNT * FOO_DELAY * 1.2):
        with mp_generator(foo, n_jobs=3) as g:
            for x in g:
                # need to cast to array because it's a memmap-ed array
                sums.add(np.array(x).sum())

    iters = (foo(i, 0) for i in range(N_JOBS))
    sums_serial = set(z.sum() for i in iters for z in i)
    assert sums == sums_serial, dict(
        sums=sums,
        sums_serial=sums_serial
    )


def foo2(index):
    for i in range(FOO_COUNT):
        yield i
    raise Exception("this exception should happen")


def test_mp_generator_exception():
    try:
        with mp_generator(foo2, n_jobs=3) as g:
            for _ in g:
                pass
    except Exception as e:
        assert e.message == "this exception should happen", e.message
    else:
        assert False, "Exception should be thrown"

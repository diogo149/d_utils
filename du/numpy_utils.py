import tempfile
import contextlib
import numpy as np

from . import utils


def constant_value_array(value, shape, dtype=float):
    """
    returns a constant valued array with a given shape and dtype
    """
    # doing it this way to make sure the resulting array has the proper dtype
    res = np.zeros(shape, dtype=dtype)
    res[:] = value
    return res


def expand_dim(arr, target, axis=0, fill_value=0.0):
    """Expands an array along a given dimension so that the size along
    the dimension is equal to a target size
    """
    shape = list(arr.shape)
    original_size = shape[axis]
    assert original_size <= target
    if original_size == target:
        return arr
    shape[axis] = target
    to_fill = constant_value_array(fill_value, shape, arr.dtype)
    indices = [slice(0, original_size,) if i == axis else slice(None)
               for i in range(len(shape))]
    to_fill[indices] = arr
    return to_fill


def memmap_concatenate(filename, arr_list, axis=0):
    """
    concatenates a list of arrays into a single memmap-ed file (for when a
    copy of the arrays wouldn't fit in memory)
    """
    f = arr_list[0]

    # verify shapes are fine
    first_shape = f.shape
    shapes = [arr.shape for arr in arr_list]
    for arr in arr_list:
        shape = arr.shape
        assert f.dtype == arr.dtype
        assert len(first_shape) == len(shape)
        assert first_shape[:axis] == shape[:axis]
        assert first_shape[axis + 1:] == shape[axis + 1:]

    dim_total = sum(shape[axis] for shape in shapes)
    final_shape = list(first_shape)
    final_shape[axis] = dim_total
    final_shape = tuple(final_shape)

    tmp = np.memmap(filename,
                    dtype=f.dtype,
                    mode='w+',
                    shape=final_shape)

    start_idx = 0
    slices = [slice(None) for _ in first_shape]
    for arr in arr_list:
        end_idx = start_idx + arr.shape[axis]
        slices[axis] = slice(start_idx, end_idx)
        tmp[slices] = arr[:]
        start_idx = end_idx

    return tmp


@contextlib.contextmanager
def memmap_concatenate_cm(arr_list, axis=0):
    """
    concatenates a list of arrays into a single memmap-ed array (for when a
    copy of the arrays wouldn't fit in memory)
    """
    with tempfile.TemporaryFile() as f:
        mmap = memmap_concatenate(f, arr_list, axis)
        yield mmap


def to_ranking(arr):
    """
    http://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
    """
    order = arr.argsort()
    ranks = order.argsort()
    return ranks


def summarize_numerical(title, nums, log_fn=utils.simple_info):
    """
    logs and summarizes a sequence of numbers

    TODO return a map of the summary instead?
    """
    arr = np.array(nums)
    nans = np.isnan(arr)
    nan_count = nans.sum()
    if nan_count > 0:
        log_fn("%s nans: %d", title, nan_count)
        arr_no_nan = arr[~nans]
    else:
        arr_no_nan = arr
    log_fn("%s mean: %f", title, np.mean(arr_no_nan))
    log_fn("%s std: %f", title, np.std(arr_no_nan))
    for p in [0, 10, 25, 50, 75, 90, 100]:
        log_fn("%s p%d: %f", title, p, np.percentile(arr_no_nan, p))


@contextlib.contextmanager
def printoptions(**kwargs):
    original = np.get_printoptions()
    try:
        np.set_printoptions(**kwargs)
        yield
    finally:
        np.set_printoptions(**original)


def _combine_stats(n_1, mu_1, sigma2_1, n_2, mu_2, sigma2_2):
    """
    based on:
    https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-mean
    http://stats.stackexchange.com/questions/55999/is-it-possible-to-find-the-combined-standard-deviation
    """
    new_n = n_1 + n_2
    new_mu = (n_1 * mu_1 + n_2 * mu_2) / new_n
    term1 = n_1 * (sigma2_1 + mu_1 ** 2)
    term2 = n_2 * (sigma2_2 + mu_2 ** 2)
    new_sigma2 = (term1 + term2) / new_n - new_mu ** 2
    return new_n, new_mu, new_sigma2


def generator_stats(gen, axis=None):
    """
    takes in a generator of arrays and returns the count, mean, and variance
    across the specified axis
    """
    n = 0
    mu = 0
    sigma2 = 0
    for x in gen:
        mu_sample = x.mean(axis=axis)
        sigma2_sample = x.var(axis=axis)
        n_sample = x.size // mu_sample.size
        n, mu, sigma2 = _combine_stats(n,
                                       mu,
                                       sigma2,
                                       n_sample,
                                       mu_sample,
                                       sigma2_sample)
    return n, mu, sigma2


def generator_std(gen, axis=None):
    """
    takes in a generator of arrays and returns the std across the specified
    axis
    """
    return np.sqrt(generator_stats(gen, axis)[2])

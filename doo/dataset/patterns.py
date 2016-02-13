"""
useful combinations of DatasetDSL operators meant for use with
DatasetDSL.apply

could also be seen as a testing ground for future DatasetDSL operators
"""

import itertools

from .. import random_utils
from .. import parallel
from .. import utils

from . import dsl


def joblib_cache(ds,
                 dirname,
                 version,
                 numpy_keys=None,
                 mmap_mode=None,
                 eager=True):
    """
    chunks data together, converting values in `keys` into numpy arrays,
    (for more efficient storage) then dechunks them afterwards

    dirname:
    directory where to store the cache

    keys:
    which keys should be converted into numpy arrays
    """
    if numpy_keys is None:
        numpy_keys = []

    return ds.numpy_chunk(
        keys=numpy_keys,
    ).cache(
        dirname=dirname,
        version=version,
        backend="joblib",
        eager=eager,
        mmap_mode=mmap_mode
    ).dechunk(
    )


def parallel_copies(ds,
                    n_copies,
                    dirname=None,
                    try_shm=True,
                    mmap_mode=None):
    """
    runs dataset on other process, reading it in to the main process on a
    separate thread

    if dirname is given, also serializes the dataset with joblib before
    sending to other processes
    """
    if n_copies == -1:
        return ds
    if dirname is not None:
        ds = ds.to_joblib_serialized(
            directory=dirname,
            mmap_mode=None,
        )
    return ds.to_other_process(
        n_jobs=n_copies,
        try_shm=try_shm,
        mmap_mode=mmap_mode,
    ).to_threaded_reader(
    )


def _does_count_match_process_num(count):
    process_idx = parallel.multiprocessing_generator.PROCESS_IDXS[-1]
    total_processes = parallel.multiprocessing_generator.PROCESS_NUMS[-1]
    return (count % total_processes) == process_idx


def papply(ds,
           fn,
           args=None,
           kwargs=None,
           n_jobs=1,
           try_shm=True,
           mmap_mode=None):
    """
    applies the transformations of a function from DatasetDSL to DatasetDSL in
    parallel

    WARNING: will reorder the items
    """
    random_key = random_utils.rand_alnum(15)
    return ds.zip_assoc(
        out=random_key,
        iterator=itertools.count(),
    ).filter(
        key=random_key,
        fn=_does_count_match_process_num,
    ).dissoc(
        key=[random_key],
    ).apply(
        fn=fn,
        args=args,
        kwargs=kwargs,
    ).apply(
        fn=parallel_copies,
        kwargs=dict(
            n_copies=n_jobs,
            try_shm=try_shm,
            mmap_mode=mmap_mode,
        )
    )


def _fn_to_subdataset_apply_fn(fn, args, kwargs, in_datamap):
    ds = dsl.from_list(
        [in_datamap]
    ).apply(
        fn=fn,
        args=args,
        kwargs=kwargs,
    )
    with ds as g:
        for data in g:
            yield data


def subdataset_apply(ds, fn, key=None, out=None, args=None, kwargs=None):
    """
    treats each element as its own dataset for the duration of the given
    function

    useful when you want to apply transformations that affect multiple
    elements
    """
    return ds.mapcat(
        fn=utils.partial(_fn_to_subdataset_apply_fn, fn, args, kwargs),
        key=key,
        out=out,
    )

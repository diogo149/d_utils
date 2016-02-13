import os
import tempfile
import functools

import joblib

from . import utils


def int_hash(obj, coerce_mmap=True):
    """
    return hash of an object as an int
    """
    return int(joblib.hash(obj, coerce_mmap=coerce_mmap), 16)


def dump_tempfile(value, compress=0, cache_size=100, dir=None):
    """Dumps value to (a) temporary file(s) and returns the list of filenames
    where the data is stored, with the first filename being the main one.
    """
    temp_file = tempfile.mktemp(dir=dir)
    res = joblib.dump(value,
                      temp_file,
                      compress=compress,
                      cache_size=cache_size)
    assert temp_file == res[0]
    return res


def dump_exists(dirname):
    """
    whether or not dump_dir has dumped to the given dirname
    """
    filename = os.path.join(dirname, "main.pkl")
    return os.path.exists(filename)


def dump_dir(value,
             dirname,
             compress=0,
             cache_size=100,
             mkdir=True,
             overwrite=True):
    """
    dumps in a directory instead of a specific file to keep the filesystem
    cleaner
    """
    # check if dirname exists
    if not os.path.isdir(dirname):
        if mkdir:
            os.mkdir(dirname)
        else:
            raise Exception("Dir not found: %s" % dirname)

    filename = os.path.join(dirname, "main.pkl")
    # check if filename exists
    if os.path.exists(filename) and not overwrite:
        raise Exception("File already exists: %s" % filename)

    return joblib.dump(value, filename)


def load_dir(dirname, mmap_mode=None):
    """
    load from dirname dumped with dump_dir
    """
    filename = os.path.join(dirname, "main.pkl")
    return joblib.load(filename, mmap_mode=mmap_mode)


def eager_cache(dirname, mmap_mode=None):
    """
    similar to utils.persistent_cache, but caches eagerly (when the script
    is read, rather than when the function is called)

    do NOT use in library code

    TODO keep in memory flag to save the result in memory
    """
    def decorator(func):
        # TODO more sophisticated checking if cache should be written to
        # eg. copy joblib.Memory
        if not os.path.exists(dirname):
            res = func()
            dump_dir(res, dirname)
            del res

        # has to be no arg function to be cached properly
        @functools.wraps(func)
        def inner():
            return load_dir(dirname, mmap_mode)

        return inner
    return decorator


# copied on 20150103 from:
# https://github.com/joblib/joblib/blob/master/joblib/numpy_pickle.py
# NOTE: only the try-except read is added
# needs to be defined in the top level so that it can be serialized
def _monkey_patched_read(self, unpickler):
    "Reconstruct the array"
    filename = os.path.join(unpickler._dirname, self.filename)
    # Load the array from the disk
    np_ver = [int(x) for x in unpickler.np.__version__.split('.', 2)[:2]]
    if np_ver >= [1, 3]:
        try:
            array = unpickler.np.load(filename,
                                      mmap_mode=unpickler.mmap_mode)
        except ValueError as e:
            if e.message == ("Array can't be memory-mapped: "
                             "Python objects in dtype."):
                utils.warn_once(
                    "Array with Python objects can't be memory-mapped.")
                array = unpickler.np.load(filename, mmap_mode=None)
            else:
                raise e
    else:
        # Numpy does not have mmap_mode before 1.3
        array = unpickler.np.load(filename)
    # Reconstruct subclasses. This does not work with old
    # versions of numpy
    if (hasattr(array, '__array_prepare__')
            and not self.subclass in (unpickler.np.ndarray,
                                      unpickler.np.memmap)):
        # We need to reconstruct another subclass
        new_array = unpickler.np.core.multiarray._reconstruct(
            self.subclass, (0,), 'b')
        new_array.__array_prepare__(array)
        array = new_array
    return array

NDArrayWrapper_read = joblib.numpy_pickle.NDArrayWrapper.read
# TODO delete
# for backwards compatibility when using a monkey patched class
MonkeyPatchedNDArrayWrapper = joblib.numpy_pickle.NDArrayWrapper


def monkey_patch_flexible_mmap_mode():
    """
    Monkey patches joblib loading so that a numpy array of objects with a
    non-None mmap mode still works.

    WARNING: this means that the object array is not memory mapped and will be
    loaded into memory.

    This is more than just being lazy, because when pickling a complex nested
    datastructure with some object arrays and some non-object arrays and you
    want the benefit of mmap-ing the non-object arrays
    """
    joblib.numpy_pickle.NDArrayWrapper.read = _monkey_patched_read


def unmonkey_patch_flexible_mmap_mode():
    joblib.numpy_pickle.NDArrayWrapper.read = NDArrayWrapper_read


class HashableDict(dict):

    def __hash__(self):
        return int_hash(self)


class HashableSet(set):

    def __hash__(self):
        return int_hash(self)


class HashableList(list):

    def __hash__(self):
        return int_hash(self)


def to_hashable(v):
    """
    return a hashable version of the input data

    NOTE: this may be incomplete
    """
    if isinstance(v, list):
        return HashableList(v)
    elif isinstance(v, dict):
        return HashableDict(v)
    elif isinstance(v, set):
        return HashableSet(v)
    else:
        return v

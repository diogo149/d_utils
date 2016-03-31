import collections
import itertools
import copy
import time


import six
import numpy as np
import joblib

from .. import utils
from ..utils import toolz
from .. import timer_utils

from . import base
from . import constructors
from . import higher_order


# ################################ transforms ################################


def _key_values(in_datamap, key):
    # handling polymorphism in key values
    if key is None:
        in_vals = [in_datamap]
    elif isinstance(key, six.string_types):
        in_vals = [in_datamap[key]]
    elif isinstance(key, (list, tuple)):
        in_vals = [in_datamap[k] for k in key]
    else:
        raise ValueError("Can't handle key: %s" % key)

    return in_vals


def _make_keys_list(key):
    if key is None:
        raise ValueError("Can't handle None key")
    elif isinstance(key, six.string_types):
        return [key]
    elif isinstance(key, list):
        return key
    elif isinstance(key, tuple):
        return list(key)
    else:
        raise ValueError("Can't handle key: %s" % key)


def transform_mapcat(in_gen, fn, key=None, out=None, args=None, kwargs=None):
    assert args is None or isinstance(args, (list, tuple))
    assert kwargs is None or isinstance(kwargs, dict)
    assert key is None or isinstance(key, (six.string_types, list, tuple))
    assert out is None or isinstance(out, (six.string_types, list, tuple))

    args = list(args or [])
    kwargs = kwargs or {}
    for in_datamap in in_gen:
        in_vals = _key_values(in_datamap, key)
        out_vals = fn(*(in_vals + args), **kwargs)
        for out_val in out_vals:
            if out is None:
                out_datamap = out_val
            elif isinstance(out, six.string_types):
                out_datamap = toolz.assoc(in_datamap, out, out_val)
            elif isinstance(out, (list, tuple)):
                assert len(out) == len(out_val)
                # using assoc to make a copy
                out_datamap = toolz.assoc(in_datamap, out[0], out_val[0])
                for k, v in zip(out[1:], out_val[1:]):
                    out_datamap[k] = v
            else:
                raise ValueError("Can't handle out: %s" % out)

            yield out_datamap


def transform_mapcat_key(in_gen, fn, key, *args, **kwargs):
    # set out to key
    return transform_mapcat(in_gen, fn, key, key, *args, **kwargs)


def _map_fn_to_mapcat_fn(fn, *args, **kwargs):
    """
    takes in a function for mapping, and arguments to apply, and returns
    a result for a mapcat

    NOTE:
    - this is a separate function for serializability
    """
    res = fn(*args, **kwargs)
    return [res]


def transform_map(in_gen, fn, *args, **kwargs):
    mapcat_fn = utils.partial(_map_fn_to_mapcat_fn, fn)
    return transform_mapcat(in_gen, mapcat_fn, *args, **kwargs)


def transform_map_key(in_gen, fn, key, *args, **kwargs):
    # set out to key
    return transform_map(in_gen, fn, key, key, *args, **kwargs)


def transform_filter(in_gen, fn, key=None, args=None, kwargs=None):
    """
    NOTE: partly copy-pasted from transform_mapcat
    """
    assert args is None or isinstance(args, (list, tuple))
    assert kwargs is None or isinstance(kwargs, dict)
    assert key is None or isinstance(key, (six.string_types, list, tuple))

    args = list(args or [])
    kwargs = kwargs or {}
    for in_datamap in in_gen:
        in_vals = _key_values(in_datamap, key)
        out_vals = fn(*(in_vals + args), **kwargs)
        if out_vals:
            yield in_datamap


def transform_remove(in_gen, fn, *args, **kwargs):
    return transform_filter(in_gen,
                            toolz.complement(fn),
                            *args,
                            **kwargs)


def _fn_to_pmapcat_fn(in_datamap, fn, key, out, args, kwargs):
    in_vals = _key_values(in_datamap, key)
    out_vals = fn(*(in_vals + args), **kwargs)

    res = []
    for out_val in out_vals:
        if out is None:
            assert isinstance(out_val, dict)
            out_datamap = out_val
        elif isinstance(out, six.string_types):
            out_datamap = toolz.assoc(in_datamap, out, out_val)
        elif isinstance(out, (list, tuple)):
            # using assoc to make a copy
            out_datamap = toolz.assoc(in_datamap, out[0], out_val[0])
            for k, v in zip(out[1:], out_val[1:]):
                out_datamap[k] = v
        else:
            raise ValueError("Can't handle out: %s" % out)

        res.append(out_datamap)

    return res


def transform_pmapcat(in_gen,
                      fn,
                      key=None,
                      out=None,
                      args=None,
                      kwargs=None,
                      backend="joblib",
                      joblib_n_jobs=-1,
                      pool=None,
                      pool_fn="imap_unordered",
                      pool_chunksize=1):
    assert kwargs is None or isinstance(kwargs, dict)
    assert key is None or isinstance(key, (six.string_types, list, tuple))
    assert out is None or isinstance(out, (six.string_types, list, tuple))

    args = list(args or [])
    kwargs = kwargs or {}

    pmapcat_fn = utils.partial(_fn_to_pmapcat_fn,
                               fn=fn,
                               key=key,
                               out=out,
                               args=args,
                               kwargs=kwargs)
    if backend == "joblib":
        # TODO set 'pre_dispatch' kwarg
        out_datamap_gens = joblib.Parallel(n_jobs=joblib_n_jobs)(
            joblib.delayed(pmapcat_fn)(in_datamap)
            for in_datamap in in_gen
        )
    elif backend == "pool":
        assert pool is not None
        if pool_fn == "imap_unordered":
            pool_map_fn = pool.imap_unordered
        elif pool_fn == "imap":
            pool_map_fn = pool.imap
        elif pool_fn == "map":
            pool_map_fn = pool.map
        else:
            raise ValueError("Pool function not found: %s" % pool_map_fn)

        out_datamap_gens = pool_map_fn(
            func=pmapcat_fn,
            iterable=in_gen,
            chunksize=pool_chunksize
        )
    else:
        raise Exception("Backend not found: %s" % backend)

    for out_vals in out_datamap_gens:
        for out_val in out_vals:
            yield out_val


def transform_select_keys(in_gen, key):
    keys = _make_keys_list(key)
    key_set = set(keys)
    for in_datamap in in_gen:
        out_datamap = in_datamap
        for key in in_datamap.keys():
            if key not in key_set:
                out_datamap = toolz.dissoc(out_datamap, key)
        yield out_datamap


def transform_dissoc(in_gen, key):
    keys = _make_keys_list(key)
    for in_datamap in in_gen:
        out_datamap = in_datamap
        for k in keys:
            out_datamap = toolz.dissoc(out_datamap, k)
        yield out_datamap


def transform_rename(in_gen, key, out):
    """
    if key and out are both sequences, renames them in order (ie. 3rd in key
    gets renamed to 3rd in out)
    """
    assert (isinstance(key, six.string_types) and isinstance(out, six.string_types)
            or isinstance(key, (list, tuple)) and isinstance(out, (list, tuple)))
    keys = _make_keys_list(key)
    outs = _make_keys_list(out)

    assert len(keys) == len(outs)
    pairs = zip(keys, outs)
    for in_datamap in in_gen:
        out_datamap = {}
        out_datamap.update(in_datamap)
        for old, new in pairs:
            out_datamap.pop(old)
            out_datamap[new] = in_datamap[old]
        yield out_datamap


def transform_assoc_constant(in_gen, out, val):
    assert isinstance(out, six.string_types)  # TODO handle list case
    for in_datamap in in_gen:
        out_datamap = toolz.assoc(in_datamap, out, val)
        yield out_datamap


def transform_zip_assoc(in_gen, out, iterator):
    assert isinstance(out, six.string_types)  # TODO handle list case
    # can implement with zip_merge
    for in_datamap, iter_val in itertools.izip(in_gen, iterator):
        out_datamap = toolz.assoc(in_datamap, out, iter_val)
        yield out_datamap


def transform_zip_merge(in_gen, map_iterator):
    # can implement with zip_merge_with(second, ...)
    for in_datamap, iter_val in itertools.izip(in_gen, map_iterator):
        out_datamap = toolz.merge(in_datamap, iter_val)
        yield out_datamap


def transform_zip_merge_with(in_gen, fn, map_iterator):
    for in_datamap, iter_val in itertools.izip(in_gen, map_iterator):
        out_datamap = toolz.merge_with(fn, in_datamap, iter_val)
        yield out_datamap


def transform_do(in_gen, fn, key=None, args=None, kwargs=None):
    # NOTE: somewhat copy-pasted from transform_mapcat
    assert args is None or isinstance(args, (list, tuple))
    assert kwargs is None or isinstance(kwargs, dict)
    assert key is None or isinstance(key, (six.string_types, list, tuple))

    args = list(args or [])
    kwargs = kwargs or {}
    for in_datamap in in_gen:
        in_vals = _key_values(in_datamap, key)
        fn(*(in_vals + args), **kwargs)
        yield in_datamap


def transform_take(in_gen, n):
    return toolz.take(n, in_gen)


def transform_repeat_each(in_gen, n):
    for in_datamap in in_gen:
        for _ in xrange(n):
            yield in_datamap


def transform_chunk(in_gen, chunk_size=-1, drop_remainder=True):
    """
    drop_remainder:
    whether or not to drop the last chunks (if the total number of elements
    is not divisible by chunk_size)
    """
    if chunk_size == -1:
        partitioned = [in_gen]
    else:
        if drop_remainder:
            partitioned = toolz.partition(chunk_size, in_gen)
        else:
            partitioned = toolz.partition_all(chunk_size, in_gen)
    for in_datamaps in partitioned:
        in_datamaps = list(in_datamaps)
        out_datamap = collections.defaultdict(list)
        key_set = set(in_datamaps[0].keys())
        for in_datamap in in_datamaps:
            assert set(in_datamap.keys()) == key_set
            for k, v in in_datamap.items():
                out_datamap[k].append(v)
        yield dict(out_datamap)


def transform_dechunk(in_gen):
    for in_datamap in in_gen:
        lens = map(len, in_datamap.values())
        assert max(lens) == min(lens), dict(max=max(lens), min=min(lens))
        for idx in range(len(in_datamap.values()[0])):
            yield {k: v[idx] for k, v in in_datamap.items()}


def transform_numpy_chunk(in_gen, keys, chunk_size=-1, drop_remainder=True):
    keys = _make_keys_list(keys)
    for in_datamap in transform_chunk(in_gen,
                                      chunk_size,
                                      drop_remainder=drop_remainder):
        out_datamap = in_datamap
        for key in keys:
            out_datamap = toolz.assoc(out_datamap,
                                      key,
                                      np.array(out_datamap[key]))
        yield out_datamap


def transform_random_sample(in_gen,
                            rng=None,
                            count=-1,
                            title=None):
    """
    title:
    optionally provide a title to time realizing the whole sequence
    """
    if rng is None:
        rng = np.random

    # NOTE: in_gen must be finite
    if title is None:
        in_datamaps = list(in_gen)
    else:
        with timer_utils.timer("random_sample[%s]" % title):
            in_datamaps = list(in_gen)
    if len(in_datamaps) > 0:
        utils.debug("Sampling from %d datamaps", len(in_datamaps))
        current_count = 0
        while True:
            if current_count == count:
                break
            current_count += 1
            idx = rng.randint(len(in_datamaps))
            # make a copy of the datamap, because if this datamap is mutated
            # this can cause unwanted memory usage due to keeping the
            # original datamaps forever
            yield dict(in_datamaps[idx])
    else:
        utils.info("No datamaps to sample from found")


def transform_sort(in_gen,
                   key=None,
                   fn=None,
                   args=None,
                   kwargs=None,
                   title=None,
                   reverse=False):
    """
    sorts a generator based on the given key

    NOTE: partly copy-pasted from transform_mapcat
    """
    assert args is None or isinstance(args, (list, tuple))
    assert kwargs is None or isinstance(kwargs, dict)
    assert key is None or isinstance(key, (six.string_types, list, tuple))
    args = list(args or [])
    kwargs = kwargs or {}
    assert (key is not None) or (fn is not None), "Can't sort on maps"

    def sort_fn(in_datamap):
        vals = _key_values(in_datamap, key)
        if fn is None:
            return tuple(vals)
        else:
            return fn(*(vals + args), **kwargs)

    # NOTE: in_gen must be finite
    if title is None:
        in_datamaps = list(in_gen)
        in_datamaps.sort(key=sort_fn, reverse=reverse)
    else:
        with timer_utils.timer("sort[%s]" % title):
            in_datamaps = list(in_gen)
            in_datamaps.sort(key=sort_fn, reverse=reverse)

    for in_datamap in in_datamaps:
        yield in_datamap


def transform_log_hash_all(in_gen, title=None, coerce_mmap=True):
    """
    realizes the whole sequence of datamaps, and logs the hash of
    the whole sequence

    use case:
    - checking if a dataset is deterministic across runs/machines
    """
    if title is None:
        in_datamaps = list(in_gen)
        h = joblib.hash(in_datamaps, coerce_mmap=coerce_mmap)
        utils.info("hash=%s" % h)
    else:
        with timer_utils.timer("hash[%s]" % title):
            in_datamaps = list(in_gen)
            h = joblib.hash(in_datamaps, coerce_mmap=coerce_mmap)
            utils.info("hash[%s]=%s" % (title, h))

    for in_datamap in in_datamaps:
        yield in_datamap


def transform_log_hash_each(in_gen, title=None, coerce_mmap=True):
    """
    logs the hash of each datamap in the sequence

    use case:
    - checking if a dataset is deterministic across runs/machines
    """
    for in_datamap in in_gen:
        h = joblib.hash(in_datamap, coerce_mmap=coerce_mmap)
        if title is None:
            utils.info("hash=%s" % h)
        else:
            utils.info("hash[%s]=%s" % (title, h))
        yield in_datamap

DEFAULT_TIC_KEY = "__tic_time"


def transform_tic(in_gen, out=None):
    """
    stores current time in the given key
    """
    if out is None:
        out = DEFAULT_TIC_KEY
    assert isinstance(out, six.string_types)
    for in_datamap in in_gen:
        yield toolz.assoc(in_datamap, out, time.time())


def transform_toc(in_gen, title, key=None, overrides=None, dissoc=False):
    """
    times durations since tic was called with key
    """
    if key is None:
        key = DEFAULT_TIC_KEY
    assert isinstance(key, six.string_types)
    for in_datamap in in_gen:
        tic_time = in_datamap[key]
        timer_utils.DEFAULT_TIMER.save_time(key=title,
                                            start_time=tic_time,
                                            overrides=overrides)
        if dissoc:
            yield toolz.dissoc(in_datamap, key)
        else:
            yield in_datamap


def transform_count(in_gen, title=None):
    """
    counts number of datamaps and prints the count out
    """
    count = 0
    for in_datamap in in_gen:
        count += 1
        yield in_datamap
    if title is not None:
        print("%s count: %d" % (title, count))
    else:
        print("count: %d" % count)


def transform_doall(in_gen, title=None, discard=True, n_iter=None):
    """
    evaluates sequence and optionally discards output. optionally logs times
    if a title is given
    """
    res = []
    if title is None:
        for in_datamap in in_gen:
            if not discard:
                res.append(in_datamap)
    else:
        with timer_utils.LoopTimer(title, n_iter=n_iter) as lt:
            in_gen = higher_order._to_generator(in_gen)
            while True:
                try:
                    in_datamap = in_gen.next()
                    lt.end_iter()
                    if not discard:
                        res.append(in_datamap)
                except StopIteration:
                    break
    return res


def transform_repeat_all(in_gen):
    in_datamaps = []
    # NOTE: we make a copy of the datamap, because if this datamap is mutated
    # this can cause unwanted memory usage due to keeping the
    # original datamaps forever
    for in_datamap in in_gen:
        in_datamaps.append(in_datamap)
        yield dict(in_datamap)
    utils.debug("repeat_all finished evaluating all %d original datamaps"
                % len(in_datamaps))
    while True:
        for in_datamap in in_datamaps:
            yield dict(in_datamap)

# ################################### dsl ################################


class DatasetDSL(base.Dataset):

    def __init__(self, dataset):
        assert isinstance(dataset, base.Dataset)
        if isinstance(dataset, DatasetDSL):
            self.dataset = dataset.dataset
        else:
            self.dataset = dataset

    def _open(self):
        return self.dataset.open()

    def _close(self):
        return self.dataset.close()

    def _stateless_transform(self, transform, *args, **kwargs):
        return DatasetDSL(higher_order.StatelessTransformDataset(self.dataset,
                                                                 transform,
                                                                 *args,
                                                                 **kwargs))

    def map(self, *args, **kwargs):
        """Apply a map fn to the objects.

        @param fn The function to apply.
        @param key If supplied, give the fn the value of the key instead of the
            whole object.  If this is a list, give the function an arg for each
            item in key.
        @param out If supplied (with key), store the return value of the fn in
            this key.  If out is a tuple, store the elements of the return
            value tuple in the keys in out.
        """
        return self._stateless_transform(transform_map,
                                         *args,
                                         **kwargs)

    def map_key(self, *args, **kwargs):
        """Apply a map fn to the value for object's key.

        The mapped value will be stored in `key`.  This is the same as calling
        `map` with the same value for `key` and `out`.

        @param key The key of the value to map.
        @param fn The function to use for the map.
        """
        return self._stateless_transform(transform_map_key,
                                         *args,
                                         **kwargs)

    def mapcat(self, *args, **kwargs):
        """Perform a map operation followed by a flatten on the output.

        The map operation should return an iterable.  For each element in the
        iterable, output a copy of the object with that element as the value in
        that `out` key.  Note that this means returning an empty list will
        delete that object.
        """
        return self._stateless_transform(transform_mapcat,
                                         *args,
                                         **kwargs)

    def mapcat_key(self, *args, **kwargs):
        """Perform a mapcat operation using `out` = `key`."""

        return self._stateless_transform(transform_mapcat_key,
                                         *args,
                                         **kwargs)

    def filter(self, *args, **kwargs):
        """Only allow items through that evaluate as true with the provided fn."""
        return self._stateless_transform(transform_filter,
                                         *args,
                                         **kwargs)

    def remove(self, *args, **kwargs):
        """Drop items that evaluate as true with the provided fn."""
        return self._stateless_transform(transform_remove,
                                         *args,
                                         **kwargs)

    def pmapcat(self, *args, **kwargs):
        return self._stateless_transform(transform_pmapcat,
                                         *args,
                                         **kwargs)

    def select_keys(self, *args, **kwargs):
        """Only allow the provided keys to pass through.

        The output will drop all keys not in the supplied key list.
        @param keys: (list) the keys to pass through.
        """
        return self._stateless_transform(transform_select_keys,
                                         *args,
                                         **kwargs)

    def dissoc(self, *args, **kwargs):
        """Drop the provided keys from the input.

        The output will drop the keys in the supplied key list.
        @param keys: (list) the keys to drop.
        """
        return self._stateless_transform(transform_dissoc,
                                         *args,
                                         **kwargs)

    def rename(self, *args, **kwargs):
        """Rename the provided in_keys to the provided out_keys.

        If in_key are a list/tuple, out_key must be a list/tuple of the same
        length.  In this case, in_key[i] will be mapped to out_key[i].

        @param in_key (str/list/tuple) Key(s) to rename.
        @param out_key (str/list/tuple) Rename to this key.
        """
        return self._stateless_transform(transform_rename,
                                         *args,
                                         **kwargs)

    def zip_assoc(self, *args, **kwargs):
        return self._stateless_transform(transform_zip_assoc,
                                         *args,
                                         **kwargs)

    def zip_merge(self, *args, **kwargs):
        return self._stateless_transform(transform_zip_merge,
                                         *args,
                                         **kwargs)

    def zip_merge_with(self, *args, **kwargs):
        return self._stateless_transform(transform_zip_merge_with,
                                         *args,
                                         **kwargs)

    def do(self, *args, **kwargs):
        """Perform the given fn on the object.

        This passes the object through after performing `fn`.  Since the object
        is passed through, if the function mutates the object the output will be
        mutated.

        The kwarg `key` works as normal.

        @param fn (function): Function to apply to each object.
        @param key (str, kwarg): Use the value of key instead of the object.
        """
        return self._stateless_transform(transform_do,
                                         *args,
                                         **kwargs)

    def assoc_constant(self, *args, **kwargs):
        """Associate to the given key the given constant.

        @param key: (string) Key to store constant in.
        @param constant: (sting|number) Constant to store.

        Use without kwargs, eg `assoc_constant('a_name', 10)`
        """
        return self._stateless_transform(transform_assoc_constant,
                                         *args,
                                         **kwargs)

    def take(self, *args, **kwargs):
        """Limit the datastream to the first n elements.

        @param n Number of elements to take.
        """
        return self._stateless_transform(transform_take,
                                         *args,
                                         **kwargs)

    def repeat_each(self, *args, **kwargs):
        return self._stateless_transform(transform_repeat_each,
                                         *args,
                                         **kwargs)

    def chunk(self, *args, **kwargs):
        """Chunk incoming datasets into a dataset of lists.

        This chunks the incoming datasets into datasets of chunk_size,
        with list-value keys.  The keys specified in `keys` will be
        converted into ndarrays.
        @param chunk_size (int, kwarg, default -1) How many datasets to chunk
            into a single dataset.  -1 means chunk all datasets into one.
        @param drop_remainder (bool, kwarg, default True) Drop any datasets
            that don't fit into a chunk.
        """
        return self._stateless_transform(transform_chunk,
                                         *args,
                                         **kwargs)

    def dechunk(self, *args, **kwargs):
        """Converts a list-valued dataset into individual datasets.

        This undoes the chunk operation (although you might lose datasets if
        drop_remainder==True).
        """
        return self._stateless_transform(transform_dechunk,
                                         *args,
                                         **kwargs)

    def numpy_chunk(self, *args, **kwargs):
        """Chunk incoming datasets into a dataset of lists and ndarrays.

        This chunks the incoming datasets into datasets of chunk_size,
        with list-value keys.  The keys specified in `keys` will be
        converted into ndarrays.
        @param keys (list of str) Keys to convert into ndarrays.
        @param chunk_size (int, kwarg, default -1) How many datasets to chunk
            into a single dataset.  -1 means chunk all datasets into one.
        @param drop_remainder (bool, kwarg, default True) Drop any datasets
            that don't fit into a chunk.
        """
        return self._stateless_transform(transform_numpy_chunk,
                                         *args,
                                         **kwargs)

    def random_sample(self, *args, **kwargs):
        """Take a random sample (with replacement) of the dataset.

        This greedily evaluates the preceeding dataset, then lazily supplies
        random samples from it.

        @param rng (kwarg default None) Random number generator.  None means
            create a new one.
        @param count (int kwarg default -1) How many samples to take.  If -1,
            take an infinite number.
        @param title (str kwarg) optionally provide a title to time realizing
            the whole sequence
        """
        return self._stateless_transform(transform_random_sample,
                                         *args,
                                         **kwargs)

    def sort(self, *args, **kwargs):
        return self._stateless_transform(transform_sort,
                                         *args,
                                         **kwargs)

    def log_hash_all(self, *args, **kwargs):
        return self._stateless_transform(transform_log_hash_all,
                                         *args,
                                         **kwargs)

    def log_hash_each(self, *args, **kwargs):
        return self._stateless_transform(transform_log_hash_each,
                                         *args,
                                         **kwargs)

    def tic(self, *args, **kwargs):
        return self._stateless_transform(transform_tic,
                                         *args,
                                         **kwargs)

    def toc(self, *args, **kwargs):
        return self._stateless_transform(transform_toc,
                                         *args,
                                         **kwargs)

    def count(self, *args, **kwargs):
        return self._stateless_transform(transform_count,
                                         *args,
                                         **kwargs)

    def doall(self, *args, **kwargs):
        return self._stateless_transform(transform_doall,
                                         *args,
                                         **kwargs)

    def repeat_all(self, *args, **kwargs):
        return self._stateless_transform(transform_repeat_all,
                                         *args,
                                         **kwargs)

    def to_other_process(self, *args, **kwargs):
        return DatasetDSL(higher_order.OtherProcessDataset(self.dataset,
                                                           *args,
                                                           **kwargs))

    def to_joblib_serialized(self, *args, **kwargs):
        return DatasetDSL(higher_order.JoblibSerializedDataset(self.dataset,
                                                               *args,
                                                               **kwargs))

    def to_threaded_reader(self, *args, **kwargs):
        return DatasetDSL(higher_order.ThreadedReaderDataset(self.dataset,
                                                             *args,
                                                             **kwargs))

    def cache(self, *args, **kwargs):
        return DatasetDSL(higher_order.cached_dataset(self.dataset,
                                                      *args,
                                                      **kwargs))

    def copy(self):
        """Return a copy of this dataset."""
        return copy.deepcopy(self)

    def map_each_key(self, keys, *args, **kwargs):
        """
        Apply the given fn to each key in 'keys'
        @param keys (list): list of keys on which to apply 'fn'
        @param fn (function): a function which takes in a single argument
                              and returns a single value
        """
        out_dset = self
        for key in keys:
            out_dset = out_dset._stateless_transform(transform_map_key, *args,
                                                     key=key, **kwargs)
        return out_dset

    def apply(self, fn, args=None, kwargs=None):
        """
        calls function with current DatasetDSL as the first argument
        """
        assert args is None or isinstance(args, (list, tuple))
        assert kwargs is None or isinstance(kwargs, dict)

        args = list(args or [])
        kwargs = kwargs or {}
        return fn(self, *args, **kwargs)

    def _to_generator(self):
        """Convert dataset into a generator.

        WARNING: if the generator isn't fully realized, the cleanup won't occur
        """
        with self.dataset as g:
            for datamap in g:
                yield datamap

    def to_list(self):
        """Converts the datastream to a list.

        WARNING: This will greedily consume the dataset, which may be an
        expensive operation."""
        return list(self._to_generator())

    def to_fn(self):
        """
        converts a dataset whose base dataset is a promise into a function
        that takes in a dataset, delivers the promise, and returns the
        original/current dataset
        """
        return higher_order.PromiseFunction(self)

# ############################ sugar constructors ############################


def from_list(datamaps):
    return DatasetDSL(constructors.FromListDataset(datamaps))


def from_generator(generator):
    return DatasetDSL(constructors.FromGeneratorDataset(generator))


def from_generator_fn(generator_fn):
    return DatasetDSL(constructors.FromGeneratorFnDataset(generator_fn))


def from_joblib_dir(*args, **kwargs):
    return DatasetDSL(constructors.FromJoblibDirectoryDataset(*args, **kwargs))


def from_joblib_cached_dataset(*args, **kwargs):
    return DatasetDSL(constructors.FromJoblibCachedDataset(*args, **kwargs))


def promise():
    return DatasetDSL(higher_order.PromiseDataset())


def multi_dataset(*args, **kwargs):
    return DatasetDSL(higher_order.MultiDataset(*args, **kwargs))

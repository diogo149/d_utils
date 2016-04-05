import shutil
import numpy as np

from .. import utils
from ..utils import toolz
from .. import timer_utils
from .. import parallel
from .. import joblib_utils

from . import base

# ########################## higher-order datasets #####################


class StatelessTransformDataset(base.Dataset):

    def __init__(self, dataset, transform, *args, **kwargs):
        """
        a transform on a dataset that requires no setup or teardown
        """
        self.dataset = dataset
        self.transform = transform
        self.args = args
        self.kwargs = kwargs

    def _open(self):
        in_gen = self.dataset.open()
        return self.transform(in_gen, *self.args, **self.kwargs)

    def _close(self):
        return self.dataset.close()


class JoblibSerializedDataset(base.Dataset):

    def __init__(self, dataset, directory, mmap_mode=None):
        """
        Dataset wrapper that serialized the dataset with joblib instead of
        a normal pickle
        """
        self.dataset = dataset
        self.directory = directory
        self.mmap_mode = mmap_mode

    def _open(self):
        return self.dataset.open()

    def _close(self):
        return self.dataset.close()

    def __getstate__(self):
        joblib_utils.dump_dir(self.dataset, self.directory)
        state = super(JoblibSerializedDataset, self).__getstate__()
        del state['dataset']
        return state

    def __setstate__(self, state):
        res = super(JoblibSerializedDataset, self).__setstate__(state)
        self.dataset = joblib_utils.load_dir(self.directory,
                                             mmap_mode=self.mmap_mode)
        return res


def _generate_from_dataset(dataset):
    with dataset as g:
        for x in g:
            yield x


class OtherProcessDataset(base.Dataset):

    def __init__(self,
                 dataset,
                 n_jobs=1,
                 buffer_size=1,
                 daemon_children=True,
                 mmap_mode='r',
                 try_shm=True):
        """
        prepares a dataset chunks in a different process before they are
        requested
        """
        self.dataset = dataset
        self.n_jobs = n_jobs
        self.buffer_size = buffer_size
        self.daemon_children = daemon_children
        self.mmap_mode = mmap_mode
        self.try_shm = try_shm

    def _open(self):
        self.mp_gen_ = parallel.rng_mp_generator(
            utils.partial(_generate_from_dataset, self.dataset),
            n_jobs=self.n_jobs,
            buffer_size=self.buffer_size,
            daemon_children=self.daemon_children,
            mmap_mode=self.mmap_mode,
            try_shm=self.try_shm
        )
        self.mp_gen_.open()
        return (x for x in self.mp_gen_)

    def _close(self):
        self.mp_gen_.close()
        # cleanup
        del self.mp_gen_


class ThreadedReaderDataset(base.Dataset):

    def __init__(self, dataset, buffer_size=2):
        """
        allows data to be read from an input dataset on a separate thread

        good in combination with the OtherProcessDataset with mmap_mode=None
        and a task running that releases the GIL (such as running code on
        the GPU) so that the deserialization can occur without any penalty on
        the main thread
        """
        self.dataset = dataset
        self.buffer_size = buffer_size

    def _open(self):
        self.base_generator_ = self.dataset.open()
        self.threaded_generator_ = parallel.ThreadedGenerator(
            generator=self.base_generator_,
            buffer_size=self.buffer_size
        )
        self.generator_ = self.threaded_generator_.open()
        return self.generator_

    def _close(self):
        self.dataset.close()
        self.threaded_generator_.close()
        del self.generator_
        del self.base_generator_
        del self.threaded_generator_


def random_sample_generators(generators,
                             weights=None,
                             rng=None):
    """
    weights: probabilities for each generator
    """
    if rng is None:
        rng = np.random
    idxs = np.arange(len(generators))
    while generators:
        idx = rng.choice(idxs, p=weights)
        gen = generators[idx]
        try:
            yield gen.next()
        except StopIteration:
            generators.pop(idx)
            if weights is not None:
                # OPTIMIZE
                weights = list(weights)
                weights.pop(idx)
                weights = np.array(weights)
                weights /= weights.sum()
            idxs = np.arange(len(generators))


def combine_generators(generators, strategy, strategy_opts):
    if strategy == "random":
        return random_sample_generators(generators, **strategy_opts)
    elif strategy == "concat":
        return toolz.concat(generators)
    else:
        raise ValueError("Strategy not found: %s" % strategy)


def _to_generator(g):
    if hasattr(g, "next"):
        return g
    else:
        return (x for x in g)


class MultiDataset(base.Dataset):

    def __init__(self, datasets, strategy, strategy_opts=None):
        """
        dataset for alternating datamaps between multiple datasets
        Strategies are "random" or "concat".
        Random will randomly pull from each dataset, removing datasets from the mix
        as they become exhausted. It will not reorder within the dataset.
        Note: If you want to randomly sample from the input datasets, you must apply a random_sample before passing to MultiDataset.
        Concat will fully exhaust each dataset in order. If a dataset is infinite, the next datasets will never be used.
        """
        self.datasets = datasets
        self.strategy = strategy
        self.strategy_opts = strategy_opts

    def _open(self):
        self.generators_ = [_to_generator(ds.open()) for ds in self.datasets]
        strategy_opts = self.strategy_opts or {}
        self.generator_ = combine_generators(self.generators_,
                                             self.strategy,
                                             strategy_opts)
        return self.generator_

    def _close(self):
        for ds in self.datasets:
            ds.close()
        del self.generators_
        del self.generator_


class PromiseDataset(base.Dataset):

    """
    a dataset that may be initialized with an inner dataset at a future time
    """

    def __init__(self):
        self.is_realized = False

    def deliver(self, ds):
        assert isinstance(ds, base.Dataset)
        assert not self.is_realized
        self.is_realized = True
        self.dataset = ds
        return self

    def _open(self):
        assert self.is_realized
        return self.dataset.open()

    def _close(self):
        return self.dataset.close()


class PromiseFunction(object):

    def __init__(self, dataset):
        assert isinstance(dataset, base.Dataset)
        self.dataset = dataset

    def __call__(self, dataset):
        base_ds = self.dataset
        while hasattr(base_ds, "dataset"):
            base_ds = base_ds.dataset
        assert isinstance(base_ds, PromiseDataset)
        base_ds.deliver(dataset)
        return self.dataset

# ############################# cached datasets ########################


class JoblibCachedDataset(base.Dataset):

    def __init__(self,
                 dataset,
                 dirname,
                 version=0,
                 eager=False,
                 mmap_mode=None):
        """
        a dataset whose data is cached to and read from disk

        params:
        - dataset : a finite dataset
        - dirname : directory to store the dataset
        - version : a version identifier to know when to invalidate the cache
                    for a directory
        - eager : whether or not to cache the dataset immediately when the
                  CachedDataset is created (eg. so that the same work isn't
                  duplicated when sent to multiple processes)
        """
        self.dataset = dataset
        self.dirname = dirname
        self.version = version
        self.eager = eager
        self.mmap_mode = mmap_mode
        if self.eager:
            self.write_cache()

    def _open(self):
        self.write_cache()
        return self.read_cache()["data"]

    def write_cache(self):
        if joblib_utils.dump_exists(self.dirname):
            cached = self.read_cache()
            if cached["version"] != self.version:
                shutil.rmtree(self.dirname)
                del cached  # free memory
            else:
                return
        title = ("Writing dataset %s (version=%s) to cache"
                 % (self.dirname, self.version))
        with timer_utils.timer(title, threshold=0.1):
            with self.dataset as g:
                utils.info(title)
                datamaps = list(g)
                joblib_utils.dump_dir(dict(version=self.version,
                                           data=datamaps),
                                      self.dirname)

    def read_cache(self):
        return joblib_utils.load_dir(self.dirname, mmap_mode=self.mmap_mode)


def cached_dataset(dataset, backend, **kwargs):
    if backend == "joblib":
        return JoblibCachedDataset(dataset, **kwargs)
    else:
        raise ValueError("Unknown dataset cache backend: %s" % backend)

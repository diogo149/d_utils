from __future__ import print_function
import os
import shutil
import multiprocessing as mp
import joblib
import pickle
import numpy as np
from .. import joblib_utils
from .. import io_utils
from .. import utils

# lists of the current process indices, and the total number of processes
# each split had
# a list allows recursively adding more processes, and seeing the hierarchy
# in which they were applied
PROCESS_IDXS = [0]
PROCESS_NUMS = [1]


def _run_gen(generator_fn_str, index, queue, temp_dir):
    PROCESS_IDXS.append(index)
    try:
        generator_fn = pickle.loads(generator_fn_str)
        for res in generator_fn(index):
            files = joblib_utils.dump_tempfile(res,
                                               dir=temp_dir)
            del res  # free result because it's already persisted
            queue.put(files)
        queue.put(None)
    except Exception as e:
        utils.exception("Exception on separate process: %s" % e)
        queue.put(e)
    except KeyboardInterrupt:
        utils.exception("KeyboardInterrupt on separate process")
        queue.put(Exception("KeyboardInterrupt on separate process"))


class _MultiprocessingGenerator(object):

    def __init__(self,
                 func,
                 n_jobs,
                 buffer_size,
                 daemon_children=True,
                 mmap_mode='r',
                 try_shm=True):
        """
        NOTE: if using multiprocessing in the generator, disable making
        each child a daemon because daemon processes
        supposedly can't have children (which would prevent the other
        process from running parallel loops) - this can affect process cleanup
        and would not be recommended
        """
        self.func = func
        self.n_jobs = n_jobs
        self.buffer_size = buffer_size
        self.daemon_children = daemon_children
        self.mmap_mode = mmap_mode
        self.try_shm = try_shm

    def open(self):
        self.running_jobs = self.n_jobs
        self.queue = mp.Queue(self.buffer_size)
        self.func_str = pickle.dumps(self.func)
        self.processes = []
        self.temp_dir = io_utils.mkdtemp(self.try_shm)
        utils.info("Creating dir %s for multiprocessing generator",
                   self.temp_dir)

        PROCESS_NUMS.append(self.n_jobs)
        for i in range(self.n_jobs):
            p = mp.Process(target=_run_gen,
                           args=(self.func_str,
                                 i,
                                 self.queue,
                                 self.temp_dir))
            self.processes.append(p)
            if self.daemon_children:
                p.daemon = True
            p.start()
        PROCESS_NUMS.pop()

    def _decrement_processes(self):
        self.running_jobs -= 1
        if self.running_jobs == 0:
            raise StopIteration

    def __iter__(self):
        return self

    def next(self):
        while True:
            files = self.queue.get()
            if files is None:
                self._decrement_processes()
            elif isinstance(files, Exception):
                raise files
            else:
                break
        res = joblib.load(files[0], mmap_mode=self.mmap_mode)
        for file_name in files:
            os.remove(file_name)
        return res

    def close(self):
        for p in self.processes:
            p.terminate()
        # there appears to be a race condition with files being deleted in
        # a different thread (in the next function), which causes this to break
        # thus ignoring errors
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        assert not os.path.exists(self.temp_dir)

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, tb):
        self.close()
        # don't supress any exception
        return False


class DummyGenerator(object):

    def __init__(self, generator_fn):
        """
        version of that runs the generator on the same process
        for testing if something works
        """
        self.generator_fn = generator_fn
        self.gen_ = self.generator_fn(0)

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def next(self):
        return self.gen_.next()

    def __exit__(self, type, value, tb):
        return False

    def open(self):
        pass

    def close(self):
        pass


def mp_generator(generator_fn,
                 n_jobs=1,
                 buffer_size=1,
                 daemon_children=True,
                 mmap_mode='r',
                 try_shm=True):
    """
    Parameters
    ----------
    generator_fn : function
                   single argument function that takes in the processor index
                   and returns a generator. NOTE: the same function is run on
                   each process

    n_jobs : integer, optional (default=1)
             number of producer processes to run f on

    buffer_size : integer, optional (default=1)
                  number of waiting results to be kept in a queue

    daemon_children : boolean, optional (default=True)
                      whether or not children processes are daemons

    Returns
    -------
    g : context manager that returns a generator
    """
    if n_jobs < 1:
        return DummyGenerator(generator_fn)
    else:
        return _MultiprocessingGenerator(generator_fn,
                                         n_jobs=n_jobs,
                                         buffer_size=buffer_size,
                                         daemon_children=daemon_children,
                                         mmap_mode=mmap_mode,
                                         try_shm=try_shm)


def _rng_generator_fn(generator_fn, index):
    np.random.seed(index)
    return generator_fn()


def rng_mp_generator(generator_fn,
                     n_jobs=1,
                     buffer_size=1,
                     daemon_children=True,
                     mmap_mode='r',
                     try_shm=True):
    """Like mp_generator, but seeds the
    numpy random number generator for each process.
    """
    return mp_generator(utils.partial(_rng_generator_fn, generator_fn),
                        n_jobs=n_jobs,
                        buffer_size=buffer_size,
                        daemon_children=daemon_children,
                        mmap_mode=mmap_mode,
                        try_shm=try_shm)


if __name__ == "__main__":
    # python -m d.parallel.multiprocessing_generator

    import time

    val = 1

    def foo(index):
        np.random.seed(index)
        for i in range(10):
            print("On %d : %d" % (index, i))
            print("Sleeping for %d seconds" % val)
            time.sleep(val)
            z = np.random.randn(1000, 4000)
            print("Awake with %f!" % z.sum())
            yield z

    with mp_generator(foo, n_jobs=3) as g:
        for x in g:
            print(x.sum())
            time.sleep(1)

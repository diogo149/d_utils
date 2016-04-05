import fcntl
import contextlib
import os
import tempfile
import subprocess
import shutil
import csv
import sys

import six
import joblib
import yaml
import numpy as np

from . import utils

if utils.config["json"]["use_simplejson"]:
    import simplejson as json
else:
    import json


def guarantee_exists(filename):
    """
    creates a file if it doesn't exist yet
    """
    if not os.path.exists(filename):
        with open(filename, 'a'):
            pass


def guarantee_dir_exists(dirname):
    """
    creates a directory if it doesn't exist yet
    """
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError as e:
            if e.errno != 17 or e.strerror != "File exists":
                raise
    assert os.path.isdir(dirname)


class file_lock(object):

    """
    Blocking lock using a file
    http://blog.vmfarms.com/2011/03/cross-process-locking-and.html
    """

    def __init__(self, filename):
        self.filename = filename

    def __enter__(self):
        # This will create it if it does not exist already
        self.handle = open(self.filename, 'w')
        # Bitwise OR fcntl.LOCK_NB if you need a non-blocking lock
        fcntl.flock(self.handle, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()
        # don't supress any exception
        return False


@contextlib.contextmanager
def tmp_chmod(filename, mode):
    """
    temporarily change permissions of a file
    """
    assert isinstance(mode, int)
    assert (mode & 0o777) == mode
    initial_mode = os.stat(filename).st_mode & 0o777
    try:
        os.chmod(filename, mode)
        yield
    finally:
        os.chmod(filename, initial_mode)


def tmp_chmod_no_write(filename):
    """
    temporarily change permissions of a file to not have write permissions
    """
    guarantee_exists(filename)
    initial_mode = os.stat(filename).st_mode & 0o777
    mode = initial_mode & ~0o222
    return tmp_chmod(filename, mode)


@contextlib.contextmanager
def human_file_lock(filename):
    """
    locking a file for multiple python processes and to prevent humans from
    editting
    """
    lock_filename = filename + ".lock"
    with file_lock(lock_filename):
        with tmp_chmod_no_write(filename):
            # TODO yield instream and outstream
            yield


@contextlib.contextmanager
def atomically_writable_file(filename, force_overwrite=False):
    """
    yields a file handle to be written to that atomically writes/overwrites
    the filename only on close
    """
    try:
        # need to exit out of the tempfile contextmanager so that anything
        # written to the file gets flushed to disk before moving the file
        # (one could also call f.flush())
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_filename = f.name
            yield f
    finally:
        if force_overwrite:
            # uses subprocess.call instead of shutil.move because the
            # latter
            # doesn't allow overwriting a non-writeable file. os.rename
            # also fails
            # with an error "Invalid cross-device link"
            subprocess.call(["mv", "-f", tmp_filename, filename])
        else:
            shutil.move(tmp_filename, filename)


@contextlib.contextmanager
def clear_file_after(filename,
                     assert_doesnt_exist_before=True):
    """
    clears the file on exit, optionally asserting that it doesn't exist
    beforehand

    useful for named temporary files / testing
    """
    if assert_doesnt_exist_before:
        assert not os.path.exists(filename)

    try:
        yield
    finally:
        if os.path.exists(filename):
            if os.path.isdir(filename):
                shutil.rmtree(filename)
            else:
                os.remove(filename)
        assert not os.path.exists(filename)


def csv_to_dicts(filename):
    """
    read in a csv file into a list of dicts
    """
    with open(filename) as f:
        return list(csv.DictReader(f))


@contextlib.contextmanager
def chdir(dirname=None):
    """
    temporarily change directory
    """
    curdir = os.getcwd()
    try:
        if dirname is not None:
            os.chdir(dirname)
        yield
    finally:
        os.chdir(curdir)


@contextlib.contextmanager
def temporary_directory(directory_path):
    """
    create a temporary directory that is deleted afterwards
    """
    os.mkdir(directory_path)
    try:
        yield
    finally:
        shutil.rmtree(directory_path)


def mkdtemp(try_shm=True):
    """
    make a directory in a temporary folder (does not clean it up)
    """
    directory = None

    # try using shared memory
    if try_shm:
        shm = joblib.pool.SYSTEM_SHARED_MEM_FS
        if os.path.exists(shm):
            template = ("Using shared memory: %s may have to"
                        " be manually cleaned")
            utils.warn_once(template % shm)
            directory = shm

    return tempfile.mkdtemp(prefix='tmp-%s-' % os.environ['USER'],
                            dir=directory)


@contextlib.contextmanager
def NamedTemporaryDirectory(try_shm=False):
    """
    temporarily create a directory and clean it up afterwards
    """
    try:
        dirname = mkdtemp(try_shm)
        yield dirname
    finally:
        shutil.rmtree(dirname)


class Tee(object):

    """
    mostly from:
    http://stackoverflow.com/questions/616645/how-do-i-duplicate-sys-stdout-to-a-log-file-in-python#
    """

    def __init__(self,
                 name,
                 mode,
                 target="stdout",
                 auto_flush=False):
        assert target in {"stdout", "stderr"}
        self.name_ = name
        self.mode_ = mode
        self.auto_flush_ = auto_flush
        self.tee_stdout_ = target == "stdout"
        self.tee_stderr_ = target == "stderr"
        self._opened = False

    def __getattr__(self, name):
        if self.tee_stdout_:
            return getattr(self._stdout, name)
        elif self.tee_stderr_:
            return getattr(self._stderr, name)
        else:
            # this should never happen
            raise ValueError

    def __enter__(self):
        self._file = open(self.name_, self.mode_)
        if self.tee_stdout_:
            self._stdout = sys.stdout
            sys.stdout = self
        if self.tee_stderr_:
            self._stderr = sys.stderr
            sys.stderr = self
        self._opened = True
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tee_stdout_:
            sys.stdout = self._stdout
        if self.tee_stderr_:
            sys.stderr = self._stderr
        self._file.close()
        self._opened = False
        # don't supress any exception
        return False

    def write(self, data):
        if self._opened:
            self._file.write(data)
        if self.tee_stdout_:
            self._stdout.write(data)
        if self.tee_stderr_:
            self._stderr.write(data)
        if self.auto_flush_:
            self.flush()

    def flush(self):
        if self._opened:
            self._file.flush()
        if self.tee_stdout_:
            self._stdout.flush()
        if self.tee_stderr_:
            self._stderr.flush()


# ################################### json ###################################


class NumpyAwareJSONEncoder(json.JSONEncoder):

    """
    mostly from:
    http://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    """

    def default(self, obj):
        if isinstance(obj, (np.ndarray, np.number)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def json_dumps(obj, **kwargs):
    """
    numpy aware json dump function
    """
    return json.dumps(obj, cls=NumpyAwareJSONEncoder, **kwargs)


def json_dump(obj, filename, **kwargs):
    """
    easy way of dumping object to json file
    """
    with open(filename, 'w') as f:
        f.write(json_dumps(obj, **kwargs))

json_loads = json.loads

json_load = json.load


# ################################### yaml ###################################


def yaml_dump_all(objs,
                  stream=None,
                  safe=True,
                  default_flow_style=False,
                  retry_coerce_json=True):
    """
    numpy aware yaml dump function, to dump a list of objects to a file

    stream:
    if None, returns a string
    if a string, treat as a file name
    else, treat it as a stream and write to it

    safe:
    produce only basic yaml tags

    http://pyyaml.org/wiki/PyYAML
    http://pyyaml.org/wiki/PyYAMLDocumentation
    """
    dump_fn = yaml.safe_dump_all if safe else yaml.dump_all
    # writing to a string instead of directly to a stream because if an
    # exception occurs, the yaml would still be partially written to the file
    try:
        as_string = dump_fn(objs, default_flow_style=default_flow_style)
    except (TypeError,
            # seems to only occur for numpy.int64
            yaml.representer.RepresenterError) as e:
        if retry_coerce_json:
            new_objs = json.loads(json_dumps(objs))
            as_string = dump_fn(new_objs,
                                default_flow_style=default_flow_style)
        else:
            raise e

    if stream is None:
        return as_string
    elif isinstance(stream, six.string_types):
        with open(stream, 'w') as f:
            f.write(as_string)
    else:
        stream.write(as_string)


def yaml_dumps_all(objs, **kwargs):
    """
    numpy aware yaml dump function that dumps a list of objects to a string

    see yaml_dump_all for arguments
    """
    # verify that stream argument is properly set
    assert "stream" not in kwargs or kwargs["stream"] is None
    return yaml_dump_all(objs, **kwargs)


def yaml_dump(obj, stream, **kwargs):
    """
    numpy aware yaml dump function that dumps an object to a stream / file

    see yaml_dump_all for arguments
    """
    return yaml_dump_all([obj], stream, **kwargs)


def yaml_dumps(obj, **kwargs):
    """
    numpy aware yaml dump function that dumps an object to a string
    """
    return yaml_dumps_all([obj], stream=None, **kwargs)


def yaml_load_all(stream=None,
                  as_string=None,
                  safe=True,
                  **kwargs):
    """
    wrapping the normal yaml functions because they take in either file
    handles or representations as strings, which is somewhat inconsistent
    with normal python apis where file handles and filenames as strings are
    usually interchangable

    if stream is a string, it is treated as a filename
    """
    if isinstance(stream, six.string_types):
        with open(stream) as f:
            return yaml_load_all(stream=f,
                                 as_string=as_string,
                                 safe=safe,
                                 **kwargs)
    assert (bool(stream)
            # because an empty string is treated as false
            + (bool(as_string) or as_string == "")
            == 1), "Only 1 must be set"
    load_fn = yaml.safe_load_all if safe else yaml.load_all
    if stream:
        gen = load_fn(stream, **kwargs)
    elif as_string or as_string == "":
        gen = load_fn(as_string, **kwargs)
    else:
        # should never get here
        raise ValueError

    return list(gen)


def yaml_loads_all(as_string, **kwargs):
    """
    load list of objects from string
    """
    return yaml_load_all(as_string=as_string, **kwargs)


def yaml_load(*args, **kwargs):
    """
    load object from file/stream
    """
    return yaml_load_all(*args, **kwargs)[0]


def yaml_loads(*args, **kwargs):
    """
    load object from string
    """
    return yaml_loads_all(*args, **kwargs)[0]

"""
trial is a way of creating a project

features:
- copies the source of the file you're using to run it
  (to have a copy of the file)
  (WARNING: there can be a race condition if the file changes before the trial
   code is run - the earlier you import du, the less likely this is to
   occur, due to the cache_inspect function)
- creates a directory where files for the run can be stored without worrying
  about collisions
- that directory also has a temporary directory that is cleared out
- allows storing intermediate values, to later be retrieved
- seeds random number generator
- creates a summary file (in trial.yml) with all of the important stored
  parameters

WARNING: when canceling a trial run, don't spam CTRL-C, since that can cancel
the clean up operations too

eg.
# in some file:
with run_trial("some_trial_name") as trial:
    # printed stuff is stored to _trials_dir_/some_trial_name/1/log.txt
    print "Hello world"
    # storing values
    trial.store_important("foo", 1)
    trial.store_important("foo", 2)
    trial.get_important("foo")  # returns [1, 2]
"""

import os
import sys
import re
import logging
import shutil
import datetime
import inspect
import importlib
import contextlib
from collections import defaultdict

import six

from . import utils, random_utils, io_utils, yaml_db
from .utils import toolz


def cache_inspect():
    """
    NOTE: seems to cache source code, so that future calls to
    inspect.getsource return the older sources, rather than the
    source when getsource is called
    """
    for record in inspect.getouterframes(inspect.currentframe()):
        frame = record[0]
        module = inspect.getmodule(frame)
        if module is not None:
            inspect.getsource(module)

cache_inspect()


def get_next_iteration_num(path):
    in_path = os.listdir(path)
    in_path_nums = filter(lambda x: re.match(r"\d+", x), in_path)
    if len(in_path_nums) == 0:
        return 1
    else:
        existing_iterations = map(int, in_path_nums)
        return max(existing_iterations) + 1


def trial_lock(trial_base_path):
    """
    only for making sure no race conditions when adding new iterations
    """
    return io_utils.file_lock(os.path.join(trial_base_path, ".lock"))


def trial_db_transaction(trial_base_path):
    # NOTE: this returns a context manager, since
    # yaml_db.db_transaction is a context manager
    db_path = os.path.join(trial_base_path, "trial.yml")
    return yaml_db.db_transaction(db_path)


@contextlib.contextmanager
def trial_db_iteration_transaction(trial_base_path, iteration_num):
    assert isinstance(iteration_num, int)
    with trial_db_transaction(trial_base_path) as db:
        iter_maps = filter(lambda x: x["iteration_num"] == iteration_num, db)
        assert len(iter_maps) <= 1
        if len(iter_maps) == 0:
            iter_map = dict(
                iteration_num=iteration_num,
            )
            db.append(iter_map)
        elif len(iter_maps) == 1:
            iter_map = iter_maps[0]
        else:
            raise ValueError("Wrong number (%d) of maps with the same "
                             "iteration number: %s"
                             % (len(iter_maps), iter_maps))

        yield iter_map


def path_in_trial_dir(trial_base_path, iteration_num, *ps):
    assert isinstance(iteration_num, int)
    return os.path.join(trial_base_path, str(iteration_num), *ps)


def create_trial_dir(trial_base_path, iteration_num, replace_strategy):
    """
    creates a new directory named 'iteration_num' under 'trial_base_dir',
    automatically selecting the next sequential iteration number if
    iteration_num is None.

    returns iteration_num because it needs to be chosen with the trial locked
    (if not specified)
    """
    with trial_lock(trial_base_path):
        if iteration_num is None:
            iteration_num = get_next_iteration_num(trial_base_path)
        assert iteration_num > 0 and isinstance(iteration_num, int)
        trial_path = path_in_trial_dir(trial_base_path, iteration_num)
        # check if already existing, if so handle appropriately
        if os.path.exists(trial_path):
            if replace_strategy == "force":
                replace = True
            elif replace_strategy == "ask":
                in_str = None
                while in_str not in ["y", "n"]:
                    in_str = six.moves.input(
                        ("%s exists, would you like "
                         "to overwrite? (y/n) ") % trial_path)
                replace = in_str == "y"
            elif replace_strategy is None:
                replace_strategy = False
            else:
                raise ValueError("replace strategy %s not found"
                                 % replace_strategy)
            if replace:
                shutil.rmtree(trial_path)
            else:
                raise ValueError("Trial already exists: %s" % trial_path)
        os.mkdir(trial_path)
    return iteration_num


@contextlib.contextmanager
def temporarily_add_file_logger_to(filename, loggers):
    """
    adds a FileLogger to each of the given loggers, which causes the
    logger to log to that file as well

    eg.
    >>> with temporarily_add_file_logger_to("log.txt", [logger]):
    >>>     logger.info("foo")
    """
    # create log file
    io_utils.guarantee_exists(filename)
    # add log handler
    file_logger = logging.FileHandler(filename)
    try:
        for logger in loggers:
            file_logger.setFormatter(logger.handlers[0].formatter)
            logger.addHandler(file_logger)
        yield
    finally:
        # remove log handler
        for logger in loggers:
            logger.removeHandler(file_logger)


@contextlib.contextmanager
def temporarily_add_to_path(directory_path):
    if directory_path in sys.path:
        yield
    else:
        sys.path.insert(0, directory_path)
        try:
            yield
        finally:
            sys.path.remove(directory_path)


class TrialState(object):

    def __init__(self,
                 trial_name,
                 iteration_num,
                 trials_dir=None):
        """
        - container for global state of the current trial
          - eg. trial specific directory for temporary files
        - can store parameters for the current trial, as well as output values
        """
        if trials_dir is None:
            trials_dir = utils.config["trial"]["trials_dir"]
        self.trials_dir = trials_dir
        self.trial_name = trial_name
        self.iteration_num = iteration_num

        self.trial_base_path_ = os.path.join(trials_dir, trial_name)
        self.store_file_ = path_in_trial_dir(self.trial_base_path_,
                                             self.iteration_num,
                                             "stored.yml")
        self.src_path_ = path_in_trial_dir(self.trial_base_path_,
                                           self.iteration_num,
                                           "src")
        self.files_path_ = path_in_trial_dir(self.trial_base_path_,
                                             self.iteration_num,
                                             "files")
        self.tmp_path_ = path_in_trial_dir(self.trial_base_path_,
                                           self.iteration_num,
                                           "tmp")
        self.state_ = dict(
            important=defaultdict(list),
            unimportant=defaultdict(list),
        )

    def tmp_path(self, *ps):
        """
        creates a file path in the experiment's tmp directory
        """
        assert os.path.isdir(self.tmp_path_)
        return os.path.join(self.tmp_path_, *ps)

    def file_path(self, *ps):
        """
        creates a file path in the experiment's file directory
        """
        assert ps
        return os.path.join(self.files_path_, *ps)

    def src_path(self, *ps):
        """
        creates a file path in the experiment's src directory
        """
        assert ps
        return os.path.join(self.src_path_, *ps)

    def module(self, module_name):
        import warnings
        warnings.warn("TrialState.module is deprecated, "
                      "use TrialState.load_module instead")
        return self.load_module(module_name)

    def load_module(self, module_name):
        with temporarily_add_to_path(self.src_path_):
            utils.simple_debug("(Trial:%s:%d) loading module: %s",
                               self.trial_name,
                               self.iteration_num,
                               module_name)
            return importlib.import_module(module_name)

    def store(self, key, value, important=False, silent=False):
        if not silent:
            utils.simple_debug("(Trial:%s:%d) %s = %s",
                               self.trial_name,
                               self.iteration_num,
                               key,
                               value)
        self.state_[
            "important" if important else "unimportant"
        ][key].append(value)

    def store_important(self, key, value, **kwargs):
        return self.store(key, value, important=True, **kwargs)

    def get(self, key, important=False):
        return self.state_["important" if important else "unimportant"][key]

    def get_important(self, key):
        return self.get(key, important=True)

    def delete(self):
        """
        removes the current trial from both the yaml_db and deletes the trial
        directory
        """
        # delete trial from yaml_db
        with trial_db_transaction(self.trial_base_path_) as db:
            indices = []
            for idx, m in enumerate(db):
                if m["iteration_num"] == self.iteration_num:
                    indices.append(idx)
            assert len(indices) == 1
            db.pop(indices[0])
        # delete trial directory
        shutil.rmtree(path_in_trial_dir(self.trial_base_path_,
                                        self.iteration_num))

    def dump(self):
        """
        saves state to trial (both stored.yml and trial.yml)
        """
        # dump state to stored.yml
        io_utils.yaml_dump(self.state_, self.store_file_)
        # commit to trial.yml yaml_db
        self.commit()

    def load(self):
        """
        NOTE: this is somewhat inconsistent with TrialState knowing where/how
        to load itself, while run_trial knows how to save it
        (so the logic of where the file is is in 2 different places)
        """
        state = io_utils.yaml_load(self.store_file_)
        # NOTE: doesn't load as defaultdict's
        self.state_ = state

    def commit(self):
        with trial_db_iteration_transaction(self.trial_base_path_,
                                            self.iteration_num) as m:
            m["important"] = self.state_["important"]


@contextlib.contextmanager
def _run_trial_internal(trial_name,
                        iteration_num=None,
                        description=None,
                        snippets=None,
                        trials_dir=utils.config["trial"]["trials_dir"],
                        loggers="default",
                        random_seed=42,
                        trial_runner_string=None,
                        replace_strategy=None):
    """
    trial_name:
    name of the trial as a string

    iteration_num:
    integer of which the iteration in the current trial

    description:
    any json encodable object (eg. string or list)

    snippets:
    list of pairs of a snippet name (used to import the snippet) and
    a source file used in this trial
    """
    # handling default values
    if description is None:
        description = []
    if snippets is None:
        snippets = []
    # expand the trials_dir
    trials_dir = os.path.realpath(trials_dir)
    if loggers == "default":
        loggers = [utils.DEFAULT_LOGGER,
                   utils.SIMPLE_LOGGER]

    # validation
    assert re.match(r'^[A-Za-z0-9_\-]+$', trial_name), trial_name
    for snippet_name, snippet_path in snippets:
        assert isinstance(snippet_name, str)
        assert "." not in snippet_name
        assert isinstance(snippet_path, str)
    snippet_names = set([snippet[0] for snippet in snippets])
    assert len(snippets) == len(snippet_names),\
        "Snippet names must be unique"
    assert "trial_runner" not in snippet_names

    # make trials dir if doesn't exist
    io_utils.guarantee_dir_exists(trials_dir)

    trial_base_path = os.path.join(trials_dir, trial_name)
    # make trial base dir  if doesn't exist
    io_utils.guarantee_dir_exists(trial_base_path)

    # make yaml_db if doesn't exist
    io_utils.guarantee_exists(os.path.join(trial_base_path, "trial.yml"))

    start_date = datetime.datetime.now()

    iteration_num = create_trial_dir(trial_base_path,
                                     iteration_num,
                                     replace_strategy)

    path_in_this_trial_dir = utils.partial(path_in_trial_dir,
                                           trial_base_path,
                                           iteration_num)
    # file paths
    src_path = path_in_this_trial_dir("src")
    tmp_path = path_in_this_trial_dir("tmp")
    files_path = path_in_this_trial_dir("files")
    params_path = path_in_this_trial_dir("params.yml")
    log_path = path_in_this_trial_dir("log.txt")

    # create directories
    for dirname in [src_path,
                    files_path]:
        os.mkdir(dirname)

    # copy files
    for snippet_name, snippet_path in snippets:
        new_snippet_path = path_in_this_trial_dir(
            "src", snippet_name + ".py")
        shutil.copy(snippet_path, new_snippet_path)

    # write down the string to create the trial
    if trial_runner_string is not None:
        trial_runner_path = path_in_this_trial_dir("src", "trial_runner.py")
        assert not os.path.exists(trial_runner_path)
        with open(trial_runner_path, 'w') as f:
            f.write(trial_runner_string)

    # writing description to trial db
    with trial_db_iteration_transaction(trial_base_path, iteration_num) as m:
        m["description"] = description

    # create trial state
    trial = TrialState(trial_name=trial_name,
                       iteration_num=iteration_num,
                       trials_dir=trials_dir)

    utils.simple_info("Running trial %s:%d on pid %d"
                      % (trial_name, iteration_num, os.getpid()))

    try:
        with random_utils.seed_random(random_seed):
            with io_utils.temporary_directory(tmp_path):
                with temporarily_add_file_logger_to(log_path, loggers):
                    # capture stdout and stderr as well as loggers,
                    # so that all printing gets logged
                    with io_utils.Tee(log_path,
                                      "a",
                                      "stderr",
                                      auto_flush=True):
                        with io_utils.Tee(log_path,
                                          "a",
                                          "stdout",
                                          auto_flush=True):
                            # execute trial
                            yield trial
    finally:
        # save params / state to persistent storage
        trial.dump()
        io_utils.yaml_dump(dict(
            trial_name=trial_name,
            iteration_num=iteration_num,
            snippets=snippets,
            description=description,
            random_seed=random_seed,
            start_date=str(start_date),
            end_date=str(datetime.datetime.now())
        ),
            params_path)


def run_trial(*args, **kwargs):
    """
    wrapper around _run_trial_internal that reads the file that this function
    was called from and saves its contents as a string

    see docstring of _run_trial_internal for arguments
    """
    assert "trial_runner_string" not in kwargs

    # read file that this function is called from
    # ---
    # this might be sketchy using inspect and is not REPL
    # friendly
    current_frame = inspect.currentframe()
    outer_frames = inspect.getouterframes(current_frame)
    caller_frames = toolz.thread_last(
        outer_frames,
        # not the first frame
        (toolz.drop, 1),
        # not in contextlib
        (filter, lambda x: "contextlib.py" not in x[1]),
    )
    caller_frame_tuple = caller_frames[0]
    caller_frame = caller_frame_tuple[0]
    runner_str = inspect.getsource(inspect.getmodule(caller_frame))
    # just a sanity check that it worked
    assert "run_trial(" in runner_str
    return _run_trial_internal(*args, trial_runner_string=runner_str, **kwargs)


def run_trial_function(trial_function, args=None, kwargs=None, **_kwargs):
    """
    wrapper around _run_trial_internal that saves the source code of the given
    function as a string, and calls the function with a TrialState object

    args:
    positional arguments to pass into trial_function

    kwargs:
    keyword arguments to pass into trial_function

    see docstring of _run_trial_internal for arguments
    """
    assert "trial_runner_string" not in _kwargs
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    func_str = "".join(inspect.getsourcelines(trial_function)[0])
    with _run_trial_internal(trial_runner_string=func_str,
                             **_kwargs) as trial:
        return trial_function(trial, *args, **kwargs)

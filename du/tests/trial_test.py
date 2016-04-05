import os
import random
import time

import numpy as np

import du
import du._test_utils
from du._test_utils import equal, hash_equal


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def in_path(*ps):
    return os.path.join(CURRENT_DIR, *ps)


def in_trials_dir(*ps):
    return os.path.join(du.config["trial"]["trials_dir"], *ps)


def test_trial_exception():
    with du.io_utils.clear_file_after(in_trials_dir()):
        try:
            with du.trial.run_trial(
                    "foobar",
                    description="this is a test",
                    snippets=[["foofoo", in_path("trial", "foo.py")],
                              ["bar", in_path("trial", "bar.py")]]
            ) as trial:
                foo = trial.module("foofoo")
                bar = trial.module("bar").Bar(foo)
                bar.run()
        except ValueError as e:
            assert e.message == "foobar", e
            assert bar.bleh == 3
        else:
            assert False


def test_trial_files_path():
    secret_string = "floop"
    with du.io_utils.clear_file_after(in_trials_dir()):
        with du.trial.run_trial("foobar", 1) as trial:
            with open(trial.file_path("choo"), 'w') as f:
                f.write(secret_string)

        # loading file written in trial
        trial2 = du.trial.TrialState("foobar", 1)
        with open(trial2.file_path("choo")) as f:
            equal(secret_string,
                  f.read())


def test_trial_stored():
    with du.io_utils.clear_file_after(in_trials_dir()):
        with du.trial.run_trial("foobar", 1) as trial:
            trial.store("foo", 42)
            trial.store("foo", 43)
            trial.store("bar", [1, 2, 3])
            trial.store_important("foo", 44)

        trial2 = du.trial.TrialState("foobar", 1)
        equal(trial2.get("foo"), [])
        trial2.load()
        equal(trial2.get("foo"), [42, 43])
        equal(trial2.get("bar"), [[1, 2, 3]])
        equal(trial2.get_important("foo"), [44])


def test_trial_db():
    with du.io_utils.clear_file_after(in_trials_dir()):
        with du.trial.run_trial("foobar", 1) as trial:
            trial.store("foo", 42)
            trial.store_important("foo", 44)

        with du.trial.trial_db_iteration_transaction(
                in_trials_dir("foobar"), 1) as m:
            equal(m["important"], {"foo": [44]})

        trial.store_important("foo", 100)

        with du.trial.trial_db_iteration_transaction(
                in_trials_dir("foobar"), 1) as m:
            equal(m["important"], {"foo": [44]})

        trial.dump()
        with du.trial.trial_db_iteration_transaction(
                in_trials_dir("foobar"), 1) as m:
            equal(m["important"], {"foo": [44, 100]})


def test_trial_delete():
    with du.io_utils.clear_file_after(in_trials_dir()):
        with du.trial.run_trial("foobar", 55):
            pass

        with du.trial.run_trial("foobar", 100):
            pass

        with du.trial.trial_db_transaction(in_trials_dir("foobar")) as db:
            assert len(db) == 2
            assert db[0]["iteration_num"] == 55
            assert db[1]["iteration_num"] == 100

        du.trial.TrialState("foobar", 55).delete()

        with du.trial.trial_db_transaction(in_trials_dir("foobar")) as db:
            assert len(db) == 1
            assert db[0]["iteration_num"] == 100


def test_trial_random_seed():
    floop = []

    def foo():
        floop.append((np.random.randn(10), random.gauss(3, 4.5)))

    with du.io_utils.clear_file_after(in_trials_dir()):
        with du.trial.run_trial("foobar", random_seed=1):
            foo()

        with du.trial.run_trial("foobar", random_seed=1):
            foo()

        with du.trial.run_trial("foobar", random_seed=2):
            foo()

    hash_equal(floop[0], floop[1])
    try:
        hash_equal(floop[1], floop[2])
    except AssertionError:
        pass
    else:
        assert False


def test_run_trial_function():
    secret_string = "floop"

    def trial_fn(trial):
        with open(trial.file_path("choo"), 'w') as f:
            f.write(secret_string)

    with du.io_utils.clear_file_after(in_trials_dir()):
        du.trial.run_trial_function(trial_fn,
                                    trial_name="foobar",
                                    iteration_num=1)

        # loading file written in trial
        trial2 = du.trial.TrialState("foobar", 1)
        with open(trial2.file_path("choo")) as f:
            equal(secret_string,
                  f.read())

        # make sure function string exists
        with open(trial2.src_path("trial_runner.py")) as f:
            content = f.read()
        # these parts of the function should be in there:
        assert """def trial_fn(trial):""" in content
        assert """with open(trial.file_path("choo"), 'w') as f:""" in content
        assert """f.write(secret_string)""" in content
        # this part should not be:
        assert "run_trial(" not in content


def test_run_trial_function_args_kwargs():
    state = [False]

    def trial_fn(trial, x, y):
        equal(x, 3)
        equal(y, 4)
        # to check that the function is called
        state[0] = True

    with du.io_utils.clear_file_after(in_trials_dir()):
        du.trial.run_trial_function(trial_fn,
                                    args=(3,),
                                    kwargs=dict(y=4),
                                    trial_name="foobar",
                                    iteration_num=1)
    equal(state, [True])


def test_run_trial_race_condition():
    template = """
import time
import du
time.sleep(2)
with du.trial.run_trial("thisisatest"):
    pass
"""
    with du.io_utils.temporary_directory("__race_condition"):
        with du.io_utils.chdir("__race_condition"):
            with open("tmp.py", "w") as f:
                f.write(template)
            with open("tmp.py") as f:
                assert f.read() == template
            os.system("python tmp.py &")
            time.sleep(1)
            with open("tmp.py", "w") as f:
                f.write("FOO")
            time.sleep(2)
            with open("_trials_dir_/thisisatest/1/src/trial_runner.py") as f:
                assert f.read() == template

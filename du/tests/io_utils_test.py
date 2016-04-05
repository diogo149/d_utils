import os

import numpy as np

import du.io_utils

import nose.tools as nt

DEFAULT_FILE_NAME = "__io_utils_test_default_file_name__"


def test_guarantee_exists():
    filename = DEFAULT_FILE_NAME
    with du.io_utils.clear_file_after(filename):
        du.io_utils.guarantee_exists(filename)
        assert os.path.isfile(filename)
        du.io_utils.guarantee_exists(filename)
        assert os.path.isfile(filename)


def test_guarantee_dir_exists():
    filename = DEFAULT_FILE_NAME
    with du.io_utils.clear_file_after(filename):
        du.io_utils.guarantee_dir_exists(filename)
        assert os.path.isdir(filename)
        du.io_utils.guarantee_dir_exists(filename)
        assert os.path.isdir(filename)


def test_json_dumps_numpy_array():
    assert du.io_utils.json_dumps(np.array([4])) == "[4]"


def test_json_dumps_numpy_array_element():
    """
    this actually fails with PyYAML
    """
    assert du.io_utils.json_dumps(np.array([4])[0]) == "4"
    assert du.io_utils.json_dumps(np.array([4], dtype=np.int64)[0]) == "4"


def test_yaml_loading_and_dumping():
    # writing to a string
    arr = np.array([1, 2, 3])
    nt.eq_(du.io_utils.yaml_loads(du.io_utils.yaml_dumps(arr)),
           [1, 2, 3])

    # writing to a file
    filename = DEFAULT_FILE_NAME
    assert not os.path.isfile(filename)

    du.io_utils.yaml_dump(np.array([4, 5, 6]), filename)
    nt.eq_(du.io_utils.yaml_load(filename),
           [4, 5, 6])

    os.remove(filename)
    assert not os.path.isfile(filename)


def test_tmp_chmod_no_write():
    filename = DEFAULT_FILE_NAME
    try:
        # testing that reading works
        with du.io_utils.tmp_chmod_no_write(filename):
            with open(filename) as f:
                f.read()
        # testing that attempting to write errors out
        try:
            with du.io_utils.tmp_chmod_no_write(filename):
                with open(filename, 'w') as f:
                    f.write("blah")
            assert False, "Shouldn't make it here"
        except IOError:
            pass
    finally:
        os.remove(filename)


def test_atomically_writable_file():
    filename = DEFAULT_FILE_NAME
    msg = "secret message"

    def read_file():
        with open(filename) as f:
            return f.read()

    try:
        with du.io_utils.tmp_chmod_no_write(filename):
            # file should be empty
            assert read_file() == ""
            with du.io_utils.atomically_writable_file(
                    filename,
                    force_overwrite=True) as f:
                f.write(msg)
                # file should still be empty
                assert read_file() == ""
            # file should have content now
            assert read_file() == msg
        # file should still have content
        assert read_file() == msg
    finally:
        os.remove(filename)

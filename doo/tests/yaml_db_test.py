import multiprocessing as mp

import doo.yaml_db
import doo._test_utils
from doo._test_utils import eq_

DEFAULT_FILE_NAME = "__yaml_db_default_file_name__"


def test_write_then_read():
    with doo.io_utils.clear_file_after(DEFAULT_FILE_NAME + ".lock"):
        with doo.io_utils.clear_file_after(DEFAULT_FILE_NAME):
            with doo.yaml_db.db_transaction(DEFAULT_FILE_NAME) as db:
                db.append([3, 4, 5])
            eq_(doo.yaml_db.read_db(DEFAULT_FILE_NAME),
                [[3, 4, 5]])


def _test_parallel_writers_write_to_db(_):
    print("foo")
    for i in range(10):
        with doo.yaml_db.db_transaction(DEFAULT_FILE_NAME) as db:
            db.append(i)


@doo._test_utils.slow
def test_parallel_writers():
    with doo.io_utils.clear_file_after(DEFAULT_FILE_NAME + ".lock"):
        with doo.io_utils.clear_file_after(DEFAULT_FILE_NAME):
            pool = mp.Pool(10)
            pool.map(_test_parallel_writers_write_to_db, range(10))

            db = doo.yaml_db.read_db(DEFAULT_FILE_NAME)

            eq_(doo.toolz.frequencies(db),
                {i: 10 for i in range(10)})

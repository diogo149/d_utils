import multiprocessing as mp

import du.yaml_db
import du._test_utils
from du._test_utils import eq_

DEFAULT_FILE_NAME = "__yaml_db_default_file_name__"


def test_write_then_read():
    with du.io_utils.clear_file_after(DEFAULT_FILE_NAME + ".lock"):
        with du.io_utils.clear_file_after(DEFAULT_FILE_NAME):
            with du.yaml_db.db_transaction(DEFAULT_FILE_NAME) as db:
                db.append([3, 4, 5])
            eq_(du.yaml_db.read_db(DEFAULT_FILE_NAME),
                [[3, 4, 5]])


def _test_parallel_writers_write_to_db(_):
    print("foo")
    for i in range(10):
        with du.yaml_db.db_transaction(DEFAULT_FILE_NAME) as db:
            db.append(i)


@du._test_utils.slow
def test_parallel_writers():
    with du.io_utils.clear_file_after(DEFAULT_FILE_NAME + ".lock"):
        with du.io_utils.clear_file_after(DEFAULT_FILE_NAME):
            pool = mp.Pool(10)
            pool.map(_test_parallel_writers_write_to_db, range(10))

            db = du.yaml_db.read_db(DEFAULT_FILE_NAME)

            eq_(du.toolz.frequencies(db),
                {i: 10 for i in range(10)})

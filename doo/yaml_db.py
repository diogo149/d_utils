import contextlib

from . import io_utils


def read_db(filename):
    """
    for a read-only snapshot of the db
    """
    with open(filename) as f:
        content = f.read()
        return list(io_utils.yaml_loads_all(content))


@contextlib.contextmanager
def db_transaction(filename):
    """
    returns the db as a mutable list, which is then written back to the
    file on exit
    """
    with io_utils.human_file_lock(filename):
        with io_utils.atomically_writable_file(filename,
                                               force_overwrite=True) as f:
            data = read_db(filename)
            try:
                yield data
            finally:
                io_utils.yaml_dump_all(data,
                                       f,
                                       safe=True,
                                       default_flow_style=False)

import os
import shutil
import getpass
import du


def copy_disk_dir_to_cache_loc(disk_dir, cache_dir, cache_loc):
    """
    takes in a full path to a directory on disk and a relative path
    to a target directory, and copies the disk dir to
    cache_loc/$USER/`cache_dir`
    (if that directory is not yet present)

    returns the full path of the new directory

    use case:
    - copying directory to /dev/shm for faster reading
    """
    assert "/" not in cache_dir
    user = getpass.getuser()
    base_cache_dir = os.path.join(cache_loc, user)
    du.io_utils.guarantee_dir_exists(base_cache_dir)
    data_dir = os.path.join(base_cache_dir, cache_dir)
    if not os.path.isdir(data_dir):
        du.info("Copying %s to %s" % (disk_dir, data_dir))
        du.info("Warning: this will create a directory that will NOT be "
                "cleaned up automatically.")
        with du.timer("Copying %s to %s" % (disk_dir, data_dir)):
            res = os.system("rsync -r %s %s" % (disk_dir, base_cache_dir))
        if res != 0:
            du.info("Copying failed, attempting to delete incomplete data")
            shutil.rmtree(data_dir)
            raise Exception("Copying failed")
    return data_dir

import os
import urllib

from .. import io_utils

DATASETS_REPO_BASE = "https://raw.githubusercontent.com/diogo149/datasets/master/"


def try_download_file(url, path):
    if not os.path.isfile(path):
        base_dir = os.path.dirname(path)
        io_utils.guarantee_dir_exists(base_dir)
        print("Downloading {} to {}".format(url, path))
        urllib.urlretrieve(url, path)

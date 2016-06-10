"""
TODO
text8
http://mattmahoney.net/dc/text8.zip

wiki8
http://mattmahoney.net/dc/enwik8.zip
"""

import os
import urllib
import numpy as np

from .. import io_utils

DATASETS_REPO_BASE = "https://raw.githubusercontent.com/diogo149/datasets/master/"


def _try_download_file(url, path):
    if not os.path.isfile(path):
        base_dir = os.path.dirname(path)
        io_utils.guarantee_dir_exists(base_dir)
        print("Downloading {} to {}".format(url, path))
        urllib.urlretrieve(url, path)


def penn_treebank_char(dtype, base_dir="~/penn_treebank_char"):
    ptb_base = DATASETS_REPO_BASE + "penn_treebank/"
    files = ["ptb.char.%s.txt" % split
             for split in ("train", "valid", "test")]
    chars_list = []
    for filename in files:
        full_file = os.path.join(base_dir, filename)
        _try_download_file(url=ptb_base + filename,
                           path=full_file)
        with open("ptb.char.valid.txt") as f:
            chars = f.read().strip().split(" ")
        chars_list.append(chars)
    char_set = set(char for chars in chars_list for char in chars)
    char2idx = {char: idx for idx, char in enumerate(char_set)}
    assert len(char2idx) == 50
    train, valid, test = [np.array([char2idx[char] for char in chars],
                                   dtype=dtype)
                          for chars in chars_list]
    return [{"x": train}, {"x": valid}, {"x": test}]

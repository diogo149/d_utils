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


def one_hot_x(datamap, vocabulary_size, dtype):
    x = datamap["x"]
    new_x = np.zeros((len(x), vocabulary_size), dtype=dtype)
    new_x[np.arange(len(x)), x] = 1
    return {"x": new_x}


def batch_and_split(datamap, batch_size, sequence_length):
    total_length, = set(map(len, datamap.values()))
    assert total_length % batch_size == 0
    split_length = total_length / batch_size
    split_datamap = {k: v.reshape(batch_size, split_length, *v.shape[1:])
                     for k, v in datamap.items()}
    batches = []
    for idx in range(int(np.ceil(split_length / float(sequence_length)))):
        idxs = slice(idx * sequence_length, (idx + 1) * sequence_length)
        batches.append({k: v[:, idxs] for k, v in split_datamap.items()})
    return batches


def add_unsupervised_sequential_y(datamap):
    assert {"x"} == set(datamap.keys())
    x = datamap["x"]
    new_x = x[:-1]
    new_y = x[1:]
    return {"x": new_x, "y": new_y}


def truncate_to_batch_size(datamap, batch_size):
    total_length, = set(map(len, datamap.values()))
    new_length = (total_length // batch_size) * batch_size
    return {k: v[:new_length] for k, v in datamap.items()}


def penn_treebank_char(dtype, base_dir="~/penn_treebank_char"):
    base_dir = os.path.expanduser(base_dir)
    ptb_base = DATASETS_REPO_BASE + "penn_treebank/"
    files = ["ptb.char.%s.txt" % split
             for split in ("train", "valid", "test")]
    chars_list = []
    for filename in files:
        full_file = os.path.join(base_dir, filename)
        _try_download_file(url=ptb_base + filename,
                           path=full_file)
        with open(full_file) as f:
            chars = f.read().strip().split(" ")
        chars_list.append(chars)
    char_set = set(char for chars in chars_list for char in chars)
    char2idx = {char: idx for idx, char in enumerate(char_set)}
    assert len(char2idx) == 50
    train, valid, test = [np.array([char2idx[char] for char in chars],
                                   dtype=dtype)
                          for chars in chars_list]
    return [{"x": train}, {"x": valid}, {"x": test}]


def text8():
    pass


def wiki8():
    pass

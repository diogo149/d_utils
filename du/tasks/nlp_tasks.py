import os
import string
import subprocess
import numpy as np

from . import tasks_utils


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
    ptb_base = tasks_utils.DATASETS_REPO_BASE + "penn_treebank/"
    files = ["ptb.char.%s.txt" % split
             for split in ("train", "valid", "test")]
    chars_list = []
    for filename in files:
        full_file = os.path.join(base_dir, filename)
        tasks_utils.try_download_file(url=ptb_base + filename,
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


def penn_treebank_word(dtype, base_dir="~/penn_treebank_word"):
    base_dir = os.path.expanduser(base_dir)
    ptb_base = tasks_utils.DATASETS_REPO_BASE + "penn_treebank/"
    files = ["ptb.%s.txt" % split
             for split in ("train", "valid", "test")]
    words_list = []
    for filename in files:
        full_file = os.path.join(base_dir, filename)
        tasks_utils.try_download_file(url=ptb_base + filename,
                                      path=full_file)
        with open(full_file) as f:
            words = f.read().replace("\n", "<eos>").split()
        words_list.append(words)
    word_set = set(word for words in words_list for word in words)
    word2idx = {word: idx for idx, word in enumerate(word_set)}
    assert len(word2idx) == 10000
    train, valid, test = [np.array([word2idx[word] for word in words],
                                   dtype=dtype)
                          for words in words_list]
    return [{"x": train}, {"x": valid}, {"x": test}]


def text8(dtype, base_dir="~/text8"):
    base_dir = os.path.expanduser(base_dir)
    file_path = os.path.join(base_dir, "text8")
    if not os.path.exists(file_path):
        zip_path = os.path.join(base_dir, "text8.zip")
        tasks_utils.try_download_file(url="http://mattmahoney.net/dc/text8.zip",
                                      path=zip_path)
        subprocess.call(["unzip", zip_path, "-d", base_dir])

    with open(file_path) as f:
        data = f.read()

    num_total = int(1e8)
    num_train = int(9e7)
    num_valid = int(5e6)
    num_test = int(5e6)

    char2idx = {char: idx for idx, char in enumerate(" " + string.lowercase)}

    assert len(data) == num_total
    assert num_train + num_valid + num_test == num_total

    train = np.zeros(num_train, dtype=dtype)
    for idx in range(0, num_train):
        train[idx] = char2idx[data[idx]]
    valid = np.zeros(num_valid, dtype=dtype)
    for idx in range(0, num_valid):
        valid[idx] = char2idx[data[idx + num_train]]
    test = np.zeros(num_test, dtype=dtype)
    for idx in range(0, num_test):
        test[idx] = char2idx[data[idx + num_train + num_valid]]

    return [{"x": train}, {"x": valid}, {"x": test}]


def enwik8(dtype, base_dir="~/enwik8"):
    base_dir = os.path.expanduser(base_dir)
    file_path = os.path.join(base_dir, "enwik8")
    if not os.path.exists(file_path):
        zip_path = os.path.join(base_dir, "enwik8.zip")
        tasks_utils.try_download_file(url="http://mattmahoney.net/dc/enwik8.zip",
                                      path=zip_path)
        subprocess.call(["unzip", zip_path, "-d", base_dir])

    with open(file_path) as f:
        data = f.read()

    num_total = int(1e8)
    num_train = int(9e7)
    num_valid = int(5e6)
    num_test = int(5e6)

    byte2idx = {}

    assert len(data) == num_total
    assert num_train + num_valid + num_test == num_total

    train = np.zeros(num_train, dtype=dtype)
    for idx in range(0, num_train):
        byte = data[idx]
        if byte not in byte2idx:
            byte2idx[byte] = len(byte2idx)
        train[idx] = byte2idx[byte]
    assert len(byte2idx) == 205
    valid = np.zeros(num_valid, dtype=dtype)
    for idx in range(0, num_valid):
        valid[idx] = byte2idx[data[idx + num_train]]
    test = np.zeros(num_test, dtype=dtype)
    for idx in range(0, num_test):
        test[idx] = byte2idx[data[idx + num_train + num_valid]]

    return [{"x": train}, {"x": valid}, {"x": test}]

import os
import urllib
import numpy as np
import sklearn.model_selection

from .. import io_utils

DATASETS_REPO_BASE = "https://raw.githubusercontent.com/diogo149/datasets/master/"


def try_download_file(url, path):
    if not os.path.isfile(path):
        base_dir = os.path.dirname(path)
        io_utils.guarantee_dir_exists(base_dir)
        print("Downloading {} to {}".format(url, path))
        urllib.urlretrieve(url, path)


def shuffle_datamap(datamap, random_state=None):
    lens = map(len, datamap.values())
    # make sure all have the same length
    assert len(set(lens)) == 1

    order = np.arange(lens[0])
    rng = np.random.RandomState(seed=random_state)
    rng.shuffle(order)
    return {k: v[order] for k, v in datamap.items()}


def train_test_split_datamap(datamap,
                             test_size=None,
                             train_size=None,
                             random_state=None,
                             stratify=None):
    """
    test_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the test split. If
        int, represents the absolute number of test samples. If None,
        the value is automatically set to the complement of the train size.
        If train size is also None, test size is set to 0.25.

    train_size : float, int, or None (default is None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.

    random_state : int or RandomState
        Pseudo-random number generator state used for random sampling.

    stratify : key into datamap or None
        If not None, data is split in a stratified fashion, using the value
        of the key in the datamap as the class labels.
    """
    keys = list(datamap.keys())
    values = [datamap[k] for k in keys]
    if stratify is not None:
        stratify = datamap[stratify]
    new_values = sklearn.model_selection.train_test_split(
        *values,
        test_size=test_size,
        train_size=train_size,
        random_state=random_state,
        stratify=stratify)
    train_datamap = dict(zip(keys, new_values[::2]))
    test_datamap = dict(zip(keys, new_values[1::2]))
    return train_datamap, test_datamap


def split_datamap(datamap, size):
    """
    splits a datamap into multiple datamaps along the first dimension
    """
    total_size = len(datamap.values()[0])
    for v in datamap.values():
        assert len(v) == total_size
    splits = []
    for idx in range(int(np.ceil(float(total_size) / size))):
        tmp = {}
        for k, v in datamap.items():
            assert len(v) == total_size
            tmp[k] = v[size * idx: size * (idx + 1)]
        splits.append(tmp)
    return splits


def random_sample_datamap(datamap, size):
    """
    returns a generator that samples from a datamap along the first dimension
    """
    total_size = len(datamap.values()[0])
    for v in datamap.values():
        assert len(v) == total_size
    while True:
        res = {k: [] for k in datamap}
        for _ in range(size):
            # TODO allow seeding rng
            idx = np.random.randint(total_size)
            for k, v in datamap.items():
                res[k].append(v[idx])
        # TODO should this always cast to an array
        res = {k: np.array(v) for k, v in res.items()}
        yield res

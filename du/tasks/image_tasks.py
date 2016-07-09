import os
import pickle
import subprocess
import numpy as np
import scipy.io
import sklearn.datasets
import sklearn.cross_validation

from . import tasks_utils


def _shuffle_batch_axis(datamap, random_state):
    lens = map(len, datamap.values())
    # make sure all have the same length
    assert len(set(lens)) == 1

    order = np.arange(lens[0])
    rng = np.random.RandomState(seed=random_state)
    rng.shuffle(order)
    return {k: v[order] for k, v in datamap.items()}


# ################################## mnist ##################################


def mnist(dtype,
          include_valid_split=True,
          random_state=42,
          shuffle_train=True):
    """
    x is in [0, 1] with shape (b, 1, 28, 28) and dtype floatX
    y is an int32 vector in range(10)
    """
    raw = sklearn.datasets.fetch_mldata('MNIST original')
    # rescaling to [0, 1] instead of [0, 255]
    x = raw['data'].reshape(-1, 1, 28, 28).astype(dtype) / 255.0
    y = raw['target'].astype("int32")
    test = {"x": x[60000:], "y": y[60000:]}
    if include_valid_split:
        # NOTE: train data is initially in order of 0 through 9
        x1, x2, y1, y2 = sklearn.cross_validation.train_test_split(
            x[:60000],
            y[:60000],
            random_state=random_state,
            test_size=10000)
        train = {"x": x1, "y": y1}
        valid = {"x": x2, "y": y2}
        if shuffle_train:
            train = _shuffle_batch_axis(train, random_state)
        # NOTE: test data is in order of 0 through 9
        return train, valid, test
    else:
        train = {"x": x[:60000], "y": y[:60000]}
        if shuffle_train:
            train = _shuffle_batch_axis(train, random_state)
        return train, test


def cluttered_mnist(base_dir="~/cluttered_mnist"):
    base_dir = os.path.expanduser(base_dir)
    # use the one from lasagne:
    # https://github.com/Lasagne/Recipes/blob/master/examples/spatial_transformer_network.ipynb
    url = ("https://s3.amazonaws.com/lasagne/recipes/"
           "datasets/mnist_cluttered_60x60_6distortions.npz")
    path = os.path.join(base_dir, "mnist_cluttered_60x60_6distortions.npz")
    tasks_utils.try_download_file(url, path)
    data = np.load(path)
    X_train, X_valid, X_test = [data[n].reshape((-1, 1, 60, 60))
                                for n in ["x_train", "x_valid", "x_test"]]
    y_train, y_valid, y_test = [np.argmax(data[n], axis=-1).astype('int32')
                                for n in ["y_train", "y_valid", "y_test"]]
    train = {"x": X_train, "y": y_train}
    valid = {"x": X_valid, "y": y_valid}
    test = {"x": X_test, "y": y_test}
    return train, valid, test

# ################################### svhn ###################################


def svhn(dtype,
         base_dir="~/svhn",
         include_extras=False,
         include_valid_split=False):
    """
    loads svhn data from http://ufldl.stanford.edu/housenumbers/
    """
    # TODO
    assert not include_extras
    assert not include_extras
    base_dir = os.path.expanduser(base_dir)

    def load_mat(filename):
        mat = scipy.io.loadmat(filename)
        # has shape: (32, 32, 3, num_examples)
        x = mat["X"]
        # has shape: (num_examples, 1)
        y = mat["y"]

        # reformat
        x = x.transpose(3, 2, 0, 1).astype(dtype) / 255.0
        # TODO flag for y dtype
        y = y.ravel().astype("int32")

        return x, y

    def download_and_load_mat(url):
        path = os.path.basename(url)
        path = os.path.join(base_dir, path)
        tasks_utils.try_download_file(url=url,
                                      path=path)
        return load_mat(path)

    train_x, train_y = download_and_load_mat(
        "http://ufldl.stanford.edu/housenumbers/train_32x32.mat")
    test_x, test_y = download_and_load_mat(
        "http://ufldl.stanford.edu/housenumbers/test_32x32.mat")
    # TODO flag to use extras:
    # "http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"

    train = {"x": train_x, "y": train_y}
    test = {"x": test_x, "y": test_y}

    return train, test

# ############################### CIFAR-10/100 ###############################


def cifar10(dtype, random_state=42, base_dir="~/cifar10", include_valid_split=True):
    """
    x is in [0, 1] with shape (b, 3, 32, 32) and dtype floatX
    y is an int32 vector in range(10)
    """
    URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    base_dir = os.path.expanduser(base_dir)
    batch_dir = os.path.join(base_dir, "cifar-10-batches-py")
    test_batch = os.path.join(batch_dir, "test_batch")
    if not os.path.isfile(test_batch):
        tar_gz = os.path.join(base_dir, "cifar-10-python.tar.gz")
        tasks_utils.try_download_file(URL, tar_gz)
        subprocess.call(["tar", "xvzf", tar_gz, "-C", base_dir])

    def read_batch(filename):
        with open(filename, 'rb') as f:
            raw = pickle.load(f)
        x = raw["data"].reshape(-1, 3, 32, 32).astype(dtype) / 255.0
        y = np.array(raw["labels"], dtype="int32")
        return x, y

    # read test data
    test_x, test_y = read_batch(test_batch)
    test = {"x": test_x, "y": test_y}
    # read train+valid data
    xs, ys = [], []
    for i in range(1, 6):
        x, y = read_batch(os.path.join(batch_dir, "data_batch_%d" % i))
        xs.append(x)
        ys.append(y)
    # combine train+valid data
    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if include_valid_split:
        # split train and valid
        x1, x2, y1, y2 = sklearn.cross_validation.train_test_split(
            x,
            y,
            random_state=random_state,
            test_size=10000)
        train = {"x": x1, "y": y1}
        valid = {"x": x2, "y": y2}
        return train, valid, test
    else:
        train = {"x": x, "y": y}
        return train, test


def cifar100(dtype,
             random_state=42,
             base_dir="~/cifar100",
             fine_label_key="y",
             coarse_label_key=None,
             include_valid_split=True):
    """
    x is in [0, 1] with shape (b, 3, 32, 32) and dtype floatX
    y is an int32 vector in range(100)
    """
    URL = "http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    base_dir = os.path.expanduser(base_dir)
    batch_dir = os.path.join(base_dir, "cifar-100-python")
    train_file = os.path.join(batch_dir, "train")
    test_file = os.path.join(batch_dir, "test")
    if not os.path.isfile(test_file):
        tar_gz = os.path.join(base_dir, "cifar-100-python.tar.gz")
        tasks_utils.try_download_file(URL, tar_gz)
        subprocess.call(["tar", "xvzf", tar_gz, "-C", base_dir])

    def read_file(filename):
        with open(filename, 'rb') as f:
            raw = pickle.load(f)

        res = {}

        res["x"] = raw["data"].reshape(-1, 3, 32, 32).astype(dtype) / 255.0

        if fine_label_key is not None:
            res[fine_label_key] = np.array(raw["fine_labels"], dtype="int32")

        if coarse_label_key is not None:
            res[coarse_label_key] = np.array(
                raw["coarse_labels"], dtype="int32")

        return res

    # read test data
    test = read_file(test_file)
    # read train data
    old_train = read_file(train_file)
    # split train and valid
    if include_valid_split:
        train_idx, valid_idx = iter(sklearn.cross_validation.ShuffleSplit(
            len(old_train["x"]),
            n_iter=1,
            test_size=10000,
            random_state=random_state)).next()
        train = {k: v[train_idx] for k, v in old_train.iteritems()}
        valid = {k: v[valid_idx] for k, v in old_train.iteritems()}
        return train, valid, test
    else:
        return old_train, test

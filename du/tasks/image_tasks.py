import os
import pickle
import subprocess
import numpy as np
import scipy.io
import sklearn.datasets
import sklearn.model_selection

from . import tasks_utils


def subtract_per_pixel_mean(datamaps):
    # assume first is train
    pixel_mean = datamaps[0]["x"].mean(axis=0, keepdims=True)
    new_datamaps = []
    for dm in datamaps:
        dm = dict(dm)  # make a copy
        dm["x"] = dm["x"] - pixel_mean
        new_datamaps.append(dm)
    return new_datamaps


# ################################## mnist ##################################


def mnist(x_dtype,
          y_dtype,
          include_valid_split=True,
          random_state=42,
          shuffle_train=True):
    """
    x is in [0, 1] with shape (b, 1, 28, 28) and dtype floatX
    y is an int32 vector in range(10)
    """
    raw = sklearn.datasets.fetch_mldata('MNIST original')
    # rescaling to [0, 1] instead of [0, 255]
    x = raw['data'].reshape(-1, 28, 28, 1).astype(x_dtype) / 255.0
    y = raw['target'].astype(y_dtype)
    # NOTE: train data is initially in order of 0 through 9
    train = {"x": x[:60000], "y": y[:60000]}
    # NOTE: test data is in order of 0 through 9
    test = {"x": x[60000:], "y": y[60000:]}

    if shuffle_train:
        train = tasks_utils.shuffle_datamap(train, random_state=random_state)

    if include_valid_split:
        train, valid = tasks_utils.train_test_split_datamap(
            train,
            test_size=10000,
            random_state=random_state,
            stratify="y")
        return train, valid, test
    else:
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
    X_train, X_valid, X_test = [data[n].reshape((-1, 60, 60, 1))
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
    # make this match cifar10 interface
    assert False
    # TODO
    assert not include_extras
    base_dir = os.path.expanduser(base_dir)

    def load_mat(filename):
        mat = scipy.io.loadmat(filename)
        # has shape: (32, 32, 3, num_examples)
        x = mat["X"]
        # has shape: (num_examples, 1)
        y = mat["y"]

        # reformat
        x = x.transpose(3, 0, 1, 2).astype(dtype) / 255.0
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


def gen_standard_cifar10_augmentation(datamap):
    x = datamap["x"]
    y = datamap["y"]
    # create mirrored images
    x = np.concatenate((x, x[..., ::-1]))
    y = np.concatenate((y, y))
    # allocate ndarray like x (before padding)
    x_epoch = np.zeros_like(x)
    # pad feature arrays with 4 pixels on each side
    x = np.pad(x, ((0, 0), (4, 4), (4, 4), (0, 0)), mode=str("constant"))
    num_images = len(x)
    while True:
        # shuffle
        indices = np.arange(num_images)
        # random cropping of 32x32
        np.random.shuffle(indices)
        y_epoch = y[indices]
        crops = np.random.random_integers(0, high=8, size=(num_images, 2))
        for i in range(num_images):
            idx = indices[i]
            crop1, crop2 = crops[i]
            x_epoch[i] = x[idx, crop1:crop1 + 32, crop2:crop2 + 32, :]
        yield {"x": x_epoch, "y": y_epoch}


def cifar10(x_dtype,
            y_dtype,
            random_state=42,
            base_dir="~/cifar10",
            include_valid_split=True):
    """
    x is in [0, 1] with shape (b, 3, 32, 32) and dtype floatX
    y is an int vector in range(10)
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
        x = raw["data"].reshape(-1, 3, 32, 32).astype(x_dtype) / 255.0
        # transpose into TF format
        x = x.transpose(0, 2, 3, 1)
        y = np.array(raw["labels"], dtype=y_dtype)
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
    train = {"x": np.concatenate(xs, axis=0), "y": np.concatenate(ys, axis=0)}

    if include_valid_split:
        train, valid = tasks_utils.train_test_split_datamap(
            train,
            test_size=10000,
            random_state=random_state,
            stratify="y")
        return train, valid, test
    else:
        return train, test


def cifar100(dtype,
             random_state=42,
             base_dir="~/cifar100",
             fine_label_key="y",
             coarse_label_key=None,
             include_valid_split=True):
    # make this match cifar10 interface
    assert False
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
        # transpose into TF format
        res["x"] = res["x"].transpose(0, 2, 3, 1)

        if fine_label_key is not None:
            res[fine_label_key] = np.array(raw["fine_labels"], dtype="int32")

        if coarse_label_key is not None:
            res[coarse_label_key] = np.array(
                raw["coarse_labels"], dtype="int32")

        return res

    # read test data
    test = read_file(test_file)
    # read train data
    train = read_file(train_file)

    if include_valid_split:
        train, valid = tasks_utils.train_test_split_datamap(
            train,
            test_size=10000,
            random_state=random_state,
            stratify="y")
        return train, valid, test
    else:
        return train, test

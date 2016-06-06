import numpy as np
import sklearn.datasets
import sklearn.cross_validation


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

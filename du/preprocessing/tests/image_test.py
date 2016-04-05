import numpy as np
import scipy.misc
from du.preprocessing.image import (get_block_with_center_and_shape,
                                    strided_downsample,
                                    resize_antialias)
from du._test_utils import numpy_equal, numpy_almost_equal, equal, raises

lena_raw = scipy.misc.lena()
lena = lena_raw / 255.0


def test_get_block_with_center_and_shape_1():
    numpy_equal(
        [[1]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(0, 0),
            shape=(1, 1),
            fill_value=0)
    )


def test_get_block_with_center_and_shape_2():
    numpy_equal(
        [[5]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(1, 1),
            shape=(1, 1),
            fill_value=0)
    )


def test_get_block_with_center_and_shape_3():
    numpy_equal(
        [[0, 0],
         [0, 1]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(0, 0),
            shape=(2, 2),
            fill_value=0)
    )


def test_get_block_with_center_and_shape_4():
    numpy_equal(
        [[0, 0, 0],
         [0, 1, 2],
         [0, 4, 5]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(0, 0),
            shape=(3, 3),
            fill_value=0)
    )


def test_get_block_with_center_and_shape_5():
    numpy_equal(
        [[6, 6, 6],
         [6, 1, 2],
         [6, 4, 5]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(0, 0),
            shape=(3, 3),
            fill_value=6)
    )


def test_get_block_with_center_and_shape_6():
    numpy_equal(
        [[5, 6],
         [8, 9]],
        get_block_with_center_and_shape(
            np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]]),
            center=(2, 2),
            shape=(2, 2),
            fill_value=0)
    )


def test_strided_downsample():
    x = np.random.randn(12, 13, 14)
    equal(
        strided_downsample(x, (3, 4, 5)).shape,
        (4, 4, 3)
    )
    numpy_equal(
        strided_downsample(x, (3, 4, 5)),
        x[0:12:3, 0:13:4, 0:14:5]
    )
    numpy_equal(
        strided_downsample(x, (3, 4, 5)),
        x[[0, 3, 6, 9]][:, [0, 4, 8, 12]][:, :, [0, 5, 10]]
    )


@raises(Exception)
def test_resize_antialias1():
    # doesn't work for int
    resize_antialias(lena_raw, lena_raw.shape)


def test_resize_antialias2():
    numpy_almost_equal(resize_antialias(lena, lena.shape),
                       lena)


def test_resize_antialias3():
    # works for float32
    numpy_almost_equal(resize_antialias(lena.astype(np.float32), lena.shape),
                       lena)


def test_resize_antialias4():
    # works for large values
    numpy_almost_equal(resize_antialias(lena * 1000, lena.shape) / 1000,
                       lena)


def test_resize_antialias5():
    # works for negative values
    numpy_almost_equal(resize_antialias(lena * -1000, lena.shape) / -1000,
                       lena)


def test_resize_antialias6():
    # can resize then transform and results in the same value
    numpy_almost_equal(resize_antialias(lena * -1000, (50, 50)) / -1000,
                       resize_antialias(lena, (50, 50)))

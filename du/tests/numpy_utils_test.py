import tempfile
import numpy as np
import du
from du.numpy_utils import (expand_dim,
                            memmap_concatenate,
                            constant_value_array)
from du._test_utils import equal


def test_expand_dim():
    z = np.zeros((2, 2))
    x = np.arange(4).reshape(2, 2)
    assert np.alltrue(np.vstack([x, z]) == expand_dim(x, 4, axis=0))
    assert np.alltrue(np.hstack([x, z]) == expand_dim(x, 4, axis=1))


def test_memmap_concatenate():
    x = np.random.randn(3, 3)
    l = [x, x, x]
    with tempfile.TemporaryFile() as f:
        assert np.alltrue(memmap_concatenate(f, l, 0) == np.concatenate(l, 0))
        assert np.alltrue(memmap_concatenate(f, l, 1) == np.concatenate(l, 1))


def test_constant_value_array():
    equal(7.5, constant_value_array(1.5, 5).sum())
    equal(5, constant_value_array(1.5, 5, int).sum())


def test_generator_std():
    x1 = np.random.rand(100, 12)
    x2 = 0.32 * np.random.rand(100, 34) + 2.0
    x3 = 1.5 * np.random.rand(100, 100) - 1.0
    xs = [x1, x2, x3]
    np.testing.assert_allclose(du.numpy_utils.generator_std(xs, axis=1),
                               np.concatenate(xs, axis=1).std(axis=1))

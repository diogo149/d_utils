from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
from du.sandbox import fcn_utils


def test_block_apply_with_border_simple():
    X = np.random.randn(6)
    print("X: ", X)
    test_cases = []
    for bs in range(1, 8):
        for bd in [0, 1, 2, 4, 8]:
            for batch in [None, 1, 3, 7]:
                test_cases.append(((bs,), (bd,), batch))
    for block_shape, border, batch_size in test_cases:
        res = fcn_utils.block_apply_with_border(
            arr=X,
            fn=lambda x: x,
            block_shape=block_shape,
            border=border,
            batch_size=batch_size)
        print(block_shape, border, batch_size)
        np.testing.assert_equal(X, res)


def test_block_apply_with_border():
    X = np.random.randn(32, 32, 32)
    for block_shape, border, batch_size in [((32, 32, 32), (0, 0, 0), None),
                                            ((32, 32, 32), (0, 0, 0), 1),
                                            ((32, 32, 32), (0, 0, 0), 2),
                                            ((5, 5, 5), (0, 0, 0), 3),
                                            ((5, 5, 5), (2, 2, 2), 3),
                                            ((2, 3, 4), (2, 2, 2), None),
                                            ((1, 2, 3), (0, 0, 0), None)]:
        res = fcn_utils.block_apply_with_border(
            arr=X,
            fn=lambda x: x,
            block_shape=block_shape,
            border=border,
            batch_size=batch_size)
        print(block_shape, border, batch_size)
        np.testing.assert_equal(X, res)

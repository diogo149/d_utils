"""
utilities for test time augmentation
"""

import numpy as np
import du
import du.data_augmentation.image2d


def affine_2d_tta(test_fn,
                  x,
                  affine_params,
                  n_iter=10,
                  mean_weight=1,
                  median_weight=1,
                  shuffle=True):
    """
    test_fn:
    function that takes in a bc01 image and returns a numpy array of
    predictions

    x:
    bc01 format image

    affine_params:
    dict of keyword arguments for du.data_augmentation.image2d.random_affine
    """

    def augment(img):
        img = img.swapaxes(0, 2)
        img = du.data_augmentation.image2d.random_affine(
            img,
            **affine_params
        )
        return img.swapaxes(0, 2)

    all_preds = []
    for _ in range(n_iter):
        idxs = np.arange(len(x))
        # shuffle the array
        # eg. for architectures that use batch-wise normalization at test time
        if shuffle:
            np.random.shuffle(idxs)
        new_x = np.array([augment(i) for i in x])
        res = test_fn(new_x[idxs])
        inverse_idxs = idxs.argsort()
        preds = res[inverse_idxs]
        all_preds.append(preds)

    all_preds = np.array(all_preds)
    median = np.median(all_preds, axis=0)
    mean = np.mean(all_preds, axis=0)
    Z = (median_weight + mean_weight)
    return (median * median_weight + mean * mean_weight) / Z

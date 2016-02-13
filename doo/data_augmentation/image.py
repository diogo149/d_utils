"""
functions for applying random perturbations to data for the purpose of
adding more variation to computer vision datasets
"""

import numpy as np
from ..preprocessing import image as preprocessing_image


def transpose_axes(data, rng=None):
    """
    randomly transposes axes of a ndimage
    """
    if rng is None:
        rng = np.random
    x = np.arange(len(data.shape))
    rng.shuffle(x)
    return data.transpose(*x)


def flip_axes(data, rng=None, prob=0.5):
    """
    randomly flips (reverses the order) along the axes of a ndimage
    """
    if rng is None:
        rng = np.random
    if not isinstance(prob, (tuple, list)):
        prob = (prob,) * data.ndim
    flips = [-1 if rng.rand() < p else 1 for p in prob]
    slices = [slice(None, None, flip) for flip in flips]
    return data[slices]


def multi_flip_axes(imgs, rng=None, prob=0.5):
    """
    randomly flips (reverses the order) along the axes of several ndimages
    (the same flipping for each ndimage)
    """
    for img in imgs:
        assert img.ndim == imgs[0].ndim, dict(
            ndims=[img.ndim for img in imgs]
        )
    if rng is None:
        rng = np.random
    if not isinstance(prob, (tuple, list)):
        prob = (prob,) * imgs[0].ndim
    flips = [-1 if rng.rand() < p else 1 for p in prob]
    slices = [slice(None, None, flip) for flip in flips]
    return [img[slices] for img in imgs]


def random_block(data, shape, rng=None):
    """
    Returns a random block of the original data of shape `shape` that
    fully fits within the origin data block
    """
    if rng is None:
        rng = np.random
    d_shape = data.shape
    assert len(d_shape) == len(shape)
    corner = [rng.randint(diff + 1)
              for diff in (np.array(d_shape) - np.array(shape))]
    slices = [slice(c, c + size) for c, size in zip(corner, shape)]
    return data[slices]


def random_block_around_center(data,
                               center,
                               shape,
                               noise_type="randint",
                               noise_magnitude=0,
                               fill_value=0,
                               rng=None):
    """
    Returns a random block of the original data of shape `shape` that
    fully fits within the origin data block, with the distance from the center
    being parameterized by the given noise
    """
    if rng is None:
        rng = np.random
    try:
        len(noise_magnitude)
    except TypeError:
        noise_magnitude = [noise_magnitude] * len(center)

    if noise_type == "randint":
        noise = [rng.randint(-m, m + 1) for m in noise_magnitude]
        new_center = np.array(center) + noise
    else:
        raise ValueError("Unsupported noise: %s" % noise_type)
    new_center = [int(np.round(coord)) for coord in new_center]
    return preprocessing_image.get_block_with_center_and_shape(
        data,
        new_center,
        shape,
        fill_value=fill_value
    )

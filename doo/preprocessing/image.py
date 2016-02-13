"""
functions for deterministically preprocessing images mostly for the
consumption of computer vision algorithms
"""

from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals


import math
import numpy as np
from scipy import ndimage
import PIL

from .. import numpy_utils


def get_block_with_corner_and_shape(data, corner, shape, fill_value=0):
    """
    takes the block with the given top corner and shape from the input data
    if the input data block is too small, pad it with the provided fill value
    """
    assert isinstance(shape, tuple), shape
    assert len(shape) == len(data.shape), {
        "shape": shape,
        "data.shape": data.shape,
    }

    maxed_corner = [max(0, c) for c in corner]
    slices = [slice(mc, c + size)
              for c, mc, size in zip(corner, maxed_corner, shape)]
    # index with tuple allows slice to work with hdf5 dataset
    sliced = data[tuple(slices)]
    if sliced.shape == shape:
        res = sliced
    else:
        res = numpy_utils.constant_value_array(fill_value, shape, data.dtype)
        inner_slices = [
            slice(mc - c, mc - c + s)
            for c, mc, s in zip(corner, maxed_corner, sliced.shape)
        ]
        res[tuple(inner_slices)] = sliced
    assert res.shape == shape, dict(
        res_shape=res.shape,
        desired_shape=shape,
        sliced_shape=sliced.shape,
        slices=slices
    )
    return res


def get_block_with_center_and_shape(data, center, shape, fill_value=0):
    """
    takes the block with a given shape from the input data block at the
    given center. if the input data block is too small, pad it with the
    provided fill value
    """
    # validate that center is in data
    for idx, (limit, coord) in enumerate(zip(data.shape, center)):
        assert 0 <= coord < limit, dict(
            idx=idx,
            limit=limit,
            coord=coord,
            data_shape=data.shape,
            center=center
        )
    corner = [c - size // 2 for c, size in zip(center, shape)]
    return get_block_with_corner_and_shape(data=data,
                                           corner=corner,
                                           shape=shape,
                                           fill_value=fill_value)


def take_center_block(data, shape, fill_value=0):
    """
    takes the center block of the given data block, with the given shape
    """
    if data.shape == tuple(shape):
        return data
    center = tuple([s // 2 for s in data.shape])
    return get_block_with_center_and_shape(data,
                                           center=center,
                                           shape=shape,
                                           fill_value=fill_value)


def rescale(data, zoom_factor):
    """
    Up or down sample by a given factor
    """
    return ndimage.interpolation.zoom(input=data, zoom=zoom_factor)


def _rotate(data, radians, axes, reshape, mode):
    return ndimage.rotate(input=data,
                          angle=math.degrees(radians),
                          axes=axes,
                          mode=mode,
                          reshape=reshape)


def rotate(data, radians, axes=(1, 0), mode="reflect"):
    """
    rotates around the center and returns an image of the same size as the
    input image - this means that some amount of the input image is
    excluded (the corners)

    Parameters
    ----------
    data : ndarray

    radians : int
              number of radians to rotate the data

    axes : tuple of integers (default=(1, 0))
           pair that specifies which axes to rotate over
    """
    return _rotate(data, radians, axes, False, mode)


def rotate_take_center(data, radians, shape, axes=(1, 0), mode="reflect"):
    """
    rotates around the center and returns an image of the given size from
    the center of the rotated image

    Parameters
    ----------
    data : ndarray

    radians : int
              number of radians to rotate the data

    shape : tuple of integers
            desired shape of the output data

    axes : tuple of integers (default=(1, 0))
           pair that specifies which axes to rotate over
    """
    rotated = _rotate(data, radians, axes, True, mode)
    return take_center_block(rotated, shape)


def rotate_multi(data, rotations, mode="reflect"):
    """
    perform multiple rotations in a row, 1 for each pair of axes

    rotations:
    a map from the indexes of the axes to rotate over to radians to rotate
    eg. {(1, 0): 0.1, (2, 0): 0.2}

    TODO:
    Seems like this should be done with a single affine transform. Can just
    create rotation matrices for each rotation you want, then multiply those
    together and apply the result. For 3d images, would be nice to not
    reprocess multiple times.
    """
    tmp = data
    for axes, radians in rotations.items():
        tmp = rotate(tmp, radians, axes, mode=mode)
    return tmp


def strided_downsample(data, downsample_factors, offsets=0):
    """
    downsamples an image by taking every few pixels (pros: very fast, cons:
    low quality / might lead to aliasing)

    downsample_factors:
    int or tuple/list of factors to downsample by in each dimension

    offsets:
    int or tuple/list of offset locations for where downsampling begins from
    """
    if isinstance(downsample_factors, int):
        downsample_factors = (downsample_factors,) * len(data.shape)
    if isinstance(offsets, int):
        offsets = (offsets,) * len(data.shape)
    return data[[slice(offset, None, stride)
                 for offset, stride in zip(offsets, downsample_factors)]]


def resize_antialias(img, shape):
    """
    resizes an image in a way to reduce the effect of aliasing

    NOTE: doesn't seem to work when img has channels

    img:
    img must be of type float32 or float64, and returns an image of type
    float32
    """
    return np.array(PIL.Image.fromarray(img).resize(shape,
                                                    PIL.Image.ANTIALIAS))

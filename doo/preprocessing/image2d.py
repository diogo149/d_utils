"""
functions for deterministically preprocessing 2D images (or 3D with color
channels) mostly for the consumption of computer vision algorithms
"""

import math
import numpy as np
import skimage.transform
from .. import utils


def _center_coords_for_shape(shape):
    """
    returns the center of an ndimage with a given shape
    """
    return np.array(shape) / 2.0 - 0.5


def _warp_cv2(img,
              H,
              output_shape,
              mode,
              order,
              cval):
    """
    returns warped image using OpenCV2

    in a few tests, this was 5-10x faster than either of skimage's warp
    functions
    """
    # import cv2 here so that entire file doesn't have to depend on it
    from .. import cv2_utils
    # TODO handle case for other types of interpolation
    assert order == 1
    kwargs = dict(
        affine_matrix=H[:2],
        shape=output_shape,
        border_mode=mode,
        fill_value=cval,
        is_inverse_map=True
    )
    if len(img.shape) < 3 or img.shape[2] <= 4:
        # warp_affine can handle images with up to 4 channels
        return cv2_utils.warp_affine(img, **kwargs)
    else:
        # handle the case for img with many channels
        channels = img.shape[2]
        result = np.empty(output_shape + (channels,), dtype=img.dtype)
        for i in range(int(np.ceil(channels / 4.0))):
            idx = slice(i * 4, (i + 1) * 4)
            result[:, :, idx] = cv2_utils.warp_affine(img[..., idx], **kwargs)
        return result


def _warp_PIL(img,
              H,
              output_shape,
              mode,
              order,
              cval):
    """
    in some tests, 5x slower than OpenCV's affine transform
    (converting to and from PIL seems to take almost as much as performing
    the transformation)
    """
    from PIL import Image, ImageTransform
    # TODO handle filling
    assert cval == 0
    # TODO handle other modes
    assert mode == "constant"
    if order == 0:
        resample = Image.NEAREST
    elif order == 1:
        resample = Image.BILINEAR
    else:
        raise AssertionError
    transform = ImageTransform.AffineTransform(H[:2].ravel())
    return np.array(Image.fromarray(img).transform(output_shape,
                                                   transform,
                                                   resample=resample),
                    dtype=img.dtype)


def _warp_fast(img, **kwargs):
    """
    returns warped image with proper dtype
    """
    return skimage.transform._warps_cy._warp_fast(
        img,
        **kwargs
    ).astype(img.dtype)


def _warp(img, **kwargs):
    """
    returns warped image with proper dtype
    """
    return skimage.transform.warp(img, **kwargs).astype(img.dtype)


def affine_transform_fn(shape,
                        zoom=None,
                        stretch=None,
                        rotation=None,
                        shear=None,
                        translation=None,
                        output_shape=None,
                        vertical_flip=False,
                        horizontal_flip=False,
                        mode="reflect",
                        fill_value=0.0,
                        crop_center=None,
                        order=1,
                        use_cv2=True,
                        use_PIL=False):
    """
    returns a function to transform images according to the given parameters

    automatically uses skimage.transform._warps_cy._warp_fast for images w/o
    channels
    differences:
    - less parameters / customizability
    - does not work for images with color
    - a little bit faster (~15%-ish when testing it)

    shape:
    shape of the images to transform

    stretch:
    vertical stretch (to warp the aspect ratio)

    output_shape:
    desired shape of the output (default: same as input shape)

    mode:
    how to treat points outside boundary
    (default: reflect - but can be much slower than constant depending on
    amount of points past boundary)

    fill_value:
    value to fill boundary with for mode="constant"

    crop_center:
    center of the region that will be cropped
    (default: center of the image)

    order:
    order of interpolation (eg. 0=nearest neighbor, 1=bi-linear, 2=...)
    see documentation of skimage.transform.warp
    (default: 1)

    use_cv2:
    whether or not to use OpenCV warping (can be 5-10x faster)

    use_PIL:
    whether ot not ro use PIL warping
    """
    assert not (use_cv2 and use_PIL)

    if len(shape) == 2:
        fast_warp = True
    elif len(shape) == 3:
        # has color channels
        fast_warp = False
    else:
        raise ValueError

    shape = shape[:2]

    if output_shape is None:
        output_shape = shape

    # ---------------------
    # base affine transform
    # ---------------------

    if rotation is not None:
        rotation = utils.rotations_to_radians(rotation)
    if shear is not None:
        shear = utils.rotations_to_radians(shear)

    tf_kwargs = dict(
        rotation=rotation,
        shear=shear,
    )

    if translation is not None:
        # the first argument of translation changes the second axis,
        # so switch back to make it more intuitive to numpy array syntax
        vertical_translation, horizontal_translation = translation
        tf_kwargs["translation"] = (horizontal_translation,
                                    vertical_translation)

    if ((zoom is not None)
            or (stretch is not None)
            or horizontal_flip
            or vertical_flip):
        if zoom is None:
            zoom = 1
        if stretch is None:
            stretch = 1
        scale_horizontal = 1.0 / zoom
        scale_vertical = 1.0 / (zoom * stretch)
        if horizontal_flip:
            scale_horizontal *= -1
        if vertical_flip:
            scale_vertical *= -1
        tf_kwargs["scale"] = (scale_horizontal, scale_vertical)

    base_tf = skimage.transform.AffineTransform(**tf_kwargs)

    # ---------------------
    # centering/uncentering
    # ---------------------
    # by default, rotation and shearing is done relative to (0, 0), which
    # is rarely desired

    transform_center = _center_coords_for_shape(shape)

    # reverse the coordinates
    # because scikit-image takes in (x,y) in position coordinates where
    # x = axis 1, y = axis 0
    center_translation = np.array(transform_center)[::-1]
    # translate the image such that the provided center is at (0, 0)
    centering_tf = skimage.transform.SimilarityTransform(
        translation=center_translation,
    )
    # to put the original image back to where it belongs
    uncentering_tf = skimage.transform.SimilarityTransform(
        translation=-center_translation,
    )

    # apply the transformations
    tf = uncentering_tf + base_tf + centering_tf

    # --------------
    # crop centering
    # --------------
    # by default, cropping takes the top left corner, which is rarely desired
    # thus we want to translate the image such that the provided crop_center
    # will be at the center of the cropped image

    if shape != output_shape:
        if crop_center is None:
            crop_center = transform_center
        crop_center = np.array(crop_center)
        default_center = _center_coords_for_shape(output_shape)
        relative_diff = crop_center - default_center
        centering_tf = skimage.transform.SimilarityTransform(
            # reverse the order of coordinates
            translation=relative_diff[::-1],
        )
        tf = centering_tf + tf

    # ----------------------
    # applying to a function
    # ----------------------

    base_kwargs = dict(
        output_shape=output_shape,
        mode=mode,
        order=order,
        cval=fill_value,
    )
    if use_cv2:
        base_fn = _warp_cv2
        base_kwargs["H"] = tf.params
    elif use_PIL:
        base_fn = _warp_PIL
        base_kwargs["H"] = tf.params
    elif fast_warp:
        base_fn = _warp_fast
        base_kwargs["H"] = tf.params
    else:
        base_fn = _warp
        base_kwargs["inverse_map"] = tf

    return utils.partial(base_fn, **base_kwargs)


def affine_transform(img, **kwargs):
    """
    transforms an img with the given parameters (see documentation of
    affine_transform_fn)
    """
    fn = affine_transform_fn(img.shape, **kwargs)
    return fn(img)


def multi_affine_transform(imgs, **kwargs):
    """ transforms a list of images with the given parameters (see documentation
    of affine_transform_fn)
    """

    for i in range(len(imgs) - 1):
        assert imgs[i].shape == imgs[i + 1].shape
    fn = affine_transform_fn(imgs[0].shape, **kwargs)
    return map(fn, imgs)

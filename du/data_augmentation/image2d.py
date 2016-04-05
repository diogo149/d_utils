"""
functions for applying random perturbations to data for the purpose of
adding more variation to computer vision datasets, specifically for 2D images
"""

import numpy as np
import scipy.ndimage
import skimage.transform
from .. import preprocessing


def random_affine_fn(shape,
                     zoom_range=(1, 1),
                     stretch_range=(1, 1),
                     rotation_range=(0, 0),
                     shear_range=(0, 0),
                     translation_range=(0, 0),
                     vertical_flip=0,
                     horizontal_flip=0,
                     rng=None,
                     output_shape=None,
                     **kwargs):
    """
    returns a function from an image to a new image perturbed by random
    parameters in the given ranges

    zoom_range:
    pair of values to zoom between (random uniform in log scale)
    eg. (0.5, 2) with scale between half and double size

    stretch_range:
    pair of values to stretch between (random uniform in log scale)

    rotation_range:
    pair of values to rotate between (in rotations) (random uniform)

    shear_range:
    pair of values to shear between (random uniform)

    translation_range:
    takes in either a pair of values to use as a range for both axes,
    or a pair of pairs of values to use as a range for each axis
    (random uniform)
    """
    # parameter used to be a boolean for 50% probability
    assert horizontal_flip is not True
    assert vertical_flip is not True

    if rng is None:
        rng = np.random

    def uniform_rand(r):
        r_low, r_high = r
        if r_low == r_high:
            return r_low
        return rng.uniform(r_low, r_high)

    def log_uniform_rand(r):
        return np.exp(uniform_rand(np.log(r)))

    kwargs["zoom"] = log_uniform_rand(zoom_range)
    kwargs["stretch"] = log_uniform_rand(stretch_range)
    kwargs["rotation"] = uniform_rand(rotation_range)
    kwargs["shear"] = uniform_rand(shear_range)

    translation_range = np.array(translation_range)

    if translation_range.shape == (2,):
        translation_range_0 = translation_range
        translation_range_1 = translation_range
    elif translation_range.shape == (2, 2):
        translation_range_0 = translation_range[0]
        translation_range_1 = translation_range[1]

    kwargs["translation"] = (uniform_rand(translation_range_0),
                             uniform_rand(translation_range_1))

    kwargs["horizontal_flip"] = rng.uniform() < horizontal_flip
    kwargs["vertical_flip"] = rng.uniform() < vertical_flip

    return preprocessing.image2d.affine_transform_fn(
        shape=shape,
        output_shape=output_shape,
        **kwargs
    )


def random_affine(img, **kwargs):
    """
    applies a random affine transform with the given parameters (see
    documentation for random_affine_fn)
    """
    fn = random_affine_fn(img.shape, **kwargs)
    return fn(img)


def multi_random_affine(imgs, **kwargs):
    """
    applies the same random affine transform with the given parameters (see
    documentation for random_affine_fn) to a list of images
    """
    for i in range(len(imgs) - 1):
        assert imgs[i].shape == imgs[i + 1].shape
    fn = random_affine_fn(imgs[0].shape, **kwargs)
    return map(fn, imgs)


def elastic_distortion(img,
                       alpha,
                       sigma,
                       normalize_displacement=False,
                       rng=None):
    """
    alpha:
    scaling factor - controls the magnitude of the deformation
    if too large, the deformation looks affine
    (alpha=8 used for mnist)

    sigma:
    elasticity coefficient - how smooth the distortion is (higher = smoother)
    if too small, looks like a completely random field

    from "Best Practices for Convolutional Neural Networks Applied to Visual
    Document Analysis"
    """
    if rng is None:
        rng = np.random
    shape = img.shape

    displacement_field_x = scipy.ndimage.gaussian_filter(
        rng.uniform(-1, 1, size=shape), sigma=sigma).ravel()
    displacement_field_y = scipy.ndimage.gaussian_filter(
        rng.uniform(-1, 1, size=shape), sigma=sigma).ravel()

    if normalize_displacement:
        displacement_field_x /= np.linalg.norm(displacement_field_x)
        displacement_field_y /= np.linalg.norm(displacement_field_y)

    def inner(cr):
        cr[:, 0] += displacement_field_x * alpha
        cr[:, 1] += displacement_field_y * alpha
        return cr

    return skimage.transform.warp(img, inner)


def random_color_augmentation(img, sigma=0.1, rng=None):
    """
    from "ImageNet Classification with Deep Convolutional Neural Networks"
    """
    if sigma == 0:
        return img

    if rng is None:
        rng = np.random

    if img.ndim == 3:
        num_channels = img.shape[2]
    elif img.ndim == 2:
        num_channels = 1
    else:
        raise AssertionError

    reshaped_img = img.reshape(-1, num_channels)
    # calculate 3x3 convariance matrix
    cov = np.cov(reshaped_img, rowvar=0)
    # need to do this, because numpy makes the shape of a 1x1 result be ()
    cov = cov.reshape((num_channels, num_channels))
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # generate alpha and noise
    alphas = rng.normal(0, sigma, num_channels)
    noise = np.dot(eigenvectors, alphas * eigenvalues).astype(img.dtype)
    return img + noise


def random_enhancement_augmentation(img,
                                    contrast_range=None,
                                    brightness_range=None,
                                    color_range=None,
                                    sharpness_range=None,
                                    rng=None):
    """
    NOTE:
    - for PIL.ImageEnhance, a value of 1 means not changing the image at all
      - thus a reasonable range is (0.7, 1.3)
    - the enhancements in order of speed are (fastest to slowest):
      - brightness
      - contrast
      - color
      - sharpness
    """
    from PIL import Image, ImageEnhance

    if rng is None:
        rng = np.random

    # TODO factor out
    def uniform_rand(r):
        r_low, r_high = r
        if r_low == r_high:
            return r_low
        return rng.uniform(r_low, r_high)

    # FIXME make general image conversion tools (take from cv2_utils)
    as_uint8 = (img * 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[2] == 1:
        # drop last dim
        as_uint8 = as_uint8[:, :, 0]
    pil_img = Image.fromarray(as_uint8)

    # TODO parameterize order of enhancement
    order = [(contrast_range, ImageEnhance.Contrast),
             (brightness_range, ImageEnhance.Brightness),
             (color_range, ImageEnhance.Color),
             (sharpness_range, ImageEnhance.Sharpness)]
    for param_range, enhance_constructor in order:
        if param_range is not None and param_range != (1, 1):
            param_value = uniform_rand(param_range)
            pil_img = enhance_constructor(pil_img).enhance(param_value)

    res = (np.array(pil_img) / 255.0).astype(img.dtype)
    if img.ndim == 3 and img.shape[2] == 1:
        res = res[:, :, None]
    return res

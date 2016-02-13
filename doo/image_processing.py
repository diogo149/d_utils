"""
functions for processing images with more of a focus on the consumption
of output by humans (as opposed to those in preprocessing.image)
"""

import numpy as np
import scipy.ndimage


def imread_grayscale(filename):
    """
    See the following for other "mode"s:
    - http://effbot.org/imagingbook/decoder.htm
    - http://svn.effbot.org/public/tags/pil-1.1.4/libImaging/Unpack.c
    """
    return scipy.ndimage.imread(filename, mode="L")


def add_with_alpha_blending(bg, fg):
    """
    add 2 images with http://en.wikipedia.org/wiki/Alpha_compositing
    with fg on top of bg

    bg: background image in RGBA

    fg: foreground image in RGBA
    """
    bg_alpha = bg[:, :, 3:4]
    fg_alpha = fg[:, :, 3:4]
    bg_rgb = bg[:, :, :3]
    fg_rgb = fg[:, :, :3]

    out_alpha = fg_alpha + bg_alpha * (1 - fg_alpha)
    out_rgb = fg_alpha * fg_rgb + bg_alpha * (1 - fg_alpha) * bg_rgb
    alpha_factor = 1. / out_alpha
    alpha_factor[np.isinf(alpha_factor)] = 0.0
    out_rgb *= alpha_factor

    out = np.concatenate([out_rgb, out_alpha], axis=2)
    assert out.shape == fg.shape == bg.shape
    return out


def circular_mask(shape, center, radius):
    """
    creates a circular bit-mask with the given shaped, with the circle
    centered at a given point with a given radius
    """
    a, b = center
    x_max, y_max = shape
    y, x = np.ogrid[-a:x_max - a, -b:y_max - b]
    mask = x * x + y * y <= radius * radius
    return mask


def alpha_mask_inner_circle(img):
    """
    zero out the alpha channel of the pixels in the image that do not fall
    in the ellipse bound by the images border
    """
    assert len(img.shape) == 3
    assert img.shape[2] == 4  # has alpha channel
    new_img = img.copy()
    x, y = img.shape[:2]
    mask = circular_mask(new_img.shape[:2],
                         (x / 2.0, y / 2.0),
                         min(x / 2.0, y / 2.0))
    new_img[..., 3] *= mask
    return new_img

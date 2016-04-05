import pickle
import numpy as np
import scipy.misc
import cv2

from du._test_utils import equal, numpy_almost_equal
from du import cv2_utils

lena = scipy.misc.lena() / 255.0
affine_identity = np.eye(2, 3)


def test_wrap_01_to_uint8_conversion_serialization():
    equal(
        pickle.loads(pickle.dumps(cv2_utils.draw_circles)).without_conversion,
        cv2_utils.draw_circles.without_conversion)


def test_warp_affine_3d_1():
    lena3 = lena[..., np.newaxis]
    res = cv2_utils.warp_affine(lena3,
                                affine_identity)
    numpy_almost_equal(res, lena3)
    equal(res.shape, lena3.shape)


def test_warp_affine_3d_2():
    lena3 = cv2_utils.gray_to_bgr(lena)
    res3 = cv2_utils.warp_affine(lena3,
                                 affine_identity)
    res = cv2_utils.bgr_to_gray(res3)
    numpy_almost_equal(res, lena)
    equal(res3.shape, lena3.shape)
    equal(res.shape, lena.shape)


def test_warp_affine_3d_3():
    lena3 = cv2_utils.gray_to_rgba(lena)
    res3 = cv2_utils.warp_affine(lena3,
                                 affine_identity)
    res = cv2_utils.rgba_to_gray(res3)
    numpy_almost_equal(res, lena)
    equal(res3.shape, lena3.shape)
    equal(res.shape, lena.shape)


def test_warp_affine_dtypes():
    equal(np.uint8,
          cv2_utils.warp_affine((255 * lena).astype(np.uint8),
                                affine_identity).dtype)
    equal(np.float32,
          cv2_utils.warp_affine(lena.astype(np.float32),
                                affine_identity).dtype)
    equal(np.float64,
          cv2_utils.warp_affine(lena.astype(np.float64),
                                affine_identity).dtype)


def test_circles():
    img = np.zeros((60, 60))
    centers = [(5, 5),  (50, 50)]

    new_img = cv2_utils.draw_circles(img, centers,
                                     color=(255, 255, 255),
                                     radius=1,
                                     thickness=-1)

    # Make sure we didn't modify original
    np.testing.assert_allclose(img, 0.0)

    np.testing.assert_almost_equal(new_img[5, 5], 1.0)
    np.testing.assert_almost_equal(new_img[50, 50], 1.0)


def test_convert_color_space_fn1():

    @cv2_utils.wrap_01_to_uint8_conversion
    def gray_to_bgr(as_uint8):
        return cv2.cvtColor(as_uint8, cv2.COLOR_GRAY2BGR)

    np.testing.assert_equal(gray_to_bgr(lena),
                            cv2_utils.gray_to_bgr(lena))


def test_convert_color_space_fn2():

    @cv2_utils.wrap_01_to_uint8_conversion
    def bgr_to_gray(as_uint8):
        return cv2.cvtColor(as_uint8, cv2.COLOR_BGR2GRAY)

    x = np.random.rand(512, 512, 3)
    np.testing.assert_equal(bgr_to_gray(x),
                            cv2_utils.bgr_to_gray(x))


def test_convert_color_space():
    np.testing.assert_equal(cv2_utils.convert_color_space(lena, "GRAY", "BGR"),
                            cv2_utils.gray_to_bgr(lena))

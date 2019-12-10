import numpy as np
import scipy.misc

from du.preprocessing.image2d import affine_transform
from du._test_utils import (numpy_almost_equal,
                            equal,
                            numpy_allclose,
                            numpy_not_allclose,
                            numpy_not_almost_equal)

img = scipy.misc.ascent() / 255.0

def _mse(a, b):
    return np.mean((a - b) ** 2)


def test_affine_transform1():
    assert np.allclose(affine_transform(img), img)


def test_affine_transform2():
    img3d = img[..., np.newaxis]
    numpy_almost_equal(affine_transform(img3d), img3d)

    numpy_almost_equal(affine_transform(img),
                       affine_transform(img3d)[..., 0])


def test_affine_transform3():
    x = np.array([[0.0, 0.1],
                  [0.2, 0.3]])
    numpy_almost_equal(affine_transform(x,
                                        translation=[0, 1],
                                        mode="constant"),
                       np.array([[0.1, 0.0],
                                 [0.3, 0.0]]))
    numpy_almost_equal(affine_transform(x,
                                        translation=[1, 0],
                                        mode="constant"),
                       np.array([[0.2, 0.3],
                                 [0.0, 0.0]]))


def test_affine_transform_flip1():
    numpy_almost_equal(affine_transform(img, horizontal_flip=True),
                       img[:, ::-1])
    numpy_almost_equal(affine_transform(img, vertical_flip=True),
                       img[::-1])
    numpy_almost_equal(affine_transform(img,
                                        vertical_flip=True,
                                        horizontal_flip=True),
                       img[::-1, ::-1])


def test_affine_transform_flip2():
    numpy_almost_equal(img,
                       affine_transform(
                           affine_transform(img,
                                            horizontal_flip=True),
                           horizontal_flip=True
                       ))
    numpy_almost_equal(img,
                       affine_transform(
                           affine_transform(img,
                                            vertical_flip=True),
                           vertical_flip=True
                       ))
    numpy_almost_equal(img,
                       affine_transform(
                           affine_transform(img,
                                            horizontal_flip=True,
                                            vertical_flip=True),
                           horizontal_flip=True,
                           vertical_flip=True
                       ))


def test_affine_transform_dtype():
    equal(np.float32,
          affine_transform(img.astype(np.float32)).dtype)
    equal(np.float32,
          affine_transform(img.astype(np.float32), rotation=0.5).dtype)


def test_affine_transform_rotation1():
    rotated = affine_transform(img, rotation=0.5)
    numpy_not_almost_equal(img, rotated)
    numpy_almost_equal(img, affine_transform(rotated, rotation=0.5))
    numpy_almost_equal(img, affine_transform(rotated, rotation=-0.5))


def test_affine_transform_rotation2():
    numpy_almost_equal(affine_transform(img, rotation=0.25),
                       np.rot90(img, 1))
    numpy_almost_equal(affine_transform(img, rotation=0.5),
                       np.rot90(img, 2))
    numpy_almost_equal(affine_transform(img, rotation=0.75),
                       np.rot90(img, 3))
    numpy_almost_equal(affine_transform(img, rotation=1.0),
                       np.rot90(img, 4))


def test_affine_transform_shear():
    sheared = affine_transform(img, shear=0.5)
    numpy_not_almost_equal(img, sheared)
    numpy_almost_equal(img, affine_transform(sheared, shear=0.5))
    numpy_almost_equal(img, affine_transform(sheared, shear=-0.5))


def test_affine_transform_zoom1():
    half_size = affine_transform(img, zoom=0.5)
    assert _mse(img, half_size) > 0.05
    assert _mse(img, affine_transform(half_size, zoom=2.0)) < 0.005


def test_affine_transform_zoom2():
    double_size = affine_transform(img, zoom=2.0)
    numpy_not_allclose(img, double_size, atol=0.5)
    # need to crop because doubling size loses information not in center
    numpy_allclose(affine_transform(img, output_shape=(256, 256)),
                   affine_transform(double_size,
                                    zoom=0.5,
                                    output_shape=(256, 256)),
                   atol=0.5)


def test_affine_transform_stretch1():
    half_stretch = affine_transform(img, stretch=0.5)
    assert _mse(img, half_stretch) > 0.01
    # TODO what does this comment mean:
    # need to crop because stretching loses information not in center
    assert _mse(img, affine_transform(half_stretch, stretch=2.0)) < 0.001


def test_affine_transform_stretch2():
    double_stretch = affine_transform(img, stretch=2.0)
    numpy_not_allclose(img, double_stretch, atol=0.5)
    # need to crop because stretching loses information not in center
    numpy_allclose(affine_transform(img, output_shape=(256, 512)),
                   affine_transform(double_stretch,
                                    stretch=0.5,
                                    output_shape=(256, 512)),
                   atol=0.5)


def test_affine_transform_many_channels():
    n_copies = 10
    img3 = np.array([img] * n_copies).transpose(1, 2, 0)
    kwargs = dict(stretch=1.2, zoom=0.3)
    ans = affine_transform(img, **kwargs)
    res = affine_transform(img3, **kwargs)
    for i in range(n_copies):
        np.testing.assert_allclose(res[..., i], ans)

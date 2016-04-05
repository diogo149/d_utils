import numpy as np
import scipy.misc

from du.preprocessing.image2d import affine_transform
from du._test_utils import (numpy_almost_equal,
                            equal,
                            numpy_allclose,
                            numpy_not_allclose,
                            numpy_not_almost_equal)

lena = scipy.misc.lena() / 255.0


def test_affine_transform1():
    assert np.allclose(affine_transform(lena), lena)


def test_affine_transform2():
    lena3d = lena[..., np.newaxis]
    numpy_almost_equal(affine_transform(lena3d), lena3d)

    numpy_almost_equal(affine_transform(lena),
                       affine_transform(lena3d)[..., 0])


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
    numpy_almost_equal(affine_transform(lena, horizontal_flip=True),
                       lena[:, ::-1])
    numpy_almost_equal(affine_transform(lena, vertical_flip=True),
                       lena[::-1])
    numpy_almost_equal(affine_transform(lena,
                                        vertical_flip=True,
                                        horizontal_flip=True),
                       lena[::-1, ::-1])


def test_affine_transform_flip2():
    numpy_almost_equal(lena,
                       affine_transform(
                           affine_transform(lena,
                                            horizontal_flip=True),
                           horizontal_flip=True
                       ))
    numpy_almost_equal(lena,
                       affine_transform(
                           affine_transform(lena,
                                            vertical_flip=True),
                           vertical_flip=True
                       ))
    numpy_almost_equal(lena,
                       affine_transform(
                           affine_transform(lena,
                                            horizontal_flip=True,
                                            vertical_flip=True),
                           horizontal_flip=True,
                           vertical_flip=True
                       ))


def test_affine_transform_dtype():
    equal(np.float32,
          affine_transform(lena.astype(np.float32)).dtype)
    equal(np.float32,
          affine_transform(lena.astype(np.float32), rotation=0.5).dtype)


def test_affine_transform_rotation1():
    rotated = affine_transform(lena, rotation=0.5)
    numpy_not_almost_equal(lena, rotated)
    numpy_almost_equal(lena, affine_transform(rotated, rotation=0.5))
    numpy_almost_equal(lena, affine_transform(rotated, rotation=-0.5))


def test_affine_transform_rotation2():
    numpy_almost_equal(affine_transform(lena, rotation=0.25),
                       np.rot90(lena, 1))
    numpy_almost_equal(affine_transform(lena, rotation=0.5),
                       np.rot90(lena, 2))
    numpy_almost_equal(affine_transform(lena, rotation=0.75),
                       np.rot90(lena, 3))
    numpy_almost_equal(affine_transform(lena, rotation=1.0),
                       np.rot90(lena, 4))


def test_affine_transform_shear():
    sheared = affine_transform(lena, shear=0.5)
    numpy_not_almost_equal(lena, sheared)
    numpy_almost_equal(lena, affine_transform(sheared, shear=0.5))
    numpy_almost_equal(lena, affine_transform(sheared, shear=-0.5))


def test_affine_transform_zoom1():
    half_size = affine_transform(lena, zoom=0.5)
    numpy_not_allclose(lena, half_size, atol=0.5)
    numpy_allclose(lena, affine_transform(half_size, zoom=2.0), atol=0.5)


def test_affine_transform_zoom2():
    double_size = affine_transform(lena, zoom=2.0)
    numpy_not_allclose(lena, double_size, atol=0.5)
    # need to crop because doubling size loses information not in center
    numpy_allclose(affine_transform(lena, output_shape=(256, 256)),
                   affine_transform(double_size,
                                    zoom=0.5,
                                    output_shape=(256, 256)),
                   atol=0.5)


def test_affine_transform_stretch1():
    half_stretch = affine_transform(lena, stretch=0.5)
    numpy_not_allclose(lena, half_stretch, atol=0.5)
    # need to crop because stretching loses information not in center
    numpy_allclose(lena,
                   affine_transform(half_stretch, stretch=2.0),
                   atol=0.5)


def test_affine_transform_stretch2():
    double_stretch = affine_transform(lena, stretch=2.0)
    numpy_not_allclose(lena, double_stretch, atol=0.5)
    # need to crop because stretching loses information not in center
    numpy_allclose(affine_transform(lena, output_shape=(256, 512)),
                   affine_transform(double_stretch,
                                    stretch=0.5,
                                    output_shape=(256, 512)),
                   atol=0.5)


def test_affine_transform_many_channels():
    n_copies = 10
    img = np.array([lena] * n_copies).transpose(1, 2, 0)
    kwargs = dict(stretch=1.2, zoom=0.3)
    ans = affine_transform(lena, **kwargs)
    res = affine_transform(img, **kwargs)
    for i in range(n_copies):
        np.testing.assert_allclose(res[..., i], ans)

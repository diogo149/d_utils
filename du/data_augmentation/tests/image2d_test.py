import scipy.misc

from du.preprocessing.image2d import affine_transform
from du.data_augmentation.image2d import random_affine, random_affine_fn
from du._test_utils import numpy_almost_equal, raises

lena = scipy.misc.lena() / 255.0


def test_random_affine_no_args():
    numpy_almost_equal(random_affine(lena), lena)


@raises(AssertionError)
def test_random_affine_is_random():
    numpy_almost_equal(random_affine(lena, rotation_range=(-0.1, 0.1)),
                       random_affine(lena, rotation_range=(-0.1, 0.1)))


def test_random_affine_fn():
    # same fn should be deterministic
    fn = random_affine_fn(lena.shape, rotation_range=(-0.1, 0.1))
    numpy_almost_equal(fn(lena),
                       fn(lena))


def test_random_affine_deterministic():
    numpy_almost_equal(
        random_affine(lena, rotation_range=(0.1, 0.1)),
        affine_transform(lena, rotation=0.1),
    )

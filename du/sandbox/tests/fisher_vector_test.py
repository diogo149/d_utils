import numpy as np
from scipy.misc import lena
import doo
import doo.sandbox.fisher_vector


def test_fisher_vector():
    fv = doo.sandbox.fisher_vector.FisherVector(descriptor="orb")
    img = (lena() / 255.0)
    X = np.array([img, img, img])
    new_X = fv.fit_transform(X)
    doo._test_utils.equal(new_X.shape, (3, fv.num_gaussians_ * (1 + 32 * 2)))

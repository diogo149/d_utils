"""
based on https://github.com/jacobgil/pyfishervector
"""

import numpy as np
import cv2
import scipy.stats
import sklearn


class FisherVector(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, n_components=5, descriptor="orb"):
        """
        n_components:
        number of components in the GMM

        descriptor:
        feature descriptor algorithm to use
        (default = orb, because older versions of cv2 do not have sift)
        """
        self.n_components = n_components
        self.descriptor = descriptor
        self.init_state()

    def init_state(self):
        if self.descriptor == "sift":
            self.descriptor_ = cv2.SIFT()
        elif self.descriptor == "orb":
            self.descriptor_ = cv2.ORB()
        else:
            raise ValueError(self.descriptor)
        self.gmm_ = cv2.EM(self.n_components)

    def validate_imgs(self, X):
        assert len(X.shape) == 3
        # assumes images are floats between 0 and 1
        assert X.min() >= 0
        assert X.max() <= 1

    def fit(self, X, y=None):
        self.validate_imgs(X)
        self.init_state()
        train_descriptors = np.concatenate(
            [self.image_descriptor_for_image(img)
             for img in X])
        self.fit_gmm(train_descriptors)
        return self

    def transform(self, X):
        self.validate_imgs(X)
        descriptors_list = [self.image_descriptor_for_image(img)
                            for img in X]
        fisher_vectors = [self.fisher_vector_for_descriptors(descriptors)
                          for descriptors in descriptors_list]
        return np.array(fisher_vectors)

    def image_descriptor_for_image(self, img):
        """
        return descriptors for grayscale image of shape (num_descriptors, ???)
        ??? = 128 for SIFT, 32 for ORB
        (note that num_descriptors is variable)
        """
        assert len(img.shape) == 2
        # convert image to uint8
        new_img = (img * 255).astype(np.uint8)
        # discard list of keypoints
        _, descriptors = self.descriptor_.detectAndCompute(new_img,
                                                           None)
        # handle the case of no descriptors
        if descriptors is None:
            if self.descriptor == "sift":
                descriptor_length = 128
            elif self.descriptor == "orb":
                descriptor_length = 32
            else:
                raise ValueError(self.descriptor)
            descriptors = np.zeros((0, descriptor_length))
        return descriptors

    def fit_gmm(self, train_descriptors):
        """
        train_descriptors:
        N x M matrix of features

        creates a gaussian mixture model, and extract out the relevant parts
        http://docs.opencv.org/modules/ml/doc/expectation_maximization.html
        """
        self.descriptor_length = train_descriptors.shape[1]
        self.gmm_.train(train_descriptors)

        # gather state
        # ---
        # matrix with shape (n_components, M)
        means = self.gmm_.getMat("means")
        # a list of n_components matrices, each with shape (M, M)
        covs = self.gmm_.getMatVector("covs")
        # matrix of shape (1, n_components)
        weights = self.gmm_.getMat("weights")
        # ignore the first dim
        weights = weights[0]

        # filter gaussians with too small weights
        threshold = 1.0 / self.n_components
        keeps = weights > threshold
        self.gmm_means_ = means[keeps]
        self.gmm_covs_ = [cov for keep, cov in zip(keeps, covs) if keep]
        self.gmm_weights_ = weights[keeps]

        self.num_gaussians_ = len(self.gmm_weights_)
        self.gaussians_ = [scipy.stats.multivariate_normal(mean=mean, cov=cov)
                           for mean, cov in zip(self.gmm_means_,
                                                self.gmm_covs_)]

    def fisher_vector_for_descriptors(self, descriptors):
        num_descriptors, descriptor_length = descriptors.shape
        assert descriptor_length == self.descriptor_length
        assert descriptor_length == self.gmm_means_.shape[1]

        if num_descriptors == 0:
            # should this error out instead?
            return np.zeros(self.num_gaussians_ * (1 + 2 * descriptor_length))

        # probabilities
        pdfs = np.array([np.atleast_1d(g.pdf(descriptors))
                         for g in self.gaussians_]).T
        assert pdfs.shape == (num_descriptors, self.num_gaussians_)
        weighted_pdfs = pdfs * self.gmm_weights_
        assert weighted_pdfs.shape == (num_descriptors, self.num_gaussians_)
        probabilities = weighted_pdfs / weighted_pdfs.sum(axis=1)[:, None]
        assert probabilities.shape == (num_descriptors, self.num_gaussians_)

        # calculate likelihood statistics
        likelihood_statistics = []
        for moment in (0, 1, 2):
            if moment == 0:
                # no need to compute for each descriptor index, because
                # they will all be the same when raising descriptors to 0
                likelihood_moment_sums = probabilities.sum(axis=0)[..., None]
                assert likelihood_moment_sums.shape == (self.num_gaussians_, 1)
            else:
                x_moment = descriptors ** moment
                likelihood_moment_sums = np.dot(probabilities.T, x_moment)
                assert likelihood_moment_sums.shape == (self.num_gaussians_,
                                                        descriptor_length)
            likelihood_statistics.append(likelihood_moment_sums)

        # calculating components of fisher vector
        sigma = np.array([np.diagonal(cov) for cov in self.gmm_covs_])
        assert sigma.shape == (self.num_gaussians_, descriptor_length)
        s0 = likelihood_statistics[0]
        s1 = likelihood_statistics[1]
        s2 = likelihood_statistics[2]
        T = num_descriptors
        w = self.gmm_weights_[:, None]
        means = self.gmm_means_

        fisher_vector_weights = (s0 - T * w) / np.sqrt(w)
        fisher_vector_weights = fisher_vector_weights.flatten()
        assert fisher_vector_weights.shape == (self.num_gaussians_,)

        fisher_vector_means = (s1 - means * s0) / np.sqrt(w * sigma)
        fisher_vector_means = fisher_vector_means.flatten()
        assert fisher_vector_means.shape == (self.num_gaussians_
                                             * descriptor_length,)

        numerator = (s2 - 2 * means * s1 + (means ** 2 - sigma) * s0)
        denomenator = np.sqrt(2 * w) * sigma
        fisher_vector_sigma = numerator / denomenator
        fisher_vector_sigma = fisher_vector_sigma.flatten()
        assert fisher_vector_sigma.shape == (self.num_gaussians_
                                             * descriptor_length,)

        fv = np.concatenate([fisher_vector_weights,
                             fisher_vector_means,
                             fisher_vector_sigma])

        # normalize
        def normalize(fv):
            v = np.sqrt(np.abs(fv)) * np.sign(fv)
            return v / np.sqrt(np.dot(v, v))

        fv = normalize(fv)

        return fv

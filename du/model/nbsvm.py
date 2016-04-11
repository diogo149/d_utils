"""
based on implementation in https://github.com/mesnilgr/nbsvm
"""

import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC


class CountToNBFeatures(sklearn.base.BaseEstimator,
                        sklearn.base.TransformerMixin):

    """
    takes in counts and outputs naive-bayes features (as in NB-SVM)
    """

    def __init__(self, alpha=1):
        self.alpha = alpha

    def fit(self, X, y):
        assert y.ndim == 1
        assert set(y) == {0, 1}
        arr = X.toarray()  # convert from matrix to array
        pos = arr[y == 1]
        neg = arr[y == 0]
        p = pos.sum(axis=0).astype(float) + self.alpha
        p /= abs(p).sum()
        q = neg.sum(axis=0).astype(float) + self.alpha
        q /= abs(q).sum()
        r = np.log(p / q)
        self.r = r
        return self

    def transform(self, X):
        arr = X.toarray()  # convert from matrix to array
        return (arr > 0) * self.r[np.newaxis]


def NBVectorizer(alpha=1, **kwargs):
    return Pipeline([("count_vectorizer", CountVectorizer(**kwargs)),
                     ("nb_features", CountToNBFeatures(alpha=alpha))])


def NBSVM(alpha=1, vectorizer_kwargs=None, svm_kwargs=None):
    if vectorizer_kwargs is None:
        vectorizer_kwargs = {}
    if svm_kwargs is None:
        svm_kwargs = {}
    return Pipeline([("vectorizer", NBVectorizer(alpha, **vectorizer_kwargs)),
                     ("svm", LinearSVC(**svm_kwargs))])

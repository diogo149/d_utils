import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class ThresholdStandardScaler(BaseEstimator, TransformerMixin):

    """
    like sklearn.preprocssing.StandardScaler, but thresholds the possible
    values to within a provided range, in order to be less sensitive to
    outliers
    """

    def __init__(self,
                 lower_threshold=-3,
                 upper_threshold=3,
                 *args,
                 **kwargs):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        assert self.upper_threshold > self.lower_threshold
        self.trn = StandardScaler(*args, **kwargs)

    def fit(self, X, y=None):
        self.trn.fit(X)
        return self

    def transform(self, X):
        res = self.trn.transform(X)
        return np.clip(res, self.lower_threshold, self.upper_threshold)

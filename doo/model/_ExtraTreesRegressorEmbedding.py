from sklearn import ensemble, preprocessing


class ExtraTreesRegressorEmbedding(object):

    """
    A supervised version of sklearn.ensemble.RandomTreesEmbedding

    This is better than using eg. ExtraTreesRegressor because this returns
    the transformed array as an already one-hot-encoded sparse array of 1's
    instead of leaf indices
    """

    def __init__(self, *args, **kwargs):
        self.sparse_output = kwargs.pop("sparse_output", True)
        self.clf = ensemble.ExtraTreesRegressor(*args, **kwargs)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        return self.one_hot_encoder_.transform(self.clf.apply(X))

    def fit_transform(self, X, y):
        self.clf.fit(X, y)
        self.one_hot_encoder_ = preprocessing.OneHotEncoder(
            sparse=self.sparse_output)
        return self.one_hot_encoder_.fit_transform(self.clf.apply(X))

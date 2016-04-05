import numpy as np
from sklearn.ensemble import ExtraTreesClassifier


class ExtraTreesFeatureSelectionClassifier(object):

    """
    Like ExtraTreesClassifier, but also sequentially selects a subset of
    features (using feature importance of an ExtraTreesClassifier) and only
    trains on those
    """

    def __init__(self, max_num_features, shrink_rate, **kwargs):
        self.model_ = ExtraTreesClassifier(**kwargs)
        self.max_num_features = max_num_features
        self.shrink_rate = shrink_rate

    def fit(self, X, y, sample_weight=None):
        self.selected_indices_ = np.arange(X.shape[1])
        while True:
            self.model_.fit(
                X=X[:, self.selected_indices_],
                y=y,
                sample_weight=sample_weight,
            )
            # we want this after fitting the model
            if self.selected_indices_.shape[0] <= self.max_num_features:
                break

            imp = self.model_.feature_importances_
            imp_new_indexes \
                = [index for val, index in sorted(zip(imp, range(len(imp))),
                                                  reverse=True)]
            self.selected_indices_ \
                = self.selected_indices_[np.array(imp_new_indexes)]
            new_num_indices = int(self.selected_indices_.shape[0]
                                  * self.shrink_rate)
            self.selected_indices_ \
                = self.selected_indices_[:max(new_num_indices,
                                              self.max_num_features)]
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[:, self.selected_indices_])

    def predict(self, X):
        return self.model_.predict(X[:, self.selected_indices_])

import numpy as np
from sklearn.base import (  # type: ignore
    BaseEstimator,
    TransformerMixin,
    clone,
)


class ListTransformer(BaseEstimator, TransformerMixin):
    """Helper class that, given a sklearn-estimator, creates a version of that
    estimator that can be applied to a list of point clouds.
    """
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y=None):
        X_concat = [el for list in X for el in list]
        self.estimator_ = clone(self.base_estimator).fit(X_concat, y)
        return self

    def transform(self, X):
        try:
            return np.array([self.estimator_.transform(arr) for arr in X])
        except ValueError:
            return [self.estimator_.transform(arr) for arr in X]


class PersistenceImageProcessor(BaseEstimator, TransformerMixin):
    """MinMax-scales the pixel values of the persistence images and
    concatenates the flattened images corresponding to one sample.
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        n_samples = len(X)
        self._scaler_ = ListTransformer(self.scaler)
        self._scaler_.fit(X.reshape(n_samples, -1, 1))
        return self

    def transform(self, X):
        n_samples = len(X)
        X_transformed = self._scaler_.fit_transform(
            X.reshape(n_samples, -1, 1)
        ).reshape(n_samples, -1)
        return X_transformed

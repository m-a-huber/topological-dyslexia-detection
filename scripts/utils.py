import numpy as np
from sklearn.base import (  # type: ignore
    BaseEstimator,
    TransformerMixin,
    clone,
)


class ListTransformer(BaseEstimator, TransformerMixin):
    """Helper class that, given a sklearn-estimator that can be applied to a
    list of points, creates a version of that estimator that can be applied to
    a list of lists of points.
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


class PersistenceProcessor(BaseEstimator, TransformerMixin):
    """Transforms output of TimeSeriesHomology into a format suitable for
    subsequent creation of persistence images.
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [
            [
                np.array([
                    np.sort(gen)  # ensure lifetimes are positive
                    for dim in dgm
                    for gen in dim
                ]).reshape(-1, 2)
                for coord in time_series
                for dgm in coord
            ]
            for time_series in X
        ]


class PersistenceImageProcessor(BaseEstimator, TransformerMixin):
    """MinMax-scales the pixel values of the persistence images and
    concatenates the flattened images corresponding to one sample.
    """
    def __init__(self, scaler):
        self.scaler = scaler

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_samples = len(X)
        self._scaler_ = ListTransformer(self.scaler)
        X_transformed = self._scaler_.fit_transform(
            X.reshape(n_samples, -1, 1)
        ).reshape(n_samples, -1)
        return X_transformed

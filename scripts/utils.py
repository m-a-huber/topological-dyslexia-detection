import numpy as np
from scipy.stats import rv_continuous
from sklearn.base import (  # type: ignore
    BaseEstimator,
    TransformerMixin,
    clone,
)


def weight_abs1p(pt):
    """Custom weight function for persistence images that weighs points in a
    persistence diagram by lifetime plus 1.
    """
    return np.abs(pt[1]) + 1


class UniformSlope(rv_continuous):
    """Helper class to sample slopes of lines through the origin such that they
    are uniform with respect to angle of the sectors [min_slope, max_slope]."""

    def __init__(self, min_slope, max_slope, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_min = np.arctan(min_slope)
        self.theta_max = np.arctan(max_slope)

    def _rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        theta = rng.uniform(self.theta_min, self.theta_max, size=size)
        return np.tan(theta)


class UniformSlopeSym(rv_continuous):
    """Helper class to sample slopes of lines through the origin such that they
    are uniform with respect to angle of the two sectors [min_slope, max_slope]
    and [-max_slope, -min_slope]."""

    def __init__(self, min_slope, max_slope, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_min = np.arctan(min_slope)
        self.theta_max = np.arctan(max_slope)

    def _rvs(self, size=None, random_state=None):
        rng = np.random.default_rng(random_state)
        theta = rng.uniform(self.theta_min, self.theta_max, size=size)
        return rng.choice([-1, 1], size=size) * np.tan(theta)


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
    """Helper class that transforms output of TimeSeriesHomology into a format
    suitable for subsequent creation of persistence images by producing a list
    of lists of NumPy-arrays of the form
    `(n_samples, n_dgms_per_sample, n_gens_of_dgm, 2)`.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):  # noqa: ARG002
        self._is_fitted_ = True
        return self

    def transform(self, X):
        return [
            [
                np.array(
                    [
                        np.sort(gen)  # ensure lifetimes are positive
                        for dim in dgm
                        for gen in dim
                    ]
                ).reshape(-1, 2)
                for coord in time_series
                for dgm in coord
            ]
            for time_series in X
        ]


class PersistenceImageProcessor(BaseEstimator, TransformerMixin):
    """Helper class that normalizes the pixel values of the persistence images
    to the desired range and concatenates the flattened images corresponding to
    one sample.
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, y=None):  # noqa: ARG002
        self.min_x_, self.max_x_ = X.min(), X.max()
        return self

    def transform(self, X):
        X_std = (X - self.min_x_) / (self.max_x_ - self.min_x_)
        X_scaled = (
            X_std * (self.feature_range[1] - self.feature_range[0])
            + self.feature_range[0]
        )
        return X_scaled.reshape(len(X), -1)

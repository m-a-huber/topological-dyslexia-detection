import numpy as np
from scipy.stats import rv_continuous
from sklearn.base import (  # type: ignore
    BaseEstimator,
    TransformerMixin,
    clone,
)


def group_by_mean(X, groups):
    """Groups the rows of X according to the corresponding group in groups and
    returns the mean of each group as rows in a 2D array.
    """
    return np.array(
        [X[groups == group].mean(axis=0) for group in np.unique(groups)]
    )


class NestedDict(dict):
    """Helper class to create nested dictionaries of arbitrary depth."""

    def __missing__(self, key):
        value = self[key] = type(self)()
        return value

    def to_dict(self):
        return {
            k: v.to_dict() if isinstance(v, NestedDict) else v
            for k, v in self.items()
        }


class UniformSlope(rv_continuous):
    """Helper class to sample slopes of lines through the origin such that they
    are uniform with respect to angle of the sectors [min_slope, max_slope].
    """

    def __init__(self, min_slope, max_slope, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_min = np.arctan(min_slope)
        self.theta_max = np.arctan(max_slope)

    def _rvs(self, *args, size=None, random_state=None):  # noqa: ARG002
        rng = np.random.default_rng(random_state)
        theta = rng.uniform(self.theta_min, self.theta_max, size=size)
        return np.tan(theta)


class UniformSlopeSym(rv_continuous):
    """Helper class to sample slopes of lines through the origin such that they
    are uniform with respect to angle of the two sectors [min_slope, max_slope]
    and [-max_slope, -min_slope].
    """

    def __init__(self, min_slope, max_slope, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta_min = np.arctan(min_slope)
        self.theta_max = np.arctan(max_slope)

    def _rvs(self, *args, size=None, random_state=None):  # noqa: ARG002
        rng = np.random.default_rng(random_state)
        theta = rng.uniform(self.theta_min, self.theta_max, size=size)
        return rng.choice([-1, 1], size=size) * np.tan(theta)


class ListTransformer(BaseEstimator, TransformerMixin):
    """Helper class that, given a sklearn-estimator that can be applied to a
    list of points, creates a version of that estimator that can be applied to
    a list of lists of points. The estimator is applied to the flattened list
    of points, and the result is returned as a list of lists of points whose
    shape matches that of the input.
    """

    def __init__(self, base_estimator):
        self.base_estimator = base_estimator

    def fit(self, X, y=None):  # noqa: ARG002
        try:
            X_concat = np.array([el for x in X for el in x])
        except ValueError:
            X_concat = [el for x in X for el in x]
        self.estimator_ = clone(self.base_estimator).fit(X_concat, None)
        return self

    def transform(self, X):
        out = [self.estimator_.transform(arr) for arr in X]
        try:
            return np.array(out)
        except ValueError:
            return out


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


class MeanAggregator(BaseEstimator, TransformerMixin):
    """Helper class that aggregates elements of an array by replacing them with
    their mean.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):  # noqa: ARG002
        self._is_fitted_ = True
        return self

    def transform(self, X):
        return np.array([x.mean(axis=0) for x in X])


class TupleElementSelector(BaseEstimator, TransformerMixin):
    """Helper class that extracts a specific element from a tuple."""

    def __init__(self, element_idx: int):
        self.element_idx = element_idx

    def fit(self, X, y=None):  # noqa: ARG002
        self._is_fitted_ = True
        return self

    def transform(self, X):
        """Extract the element at the specified index from each sample."""
        return [x[self.element_idx] for x in X]

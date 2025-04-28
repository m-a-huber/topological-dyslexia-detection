import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else StandardScaler()

    def fit(self, X, y=None):
        all_points = np.concatenate(X)
        self.scaler.fit(all_points)
        return self

    def transform(self, X):
        return [self.scaler.transform(cloud) for cloud in X]

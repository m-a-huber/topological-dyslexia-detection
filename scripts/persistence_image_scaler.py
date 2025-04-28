from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.preprocessing import MinMaxScaler  # type: ignore


class PersistenceImageScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=None):
        self.scaler = scaler if scaler is not None else MinMaxScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X.T)
        return self

    def transform(self, X):
        return self.scaler.fit_transform(X.T).T

import gudhi as gd  # type: ignore
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing_extensions import Self


class TimeSeriesHomology(TransformerMixin, BaseEstimator):
    """Class implementing extended persistence of a time series.

    Parameters:
        homology_coeff_field (int, optional): The field coefficient over which
            homology is computed. Must be a prime number less than or equal to
            46337. Defaults to 11.
        min_persistence (float, optional): The minimum persistence value to
            take into account. Defaults to 0.0.
    """

    def __init__(
        self,
        homology_coeff_field: int = 11,
        min_persistence: float = 0.0,
        type: str = "normal",
        slope: float = 1.0,
        sigmoid_factor: float = 1.0,
    ):
        self.homology_coeff_field = homology_coeff_field
        self.min_persistence = min_persistence
        self.type = type
        self.slope = slope
        self.sigmoid_factor = sigmoid_factor

    def fit(
        self,
        X: npt.NDArray,
    ) -> Self:
        return self

    def transform(
        self,
        X: npt.NDArray | list,
    ) -> list[npt.NDArray]:
        """Computes extended persistent homology of a time series.

        Args:
            X (npt.NDArray | list): Times series given as a NumPy array of
                shape (n_time_steps, 2) or as a list that is convertible to a
                NumPy-array of shape (n_time_steps, 2), where the second axis
                contains (time, value)-pairs of the time series.

        Returns:
            list[list[npt.NDArray]]: A list of three persistence diagrams. Each
                persistence diagram is a list of NumPy-arrays of shape
                `(n_generators, 2)`, where the i-th entry of the list is an
                array containing the birth and death times of the homological
                generators in dimension i-1. In particular, the list starts
                with 0-dimensional homology and contains information from
                consecutive homological dimensions.
        """
        self.st_ = self._get_simplex_tree(X)
        dgm = self.st_.persistence(
            homology_coeff_field=self.homology_coeff_field,
            min_persistence=self.min_persistence,
            persistence_dim_max=True
        )
        return self._format_dgm(dgm)

    def _get_simplex_tree(
        self,
        X: npt.NDArray | list,
    ) -> gd.SimplexTree:
        """Constructs simplex tree from time series data.

        Args:
            X (npt.NDArray | list): Times series given as a NumPy array of
                shape (n_time_steps, 2) or as a list that is convertible to a
                NumPy-array of shape (n_time_steps, 2), where the second axis
                contains (time, value)-pairs of the time series.

        Returns:
            gudhi.Simplextree: Simplex tree encoding the sublevel set
                filtration of the time series.
        """
        X = np.array(X, dtype=np.float64)
        st = gd.SimplexTree()
        vertex_array_vertices = np.arange(len(X)).reshape([1, -1])
        filtrations_vertices = self._get_vertex_filtrations(X)
        st.insert_batch(
            vertex_array=vertex_array_vertices,
            filtrations=filtrations_vertices
        )
        vertex_array_edges = np.array([
            [i, i + 1]
            for i in range(len(X) - 1)
        ]).T
        filtrations_edges = np.maximum(
            filtrations_vertices[:-1],
            filtrations_vertices[1:]
        )
        st.insert_batch(
            vertex_array=vertex_array_edges,
            filtrations=filtrations_edges
        )
        return st

    def _get_vertex_filtrations(self, X):
        if self.type == "normal":
            return X[:, 1]
        if self.type == "sloped":
            x_min = np.min(
                X[:, 1]
            )
            x_max = np.max(
                X[:, 1]
            )
            x_mid = 0.5 * (x_min + x_max)

            def _get_filtration(a):
                # a is array of shape (2,) interpreted as containing t- and
                # x-value of time series
                return a[0] - (a[1] - x_mid) / self.slope
            return np.apply_along_axis(
                func1d=_get_filtration,
                axis=1,
                arr=X
            )
        if self.type == "sigmoidal":
            x_min = np.min(
                X[:, 1]
            )
            x_max = np.max(
                X[:, 1]
            )

            def _get_filtration(a):
                # a is array of shape (2,) interpreted as containing t- and
                # x-value of time series
                if a[1] == x_min:
                    return self.sigmoid_factor * np.inf
                if a[1] == x_max:
                    return -self.sigmoid_factor * np.inf
                else:
                    return (
                        a[0]
                        + np.log((x_max - a[1]) / (a[1] - x_min))
                        / self.sigmoid_factor
                    )
            return np.apply_along_axis(
                func1d=_get_filtration,
                axis=1,
                arr=X
            )

    def _format_dgm(
        self,
        dgm: list[tuple[int, tuple[float, float]]],
    ) -> list[npt.NDArray]:
        """Helper function to convert a persistence diagram given in the Gudhi
        format into the format suitable for plotting.

        Args:
            dgm (list[tuple[int, tuple[float, float]]]): Persistence diagram
                given in the Gudhi format, that is, a list containing tuples of
                the form (homological_dimension, (birth, death)).

        Returns:
            list[npt.NDArray]: A list of NumPy-arrays of shape
                `(n_generators, 2)`, where the i-th entry of the list is an
                array containing the birth and death times of the homological
                generators in dimension i-1. In particular, the list starts
                with 0-dimensional homology and contains information from
                consecutive homological dimensions.
        """
        def _sort_by_lifetime(dgm_formatted):
            return dgm_formatted[np.argsort(
                np.diff(dgm_formatted, axis=1).reshape(-1,)
            )]
        aux_array = np.array([
            [dim, *birth_death_pair]
            for dim, birth_death_pair in dgm
        ]).reshape(-1, 3)
        max_dim = np.max(aux_array[:, 0]) if aux_array.size > 0 else -1
        if max_dim == -1:
            return [
                np.empty((0, 2), dtype=np.float64)
            ]
        else:
            dims = np.arange(max_dim + 1)
            dgm_formatted = [
                aux_array[aux_array[:, 0] == dim][:, 1:]
                for dim in dims
            ]
            dgm_formatted = list(map(_sort_by_lifetime, dgm_formatted))
            return dgm_formatted

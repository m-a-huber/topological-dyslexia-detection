import gudhi as gd  # type: ignore
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from typing_extensions import Self


class TimeSeriesHomology(TransformerMixin, BaseEstimator):

    def __init__(
        self,
        **homology_kwargs
    ):
        self.__dict__.update(homology_kwargs)

    def fit(
        self,
        X: npt.NDArray,
    ) -> Self:
        return self

    def transform(
        self,
        X: npt.NDArray | list,
    ) -> list[list[npt.NDArray]]:
        """Computes extended persistent homology of a time series.

        Args:
            X (npt.NDArray | list): Times series given as a NumPy array of
                shape (n_time_steps,) or as a list that is convertible to a
                NumPy-array of shape (n_time_steps,).

        Returns:
            list[list[npt.NDArray]]: A list of three persistence diagrams. Each
                persistence diagram is a list of NumPy-arrays of shape
                `(n_generators, 2)`, where the i-th entry of the list is an
                array containing the birth and death times of the homological
                generators in dimension i-1. In particular, the list starts
                with 0-dimensional homology and contains information from
                consecutive homological dimensions.
        """
        _X = np.array(X, dtype=np.float64)
        st = self._get_simplex_tree(_X)
        st.extend_filtration()
        dgms = st.extended_persistence(**self.__dict__)
        return [
            self._format_dgm(dgm)
            for dgm in dgms[:3]  # last dgm is always empty
        ]

    def _get_simplex_tree(
        self,
        X: npt.NDArray | list,
    ) -> gd.SimplexTree:
        """Constructs simplex tree from time series data.

        Args:
            X (npt.NDArray | list): Times series given as a NumPy array of
                shape (n_time_steps,) or as a list that is convertible to a
                NumPy-array of shape (n_time_steps,).

        Returns:
            gudhi.Simplextree: Simplex tree encoding the sublevel set
                filtration of the time series.
        """
        st = gd.SimplexTree()
        vertex_array_vertices = np.arange(len(X)).reshape([1, -1])
        filtrations_vertices = X
        st.insert_batch(
            vertex_array=vertex_array_vertices,
            filtrations=filtrations_vertices
        )
        vertex_array_edges = np.array([
            [i, i + 1]
            for i in range(len(X) - 1)
        ]).T
        filtrations_edges = np.maximum(X[:-1], X[1:])
        st.insert_batch(
            vertex_array=vertex_array_edges,
            filtrations=filtrations_edges
        )
        return st

    def _format_dgm(
        self,
        dgm: list[tuple[int, tuple[float, float]]]
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

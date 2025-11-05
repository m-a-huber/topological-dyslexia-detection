from typing import Optional

import gudhi as gd
import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, TransformerMixin
from typing_extensions import Self


class TimeSeriesHomology(TransformerMixin, BaseEstimator):
    """Class implementing horizontal, sloped, sigmoid and arctan persistence of
    (possibly multivariate) time series.

    Parameters:
        filtration_type (str, optional): Which type of filtration to use for
            time series. Must be one of `"horizontal"`, `"sloped"`, `"sigmoid"`
            and `"arctan"`. Defaults to `"horizontal"`.
        slope (float, optional): Slope f'(0) of sweeping function f(t) used to
            construct non-horizontal filtration. This is leads to the sweeping
            function being f(t):=slope*t+0.5, f(t):=1/(1+exp(-4*slope*t)) or
            f(t):=arctan(pi*slope*t)/pi+0.5 if `filtration_type` is set to
            `"sloped"`, `"sigmoid"` or `"arctan"`, respectively. Ignored unless
            `filtration_type` is set to `"sloped"`, `"sigmoid"` or `"arctan"`.
            Defaults to `1.0`.
        padding_factor (float, optional): Factor by which to pad min-max range
            of values of time series to avoid infinite filtration values.
            Ignored unless `"filtration_type"` is set to `"sigmoid"` or
            `"arctan"`. Defaults to `0.05`.
        use_extended_persistence (bool, optional): Whether or not to compute
            extended persistence as opposed to ordinary persistence. Defaults
            to `True`.
        homology_coeff_field (int, optional): The field coefficient over which
            homology is computed. Must be a prime number less than or equal to
            46337. Defaults to `11`.
        min_persistence (float, optional): The minimum persistence value to
            take into account. Defaults to `0.0`.
        drop_infinite_persistence (bool, optional): Whether or not to drop
            homological generators with infinite lifespan from the resulting
            persistences. Ignored if `use_extended_persistence` is set to
            `True`, since in that case all generators have finite lifespan.
            Defaults to `False`.
    """

    def __init__(
        self,
        filtration_type: str = "horizontal",
        slope: float = 1.0,
        padding_factor: float = 0.05,
        use_extended_persistence: bool = True,
        homology_coeff_field: int = 11,
        min_persistence: float = 0.0,
        drop_infinite_persistence: bool = False,
    ):
        self.filtration_type = filtration_type
        self.slope = slope
        self.padding_factor = padding_factor
        self.use_extended_persistence = use_extended_persistence
        self.homology_coeff_field = homology_coeff_field
        self.min_persistence = min_persistence
        self.drop_infinite_persistence = drop_infinite_persistence

    def fit(
        self,
        X: npt.NDArray,  # noqa: ARG002
        y: Optional[None] = None,  # noqa: ARG002
    ) -> Self:
        """Does nothing, present here for API consistency with scikit-learn."""
        return self

    def transform(
        self,
        X: list[npt.NDArray],
        y: Optional[None] = None,  # noqa: ARG002
    ) -> list[list[list[list[npt.NDArray]]]]:
        """Computes persistent homology of a collection of (possibly
        multivariate) time series.

        Args:
            X (list[npt.NDArray]): A list of time series, each of which is
                given as a NumPy-array of shape (n_time_steps, n_features),
                where the last axis contains tuples of the form
                (time, value_1, ..., value_n).
            y (None, optional): Not used, present here for API consistency with
                scikit-learn.

        Returns:
            list[list[list[list[npt.NDArray]]]]: A list containing lists of
                lists of persistence diagrams of each time series, one for each
                coordinate of the value of the time series. Each list of
                persistence diagrams contains either three (if
                `use_extended_persistence` is set to `True`) or one persistence
                diagrams (otherwise). In the former case, the list contains the
                ordinary, relative and essential diagram, in this order. Each
                persistence diagram is given as a list of NumPy-arrays of shape
                `(n_generators, 2)`, where the i-th entry of the list is an
                array containing the birth and death times of the homological
                generators in dimension i-1. In particular, the list starts
                with 0-dimensional homology and contains information from
                consecutive homological dimensions.
        """
        self.simplex_tree_lists_: list[list[gd.SimplexTree]] = [
            [
                self._get_simplex_tree(time_series[:, [0, coord]])
                for coord in range(1, time_series.shape[1])
            ]
            for time_series in X
        ]
        if self.use_extended_persistence:
            for simplex_tree_list in self.simplex_tree_lists_:
                for simplex_tree in simplex_tree_list:
                    simplex_tree.extend_filtration()
            dgms_lists = [
                [
                    simplex_tree.extended_persistence(
                        homology_coeff_field=self.homology_coeff_field,
                        min_persistence=self.min_persistence,
                    )
                    for simplex_tree in simplex_tree_list
                ]
                for simplex_tree_list in self.simplex_tree_lists_
            ]
        else:
            dgms_lists = [
                [
                    [
                        simplex_tree.persistence(
                            homology_coeff_field=self.homology_coeff_field,
                            min_persistence=self.min_persistence,
                            persistence_dim_max=False,
                        )
                    ]
                    for simplex_tree in simplex_tree_list
                ]
                for simplex_tree_list in self.simplex_tree_lists_
            ]
        dgms_formatted_lists = [
            [
                [
                    self._format_dgm(dgm)
                    for dgm in dgms[:3]  # drop fourth diagram because it is
                    # either empty (in case of extended persistence) or
                    # non-existent (in case of ordinary persistence)
                ]
                for dgms in dgms_list
            ]
            for dgms_list in dgms_lists
        ]
        if (
            not self.use_extended_persistence
            and self.drop_infinite_persistence
        ):
            dgms_formatted_lists = [
                [
                    [
                        [dim[np.isfinite(dim).all(axis=1)] for dim in dgm]
                        for dgm in dgms
                    ]
                    for dgms in dgms_formatted_list
                ]
                for dgms_formatted_list in dgms_formatted_lists
            ]
        return dgms_formatted_lists

    def _get_simplex_tree(
        self,
        time_series: npt.NDArray,
    ) -> gd.SimplexTree:
        """Constructs simplex tree from a single  univariate time series.

        Args:
            time_series (npt.NDArray): Times series given as a NumPy-array of
                shape (n_time_steps, 2), where the last axis contains
                (time, value)-pairs of the time series.

        Returns:
            gudhi.Simplextree: Simplex tree encoding the sublevel set
                filtration of the time series.
        """
        time_series = np.array(time_series, dtype=np.float64)
        st = gd.SimplexTree()
        vertex_array_vertices = np.arange(len(time_series)).reshape([1, -1])
        filtrations_vertices = self._get_vertex_filtrations(time_series)
        st.insert_batch(
            vertex_array=vertex_array_vertices,
            filtrations=filtrations_vertices,
        )
        vertex_array_edges = np.array(
            [[i, i + 1] for i in range(len(time_series) - 1)]
        ).T
        filtrations_edges = np.maximum(
            filtrations_vertices[:-1], filtrations_vertices[1:]
        )
        st.insert_batch(
            vertex_array=vertex_array_edges, filtrations=filtrations_edges
        )
        return st

    def _get_vertex_filtrations(self, time_series):
        if self.filtration_type == "horizontal":
            return time_series[:, 1]
        elif self.filtration_type == "sloped":
            x_range = np.ptp(time_series, axis=0)[1]
            x_min = np.min(time_series[:, 1])
            x_max = np.max(time_series[:, 1])

            def _get_filtration(a):
                # a is array of shape (2,) interpreted as containing t- and
                # x-value of time series
                aux = (a[1] - x_min) / (x_max - x_min)
                return a[0] - ((aux - 0.5) / self.slope)

            return np.apply_along_axis(
                func1d=_get_filtration, axis=1, arr=time_series
            )
        elif self.filtration_type == "sigmoid":
            x_range = np.ptp(time_series, axis=0)[1]
            x_min = np.min(time_series[:, 1]) - self.padding_factor * x_range
            x_max = np.max(time_series[:, 1]) + self.padding_factor * x_range

            def _get_filtration(a):
                # a is array of shape (2,) interpreted as containing t- and
                # x-value of time series
                aux = (a[1] - x_min) / (x_max - x_min)
                return a[0] + (np.log((1 / aux) - 1) / (4 * self.slope))

            return np.apply_along_axis(
                func1d=_get_filtration, axis=1, arr=time_series
            )
        elif self.filtration_type == "arctan":
            x_range = np.ptp(time_series, axis=0)[1]
            x_min = np.min(time_series[:, 1]) - self.padding_factor * x_range
            x_max = np.max(time_series[:, 1]) + self.padding_factor * x_range

            def _get_filtration(a):
                # a is array of shape (2,) interpreted as containing t- and
                # x-value of time series
                aux = (a[1] - x_min) / (x_max - x_min)
                return a[0] - (
                    np.tan(np.pi * (aux - 0.5)) / (np.pi * self.slope)
                )

            return np.apply_along_axis(
                func1d=_get_filtration, axis=1, arr=time_series
            )
        else:
            raise ValueError(
                "Got invalid value for `filtration_type`, must be one of "
                "`'horizontal'`, `'sloped'`, `'sigmoid'` and `'arctan'`."
            )

    def _format_dgm(
        self,
        dgm: list[tuple[int, tuple[float, float]]],
    ) -> list[npt.NDArray]:
        """Helper function to convert a persistence diagram given in the Gudhi
        format into the format suitable for further processing and plotting.

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
            return dgm_formatted[
                np.argsort(
                    np.diff(dgm_formatted, axis=1).reshape(
                        -1,
                    )
                )
            ]

        aux_array = np.array(
            [[dim, *birth_death_pair] for dim, birth_death_pair in dgm]
        ).reshape(-1, 3)
        max_dim = np.max(aux_array[:, 0]) if aux_array.size > 0 else -1
        if max_dim == -1:
            return [np.empty((0, 2), dtype=np.float64)]
        else:
            dims = np.arange(max_dim + 1)
            dgm_formatted = [
                aux_array[aux_array[:, 0] == dim][:, 1:] for dim in dims
            ]
            dgm_formatted = list(map(_sort_by_lifetime, dgm_formatted))
            return dgm_formatted

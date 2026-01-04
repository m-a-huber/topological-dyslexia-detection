import numpy as np
import numpy.typing as npt
import polars as pl


def get_X_y_groups(
    df: pl.DataFrame,
    model_name: str,
) -> tuple[
    list | npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """Given a dataframe produced by `make_dataframes.get_df`, this function
    creates `X`, `y` and `groups` for subsequent passing to
    `StratifiedGroupKFold`.

    Args:
        df (pl.DataFrame): Input dataframe, as created by
            `make_dataframes.get_df`.
        model_name (str): The class of the model. Must be one of
            `'tsh'`, `'baseline_bjornsdottir'` and
            `'baseline_raatikainen'`.

    Returns:
        tuple[
            list | npt.NDArray,
            npt.NDArray,
            npt.NDArray,
        ]:
            Tuple containing `X`, `y` and `groups`.
    """
    if model_name == "tsh":  # time series homology
        X = [
            np.array(time_series, dtype=float)
            for time_series in df["time_series"].to_list()
        ]
    elif model_name == "tsh_aggregated":
        X = [
            [
                np.array(time_series, dtype=float)
                for time_series in time_series_list
            ]
            for time_series_list in df["time_series_list"].to_list()
        ]
    elif model_name == "baseline_bjornsdottir":
        X = (
            df.drop("READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID")
            .to_numpy()
            .astype(float)
        )
    elif model_name == "baseline_raatikainen":
        X = df.drop("READER_ID", "LABEL").to_numpy()
    elif model_name == "baseline_bjornsdottir_with_tsh":
        X_baseline = (
            df.drop(
                "READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID", "time_series"
            )
            .to_numpy()
            .astype(float)
        )
        X_tsh = [
            np.array(time_series, dtype=float)
            for time_series in df["time_series"].to_list()
        ]
        assert len(X_baseline) == len(X_tsh)
        X = [
            (x_baseline, x_tsh) for x_baseline, x_tsh in zip(X_baseline, X_tsh)
        ]
    elif model_name == "baseline_raatikainen_with_tsh_aggregated":
        X_baseline = df.drop(
            "READER_ID", "LABEL", "time_series_list"
        ).to_numpy()
        X_tsh_aggregated = [
            [
                np.array(time_series, dtype=float)
                for time_series in time_series_list
            ]
            for time_series_list in df["time_series_list"].to_list()
        ]
        assert len(X_baseline) == len(X_tsh_aggregated)
        X = [
            (x_baseline, x_tsh_aggregated)
            for x_baseline, x_tsh_aggregated in zip(
                X_baseline, X_tsh_aggregated
            )
        ]
    y = df["LABEL"].to_numpy().astype(int)
    groups = df["READER_ID"].to_numpy().astype(int)
    return X, y, groups

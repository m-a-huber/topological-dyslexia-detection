import numpy as np
import numpy.typing as npt
import polars as pl


def get_X_y_groups(
    df: pl.DataFrame,
    model_class: str,
) -> tuple[list[npt.NDArray] | npt.NDArray, npt.NDArray, npt.NDArray]:
    """Given a dataframe produced by `make_dataframes.make_df`, this function
    creates `X`, `y` and `groups` for subsequent passing to
    `StratifiedGroupKFold`.

    Args:
        df (pl.DataFrame): Input dataframe, as created by
            `make_dataframes.make_df`.
        model_class (str): The class of the model. Must be one of
            `'tda_experiment'`, `'baseline_bjornsdottir'` and
            `'baseline_raatikainen'`.

    Returns:
        tuple[list[npt.NDArray] | npt.NDArray, npt.NDArray, npt.NDArray]:
            Tuple containing `X`, `y` and `groups`.
    """
    if model_class == "tda_experiment":
        X = [
            np.array(time_series, dtype=float)
            for time_series in df["time_series"].to_list()
        ]
    elif model_class == "baseline_bjornsdottir":
        X = (
            df.drop("READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID")
            .to_numpy()
            .astype(float)
        )
    elif model_class == "baseline_raatikainen":
        X = df.drop("READER_ID", "LABEL").to_numpy()
    else:
        raise ValueError(
            "Invalid choice of `model_class`; must be one of "
            "`'tda_experiment'`, `'baseline_bjornsdottir'` and "
            "`'baseline_raatikainen'`."
        )
    y = df["LABEL"].to_numpy().astype(int)
    groups = df["READER_ID"].to_numpy().astype(int)
    return X, y, groups


def _is_sorted(lst):
    # Helper to check if a list is sorted in strictly increasing manner
    return all(lst[i] < lst[i + 1] for i in range(len(lst) - 1))


def _shift_time_series(ts):
    # Helper to shift time series to start at t=0
    assert _is_sorted([t for t, _, _ in ts])
    t_min = ts[0][0]
    return [[t - t_min, x, y] for t, x, y in ts]

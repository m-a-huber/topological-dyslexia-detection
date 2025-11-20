import numpy as np
import numpy.typing as npt
import polars as pl


def get_X_y_groups(
    df: pl.DataFrame,
    model_name: str,
) -> tuple[list[npt.NDArray] | npt.NDArray, npt.NDArray, npt.NDArray]:
    """Given a dataframe produced by `make_dataframes.get_df`, this function
    creates `X`, `y` and `groups` for subsequent passing to
    `StratifiedGroupKFold`.

    Args:
        df (pl.DataFrame): Input dataframe, as created by
            `make_dataframes.get_df`.
        model_name (str): The class of the model. Must be one of
            `'tda_experiment'`, `'baseline_bjornsdottir'` and
            `'baseline_raatikainen'`.

    Returns:
        tuple[list[npt.NDArray] | npt.NDArray, npt.NDArray, npt.NDArray]:
            Tuple containing `X`, `y` and `groups`.
    """
    if model_name == "tda_experiment":
        X = [
            np.array(time_series, dtype=float)
            for time_series in df["time_series"].to_list()
        ]
    elif model_name == "baseline_bjornsdottir":
        X = (
            df.drop("READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID")
            .to_numpy()
            .astype(float)
        )
    elif model_name == "baseline_raatikainen":
        X = df.drop("READER_ID", "LABEL").to_numpy()
    y = df["LABEL"].to_numpy().astype(int)
    groups = df["READER_ID"].to_numpy().astype(int)
    return X, y, groups

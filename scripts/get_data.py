from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from scripts import constants


def _parse_fixation_report(
    fixation_report_path: Path,
) -> pl.DataFrame:
    """Creates a dataframe from a fixation report given as in the directory
    `FixationReports` of CopCo.

    Args:
        fixation_report_path (Path): Path to the fixation report to parse.

    Returns:
        pl.DataFrame: Dataframe containing the data from the fixation report.
    """
    with open(fixation_report_path, "r", encoding="utf-8-sig") as f_in:
        header_line = f_in.readline()
    header = [
        col_name.strip('"') for col_name in header_line.strip().split("\t")
    ]
    df = pl.read_csv(
        fixation_report_path,
        separator="\t",
        skip_lines=1,
        quote_char=None,
        infer_schema=False,
        has_header=False,
        new_columns=header,
    )
    return df


def get_df(
    model_name: str,
    level: str,
    min_n_fixations: int,
    include_l2: bool,
    verbose: bool = True,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Given a `model_name`, returns a `pl.DataFrame` whose first columns
    are `READER_ID`, `LABEL`, and possibly `TRIAL_ID` and `SAMPLE_ID`
    (depending on the aggregation level of the corresponding model). The
    remaining columns contain the input data required for the corresponding
    model. If `model_name` is set to `"tsh"`, fixation sequences of length
    less than `min_n_fixations` will be discarded.

    Args:
        model_name (str): The class of the model. Must be one of
            `'tsh'`, `'baseline_bjornsdottir'`, `'baseline_raatikainen'`,
            `'baseline_bjornsdottir_with_tsh'`, or
            `'baseline_raatikainen_with_tsh'`.
        level (str): The aggregation level of the corresponding model. Must be
            one of `'trial'` and `'reader'`.
        min_n_fixations (int): If greater than 1 and `model_name` is set to
            `"tsh"`, fixation sequences of length less than
            `min_n_fixations` will be discarded. Ignored otherwise.
        include_l2 (bool): Whether or not to include data from CopCo-subjects
            whose native language is not Danish (all of these subjects are
            non-dyslexic).
        verbose (bool, optional): Whether or not to produce verbose output.
            Defaults to True.
        overwrite (bool, optional): Whether or not to overwrite existing output
            files. Defaults to False.

    Returns:
        pl.DataFrame: Output dataframe.
    """
    df_file_name = (
        f"dataframe_{model_name}_{level}_level_min_n_fixations_"
        f"{min_n_fixations}"
    )
    if include_l2:
        df_file_name += "_with_l2"
    else:
        df_file_name += "_without_l2"
    df_out_path = Path(f"./dataframes/{df_file_name}.json")
    subjects = constants.subjects_non_dys_l1 + constants.subjects_dys
    if include_l2:
        subjects += constants.subjects_non_dys_l2
    if not df_out_path.exists() or overwrite:
        if model_name == "tsh":  # time series homology
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = _parse_fixation_report(fixation_report_path)
                # Create column containing reader ID
                df_subject = df_subject.with_columns(
                    pl.lit(subject).alias("READER_ID")
                )
                # Create column containing label
                label = 1 if subject in constants.subjects_dys else 0
                df_subject = df_subject.with_columns(
                    pl.lit(label).alias("LABEL")
                )
                # Create column with sample ID
                df_subject = df_subject.rename(
                    {
                        "TRIAL_INDEX": "TRIAL_ID",
                    }
                )
                df_subject = df_subject.with_columns(
                    pl.concat_str(
                        [
                            "READER_ID",
                            "TRIAL_ID",
                        ],
                        separator="-",
                    ).alias("SAMPLE_ID"),
                )
                # Select relevant columns
                df_subject = df_subject.select(
                    [
                        "READER_ID",
                        "LABEL",
                        "TRIAL_ID",
                        "SAMPLE_ID",
                        "CURRENT_FIX_DURATION",
                        "NEXT_SAC_DURATION",
                        "CURRENT_FIX_X",
                        "CURRENT_FIX_Y",
                    ]
                )
                # Cast columns to correct types
                df_subject = df_subject.with_columns(
                    pl.col("READER_ID").cast(pl.Int64),
                    pl.col("LABEL").cast(pl.Int64),
                    pl.col("TRIAL_ID").cast(pl.Int64),
                    pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
                    pl.col("NEXT_SAC_DURATION")
                    .str.replace(".", "0")
                    .cast(pl.Float64),
                    pl.col("CURRENT_FIX_X")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                    pl.col("CURRENT_FIX_Y")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                )
                # Drop practice trials
                df_subject = df_subject.filter(pl.col("TRIAL_ID") > 10)
                # Create column containing start time of fixation
                df_subject = df_subject.with_columns(
                    pl.concat(
                        [
                            pl.Series([0.0]),
                            (
                                df_subject["CURRENT_FIX_DURATION"]
                                + df_subject["NEXT_SAC_DURATION"]
                            ).cum_sum()[:-1],
                        ]
                    ).alias("CURRENT_FIX_START")
                )
                df_subject = df_subject.rename(
                    {
                        "CURRENT_FIX_START": "current_fix_start",
                        "CURRENT_FIX_X": "current_fix_x",
                        "CURRENT_FIX_Y": "current_fix_y",
                    }
                )
                df_subject = df_subject.drop(
                    "CURRENT_FIX_DURATION",
                    "NEXT_SAC_DURATION",
                )
                subject_dfs.append(df_subject)
            df_all = pl.concat(subject_dfs)
            # Drop samples containing too few fixations
            if min_n_fixations > 1:
                df_all = df_all.filter(
                    pl.count("SAMPLE_ID").over("SAMPLE_ID") >= min_n_fixations
                )
            # Shift each time series to start at t=0
            df_all = df_all.with_columns(
                (
                    pl.col("current_fix_start")
                    - pl.col("current_fix_start").min().over("SAMPLE_ID")
                ).alias("current_fix_start")
            )
            # Combine x-, y- and t-coordinate of fixations into time series
            df_all = (
                df_all.sort("current_fix_start")
                .group_by(["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"])
                .agg(
                    pl.concat_list(
                        ["current_fix_start", "current_fix_x", "current_fix_y"]
                    ).alias("time_series")
                )
            )
            df_all = df_all.sort(
                ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
            )
            if level == "trial":
                df_out = df_all
            elif level == "reader":
                df_out = df_all.group_by(["READER_ID", "LABEL"]).agg(
                    pl.col("time_series").alias("time_series_list")
                )
        elif model_name == "baseline_bjornsdottir":
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects:
                extracted_features_path = Path(
                    f"./data_copco/ExtractedFeatures/P{subject}.csv"
                )
                df_subject = pl.read_csv(extracted_features_path)
                df_subject = df_subject.fill_null(0)
                df_subject = df_subject.with_columns(
                    pl.lit(subject).alias("READER_ID")
                )
                # Set dyslexia label
                label = 1 if subject in constants.subjects_dys else 0
                df_subject = df_subject.with_columns(
                    pl.lit(label).cast(pl.Int64).alias("LABEL")
                )
                subject_dfs.append(df_subject)
            df_all = pl.concat(subject_dfs)
            df_all = df_all.rename(
                {
                    "trialId": "TRIAL_ID",
                }
            )
            df_all = df_all.with_columns(
                pl.concat_str(
                    [
                        "READER_ID",
                        "TRIAL_ID",
                    ],
                    separator="-",
                ).alias("SAMPLE_ID")
            )
            df_all = df_all.drop(
                "word",
                "char_IA_ids",
                "part",
                "wordId",
                "sentenceId",
                "speechId",
                "paragraphId",
                "landing_position",
            )
            df_all = df_all.with_columns(pl.col("READER_ID").cast(pl.Int64))
            # Create aggregated reading measures
            aggs = []
            if level == "trial":
                for col in df_all.columns:
                    if col in ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]:
                        continue
                    aggs.extend(
                        [
                            pl.col(col).mean().alias(f"mean_{col}"),
                            pl.col(col).std().alias(f"std_{col}"),
                            pl.col(col).max().alias(f"max_{col}"),
                        ]
                    )
                df_out = df_all.group_by(
                    ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
                ).agg(aggs)
            elif level == "reader":
                df_all = df_all.drop("TRIAL_ID", "SAMPLE_ID")
                for col in df_all.columns:
                    if col in ["READER_ID", "LABEL"]:
                        continue
                    aggs.extend(
                        [
                            pl.col(col).mean().alias(f"mean_{col}"),
                            pl.col(col).std().alias(f"std_{col}"),
                            pl.col(col).max().alias(f"max_{col}"),
                        ]
                    )
                df_out = df_all.group_by(["READER_ID", "LABEL"]).agg(aggs)
        elif model_name == "baseline_raatikainen":
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = _parse_fixation_report(fixation_report_path)
                # Create column containing reader ID
                df_subject = df_subject.with_columns(
                    pl.lit(subject).alias("READER_ID")
                )
                # Create column containing label
                label = 1 if subject in constants.subjects_dys else 0
                df_subject = df_subject.with_columns(
                    pl.lit(label).alias("LABEL")
                )
                # Create column with sample ID
                df_subject = df_subject.rename(
                    {
                        "TRIAL_INDEX": "TRIAL_ID",
                    }
                )
                df_subject = df_subject.with_columns(
                    pl.concat_str(
                        [
                            "READER_ID",
                            "TRIAL_ID",
                        ],
                        separator="-",
                    ).alias("SAMPLE_ID"),
                )
                # Select relevant columns
                df_subject = df_subject.select(
                    [
                        "READER_ID",
                        "LABEL",
                        "TRIAL_ID",
                        "SAMPLE_ID",
                        "CURRENT_FIX_DURATION",
                        "NEXT_SAC_DURATION",
                        "NEXT_SAC_AMPLITUDE",
                    ]
                )
                # Cast columns to correct types
                df_subject = df_subject.with_columns(
                    pl.col("READER_ID").cast(pl.Int64),
                    pl.col("LABEL").cast(pl.Int64),
                    pl.col("TRIAL_ID").cast(pl.Int64),
                    pl.col("CURRENT_FIX_DURATION").cast(pl.Float64),
                    pl.col("NEXT_SAC_DURATION")
                    .str.replace(".", "0")
                    .cast(pl.Float64),
                    pl.col("NEXT_SAC_AMPLITUDE")
                    .str.replace(".", "0")
                    .str.replace(",", ".")
                    .cast(pl.Float64),
                )
                # Drop practice trials
                df_subject = df_subject.filter(pl.col("TRIAL_ID") > 10)
                subject_dfs.append(df_subject)
            df_all = pl.concat(subject_dfs)
            if level == "trial":
                groups = ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
            elif level == "reader":
                groups = ["READER_ID", "LABEL"]
            df_out = df_all.group_by(groups).agg(
                [
                    pl.col("CURRENT_FIX_DURATION")
                    .mean()
                    .alias("mean_fixation_duration"),
                    pl.col("NEXT_SAC_DURATION")
                    .mean()
                    .alias("mean_saccade_duration"),
                    pl.col("NEXT_SAC_AMPLITUDE")
                    .mean()
                    .alias("mean_saccade_amplitude"),
                    pl.len().cast(pl.Int64).alias("total_fixation_count"),
                ]
            )
        elif model_name in [
            "baseline_bjornsdottir_with_tsh",
            "baseline_raatikainen_with_tsh",
        ]:
            baseline_model_name = "_".join(model_name.split("_")[:2])
            df_baseline = get_df(
                model_name=baseline_model_name,
                level=level,
                min_n_fixations=min_n_fixations,
                include_l2=include_l2,
                verbose=verbose,
                overwrite=overwrite,
            )
            df_tsh = get_df(
                model_name="tsh",
                level=level,
                min_n_fixations=min_n_fixations,
                include_l2=include_l2,
                verbose=verbose,
                overwrite=overwrite,
            )
            if level == "trial":
                df_out = df_baseline.join(
                    df_tsh,
                    on=["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"],
                    how="inner",
                )
            elif level == "reader":
                df_out = df_baseline.join(
                    df_tsh,
                    on=["READER_ID", "LABEL"],
                    how="inner",
                )
        if level == "trial":
            df_out = df_out.sort(["READER_ID", "TRIAL_ID", "SAMPLE_ID"])
        elif level == "reader":
            df_out = df_out.sort(["READER_ID"])
        df_out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.write_json(df_out_path)
        if verbose:
            print(
                f"Saved dataframe for model '{model_name}' to `{df_out_path}`."
            )
    else:
        df_out = pl.read_json(df_out_path)
        if verbose:
            print(
                f"Found dataframe for model '{model_name}' at "
                f"`{df_out_path}`; not overwriting."
            )
    return df_out


def _extract_combined_features(
    df: pl.DataFrame,
    level: str,
) -> list:
    """Extract baseline and TSH features and combine them into tuples."""
    if level == "trial":
        X_baseline = (
            df.drop(
                "READER_ID",
                "LABEL",
                "TRIAL_ID",
                "SAMPLE_ID",
                "time_series",
            )
            .to_numpy()
            .astype(float)
        )
        X_tsh = [
            np.array(time_series, dtype=float)
            for time_series in df["time_series"].to_list()
        ]
    elif level == "reader":
        X_baseline = (
            df.drop("READER_ID", "LABEL", "time_series_list")
            .to_numpy()
            .astype(float)
        )
        X_tsh = [
            [
                np.array(time_series, dtype=float)
                for time_series in time_series_list
            ]
            for time_series_list in df["time_series_list"].to_list()
        ]
    assert len(X_baseline) == len(X_tsh)
    return [
        (x_baseline, x_tsh) for x_baseline, x_tsh in zip(X_baseline, X_tsh)
    ]


def get_X_y_groups(
    df: pl.DataFrame,
    model_name: str,
    level: str,
) -> tuple[
    list | npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """Given a dataframe produced by `get_data.get_df`, this function
    creates `X`, `y` and `groups` for subsequent passing to
    `StratifiedGroupKFold`.

    Args:
        df (pl.DataFrame): Input dataframe, as created by
            `get_data.get_df`.
        model_name (str): The class of the model. Must be one of
            `'tsh'`, `'baseline_bjornsdottir'`, `'baseline_raatikainen'`,
            `'baseline_bjornsdottir_with_tsh'`, or
            `'baseline_raatikainen_with_tsh'`.
        level (str): The aggregation level of the corresponding model. Must be
            one of `'trial'` and `'reader'`.

    Returns:
        tuple[
            list | npt.NDArray,
            npt.NDArray,
            npt.NDArray,
        ]:
            Tuple containing `X`, `y` and `groups`.
    """
    if model_name == "tsh":  # time series homology
        if level == "trial":
            X = [
                np.array(time_series, dtype=float)
                for time_series in df["time_series"].to_list()
            ]
        elif level == "reader":
            X = [
                [
                    np.array(time_series, dtype=float)
                    for time_series in time_series_list
                ]
                for time_series_list in df["time_series_list"].to_list()
            ]
    elif model_name == "baseline_bjornsdottir":
        if level == "trial":
            X = (
                df.drop("READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID")
                .to_numpy()
                .astype(float)
            )
        elif level == "reader":
            X = df.drop("READER_ID", "LABEL").to_numpy().astype(float)
    elif model_name == "baseline_raatikainen":
        if level == "trial":
            X = (
                df.drop("READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID")
                .to_numpy()
                .astype(float)
            )
        elif level == "reader":
            X = df.drop("READER_ID", "LABEL").to_numpy().astype(float)
    elif model_name in [
        "baseline_bjornsdottir_with_tsh",
        "baseline_raatikainen_with_tsh",
    ]:
        X = _extract_combined_features(df, level)
    y = df["LABEL"].to_numpy().astype(int)
    groups = df["READER_ID"].to_numpy().astype(int)
    return X, y, groups

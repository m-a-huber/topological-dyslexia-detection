from pathlib import Path
from typing import Optional

import polars as pl
from sklearn.model_selection import (
    StratifiedKFold,
    train_test_split,
)

from scripts import constants


def parse_fixation_report(
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


def make_df(
    model_class: str,
    min_n_fixations: int,
    include_l2: bool,
    verbose: bool = True,
    overwrite: bool = False,
) -> pl.DataFrame:
    """Given a `model_class`, returns a `pl.DataFrame` whose first four
    columns are `READER_ID`, `LABEL` and , possibly, `TRIAL_ID` and `SAMPLE_ID`
    (depending on the aggregation level of the corresponding model). The
    remaining columns contain the input data required for the corresponding
    model. If `model_class` is set to `"tda_experiment"`, the `min_n_fixations`
    fixation sequences of length less than `min_n_fixations` will be discarded.

    Args:
        model_class (str): Name of the model. Must be one of
            `'tda_experiment'`, `'baseline_bjornsdottir'`,
            `'baseline_raatikainen'`, `'baseline_benfatto'` and
            `'baseline_haller'`.
        min_n_fixations (int): If greater than 1 and `model_class` is set to
            `"tda_experiment"`, fixation sequences of length less than
            `min_n_fixations` will be discarded. Ignored otherwise.
        include_l2 (bool): Whether or not to include data from CopCo-subjects
            whose native language is not Danish (all of these subjects are
            non-dyslexic).
        verbose (bool, optional): Whether or not to produce verbose output.
            Defaults to True.
        overwrite (bool, optional): Whether or not to overwrite existing output
            files. Defaults to False.
        seed (Optional[int], optional): Random seed for reproducibility.
            Defaults to None.

    Returns:
        pl.DataFrame: Output dataframe.
    """
    df_out_path = Path(f"./dataframes/dataframe_{model_class}.json")
    subjects = constants.subjects_non_dys_l1 + constants.subjects_dys
    if include_l2:
        subjects += constants.subjects_non_dys_l2
    if not df_out_path.exists() or overwrite:
        if model_class == "tda_experiment":
            # Create df for all subjects
            subject_dfs = []
            for subject in subjects:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = parse_fixation_report(fixation_report_path)
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
            # Combine x-, y- and t-coordinate of fixations into time series
            df_out = (
                df_all.sort("current_fix_start")
                .group_by(["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"])
                .agg(
                    pl.concat_list(
                        ["current_fix_start", "current_fix_x", "current_fix_y"]
                    ).alias("time_series")
                )
            )
            # Verify that number of rows is correct
            assert len(df_out) == df_all["SAMPLE_ID"].unique().len()
        elif model_class == "baseline_bjornsdottir":
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
            # Downsample dyslexic subjects
            df_non_dys = df_all.filter(pl.col("LABEL") == 0)
            df_dys = df_all.filter(pl.col("LABEL") == 1)
            max_number_of_samples = len(df_dys["SAMPLE_ID"].unique())
            df_non_dys_grouped = df_non_dys.group_by(
                "SAMPLE_ID", maintain_order=True
            )
            df_non_dys_downsampled = pl.concat(
                [
                    df
                    for _, df in list(df_non_dys_grouped)[
                        :max_number_of_samples
                    ]
                ]
            )
            # Combine dataframes
            df_all = pl.concat([df_non_dys_downsampled, df_dys])
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
            assert df_out.shape == df_out.drop_nans().shape
            # Sort dataframe for reproducibility of final result
            df_out = df_out.sort(
                ["READER_ID", "LABEL", "TRIAL_ID", "SAMPLE_ID"]
            )
        elif model_class == "baseline_raatikainen":
            data_dict = {
                "READER_ID": [],
                "LABEL": [],
                "mean_fixation_duration": [],
                "mean_saccade_duration": [],
                "mean_saccade_amplitude": [],
                "total_fixation_count": [],
            }
            for subject in subjects:
                # Parse fixation report
                fixation_report_path = Path(
                    f"./data_copco/FixationReports/FIX_report_P{subject}.txt"
                )
                df_subject = parse_fixation_report(fixation_report_path)
                # Select relevant columns
                df_subject = df_subject.select(
                    [
                        "TRIAL_INDEX",
                        "CURRENT_FIX_DURATION",
                        "NEXT_SAC_DURATION",
                        "NEXT_SAC_AMPLITUDE",
                    ]
                )
                # Cast columns to correct types
                df_subject = df_subject.with_columns(
                    pl.col("TRIAL_INDEX").cast(pl.Int64),
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
                df_subject = df_subject.filter(pl.col("TRIAL_INDEX") > 10)
                # Set dyslexia label
                label = 1 if subject in constants.subjects_dys else 0
                # Set values of data-dictionary
                data_dict["READER_ID"].append(int(subject))
                data_dict["LABEL"].append(label)
                data_dict["mean_fixation_duration"].append(
                    df_subject["CURRENT_FIX_DURATION"].mean()
                )
                data_dict["mean_saccade_duration"].append(
                    df_subject["NEXT_SAC_DURATION"].mean()
                )
                data_dict["mean_saccade_amplitude"].append(
                    df_subject["NEXT_SAC_AMPLITUDE"].mean()
                )
                data_dict["total_fixation_count"].append(len(df_subject))
            df_out = pl.from_dict(data_dict)
        elif model_class == "baseline_benfatto":
            raise NotImplementedError()
        elif model_class == "baseline_haller":
            raise NotImplementedError()
        else:
            raise ValueError(
                "Invalid choice of `model_class`; must be one of "
                "`'tda_experiment'`, `'baseline_bjornsdottir'`, "
                "`'baseline_raatikainen'`, `'baseline_benfatto'` and "
                "`'baseline_haller'`."
            )
        df_out_path.parent.mkdir(parents=True, exist_ok=True)
        df_out.write_json(df_out_path)
        if verbose:
            print(
                f"Saved dataframe for model `'{model_class}'` to "
                f"`{df_out_path}`."
            )
    else:
        df_out = pl.read_json(df_out_path)
        if verbose:
            print(
                f"Found dataframe for model `'{model_class}'` at "
                f"`{df_out_path}`; not overwriting."
            )
    return df_out


def get_dfs_splits(
    df: pl.DataFrame,
    percentage_val_split: float,
    n_splits_train_test: int,
    check_dfs: bool = False,
    random_state: Optional[int] = None,
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Given a dataframe produced by `make_dataframes.make_df`, this function
    first determines a validation split of the dataframe, and then
    splits the portion into `n_splits_train_test` splits. These are such that
    the validation split contains the data from roughly `percentage_val_split`
    percents of all readers, and such that each subsequent split contains
    roughly an equal number of unique reader IDs. Both of these procedures are
    done in a stratified fashion, so that both the validation split as well as
    each of the subsequent splits contain roughly the same proportion of reader
    IDs corresponding to a reader with dyslexia. Moreover, it is such that
    every reader ID appears in precisely one split.

    Args:
        df (pl.DataFrame): Input dataframe.
        percentage_val_split (float): Percentage of reader IDs to appear in the
            validation split.
        n_splits_train_test (int): Number of splits to create from
            non-validation data.
        check_dfs (bool, optional): Whether or not to explicitly verify that
            each reader ID appears in exactly one split. Defaults to False.
        random_state (Optional[int], optional): Random seed for
            reproducibility. Defaults to None.

    Returns:
        tuple[pl.DataFrame, list[pl.DataFrame]]: Tuple containing the dataframe
            corresponding to the validation split, and a list containing the
            dataframes corresponding to each subsequent split.
    """
    # Select relevant sub-dataframe
    df_reader_ids_labels = df.select(
        pl.col("READER_ID"), pl.col("LABEL")
    ).unique(maintain_order=True)
    # Create validation split
    df_aux_train_test, df_aux_val = train_test_split(
        df_reader_ids_labels,
        test_size=percentage_val_split,
        stratify=df_reader_ids_labels["LABEL"],
        random_state=random_state,
    )
    # Create splits of non-validation data
    skf = StratifiedKFold(
        n_splits=n_splits_train_test,
        shuffle=True,
        random_state=random_state,
    )
    dfs_aux_train_test = [
        df_aux_train_test[split_idxs]
        for _, split_idxs in skf.split(
            df_aux_train_test, df_aux_train_test["LABEL"]
        )
    ]
    reader_ids_val = df_aux_val["READER_ID"]
    reader_idss_train_test = [
        df_aux_train_test["READER_ID"]
        for df_aux_train_test in dfs_aux_train_test
    ]
    df_val = df.filter(pl.col("READER_ID").is_in(reader_ids_val))
    dfs_train_test = [
        df.filter(pl.col("READER_ID").is_in(reader_ids_train_test))
        for reader_ids_train_test in reader_idss_train_test
    ]
    if check_dfs:  # Checks that no reader appears in more than one split
        dfs_all = [df_val, *dfs_train_test]
        reader_ids_sets = [set(df["READER_ID"].unique()) for df in dfs_all]
        assert set().union(*reader_ids_sets) == set(
            df["READER_ID"].unique()
        ), "At least one reader ID appears in no split."
        assert len(set().union(*reader_ids_sets)) == sum(
            len(s) for s in reader_ids_sets
        ), "At least one reader ID appears in more than one split."
    return df_val, dfs_train_test
